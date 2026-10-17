//! In-memory image + canonical `decode → RGBA` materialization.
//!
//! A still is a 1-frame `.exv` written/read through `exv-video`. This module holds the shared
//! in-memory [`Image`] (header + one full-size planar buffer per channel) that every layer speaks, plus
//! the ONE source of truth for turning it into interleaved linear RGBA (viewers / compositors / the AE
//! plugin) so consumers don't re-implement channel mapping divergently.

use crate::CodecError;
use exv_format::{Dtype, Header};

/// How a set of channels maps onto the four display slots R,G,B,A: THE single owner of the display channel pick
/// (the CPU RGBA assembly [`Image::to_rgba_f32`], the C ABI's fixed RGBA decode and the viewer's layer maps all
/// derive from [`Self::pick`], so every display path shows an image identically).
///
/// `ch[slot]` is the source channel index (header order) or `None`. Missing colour reads `0.0`, missing alpha
/// `1.0`; `grey` broadcasts slot 0 into R,G,B (slot 3 stays alpha).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RgbaPick {
    pub ch: [Option<usize>; 4],
    pub grey: bool,
}

impl RgbaPick {
    /// The pick over the channels `idx` (header indices), deterministic and never black while a colour candidate
    /// exists:
    /// 1. `R/G/B` by name (case-insensitive), `A` by name = alpha;
    /// 2. else `Y` = grey;
    /// 3. else the channels that are not `A`: one = grey, two or more = the first three in header order as R,G,B.
    pub fn pick(header: &Header, idx: &[usize]) -> Self {
        let mut idx = idx.to_vec();
        idx.sort_unstable(); // header order, independent of the caller's canonical sort
        let name = |i: usize| header.channels.get(i).map_or("", |c| c.name.as_str());
        let find = |want: &str| idx.iter().copied().find(|&i| name(i).eq_ignore_ascii_case(want));
        let (r, g, b, a) = (find("R"), find("G"), find("B"), find("A"));
        if r.is_some() || g.is_some() || b.is_some() {
            return Self { ch: [r, g, b, a], grey: false };
        }
        if let Some(y) = find("Y") {
            return Self { ch: [Some(y), None, None, a], grey: true };
        }
        let col: Vec<usize> = idx.iter().copied().filter(|&i| Some(i) != a).collect();
        match col.as_slice() {
            [] => Self { ch: [None, None, None, a], grey: false },
            [only] => Self { ch: [Some(*only), None, None, a], grey: true },
            [r, g, rest @ ..] => Self { ch: [Some(*r), Some(*g), rest.first().copied(), a], grey: false },
        }
    }

    /// The source channel feeding each of R,G,B,A with the grey broadcast applied (a grey pick feeds slot 0 into
    /// R,G,B) — exactly what a display path samples.
    pub fn slots(&self) -> [Option<usize>; 4] {
        if self.grey { [self.ch[0], self.ch[0], self.ch[0], self.ch[3]] } else { self.ch }
    }
}

/// In-memory image: header + one plane per channel (full-size, row-major).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub header: Header,
    pub planes: Vec<Vec<u8>>,
}

impl Image {
    /// Checked constructor.
    ///
    /// # Errors
    /// [`CodecError::Dims`] when the number of planes != the number of channels or a plane is not `w*h*esz` bytes.
    pub fn new(header: Header, planes: Vec<Vec<u8>>) -> Result<Self, CodecError> {
        if planes.len() != header.channels.len() {
            return Err(CodecError::Dims("plane count != channel count"));
        }
        for (ch, pl) in header.channels.iter().zip(&planes) {
            if header.plane_len(ch.dtype) != Some(pl.len()) {
                return Err(CodecError::Dims("plane length != width*height*element size"));
            }
        }
        Ok(Self { header, planes })
    }

    /// The display routing of ALL this image's channels ([`RgbaPick::pick`] over every channel).
    pub fn rgba_pick(&self) -> RgbaPick {
        let all: Vec<usize> = (0..self.header.channels.len()).collect();
        RgbaPick::pick(&self.header, &all)
    }

    /// The source channel of each R,G,B,A slot with the grey broadcast applied ([`RgbaPick::slots`] of
    /// [`Self::rgba_pick`]): the ONE answer to "which channels does this image display as colour" for the CPU
    /// assembly below and for hosts that route planes themselves (the C ABI's fixed RGBA decode).
    pub fn rgba_slots(&self) -> [Option<usize>; 4] {
        self.rgba_pick().slots()
    }

    /// Assemble the image into an interleaved **linear RGBA f32** buffer (`width*height*4`) — the
    /// canonical `decode → RGBA` for viewers / compositors / the AE plugin. The channel pick is
    /// [`Self::rgba_slots`] (R/G/B/A by name, else `Y` or a lone/first channel as grey, else the first three as
    /// R,G,B). Missing colour → `0.0`, missing alpha → `1.0`. Float dtypes promote losslessly; integer dtypes
    /// normalize to `[0,1]` by their max (so id/mask channels are at least viewable). Returns `(rgba, width, height)`.
    pub fn to_rgba_f32(&self) -> (Vec<f32>, u32, u32) {
        let (w, h) = (self.header.width, self.header.height);
        let n = w as usize * h as usize;
        let slots = self.rgba_slots();
        let s = |ci: usize, i: usize| sample_f32(self.planes.get(ci).map_or(&[][..], Vec::as_slice), self.header.channels[ci].dtype, i);
        let mut out = vec![0.0f32; n * 4];
        for (i, px) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            for (k, (v, absent)) in px.iter_mut().zip([0.0, 0.0, 0.0, 1.0]).enumerate() {
                *v = slots[k].map_or(absent, |ci| s(ci, i));
            }
        }
        (out, w, h)
    }

    /// Assemble into interleaved **RGBA8** (`width*height*4` bytes), clamping the linear RGBA f32 to
    /// `[0,1]` · 255 (round). No tonemapping — an HDR consumer uses [`Self::to_rgba_f32`] + its own curve.
    pub fn to_rgba8(&self) -> (Vec<u8>, u32, u32) {
        let (rgba, w, h) = self.to_rgba_f32();
        let out = rgba.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        (out, w, h)
    }
}

/// IEEE half (f16) bit pattern → f32. Re-export of the ONE shared conversion ([`crate::hdr::half_to_f32`]:
/// exact, NaN payload kept) — `pub` here so the viewer's named-layer/AOV path keeps using it.
pub use crate::hdr::half_to_f32;

/// One sample of `plane` at flat index `i` as an f32 display value (floats promote; ints normalize to
/// `[0,1]` by their max). A short/under-length plane reads `0.0` rather than panicking. `pub` so the
/// viewer's named-layer/AOV path reuses the ONE per-sample decoder instead of a divergent copy.
pub fn sample_f32(plane: &[u8], dtype: Dtype, i: usize) -> f32 {
    let esz = dtype.size();
    let Some(b) = plane.get(i * esz..i * esz + esz) else {
        return 0.0;
    };
    match dtype {
        Dtype::F16 => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
        Dtype::Bf16 => crate::hdr::bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        Dtype::U8 => b[0] as f32 / 255.0,
        Dtype::U16 => u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0,
        Dtype::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 4_294_967_295.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use exv_format::{Channel, Dtype};

    /// G1: `to_rgba_f32` resolves R/G/B/A by name (alpha → 1 when absent), a lone non-RGB channel (or
    /// `Y`) broadcasts to grey, ints normalize; `to_rgba8` clamps.
    #[test]
    fn to_rgba_resolution_and_fallbacks() {
        let hdr = |chs: Vec<Channel>| Header {
            width: 2,
            height: 1,
            frame_count: 1,
            tile_w: 2,
            tile_h: 2,
            codec_id: 1,
            codec_params: Vec::new(),
            channels: chs,
        };
        let f32p = |a: f32, b: f32| a.to_le_bytes().into_iter().chain(b.to_le_bytes()).collect::<Vec<u8>>();
        // R,G,B present, no alpha → alpha 1.0; resolved BY NAME (order-independent).
        let rgb = Image::new(
            hdr(vec![
                Channel { name: "B".into(), dtype: Dtype::F32 },
                Channel { name: "R".into(), dtype: Dtype::F32 },
                Channel { name: "G".into(), dtype: Dtype::F32 },
            ]),
            vec![f32p(0.5, 0.6), f32p(0.1, 0.2), f32p(0.3, 0.4)],
        )
        .unwrap();
        let (px, w, h) = rgb.to_rgba_f32();
        assert_eq!((w, h), (2, 1));
        assert_eq!(&px[0..4], &[0.1, 0.3, 0.5, 1.0]); // R,G,B by name, A=1
        assert_eq!(&px[4..8], &[0.2, 0.4, 0.6, 1.0]);
        // sole non-RGB channel → grey.
        let grey = Image::new(hdr(vec![Channel { name: "Z".into(), dtype: Dtype::F32 }]), vec![f32p(0.7, 0.9)]).unwrap();
        assert_eq!(&grey.to_rgba_f32().0[0..4], &[0.7, 0.7, 0.7, 1.0]);
        // U8 normalize + to_rgba8 clamp (sole non-RGB channel → grey).
        let u = Image::new(hdr(vec![Channel { name: "L".into(), dtype: Dtype::U8 }]), vec![vec![255u8, 0u8]]).unwrap();
        let (b8, ..) = u.to_rgba8();
        assert_eq!(&b8[0..4], &[255, 255, 255, 255]);
        assert_eq!(&b8[4..8], &[0, 0, 0, 255]);
    }

    /// The pick rules in priority order: names, then `Y` grey, then the non-alpha channels (one = grey, 2+ = the first
    /// three as R,G,B); alpha is by name only; subsets (layer groups) are sorted to header order.
    #[test]
    fn rgba_pick_rules() {
        let hdr = |names: &[&str]| Header {
            width: 1,
            height: 1,
            frame_count: 1,
            tile_w: 1,
            tile_h: 1,
            codec_id: 1,
            codec_params: Vec::new(),
            channels: names.iter().map(|n| Channel { name: (*n).into(), dtype: Dtype::F32 }).collect(),
        };
        let slots = |names: &[&str]| {
            let h = hdr(names);
            RgbaPick::pick(&h, &(0..names.len()).collect::<Vec<_>>()).slots()
        };
        assert_eq!(slots(&["B", "a", "R", "G"]), [Some(2), Some(3), Some(0), Some(1)], "names win, case-insensitive");
        assert_eq!(slots(&["R", "Z"]), [Some(0), None, None, None], "a partial RGB is not widened");
        assert_eq!(slots(&["Z", "Y", "A"]), [Some(1), Some(1), Some(1), Some(2)], "Y = grey, alpha kept");
        assert_eq!(slots(&["depth", "A"]), [Some(0), Some(0), Some(0), Some(1)], "lone non-alpha = grey");
        assert_eq!(slots(&["U", "V"]), [Some(0), Some(1), None, None], "two channels = R,G");
        assert_eq!(slots(&["U", "A", "V", "W", "X"]), [Some(0), Some(2), Some(3), Some(1)], "first three non-alpha");
        assert_eq!(slots(&["A"]), [None, None, None, Some(0)], "only alpha: no colour");
        assert_eq!(slots(&[]), [None; 4]);
        let h = hdr(&["U", "V", "W"]);
        assert_eq!(RgbaPick::pick(&h, &[2, 0]).slots(), [Some(0), Some(2), None, None], "subsets use header order");
    }
}
