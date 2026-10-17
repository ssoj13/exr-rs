//! Shared synthetic clips for the exv-video integration tests.

use exv_codec::image::Image;
use exv_format::{Channel, Dtype, Header};

/// 3-channel (F16, F32, U16) 48x32 frame whose content pans with `shift` (so MV search finds real
/// motion and neighbouring frames differ). `tile` = tile edge.
pub fn frame(shift: u32, tile: u32) -> Image {
    let (w, h) = (48u32, 32u32);
    let n = (w * h) as usize;
    let at = |i: usize, s: u32| ((i as u32 % w + s) * 7 + (i as u32 / w) * 3) & 0x3ff;
    let r: Vec<u8> = (0..n).flat_map(|i| (at(i, shift) as u16).to_le_bytes()).collect();
    let z: Vec<u8> = (0..n).flat_map(|i| (at(i, shift) as f32 * 0.25).to_le_bytes()).collect();
    let u: Vec<u8> = (0..n).flat_map(|i| ((at(i, shift) * 61) as u16).to_le_bytes()).collect();
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: tile,
        tile_h: tile,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![
            Channel { name: "R".into(), dtype: Dtype::F16 },
            Channel { name: "Z".into(), dtype: Dtype::F32 },
            Channel { name: "U".into(), dtype: Dtype::U16 },
        ],
    };
    Image::new(header, vec![r, z, u]).expect("img")
}

/// `n` consecutive frames (shift = frame number).
pub fn clip(n: u32, tile: u32) -> Vec<Image> {
    (0..n).map(|s| frame(s, tile)).collect()
}
