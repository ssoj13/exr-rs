//! The HTJ2K irreversible 9/7 profile (`Profile::htj2k_ldr`) through the real container: a U8+U16 still and a
//! P-frame-less clip come back within the documented +-1 LSB class at a fine q, the file records the `[2, q]`
//! params, LOD works on it, and a float channel is a typed encode error that leaves nothing half-written.

#![cfg(feature = "htj2k-backend")]

use exv_codec::image::Image;
use exv_codec::{CodecError, Profile};
use exv_format::{Channel, CodecId, Dtype, Header};
use exv_video::{VideoError, VideoIndex, read_still, read_video, write_still, write_video};

const W: u32 = 80;
const H: u32 = 56;

/// A smooth ramp plus a per-pixel dither as a U8 or U16 plane; `shift` pans it (a clip's moving content).
fn plane(dt: Dtype, shift: u32) -> Vec<u8> {
    (0..W * H)
        .flat_map(|i| {
            let v = ((i % W + shift) * 255 / W + (i / W) % 4).min(255);
            if dt == Dtype::U8 { vec![v as u8] } else { ((v * 257) as u16).to_le_bytes().to_vec() }
        })
        .collect()
}

fn image(dts: &[Dtype], shift: u32) -> Image {
    let header = Header {
        width: W,
        height: H,
        frame_count: 1,
        tile_w: 32,
        tile_h: 32,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: dts.iter().enumerate().map(|(i, &dtype)| Channel { name: format!("c{i}"), dtype }).collect(),
    };
    Image::new(header, dts.iter().map(|&d| plane(d, shift)).collect()).expect("image")
}

/// Largest absolute sample difference of two images (same channel layout).
fn worst(a: &Image, b: &Image) -> i32 {
    let v = |p: &[u8], dt: Dtype, i: usize| if dt == Dtype::U8 { i32::from(p[i]) } else { i32::from(u16::from_le_bytes([p[2 * i], p[2 * i + 1]])) };
    a.header
        .channels
        .iter()
        .enumerate()
        .flat_map(|(ci, c)| (0..a.planes[ci].len() / c.dtype.size()).map(move |i| (ci, c.dtype, i)))
        .map(|(ci, dt, i)| (v(&a.planes[ci], dt, i) - v(&b.planes[ci], dt, i)).abs())
        .max()
        .unwrap_or(0)
}

#[test]
fn still_roundtrips_within_one_lsb_and_records_params() {
    let img = image(&[Dtype::U8, Dtype::U16, Dtype::U8], 0);
    let file = write_still(&img, &Profile::htj2k_ldr(34)).expect("encode");
    let idx = VideoIndex::parse(&file).expect("parse");
    assert_eq!((idx.header.codec_id, idx.header.codec_params.clone()), (CodecId::HTJ2K.0, vec![2, 34]));
    let back = read_still(&file).expect("decode");
    assert_eq!(back.header.channels, img.header.channels);
    assert!(worst(&img, &back) <= 1, "+-1 LSB class at q=34, got {}", worst(&img, &back));
    // A coarse q is genuinely lossy (not a lossless fallback) and genuinely smaller.
    let coarse = write_still(&img, &Profile::htj2k_ldr(10)).expect("encode");
    assert!(coarse.len() < file.len() / 2, "{} vs {}", coarse.len(), file.len());
    assert!(worst(&img, &read_still(&coarse).expect("decode")) > 1);
}

/// A lossy profile keeps every P-frame tile INTRA/SKIP (open-loop temporal PRED would drift): the clip decodes,
/// every frame within tolerance, and a repeated frame costs (almost) nothing.
#[test]
fn clip_frames_stay_intra_and_decode() {
    let dts = [Dtype::U8, Dtype::U16];
    let frames: Vec<Image> = [0, 3, 3, 6].iter().map(|&s| image(&dts, s)).collect();
    let file = write_video(&frames, &Profile::htj2k_ldr(34), 4, 0, 0).expect("encode");
    let back = read_video(&file).expect("decode");
    assert_eq!(back.len(), frames.len());
    for (a, b) in frames.iter().zip(&back) {
        assert!(worst(a, b) <= 1);
    }
    let single = write_still(&frames[1], &Profile::htj2k_ldr(34)).expect("one").len();
    let two = write_video(&frames[1..3], &Profile::htj2k_ldr(34), 4, 0, 0).expect("two").len();
    assert!(two < single + single / 4, "the repeated frame must be SKIP tiles: {two} vs {single}");
}

#[test]
fn float_or_int32_channel_is_a_typed_error() {
    for dt in [Dtype::F16, Dtype::Bf16, Dtype::F32, Dtype::U32] {
        let mut img = image(&[Dtype::U8], 0);
        img.header.channels.push(Channel { name: "z".into(), dtype: dt });
        img.planes.push(vec![0u8; (W * H) as usize * dt.size()]);
        let err = write_still(&img, &Profile::htj2k_ldr(16)).expect_err("float must be refused");
        assert!(matches!(err, VideoError::Codec(CodecError::Dtype { dtype, .. }) if dtype == dt), "{dt:?}: {err:?}");
    }
}

/// The 9/7 tile keeps J2K resolution scalability (the viewer LOD seam); hdr-code tiles do not have it.
#[test]
fn lod_reads_reduced_levels() {
    use exv_codec::frame::FrameReader;
    let img = image(&[Dtype::U8, Dtype::U16], 0);
    let file = write_still(&img, &Profile::htj2k_ldr(30)).expect("encode");
    let (header, body) = exv_video::still_frame_body(&file).expect("body");
    let reader = FrameReader::open(body, header).expect("open");
    let full = reader.decode_tile_channel(0, 1).expect("full");
    let (l0, rw, rh) = reader.decode_tile_channel_level(0, 1, 0).expect("level 0");
    assert_eq!((l0, rw, rh), (full, 32, 32));
    let (l1, rw, rh) = reader.decode_tile_channel_level(0, 1, 1).expect("level 1");
    assert_eq!((rw, rh, l1.len()), (16, 16, 16 * 16 * 2));

    let hdr = write_still(&image(&[Dtype::U8], 0), &Profile::htj2k_lossy(8)).expect("hdr codes");
    let (header, body) = exv_video::still_frame_body(&hdr).expect("body");
    let reader = FrameReader::open(body, header).expect("open");
    assert!(reader.decode_tile_channel_level(0, 0, 1).is_err(), "hdr-code tiles have no pixel pyramid");
}
