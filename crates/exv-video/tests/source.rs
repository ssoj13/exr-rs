//! `ExvSource` (the native `.exv` tile source) through the real container: parity with `read_still` for every
//! codec profile, the zero-copy blob seam, INTRA-only guard, and `exv_tile::read_region` over a real file
//! (ported from the codec's `mem_gate` region tests: full grid == `read_still`, sub-region pixels, a corrupt tile
//! recorded instead of sinking the read).

mod common;

use std::sync::Arc;

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::{Channel, Dtype, Header};
use exv_tile::{Budget, Edge, OnErr, Region, TileBuf, TileError, TilePool, TileSource, read_region};
use exv_video::{ExvSource, VideoIndex, WriteOpts, read_still, still_frame_body, write_still, write_video_meta};

/// Mixed-dtype image whose pixels depend on (x, y, channel); 3x2 tiles of 64 incl. a partial right column.
fn mixed(dts: &[Dtype]) -> Image {
    let (w, h) = (150u32, 100u32);
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 64,
        tile_h: 64,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: dts.iter().enumerate().map(|(i, &dtype)| Channel { name: format!("c{i}"), dtype }).collect(),
    };
    let planes = dts
        .iter()
        .enumerate()
        .map(|(ci, d)| {
            (0..w * h)
                .flat_map(|i| {
                    let v = (i % w) * 3 + (i / w) * 5 + ci as u32 * 11;
                    match d {
                        Dtype::U8 => vec![(v & 0xff) as u8],
                        Dtype::U16 => (((v * 7) & 0xffff) as u16).to_le_bytes().to_vec(),
                        Dtype::F16 => ((v & 0x3ff) as u16).to_le_bytes().to_vec(),
                        Dtype::F32 => (v as f32 * 0.25).to_le_bytes().to_vec(),
                        _ => v.to_le_bytes().to_vec(),
                    }
                })
                .collect()
        })
        .collect();
    Image::new(header, planes).expect("image")
}

fn bytes(v: Vec<u8>) -> exv_video::Bytes {
    Arc::new(v)
}

/// Gather tile `(tx, ty)` of channel `ci` out of a full plane (the reference the source must reproduce).
fn gather(img: &Image, ci: usize, tx: u32, ty: u32) -> Vec<u8> {
    let (x, y, tw, th) = img.header.tile_rect(tx, ty);
    let esz = img.header.channels[ci].dtype.size();
    (0..th)
        .flat_map(|r| {
            let s = ((y + r) as usize * img.header.width as usize + x as usize) * esz;
            img.planes[ci][s..s + tw as usize * esz].to_vec()
        })
        .collect()
}

/// Every profile family x the dtypes it supports: each tile's `read_tile` / `read_chan` equals the plane gather of
/// the whole-image `read_still` (lossy profiles decode deterministically, so they are compared too).
#[test]
fn tiles_equal_read_still_for_every_profile() {
    let all = [Dtype::F16, Dtype::F32, Dtype::U32, Dtype::U16, Dtype::U8];
    let float = [Dtype::F16, Dtype::F32, Dtype::U32];
    let ldr = [Dtype::U8, Dtype::U16];
    let mut cases: Vec<(&str, Profile, &[Dtype])> = vec![
        ("fast", Profile::lossless_fast(), &all),
        ("wave", Profile::lossless_wave(), &all),
        ("lossy-wave", Profile::lossy_wave(4), &float),
        ("exv-rt", Profile::exv_rt_dwt(8, 0), &[Dtype::F16]),
    ];
    #[cfg(feature = "htj2k-backend")]
    cases.extend([
        ("htj2k", Profile::htj2k(), &all as &[Dtype]),
        ("htj2k-lossy", Profile::htj2k_lossy(16), &float),
        ("htj2k-ldr", Profile::htj2k_ldr(34), &ldr),
    ]);
    for (name, prof, dts) in cases {
        let img = mixed(dts);
        let file = write_still(&img, &prof).unwrap_or_else(|e| panic!("{name}: encode {e}"));
        let whole = read_still(&file).unwrap_or_else(|e| panic!("{name}: read_still {e}"));
        let src = ExvSource::open_still(bytes(file)).unwrap_or_else(|e| panic!("{name}: open {e}"));
        let g = src.grid();
        let mut buf = TileBuf::default();
        let mut one = Vec::new();
        for ty in 0..g.tiles_y() {
            for tx in 0..g.tiles_x() {
                src.read_tile(tx, ty, &mut buf).unwrap_or_else(|e| panic!("{name} ({tx},{ty}): {e}"));
                assert_eq!(src.tile_bytes(tx, ty), Ok(buf.bytes()), "{name}: exact reservation cost");
                for ci in 0..dts.len() {
                    assert_eq!(buf.planes[ci], gather(&whole, ci, tx, ty), "{name} tile ({tx},{ty}) ch {ci}");
                    src.read_chan(tx, ty, ci, &mut one).expect("chan");
                    assert_eq!(one, buf.planes[ci], "{name}: read_chan == read_tile plane ({tx},{ty}) ch {ci}");
                }
            }
        }
    }
}

#[test]
fn tile_blob_is_a_zero_copy_slice_of_the_file() {
    let img = mixed(&[Dtype::F16, Dtype::F32]);
    let file = write_still(&img, &Profile::lossless_fast()).expect("encode");
    let data = bytes(file);
    let src = ExvSource::open_still(data.clone()).expect("open");
    let all: &[u8] = (*data).as_ref();
    let range = all.as_ptr_range();
    for ty in 0..src.grid().tiles_y() {
        for tx in 0..src.grid().tiles_x() {
            let b = src.tile_blob(tx, ty).expect("blob");
            assert!(!b.is_empty() && range.contains(&b.as_ptr()) && range.contains(&b.as_ptr().wrapping_add(b.len() - 1)), "blob lies inside the mapped bytes");
        }
    }
    assert!(matches!(src.tile_blob(9, 0), Err(TileError::Oob { .. })));
    assert!(matches!(src.read_tile(0, 9, &mut TileBuf::default()), Err(TileError::Oob { .. })));
}

/// A P frame is not tile-independent: opening it is a typed `NotIntra`, frame 0 (I) opens.
#[test]
fn non_intra_frames_are_refused() {
    let clip = common::clip(3, 16);
    let opts = WriteOpts { gop: 3, ..WriteOpts::default() };
    let file = write_video_meta(&clip, &Profile::lossless_fast(), &opts).expect("encode");
    let idx = VideoIndex::parse(&file).expect("parse");
    let data = bytes(file);
    assert!(ExvSource::open(data.clone(), &idx, 0).is_ok());
    assert_eq!(ExvSource::open(data.clone(), &idx, 1).err(), Some(TileError::NotIntra { frame: 1 }));
    assert!(matches!(ExvSource::open(data, &idx, 7), Err(TileError::Limit(_))));
}

/// `read_region` over the FULL grid equals `read_still`; a sub-region matches the absolute pixels; the pool path
/// returns the same bytes and hits on a re-read.
#[test]
fn region_full_equals_read_still_and_subregion_matches() {
    let img = mixed(&[Dtype::F16, Dtype::F16, Dtype::F32, Dtype::U32]);
    let file = write_still(&img, &Profile::lossless_fast()).expect("encode");
    let full = read_still(&file).expect("read_still");
    let src = ExvSource::open_still(bytes(file)).expect("open");
    let bud = Budget::new(256 << 20);
    let (w, h) = (img.header.width, img.header.height);

    let whole = read_region(&src, None, Region { x: 0, y: 0, w, h }, 0, Edge::Zero, OnErr::Fail, &bud).expect("whole");
    assert_eq!(whole.planes, full.planes, "read_region(full grid) must equal read_still");

    let r = Region { x: 64, y: 30, w: 80, h: 60 };
    let pool = TilePool::new(bud.clone());
    let sub = read_region(&src, Some(&pool), r, 0, Edge::Zero, OnErr::Fail, &bud).expect("sub");
    let ci = 3usize; // u32 channel, distinct per pixel
    for &(lx, ly) in &[(0u32, 0u32), (79, 59), (33, 41)] {
        let a = (ly as usize * r.w as usize + lx as usize) * 4;
        let f = ((r.y + ly) as usize * w as usize + (r.x + lx) as usize) * 4;
        assert_eq!(&sub.planes[ci][a..a + 4], &full.planes[ci][f..f + 4], "region px ({lx},{ly})");
    }
    let again = read_region(&src, Some(&pool), r, 0, Edge::Zero, OnErr::Fail, &bud).expect("again");
    assert_eq!(again.planes, sub.planes);
    let st = pool.stats();
    assert!(st.hits > 0 && st.loads == st.misses, "{st:?}");
}

/// A corrupt channel blob must NOT sink the read: only that (tile, channel) is zero-filled and recorded (per-channel
/// codec), the strict read aborts, and the other channels of the SAME tile survive.
#[test]
fn region_tolerates_a_corrupt_channel() {
    let img = mixed(&[Dtype::F16, Dtype::F32, Dtype::U32]);
    let nch = 3;
    let mut file = write_still(&img, &Profile::lossless_fast()).expect("encode");
    // The frame body starts with tile 0's `[chan_len[nch]:u32][ch0_blob]...`: flip channel 0's blob TAG (the byte
    // past the length table) to an unknown value so only (tile 0, ch 0) fails to decode.
    let off = {
        let (_, body) = still_frame_body(&file).expect("body");
        body.as_ptr() as usize - file.as_ptr() as usize
    };
    file[off + nch * 4] = 0xFF;

    assert!(read_still(&file).is_err(), "strict whole read aborts");
    let src = ExvSource::open_still(bytes(file.clone())).expect("open: index and container still parse");
    let (w, h) = (img.header.width, img.header.height);
    let all = Region { x: 0, y: 0, w, h };
    let bud = Budget::new(256 << 20);
    assert!(matches!(read_region(&src, None, all, 0, Edge::Zero, OnErr::Fail, &bud), Err(TileError::Decode { tx: 0, ty: 0, .. })));
    let out = read_region(&src, None, all, 0, Edge::Zero, OnErr::Record, &bud).expect("lenient");
    assert_eq!(out.failed.len(), 1, "only the corrupted tile/channel fails: {:?}", out.failed);
    assert!(matches!(out.failed[0], (0, 0, Some(0), _)));
    assert_eq!(&out.planes[0][0..2], &[0u8, 0u8], "failed channel region is zero-filled");
    // The intact channels of the corrupt tile equal the pristine image.
    assert_eq!(gather_px(&out.planes[1], w, 4), gather_px(&img.planes[1], w, 4), "ch 1 of tile 0 survives");
}

/// First `n` pixels of the first row of a 4-byte plane.
fn gather_px(p: &[u8], _w: u32, n: usize) -> Vec<u8> {
    p[..n * 4].to_vec()
}
