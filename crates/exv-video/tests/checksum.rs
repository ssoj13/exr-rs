//! Tile-blob CRC-32C (FORMAT.md §5): every profile x dtype set round-trips with checksums, a single flipped bit in a
//! tile blob is a typed `DecodeError::Checksum` on every read path (whole frame, per tile, raw GPU blob, clip scan),
//! and the same flip with `Verify::Off` shows what the checksum protects against (wrong pixels or a deeper error).

use exv_codec::Profile;
use exv_codec::frame::FrameReader;
use exv_codec::image::Image;
use exv_format::crc::Mismatch;
use exv_format::{Channel, DecodeError, DecodeLimits, Dtype, Header, Verify};
use exv_video::{BadBlob, Chans, Decoder, decode_frame, ExvSource, VideoIndex, frame_view, read_still_with, verify_checksums, write_still, write_video};

/// A `w x h` image of `dtypes` (smooth + hashed content so every codec emits real payloads), tile edge `tile`.
fn image(dtypes: &[Dtype], w: u32, h: u32, tile: u32, shift: u32) -> Image {
    let n = (w * h) as usize;
    let plane = |ci: usize, d: Dtype| -> Vec<u8> {
        let mut v = Vec::with_capacity(n * d.size());
        for i in 0..n as u32 {
            let s = ((i % w + shift) * 5 + (i / w) * 3 + ci as u32 * 13) ^ (i.wrapping_mul(2_654_435_761) >> 29);
            match d {
                Dtype::U8 => v.push(s as u8),
                Dtype::U16 => v.extend_from_slice(&(s as u16).wrapping_mul(97).to_le_bytes()),
                Dtype::F16 | Dtype::Bf16 => v.extend_from_slice(&(0x3800 | (s & 0x3FF) as u16).to_le_bytes()),
                Dtype::F32 => v.extend_from_slice(&(0.5 + (s & 0xFFF) as f32 * 0.01).to_le_bytes()),
                Dtype::U32 => v.extend_from_slice(&s.wrapping_mul(31).to_le_bytes()),
            }
        }
        v
    };
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: tile,
        tile_h: tile,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: dtypes.iter().enumerate().map(|(i, &dtype)| Channel { name: format!("c{i}"), dtype }).collect(),
    };
    Image::new(header, dtypes.iter().enumerate().map(|(i, &d)| plane(i, d)).collect()).expect("image")
}

/// `(name, profile, dtypes)`: every profile on the dtypes it codes (Irrev97 = U8/U16 only, EXV-RT's codec = F16).
fn cases() -> Vec<(&'static str, Profile, Vec<Dtype>)> {
    let mixed = vec![Dtype::F16, Dtype::U8, Dtype::U16, Dtype::F32, Dtype::U32, Dtype::Bf16];
    let mut v = vec![
        ("fast", Profile::lossless_fast(), mixed.clone()),
        ("fast-maxratio", Profile::lossless_fast().max_ratio(), mixed.clone()),
        ("wave", Profile::lossless_wave(), mixed.clone()),
        ("lossy-wave", Profile::lossy_wave(3), mixed.clone()),
        ("exv-rt-dwt", Profile::exv_rt_dwt(16, 2), vec![Dtype::F16, Dtype::F16, Dtype::U32]),
        ("exv-rt-lerp", Profile::exv_rt(8, 4, true), vec![Dtype::F16, Dtype::U16]),
    ];
    #[cfg(feature = "htj2k-backend")]
    v.extend([
        ("htj2k", Profile::htj2k(), mixed.clone()),
        ("htj2k-u8u16", Profile::htj2k(), vec![Dtype::U8, Dtype::U16]),
        ("htj2k-hdr", Profile::htj2k_lossy(8), mixed.clone()),
        ("htj2k-irrev97", Profile::htj2k_ldr(16), vec![Dtype::U8, Dtype::U16, Dtype::U8]),
    ]);
    v
}

/// Byte offset of tile `t`'s blob (frame `di` in decode order) inside the file, and its length.
fn blob_at(file: &[u8], di: usize, t: u32) -> (usize, usize) {
    let idx = VideoIndex::parse(file).expect("index");
    let body = frame_view(file, &idx, di).expect("frame").body;
    let blob = FrameReader::open(body, idx.header.clone()).expect("reader").tile_blob(t).expect("INTRA blob");
    (blob.as_ptr() as usize - file.as_ptr() as usize, blob.len())
}

/// Every profile: the clean file decodes (bit-exact when lossless) with no mismatch; ONE flipped bit in the middle of
/// tile 1's blob is `Checksum` on the whole-frame, per-tile and raw-blob paths and in the clip scan, naming the damaged
/// channel for a per-channel codec (whose other channels of that tile still decode) and no channel for a
/// multi-component HTJ2K tile; `Verify::Off` returns wrong pixels or a codec error (never the original image).
#[test]
fn bit_flip_in_a_tile_blob_is_a_typed_checksum_error() {
    let on = DecodeLimits::default();
    let off = DecodeLimits::default().with_verify(Verify::Off);
    for (name, prof, dtypes) in cases() {
        let img = image(&dtypes, 40, 24, 16, 0); // 3x2 tiles, edge tiles on both axes
        let file = write_still(&img, &prof).expect("encode");
        let clean = read_still_with(&file, &on).unwrap_or_else(|e| panic!("{name}: clean decode: {e}"));
        if prof.is_lossless() {
            assert_eq!(clean.planes, img.planes, "{name}: lossless round trip");
        }
        let idx = VideoIndex::parse(&file).expect("index");
        assert_eq!(verify_checksums(&file, &idx, 8), Ok(vec![]), "{name}: clean file has no mismatch");

        let (at, len) = blob_at(&file, 0, 1);
        let hit = at + len / 2;
        // The damaged channel: per-channel codecs checksum each sub-blob, HTJ2K the whole tile (`None`).
        let chan: Option<u16> = {
            let body = frame_view(&file, &idx, 0).expect("frame").body;
            let rd = FrameReader::open(body, idx.header.clone()).expect("reader");
            rd.tile_chans(1).ok().map(|subs| {
                let start = |b: &[u8]| b.as_ptr() as usize - file.as_ptr() as usize;
                let pos = subs.iter().position(|b| (start(b)..start(b) + b.len()).contains(&hit));
                pos.expect("the middle byte lies in a channel sub-blob") as u16
            })
        };
        let mut bad = file.clone();
        bad[hit] ^= 0x10;
        let is_tile1 = |e: &DecodeError| matches!(e, DecodeError::Checksum(m) if m.tile == 1 && m.chan == chan && m.expected != m.got);
        let e = read_still_with(&bad, &on).expect_err("verified decode must fail");
        assert!(is_tile1(&e), "{name}: whole frame: {e:?}");
        let idx = VideoIndex::parse(&bad).expect("index structure is intact");
        let found = verify_checksums(&bad, &idx, 8).expect("scan");
        assert!(matches!(found.as_slice(), [BadBlob { frame: 0, tile: 1, err }] if is_tile1(err)), "{name}: scan {found:?}");
        let src = ExvSource::open_still(std::sync::Arc::new(bad.clone())).expect("source");
        assert!(src.tile_blob(1, 0).is_err() && src.tile_blob(0, 0).is_ok(), "{name}: raw GPU blob path verifies per tile");
        let body = frame_view(&bad, &idx, 0).expect("frame").body;
        let rd = FrameReader::open(body, idx.header.clone()).expect("reader");
        assert!(is_tile1(&rd.decode_tile(1, &prof, None).expect_err("tile 1")), "{name}: per-tile decode");
        assert!(rd.decode_tile(0, &prof, None).is_ok(), "{name}: an intact tile still decodes");
        if let Some(c) = chan {
            for ci in (0..dtypes.len()).filter(|&ci| ci != c as usize) {
                assert!(rd.tile_channel_blob(1, ci).is_ok(), "{name}: ch {ci} sub-blob of the damaged tile is intact");
                assert!(rd.decode_tile_channel(1, ci).is_ok(), "{name}: ch {ci} of the damaged tile decodes");
            }
            assert!(is_tile1(&rd.decode_tile_channel(1, c as usize).expect_err("damaged channel")), "{name}: damaged channel");
        }

        // The contrast: unverified, the flip is either silently wrong pixels or an error from deep inside the codec.
        match read_still_with(&bad, &off) {
            Ok(got) => {
                assert_ne!(got.planes, clean.planes, "{name}: Verify::Off decoded the flipped blob to the clean image");
                eprintln!("{name}: Verify::Off -> Ok with WRONG pixels (silent corruption)");
            }
            Err(e) => {
                assert!(!matches!(e, DecodeError::Checksum(_)), "{name}: Verify::Off must not check CRCs: {e:?}");
                eprintln!("{name}: Verify::Off -> {e}");
            }
        }
    }
}

/// A clip: the flip in a later frame is reported at that frame's DISPLAY index by the scan and stops exactly that
/// frame's decode, and the scan stops at `max`.
#[test]
fn clip_scan_reports_frame_and_tile() {
    let prof = Profile::lossless_fast();
    let clip: Vec<Image> = (0..4).map(|s| image(&[Dtype::F16, Dtype::F32], 40, 24, 16, s * 9)).collect();
    let file = write_video(&clip, &prof, 2, 0, 0).expect("I/P clip"); // display 0 I, 1 P, 2 I, 3 P
    let idx = VideoIndex::parse(&file).expect("index");
    let di = idx.decode_idx(2).expect("frame 2");
    let (at, len) = blob_at(&file, di, 3);
    let mut bad = file.clone();
    bad[at + len / 2] ^= 0x01;
    let found = verify_checksums(&bad, &idx, 8).expect("scan");
    assert!(matches!(found.as_slice(), [BadBlob { frame: 2, tile: 3, err: DecodeError::Checksum(Mismatch { tile: 3, .. }) }]), "{found:?}");
    assert!(decode_frame(&bad, &idx, 1, Chans::All).is_ok(), "the GOP before the damage decodes");
    for k in [2, 3] {
        let r = decode_frame(&bad, &idx, k, Chans::All).map(|_| ());
        assert!(matches!(r, Err(DecodeError::Checksum(Mismatch { tile: 3, .. }))), "frame {k}: {r:?}");
    }
    assert!(Decoder::open(&bad).expect("decoder").frame(0).is_ok(), "the stateful decoder fails only the damaged GOP");
    // Damage everywhere -> the scan stops at `max`.
    let mut worse = bad.clone();
    for t in [0, 1] {
        let (a, l) = blob_at(&worse, 0, t);
        worse[a + l / 2] ^= 0x01;
    }
    assert_eq!(verify_checksums(&worse, &idx, 2).expect("scan").len(), 2);
}

/// Lenient region reads keep the checksum's granularity: a per-channel codec loses only the damaged (tile, channel), a
/// multi-component HTJ2K tile (one blob for every channel) fails as a whole tile, recorded once.
#[test]
fn lenient_region_records_damage_at_blob_granularity() {
    use exv_tile::{Budget, Edge, OnErr, Region, read_region};
    let dtypes = [Dtype::F16, Dtype::F32, Dtype::U32];
    let mut profs = vec![("fast", Profile::lossless_fast())];
    #[cfg(feature = "htj2k-backend")]
    profs.push(("htj2k", Profile::htj2k()));
    for (name, prof) in profs {
        let img = image(&dtypes, 40, 24, 16, 0);
        let file = write_still(&img, &prof).expect("encode");
        let (at, len) = blob_at(&file, 0, 0);
        let mut bad = file.clone();
        bad[at + len - 1] ^= 0x40; // the last byte: the last channel of a per-channel tile, the codestream tail of HTJ2K
        let src = ExvSource::open_still(std::sync::Arc::new(bad)).expect("open");
        let all = Region { x: 0, y: 0, w: 40, h: 24 };
        let out = read_region(&src, None, all, 0, Edge::Zero, OnErr::Record, &Budget::new(64 << 20)).expect("lenient");
        let failed: Vec<_> = out.failed.iter().map(|f| (f.0, f.1, f.2)).collect();
        if prof.backend().layout() == exv_codec::registry::TileLayout::MultiComp {
            assert_eq!(failed, vec![(0, 0, None)], "{name}: the whole tile, once");
        } else {
            assert_eq!(failed, vec![(0, 0, Some(dtypes.len() - 1))], "{name}: only the damaged channel");
        }
    }
}
