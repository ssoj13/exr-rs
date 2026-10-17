//! N3 gate: `DecodeLimits` bounds the REAL peak heap of a decode, not just the decoded pixels.
//!
//! The reader charges `planes + one tile + tile_px * sum(TileCodec::work_bytes)` (see
//! `FrameReader::check_budget`). Here, for every codec profile x dtype, a single-tile still is decoded under
//! exactly the limit the reader reports as "needed" and the process-wide peak (`exv_format::bufpool::track`) must not
//! exceed it. A decoder that starts allocating more than its `work_bytes` account fails this test.
//! (Before the accounting, lossy-wave peaked at ~9.5x the counted bytes, wave 3.5x, HTJ2K 3x.)
//!
//! Serialized: the allocator counters are process-global.

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::{Channel, DecodeError, DecodeLimits, Dtype, Header};
use exv_video::{read_still_with, write_still};

#[global_allocator]
static GLOBAL: exv_format::bufpool::track::TrackingAlloc<std::alloc::System> = exv_format::bufpool::track::TrackingAlloc::new(std::alloc::System);

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A single-tile still (`tile == image`: the worst case for the per-tile working set) of two channels of
/// `dtype`, filled with a smooth gradient plus a hash so every codec has real work to do.
fn still(dtype: Dtype, w: u32, h: u32) -> Image {
    let n = w as usize * h as usize;
    let plane = |seed: u32| -> Vec<u8> {
        let mut v = Vec::with_capacity(n * dtype.size());
        for i in 0..n as u32 {
            let (x, y) = (i % w, i / w);
            let s = x.wrapping_mul(3) ^ y.wrapping_mul(5) ^ (i.wrapping_mul(2_654_435_761) >> 27) ^ seed;
            match dtype {
                Dtype::U8 => v.push(s as u8),
                Dtype::U16 | Dtype::F16 | Dtype::Bf16 => {
                    // keep float16 patterns finite and smooth: exponent field 0x3C00.. + 10-bit mantissa
                    let m = (s & 0x3FF) as u16;
                    v.extend_from_slice(&(if dtype == Dtype::U16 { s as u16 } else { 0x3C00 | m }).to_le_bytes());
                }
                Dtype::F32 => v.extend_from_slice(&(1.0f32 + (s & 0xFFFF) as f32 * 0.001).to_le_bytes()),
                Dtype::U32 => v.extend_from_slice(&s.to_le_bytes()),
            }
        }
        v
    };
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: w,
        tile_h: h,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![Channel { name: "a".into(), dtype }, Channel { name: "b".into(), dtype }],
    };
    Image::new(header, vec![plane(1), plane(2)]).expect("well-formed")
}

#[test]
fn limit_means_real_peak() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(exv_format::bufpool::track::is_installed(), "TrackingAlloc is not the global allocator: the gate would pass vacuously");
    let mut profiles = vec![
        ("fast", Profile::lossless_fast()),
        ("fast+max_ratio", Profile::lossless_fast().max_ratio()),
        ("wave", Profile::lossless_wave()),
        ("lossy-wave", Profile::lossy_wave(4)),
        ("exv-rt-dwt", Profile::exv_rt_dwt(16, 2)),
    ];
    #[cfg(feature = "htj2k-backend")]
    profiles.extend([("htj2k", Profile::htj2k()), ("htj2k-lossy", Profile::htj2k_lossy(8))]);

    let mut over = Vec::new();
    for (name, prof) in &profiles {
        for dtype in [Dtype::U8, Dtype::F16, Dtype::F32, Dtype::U32] {
            let img = still(dtype, 512, 512);
            let data = write_still(&img, prof).expect("encode");
            // Raise the budget to what each successive check reports (container: frame bytes, then the frame
            // reader: planes + tile + working set) until the decode is admitted: the final figure is the
            // limit this stack REQUIRES for the file.
            let mut need = 0u64;
            loop {
                match read_still_with(&data, &DecodeLimits::new(need)) {
                    Err(DecodeError::Limit { need: n, .. }) if n > need => need = n,
                    other => {
                        assert!(other.is_ok(), "{name} {dtype:?}: unexpected failure at budget {need}: {:?}", other.map(|i| i.planes.len()));
                        break;
                    }
                }
            }
            // A FRESH thread: jph-rs keeps a per-thread codec pool, and the one this thread built while encoding
            // already holds its storage, which would hide the decoder's own allocation from the peak.
            let (res, peak) = exv_format::bufpool::track::measure(|| {
                std::thread::scope(|sc| sc.spawn(|| read_still_with(&data, &DecodeLimits::new(need))).join().expect("decode thread"))
            });
            assert!(res.is_ok(), "{name} {dtype:?}: decode under its own reported need {need} must succeed: {:?}", res.as_ref().map(|_| ()));
            eprintln!("{name:>15} {dtype:?}: need {need:>10} B, peak {peak:>10} B ({:.2}x)", peak as f64 / need as f64);
            // 1 MiB slack: file-size-proportional bookkeeping (tile index, header) outside the pixel model.
            if peak as u64 > need + (1 << 20) {
                over.push(format!("{name} {dtype:?}: real peak {peak} B exceeds the accounted limit {need} B"));
            }
        }
    }
    assert!(over.is_empty(), "decode peaked above its accounted limit:
{}", over.join("
"));
}

/// Worst-case HTJ2K: incompressible (hash) samples maximise jph-rs's code-block storage, and many channels
/// at small tiles expose its per-component / per-tile constants (the fixed terms of `TileCodec::work`).
/// The smooth gradient of `still` hides both.
#[cfg(feature = "htj2k-backend")]
#[test]
fn htj2k_worst_case_working_set_is_covered() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let noise = |dtype: Dtype, w: u32, h: u32, nch: usize| -> Image {
        let n = w as usize * h as usize;
        let plane = |seed: u32| -> Vec<u8> {
            let mut v = Vec::with_capacity(n * dtype.size());
            for i in 0..n as u32 {
                let s = (i.wrapping_mul(2_654_435_761) >> 20) ^ seed;
                match dtype.size() {
                    1 => v.push(s as u8),
                    2 => v.extend_from_slice(&(s as u16).to_le_bytes()),
                    _ => v.extend_from_slice(&s.to_le_bytes()),
                }
            }
            v
        };
        let header = Header {
            width: w,
            height: h,
            frame_count: 1,
            tile_w: w,
            tile_h: h,
            codec_id: 1,
            codec_params: Vec::new(),
            channels: (0..nch).map(|i| Channel { name: format!("c{i}"), dtype }).collect(),
        };
        Image::new(header, (0..nch).map(|i| plane(i as u32)).collect()).expect("well-formed")
    };
    let mut over = Vec::new();
    for dtype in [Dtype::U8, Dtype::F16, Dtype::F32, Dtype::U32] {
        for &(s, nch) in &[(64u32, 1usize), (64, 16), (64, 43), (256, 1), (256, 16), (512, 4), (1024, 1)] {
            let img = noise(dtype, s, s, nch);
            let data = write_still(&img, &Profile::htj2k()).expect("encode");
            let mut need = 0u64;
            while let Err(DecodeError::Limit { need: n, .. }) = read_still_with(&data, &DecodeLimits::new(need)) {
                assert!(n > need, "limit did not converge");
                need = n;
            }
            let (res, peak) = exv_format::bufpool::track::measure(|| {
                std::thread::scope(|sc| sc.spawn(|| read_still_with(&data, &DecodeLimits::new(need))).join().expect("decode thread"))
            });
            assert!(res.is_ok());
            eprintln!("worst {dtype:?} {s}x{s} nch{nch}: need {need:>10} B, peak {peak:>10} B ({:.2}x)", peak as f64 / need as f64);
            if peak as u64 > need {
                over.push(format!("{dtype:?} {s}x{s} nch{nch}: real peak {peak} B > accounted {need} B"));
            }
        }
    }
    assert!(over.is_empty(), "HTJ2K peaked above its accounted limit:\n{}", over.join("\n"));
}
