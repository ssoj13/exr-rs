//! Hostile-file regression net over the WHOLE read stack (container -> frame reader -> every codec).
//!
//! Valid still/clip files of every codec profile are mutated byte by byte (and truncated); each mutant is
//! pushed through the decode entry points on a worker thread with a hard deadline and a heap high-water mark
//! (`exv_format::bufpool::track`). The contract under test (FORMAT.md section 8): a corrupt file is a typed error or a decode
//! of garbage - NEVER a panic, a hang, or an allocation beyond what the caller's `DecodeLimits` account for.
//! It is the harness that found the HTJ2K sub-sampling hang (N1) and the code-block-size allocation (N2).
//!
//! The default run samples (fast enough for every `cargo test`); `EXV_FUZZ_FULL=1 cargo test -p exv-video
//! --test hostile_files -- --include-ignored` sweeps every byte of the structural regions.
//! Also here: the small-budget forged-dimensions gate (N4).

use std::panic::AssertUnwindSafe;
use std::sync::mpsc;
use std::time::Duration;

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::{Channel, DecodeError, DecodeLimits, Dtype, Header, Verify};
use exv_video::{Decoder, VideoIndex, frame_body, read_still_with, read_video_with, verify_checksums, write_still, write_video};

#[global_allocator]
static GLOBAL: exv_format::bufpool::track::TrackingAlloc<std::alloc::System> = exv_format::bufpool::track::TrackingAlloc::new(std::alloc::System);

/// The allocator counters are process-global: one measured test at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Budget handed to every decode; a mutant may legitimately use up to ~this much (the frames are tiny, so in
/// practice only forged geometry gets near it).
const LIMIT: u64 = 64 << 20;
/// Heap ceiling for one mutant: the budget plus slack for working sets that scale with the file (index,
/// clip frames), still far below what an unaccounted allocation (e.g. jph-rs's 358 MiB) would take.
const HEAP_CAP: usize = 256 << 20;
/// Per-mutant deadline; a legit decode of these files takes milliseconds, a spin takes forever.
const DEADLINE: Duration = Duration::from_secs(60);

/// A `w x h` image of `dtypes`, deterministic smooth+hashed content, tile edge `tile`.
fn image(dtypes: &[Dtype], w: u32, h: u32, tile: u32, shift: u32) -> Image {
    let n = (w * h) as usize;
    let plane = |ci: usize, d: Dtype| -> Vec<u8> {
        let mut v = Vec::with_capacity(n * d.size());
        for i in 0..n as u32 {
            let s = ((i % w + shift) * 7 + (i / w) * 3 + (ci as u32) * 11) ^ (i.wrapping_mul(2_654_435_761) >> 28);
            match d {
                Dtype::U8 => v.push(s as u8),
                Dtype::U16 => v.extend_from_slice(&(s as u16).wrapping_mul(61).to_le_bytes()),
                Dtype::F16 | Dtype::Bf16 => v.extend_from_slice(&(0x3C00 | (s & 0x3FF) as u16).to_le_bytes()),
                Dtype::F32 => v.extend_from_slice(&(1.0 + (s & 0xFFF) as f32 * 0.01).to_le_bytes()),
                Dtype::U32 => v.extend_from_slice(&s.wrapping_mul(97).to_le_bytes()),
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
    Image::new(header, dtypes.iter().enumerate().map(|(i, &d)| plane(i, d)).collect()).expect("well-formed image")
}

/// `(name, file bytes)` corpus: every codec profile x {mixed-dtype still, F16 still (EXV-RT's dtype), I/P clip,
/// I/P/B clip}; `full == false` keeps the two representative files per profile (mixed still, F16 I/P clip).
fn corpus(full: bool) -> Vec<(String, Vec<u8>)> {
    let mixed = [Dtype::F16, Dtype::U8, Dtype::U16, Dtype::F32, Dtype::U32, Dtype::Bf16];
    let half = [Dtype::F16, Dtype::F16];
    let mut profiles = vec![
        ("fast", Profile::lossless_fast()),
        ("fast-maxratio", Profile::lossless_fast().max_ratio()),
        ("wave", Profile::lossless_wave()),
        ("lossy-wave", Profile::lossy_wave(4)),
        ("exv-rt-dwt", Profile::exv_rt_dwt(16, 2)),
        ("exv-rt-lerp", Profile::exv_rt(8, 4, true)),
    ];
    #[cfg(feature = "htj2k-backend")]
    profiles.extend([("htj2k", Profile::htj2k()), ("htj2k-lossy", Profile::htj2k_lossy(8))]);
    let mut out = Vec::new();
    for (name, prof) in &profiles {
        for (set_name, set) in [("mixed", &mixed[..]), ("half", &half[..])] {
            if full || set_name == "mixed" {
                let still = image(set, 40, 24, 16, 0);
                out.push((format!("{name}/{set_name}/still"), write_still(&still, prof).expect("still")));
            }
            if full || set_name == "half" {
                let clip: Vec<Image> = (0..4).map(|s| image(set, 40, 24, 16, s)).collect();
                out.push((format!("{name}/{set_name}/ip"), write_video(&clip, prof, 2, 2, 0).expect("I/P clip")));
            }
            if full {
                let bclip: Vec<Image> = (0..6).map(|s| image(set, 40, 24, 16, s)).collect();
                out.push((format!("{name}/{set_name}/ipb"), write_video(&bclip, prof, 5, 2, 2).expect("I/P/B clip")));
            }
        }
    }
    out
}

/// Every decode entry point a host uses, under the test budget. Results are discarded on purpose: the
/// assertion is that this RETURNS (no panic / hang / over-allocation), not what it returns.
///
/// The decodes run with checksums OFF: with them on, a mutated tile blob stops at its CRC and the codecs would never
/// see hostile bytes (they still must survive them: `Verify::Off` hosts, a forged file with matching CRCs). The
/// verifying paths (`verify_checksums`, a CRC-checked still decode) run once on top.
fn battery(data: &[u8]) {
    let limits = DecodeLimits::new(LIMIT).with_verify(Verify::Off);
    if let Ok(idx) = VideoIndex::parse(data) {
        drop(verify_checksums(data, &idx, 4));
    }
    drop(read_still_with(data, &DecodeLimits::new(LIMIT)));
    drop(read_still_with(data, &limits));
    drop(read_video_with(data, &limits));
    if let Ok(mut d) = Decoder::open(data).map(|d| d.with_limits(limits)) {
        for k in 0..d.frame_count().min(8) {
            drop(d.frame(k));
        }
    }
}

/// Run [`battery`] on `data` in a worker thread; returns the heap high-water mark, or how it failed
/// (hang / panic / over the heap cap).
fn run_case(data: Vec<u8>) -> Result<usize, &'static str> {
    let (res, peak) = exv_format::bufpool::track::measure(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let ok = std::panic::catch_unwind(AssertUnwindSafe(|| battery(&data))).is_ok();
            // the receiver may have timed out and gone away
            tx.send(ok).ok();
        });
        match rx.recv_timeout(DEADLINE) {
            Ok(true) => Ok(()),
            Ok(false) => Err("panicked"),
            Err(mpsc::RecvTimeoutError::Timeout) => Err("hung (deadline exceeded)"),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err("worker died"),
        }
    });
    res?;
    if peak > HEAP_CAP { Err("heap peak above the cap") } else { Ok(peak) }
}

/// Byte values tried at every sampled offset (`full`: six, else the three that matter most: 0, 0xFF, flip).
fn variants(b: u8, full: bool) -> Vec<u8> {
    if full { vec![0x00, 0x01, 0x7F, 0x80, 0xFF, b ^ 0xFF] } else { vec![0x00, 0xFF, b ^ 0xFF] }
}

/// Offsets to mutate in a file of `len` bytes: the structural head (prelude + header + first frame header) and
/// tail (index + trailer) densely, the middle (tile blobs) by a deterministic pseudo-random sample.
fn offsets(len: usize, head: usize, tail: usize, sample: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..head.min(len)).chain(len.saturating_sub(tail)..len).collect();
    let mut x = seed | 1;
    for _ in 0..sample {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17; // xorshift64: deterministic, no rand dependency
        v.push((x % len as u64) as usize);
    }
    v
}

fn sweep(full: bool, head: usize, tail: usize, sample: usize, blob: usize) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(exv_format::bufpool::track::is_installed(), "TrackingAlloc is not the global allocator: the heap gate would pass vacuously");
    let mut failures = Vec::new();
    let (mut cases, mut worst) = (0usize, 0usize);
    for (name, file) in corpus(full) {
        let mut run = |what: String, data: Vec<u8>| {
            cases += 1;
            match run_case(data) {
                Ok(peak) => worst = worst.max(peak),
                Err(why) => failures.push(format!("{name}: {what}: {why}")),
            }
        };
        // The first frame body starts with its tile blobs: mutate their heads densely too - that is where a codec
        // stream header (HTJ2K SIZ/COD/QCD, EXV-RT block grid, rANS tables) lives, the N1/N2 class.
        let body_off = VideoIndex::parse(&file)
            .ok()
            .and_then(|idx| frame_body(&file, &idx, 0))
            .map_or(0, |(_, body)| body.as_ptr() as usize - file.as_ptr() as usize);
        let mut at_list = offsets(file.len(), head, tail, sample, 0x9E37_79B9_7F4A_7C15 ^ file.len() as u64);
        let dense = at_list.len();
        at_list.extend(body_off..(body_off + blob).min(file.len()));
        for (n, at) in at_list.into_iter().enumerate() {
            for v in variants(file[at], full) {
                let mut bad = file.clone();
                bad[at] = v;
                run(format!("byte {at} = {v:#04x}"), bad);
            }
            // truncation at the same offset (dense head/tail only: the sample would just repeat it)
            if n < head.min(file.len()) + tail.min(file.len()) || n >= dense {
                run(format!("truncated at {at}"), file[..at].to_vec());
            }
        }
    }
    eprintln!("hostile_files: {cases} mutants, worst heap peak {} KiB", worst >> 10);
    assert!(failures.is_empty(), "{} of {cases} mutants broke the contract:
{}", failures.len(), failures.join("
"));
}

/// Sampled sweep: part of every `cargo test`.
#[test]
fn mutated_files_never_hang_panic_or_overallocate() {
    sweep(false, 32, 48, 16, 150);
}

/// Dense sweep over every structural byte and a large body sample (minutes).
#[test]
#[ignore = "extended fuzz (minutes); run with --include-ignored"]
fn mutated_files_extended() {
    sweep(true, 160, 192, 400, 320);
}

/// N4: a tiny file whose header FORGES a huge image must be refused by a small budget with a typed `Limit`,
/// before any plane is allocated - not after committing (and zero-filling) a gigabyte for a 3 KB file.
#[test]
fn forged_huge_image_is_refused_by_a_small_budget() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(exv_format::bufpool::track::is_installed());
    let img = image(&[Dtype::U8], 8, 8, 8, 0);
    let mut file = write_still(&img, &Profile::lossless_fast()).expect("still");
    assert!(file.len() < 3 << 10, "the forged file stays tiny");
    // prelude = MAGIC(4) | version(1) | gop(4) | frame_count(4) = 13 bytes, then the Header (FORMAT.md 2.1):
    // width@2 height@6 tile_w@14 tile_h@18 -> one 32768-square tile (1 GiB of U8).
    for (off, v) in [(2u32, 32768u32), (6, 32768), (14, 32768), (18, 32768)] {
        file[13 + off as usize..13 + off as usize + 4].copy_from_slice(&v.to_le_bytes());
    }
    let (res, peak) = exv_format::bufpool::track::measure(|| read_still_with(&file, &DecodeLimits::new(64 << 20)));
    assert!(matches!(res, Err(DecodeError::Limit { need, max }) if need >= 1 << 30 && max == 64 << 20), "{:?}", res.map(|i| i.planes.len()));
    assert!(peak < 1 << 20, "a refused decode must not have allocated planes ({peak} B)");
}
