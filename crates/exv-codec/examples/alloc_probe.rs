//! Allocation churn per tile op (docs/OPT.md): total bytes requested by ONE `encode_tile` / `decode_tile_into` of a
//! 256x256x5ch tile (raw 1.0 MB), as a multiple of the raw bytes. Run:
//! `cargo run --release -p exv-codec --example alloc_probe`
//! The counters are process-global and every call here runs single-threaded (`RAYON_NUM_THREADS=1`).
//!
//! Two data sets: `noisy` (low-bit noise: the codecs mostly fall back to RAW, so it measures the fixed per-call
//! overhead) and `smooth` (gradients: the codecs really compress, like real plates). `dec` decodes into a warm
//! recycled [`TileBuf`] right after an ENCODE on the same thread (the pooled HTJ2K codestream switches direction);
//! `dec+` is the allocating `decode_tile` front (a fresh buffer per tile); `dec2` = a decode right after a decode
//! (the steady state of a decode-only run). `frame.*` rows: whole-frame `decode_all_into` on ALL rayon threads into
//! reused planes, bytes allocated per frame (3rd frame, every worker's pools warm).

use exv_codec::{Profile, TileBuf, decode_tile, decode_tile_into, encode_tile};
use exv_format::Dtype;
use exv_format::bufpool::track::{TrackingAlloc, total_allocated};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// Allocation COUNT (the byte totals come from `exv_format::bufpool::track`): the pinned bufpool revision has no counter.
static COUNT: AtomicUsize = AtomicUsize::new(0);

/// `TrackingAlloc` + a call counter.
struct Counting(TrackingAlloc<System>);

// SAFETY: forwards every call unchanged to the inner allocator (only adds a relaxed counter bump).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        unsafe { self.0.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { self.0.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        unsafe { self.0.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting(TrackingAlloc::new(System));

fn allocations() -> usize {
    COUNT.load(Relaxed)
}

/// The 5-channel tile: `noise` LSBs of xorshift noise on top of a smooth ramp.
fn planes(n: usize, noise: u32) -> Vec<Vec<u8>> {
    let mut s = 0x9e37_79b9u32;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    [2usize, 2, 2, 4, 4]
        .iter()
        .map(|&esz| (0..n * esz).map(|i| ((i / esz) as u32 / 7 + if noise > 0 { rnd() % noise } else { 0 }) as u8).collect())
        .collect()
}

/// Bytes allocated by the 3rd `decode_all_into` of a 2048^2 frame (256^2 tiles, all rayon threads) into the same
/// reused [`Planes`]: the steady-state per-frame churn of playback.
fn frames(dts: &[Dtype], profiles: &[(&str, Profile)]) {
    use exv_codec::frame::{Chans, FrameReader, Planes, write_frame_tiles};
    use exv_format::{Channel, DecodeLimits, Header};
    let s = 2048u32;
    let pl: Vec<Vec<u8>> = planes((s * s) as usize, 16);
    let raw: usize = pl.iter().map(Vec::len).sum();
    for (name, p) in profiles {
        let hd = Header {
            width: s,
            height: s,
            frame_count: 1,
            tile_w: 256,
            tile_h: 256,
            codec_id: p.codec_id().0,
            codec_params: p.params(),
            channels: dts.iter().enumerate().map(|(i, &dtype)| Channel { name: format!("c{i}"), dtype }).collect(),
        };
        let body = write_frame_tiles(&pl, None, &hd, p).expect("encode frame");
        let rd = FrameReader::open(&body, hd).expect("open").with_limits(DecodeLimits::new(16 << 30));
        let mut out = Planes::default();
        for _ in 0..2 {
            rd.decode_all_into(p, None, Chans::All, &mut out).expect("decode");
        }
        let (a0, n0) = (total_allocated(), allocations());
        rd.decode_all_into(p, None, Chans::All, &mut out).expect("decode");
        let (a1, n1) = (total_allocated(), allocations());
        println!(
            "frame  {name:<14} decode_all_into {:>6.3}x raw per frame ({} KiB, {} allocs)",
            (a1 - a0) as f64 / raw as f64,
            (a1 - a0) >> 10,
            n1 - n0
        );
    }
    // Per-TILE transient of the whole-frame path (what one decode lane holds beside the planes): a one-tile 256^2 x 5ch
    // frame decoded into already-fitted planes on a FRESH one-thread pool (cold thread-locals, like a lane's first
    // tile) and again on the same pool (warm: the steady state). `cost` = what `FrameReader` charges per lane.
    let pl: Vec<Vec<u8>> = planes(256 * 256, 16);
    for (name, p) in profiles {
        let hd = Header {
            width: 256,
            height: 256,
            frame_count: 1,
            tile_w: 256,
            tile_h: 256,
            codec_id: p.codec_id().0,
            codec_params: p.params(),
            channels: dts.iter().enumerate().map(|(i, &dtype)| Channel { name: format!("c{i}"), dtype }).collect(),
        };
        let body = write_frame_tiles(&pl, None, &hd, p).expect("encode tile frame");
        let rd = FrameReader::open(&body, hd).expect("open").with_limits(DecodeLimits::new(16 << 30));
        let mut out = Planes::default();
        rd.decode_all_into(p, None, Chans::All, &mut out).expect("fit");
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
        let (_, cold) = exv_format::bufpool::track::measure(|| one.install(|| rd.decode_all_into(p, None, Chans::All, &mut out).expect("cold")));
        let (_, warm) = exv_format::bufpool::track::measure(|| one.install(|| rd.decode_all_into(p, None, Chans::All, &mut out).expect("warm")));
        println!("tile   {name:<14} lane peak cold {:>6} KiB warm {:>6} KiB", cold >> 10, warm >> 10);
    }
}

fn main() {
    let n = 256 * 256usize;
    let dts = [Dtype::F16, Dtype::F16, Dtype::F16, Dtype::F32, Dtype::U32];
    let profiles = [
        ("htj2k", Profile::htj2k()),
        ("fast", Profile::lossless_fast()),
        ("wave", Profile::lossless_wave()),
        ("lossy-wave4", Profile::lossy_wave(4)),
        ("rt", Profile::exv_rt_dwt(8, 0)),
        ("lossy-htj2k16", Profile::htj2k_lossy(16)),
    ];
    frames(&dts, &profiles);
    for (set, noise) in [("noisy", 16u32), ("smooth", 0)] {
        let pl = planes(n, noise);
        let chans: Vec<&[u8]> = pl.iter().map(|p| p.as_slice()).collect();
        let raw: usize = pl.iter().map(|p| p.len()).sum();
        for (name, p) in &profiles {
            let mut buf = TileBuf::default();
            let blob = encode_tile(&chans, &dts, 256, 256, p).expect("encode"); // warm thread-local pools
            decode_tile_into(&blob, &dts, 256, 256, p, &mut buf).expect("decode");
            let a0 = total_allocated();
            let blob = encode_tile(&chans, &dts, 256, 256, p).expect("encode");
            let a1 = total_allocated();
            decode_tile_into(&blob, &dts, 256, 256, p, &mut buf).expect("decode");
            let a2 = total_allocated();
            let _ = decode_tile(&blob, &dts, 256, 256, p).expect("decode");
            let a3 = total_allocated();
            let n3 = allocations();
            decode_tile_into(&blob, &dts, 256, 256, p, &mut buf).expect("decode");
            let a4 = total_allocated();
            let n4 = allocations();
            let x = |b: usize| b as f64 / raw as f64;
            println!(
                "{set:<6} {name:<14} enc {:>5.2}x  dec {:>5.2}x  dec+ {:>5.2}x  dec2 {:>5.2}x ({} allocs)   (blob {:.2}x smaller)",
                x(a1 - a0),
                x(a2 - a1),
                x(a3 - a2),
                x(a4 - a3),
                n4 - n3,
                x(raw) / x(blob.len())
            );
        }
    }
}
