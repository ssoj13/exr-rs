//! Allocation profile of the planar read (P2-READ gate stage (a)): decoding a frame into caller-owned
//! planes must not allocate anything frame-sized. A counting global allocator records, inside the
//! measured region, the number of allocations, the total bytes and the LARGEST single allocation;
//! the gate is "no allocation anywhere near a plane" (the decode scratch of one chunk is the floor).
//! Single-threaded measurement (the counters are thread-local-free atomics: run with `--test-threads=1`
//! or accept that other tests may add noise - this file holds ONE test).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use exr_core::api::planar::{read_planes_part, Frame, PlaneDst};
use exr_core::api::{ChannelData, Image};
use exr_core::attr::{Compression, PixelType};
use exr_core::io::{MemIStream, MemOStream};
use exr_core::MultiPartInputFile;
use imath_rs::{Box2i, Half, V2i};

struct Counting;
static ON: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ON.load(Relaxed) != 0 {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size(), Relaxed);
            LARGEST.fetch_max(l.size(), Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ON.load(Relaxed) != 0 {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(n, Relaxed);
            LARGEST.fetch_max(n, Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn measure<R>(f: impl FnOnce() -> R) -> (R, usize, usize, usize) {
    COUNT.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    LARGEST.store(0, Relaxed);
    ON.store(1, Relaxed);
    let r = f();
    ON.store(0, Relaxed);
    (r, COUNT.load(Relaxed), BYTES.load(Relaxed), LARGEST.load(Relaxed))
}

#[test]
fn planar_read_allocates_nothing_plane_sized() {
    let (w, h) = (1024usize, 512usize);
    let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w as i32 - 1, y: h as i32 - 1 } };
    let mk = |f: &dyn Fn(usize) -> f32| -> Vec<Half> { (0..w * h).map(|i| Half::from(f(i))).collect() };
    let img = Image::new(dw)
        .with_channel("R", ChannelData::Half(mk(&|i| (i % 251) as f32 * 0.01)))
        .with_channel("G", ChannelData::Half(mk(&|i| (i / 7 % 311) as f32 * 0.02)))
        .with_channel("B", ChannelData::Half(mk(&|i| (i / 13 % 97) as f32 * 0.03)));
    let plane = w * h * 4;
    for comp in [Compression::NoCompression, Compression::Zip, Compression::Piz] {
        let mut out = MemOStream::new();
        img.write_to(&mut out, comp).expect("write");
        let bytes = out.into_data();
        for threads in [0, 4] {
            let mut bufs: Vec<Vec<u8>> = (0..3).map(|_| vec![0u8; plane]).collect();
            // Warm-up decode (first-use lazies), then the measured one.
            for round in 0..2 {
                let (_, count, total, largest) = measure(|| {
                    let mut is = MemIStream::new(&bytes[..]);
                    let mut mp = MultiPartInputFile::with_num_threads(&mut is, Some(threads)).expect("open");
                    let mut it = bufs.iter_mut();
                    let mut dst: Vec<PlaneDst> = ["R", "G", "B"]
                        .iter()
                        .map(|n| PlaneDst {
                            channel: n,
                            data: it.next().unwrap(),
                            x_stride: 4,
                            y_stride: w * 4,
                            type_: PixelType::Float,
                            fill: 0.0,
                        })
                        .collect();
                    let frame = Frame { origin: V2i { x: 0, y: 0 }, width: w, height: h };
                    read_planes_part(&mut mp, 0, frame, &mut dst).expect("decode");
                });
                if round == 1 {
                    eprintln!(
                        "{comp:?} threads={threads}: {count} allocs, {total} B total, largest {largest} B (plane {plane} B, file {} B)",
                        bytes.len()
                    );
// Stage (a): nothing plane-sized (a quarter of one plane is the ceiling; the per-open                    // codec scratch of the compressors is what is left until ExrReadCtx reuses it).                    assert!(                        largest * 4 <= plane,                        "{comp:?} threads={threads}: a {largest} B allocation on the read path (plane {plane} B)"                    );
                }
            }
        }
    }
}
