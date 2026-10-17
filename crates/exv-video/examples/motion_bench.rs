//! A/B micro-bench of the motion kernels (docs/OPT.md G): the current row-wise `search_global_mv` / `avg_plane`
//! against verbatim copies of the per-pixel code they replaced, on a 1920x1080 F16 plane (search radius 4 = 81
//! candidates; one B-frame mean = one plane). Interleaved runs, median + min of N (the box is shared).
//! `RAYON_NUM_THREADS=1 cargo run --release -p exv-video --example motion_bench` for the single-thread kernel
//! ratio; without it the parallel candidate search is measured too.

use std::time::Instant;

use exv_format::Dtype;
use exv_video::motion::{avg_plane, search_global_mv, shift_plane};

/// `(median, min)` ms of `reps` runs of `f` after one warm-up.
fn med(reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    f();
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    (t[t.len() / 2], t[0])
}

/// The per-pixel search that `search_global_mv` replaced (verbatim: `read()` indexing, serial candidates).
fn search_old(cur: &[u8], prev: &[u8], w: u32, h: u32, dtype: Dtype, search: i32) -> (i32, i32) {
    let esz = dtype.size();
    let (wu, hu) = (w as usize, h as usize);
    let read = |p: &[u8], idx: usize| -> i64 {
        let mut v = 0i64;
        for k in 0..esz {
            v |= (p[idx * esz + k] as i64) << (8 * k);
        }
        v
    };
    let (mut best, mut best_sad, mut best_cost) = ((0i32, 0i32), u64::MAX, i32::MAX);
    for dy in -search..=search {
        for dx in -search..=search {
            let mut sad = 0u64;
            for y in (0..hu).step_by(2) {
                let sy = (y as i64 - dy as i64).clamp(0, hu as i64 - 1) as usize;
                for x in (0..wu).step_by(2) {
                    let sx = (x as i64 - dx as i64).clamp(0, wu as i64 - 1) as usize;
                    sad += (read(cur, y * wu + x) - read(prev, sy * wu + sx)).unsigned_abs();
                }
            }
            let cost = dx.abs() + dy.abs();
            if sad < best_sad || (sad == best_sad && cost < best_cost) {
                (best_sad, best, best_cost) = (sad, (dx, dy), cost);
            }
        }
    }
    best
}

/// The per-element mean that `avg_plane` replaced (verbatim: runtime-width byte loops).
fn avg_old(pa: &[u8], pb: &[u8], esz: usize) -> Vec<u8> {
    let mut out = vec![0u8; pa.len()];
    for ((o, x), y) in out.chunks_exact_mut(esz).zip(pa.chunks_exact(esz)).zip(pb.chunks_exact(esz)) {
        let (mut va, mut vb) = (0u64, 0u64);
        for (k, (&bx, &by)) in x.iter().zip(y).enumerate() {
            va |= (bx as u64) << (8 * k);
            vb |= (by as u64) << (8 * k);
        }
        let avg = (va + vb) >> 1;
        for (k, ob) in o.iter_mut().enumerate() {
            *ob = (avg >> (8 * k)) as u8;
        }
    }
    out
}

fn main() {
    let (w, h) = (1920u32, 1080u32);
    let mut s = 0x1234_5678u32;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    for (dt, esz) in [(Dtype::F16, 2usize), (Dtype::F32, 4)] {
        let prev: Vec<u8> = (0..w as usize * h as usize * esz).map(|i| ((i / esz) as u32 / 3 + rnd() % 8) as u8).collect();
        let cur = shift_plane(&prev, w, h, dt, 2, 1).expect("shift");
        let other: Vec<u8> = prev.iter().map(|b| b.wrapping_add(5)).collect();
        // Same answers first: the A/B is only meaningful between equal functions.
        assert_eq!(search_old(&cur, &prev, w, h, dt, 4), search_global_mv(&cur, &prev, w, h, dt, 4));
        assert_eq!(avg_old(&prev, &other, esz), avg_plane(&prev, &other, esz).expect("avg"));
        let raw = prev.len() as f64 / 1e6;
        for round in 0..2 {
            let a = med(5, || {
                std::hint::black_box(search_old(&cur, &prev, w, h, dt, 4));
            });
            let b = med(5, || {
                std::hint::black_box(search_global_mv(&cur, &prev, w, h, dt, 4));
            });
            println!(
                "{dt:?} search r4 (round {round})  old med {:>8.2} min {:>8.2} ms | new med {:>8.2} min {:>8.2} ms | x{:.1} (min x{:.1})",
                a.0,
                a.1,
                b.0,
                b.1,
                a.0 / b.0,
                a.1 / b.1
            );
            let a = med(21, || drop(std::hint::black_box(avg_old(&prev, &other, esz))));
            let b = med(21, || drop(std::hint::black_box(avg_plane(&prev, &other, esz))));
            println!(
                "{dt:?} avg {raw:.1} MB (round {round})  old med {:>8.3} min {:>8.3} ms | new med {:>8.3} min {:>8.3} ms | x{:.1} (min x{:.1})",
                a.0,
                a.1,
                b.0,
                b.1,
                a.0 / b.0,
                a.1 / b.1
            );
        }
    }
}
