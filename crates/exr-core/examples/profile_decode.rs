//! Throwaway decode profiler: how long does ONE full `Image::read`
//! of a real EXR take, in RELEASE?  Answers the SIMD review's first question —
//! was the viewer's "~1s/frame" a debug build or a genuine codec cost?  If a
//! release decode is ~tens of ms, SIMD on the codec is misdirected; if it is
//! ~hundreds of ms, the codec is worth profiling deeper.
//!
//! Run (RELEASE is the point):
//! ```text
//! cargo run -p exr-core --release --example profile_decode -- D:\_demo\EXR\heart\heart.0.exr 50
//! ```
//! arg1 = path, arg2 = iterations (default 50).

use std::process::ExitCode;
use std::time::Instant;

use exr_core::Image;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: profile_decode <file.exr> [iters]");
        return ExitCode::FAILURE;
    };
    let iters: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);

    // Warm once: validates the file, fills OS cache, and reports what we decode.
    let img = match Image::read(&path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (w, h) = (img.width(), img.height());
    let nch = img.channels().count();
    // Uncompressed HALF-equivalent payload (rough throughput denominator).
    let bytes = (w as u64) * (h as u64) * (nch as u64) * 2;
    println!(
        "{path}\n  {w}x{h}, {nch} channels, {:?}; iters={iters}",
        img.compression
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        // Full facade read: container + chunk loop + codec + assemble.
        let _ = Image::read(&path).expect("decode");
    }
    let elapsed = t0.elapsed();

    let per = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let mbps = (bytes as f64 / 1.0e6) / (per / 1000.0);
    println!(
        "  decode: {per:.2} ms/frame  ({mbps:.0} MB/s uncompressed-equivalent)  total {:.2}s",
        elapsed.as_secs_f64()
    );
    ExitCode::SUCCESS
}
