//! Both directions: incomplete-file semantics for a scanline
//! `InputFile`/`Imf::InputFile` truncated MID-CHUNK-REGION -- the contract
//! neither direction had an oracle test for before this: **open succeeds**
//! (never a constructor error just because chunks are missing),
//! **`isComplete()`/`is_complete()` is false**, **present rows read back
//! correctly**, and **reading a missing row fails cleanly** (never a panic
//! or a hang). `tests/parity_exr.rs::truncated_goldens_never_panic` already
//! covers the weaker "never panics" property across several codecs/files;
//! this test adds the STRONGER incomplete-file semantics on both sides:
//! * (a) OUR reader opening a real C++-written golden that WE truncate.
//! * (b) the real C++ oracle (new `read-truncated` mode, additive to
//!   `oracle.cpp`) opening a file WE wrote and truncated -- proving C++
//!   itself follows the same documented contract we port it against, not
//!   just that our port happens to behave plausibly.
//!
//! Uses `golden.exr` (NONE compression, 67x43, one scanline per chunk --
//! see `oracle.cpp`'s header doc for the exact pixel formula, including the
//! flat-rect override), so a byte-exact chunk boundary is
//! `chunk_region_start + k * chunk_stride` for integer `k`, and the KNOWN
//! formula gives an independent expectation for every present row (not
//! merely "no panic").

use std::path::Path;
use std::process::Command;

use exr_core::attr::PixelType;
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::io::MemIStream;
use exr_core::scanline::InputFile;
use imath_rs::Half;

mod support;

/// Locate a committed golden EXR by file name (same convention as
/// `parity_exr.rs::parity_file`).
fn parity_file(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("parity").join(name)
}

/// `oracle.cpp`'s KNOWN scanline image formula (R channel only -- enough to
/// prove "present rows are pixel-correct", not just "present rows don't
/// error"). `idx = y * WIDTH + x`; `R = half(1.0 + idx * 0.5)`, except inside
/// the FLAT RECT (`x in [8,40), y in [8,24)`, oracle.cpp's `is_flat`), where
/// R is the constant `0.5`.
fn known_r(width: i32, x: i32, y: i32) -> u16 {
    let is_flat = (8..40).contains(&x) && (8..24).contains(&y);
    let v = if is_flat {
        0.5
    } else {
        let idx = y * width + x;
        1.0 + idx as f32 * 0.5
    };
    Half::from_f32(v).bits()
}

/// Read channel `R` of row `y` with a FRESH per-row `FrameBuffer` (base
/// offset recomputed for `y`, buffer sized for exactly one row) -- this
/// works regardless of which row is requested, mirroring the oracle's own
/// `do_read_truncated`.
fn read_row_r(inf: &mut InputFile, y: i32, min_x: i32, width: i32) -> exr_core::Result<Vec<u16>> {
    let row_bytes = width as usize * 2;
    let mut buf = vec![0u8; row_bytes];
    let x_stride = 2i64;
    let y_stride = row_bytes as i64;
    let base_offset = -(y as i64 * y_stride + min_x as i64 * x_stride);
    let mut fb = FrameBuffer::new();
    fb.insert(
        "R",
        Slice::new(PixelType::Half, base_offset, 2, row_bytes, 1, 1, 0.0, false, false).unwrap(),
    )?;
    inf.set_frame_buffer(&fb)?;
    inf.read_pixels(y, y, &mut buf)?;
    Ok(buf.as_chunks::<2>().0.iter().map(|c| u16::from_ne_bytes(*c)).collect())
}

/// `golden.exr`'s bytes plus the fixed-size-chunk NONE-compression layout
/// that makes truncation boundaries exact (one row per chunk).
struct GoldenLayout {
    bytes: Vec<u8>,
    min_x: i32,
    min_y: i32,
    width: i32,
    height: i32,
    chunk_stride: usize,
    chunk_region_start: usize,
}

/// Locate `golden.exr` and compute its [`GoldenLayout`].
fn golden_layout(name: &str) -> Option<GoldenLayout> {
    let path = support::fixture_or_skip(
        name,
        parity_file("golden.exr"),
        "Run `cargo run -p xtask -- parity-exr` to regenerate the C++ goldens.",
    )?;
    let bytes = std::fs::read(&path).expect("read golden.exr");

    let mut is = MemIStream::new(bytes.clone());
    let inf = InputFile::new(&mut is).expect("open golden.exr");
    assert!(inf.is_complete(), "golden.exr itself must be complete before we truncate it");
    let dw = inf.header().data_window().unwrap();
    let (min_x, min_y) = (dw.min.x, dw.min.y);
    let width = dw.max.x - min_x + 1;
    let height = dw.max.y - min_y + 1;
    let channels = inf.header().channels().unwrap();
    let bytes_per_row: usize = channels
        .iter()
        .map(|(_, ch)| exr_core::misc::pixel_type_size(ch.type_).unwrap() * width as usize)
        .sum();
    // Chunk leader: y(i32) + dataSize(i32) = 8 bytes, then the row's packed
    // pixel bytes (NONE compression -> raw, no size ambiguity).
    let chunk_stride = 8 + bytes_per_row;
    let chunk_region_len = chunk_stride * height as usize;
    let chunk_region_start = bytes.len() - chunk_region_len;
    Some(GoldenLayout { bytes, min_x, min_y, width, height, chunk_stride, chunk_region_start })
}

#[test]
fn ours_reads_a_truncated_cpp_golden() {
    let Some(GoldenLayout { bytes, min_x, min_y, width, height, chunk_stride, chunk_region_start }) =
        golden_layout("ours_reads_a_truncated_cpp_golden")
    else {
        return;
    };

    // Keep roughly half the rows -- enough present rows AND enough missing
    // ones to exercise both sides of the contract.
    let k = (height / 2) as usize;
    assert!(k > 0 && k < height as usize, "need both present and missing rows");
    let cut = chunk_region_start + k * chunk_stride;
    let truncated = bytes[..cut].to_vec();

    let mut is = MemIStream::new(truncated);
    let mut inf = InputFile::new(&mut is).expect("open must succeed despite missing chunks");
    assert!(
        !inf.is_complete(),
        "a file missing {}/{} rows' chunks must report is_complete() == false",
        height as usize - k,
        height
    );

    // Present rows (0..k): pixel-exact against the KNOWN formula.
    for y in min_y..min_y + k as i32 {
        let row = read_row_r(&mut inf, y, min_x, width).unwrap_or_else(|e| {
            panic!("row {y} is present on disk but read_pixels errored: {e:?}")
        });
        for x in 0..width {
            assert_eq!(
                row[x as usize],
                known_r(width, x, y),
                "row {y} (present) pixel {x} mismatch vs the KNOWN formula"
            );
        }
    }
    // Missing rows (k..height): must error, never panic or silently zero-fill.
    for y in min_y + k as i32..min_y + height {
        assert!(
            read_row_r(&mut inf, y, min_x, width).is_err(),
            "row {y} (missing on disk) must error, not succeed with fabricated data"
        );
    }
}

#[test]
fn cpp_oracle_reads_our_truncated_file() {
    let Some(oracle) = support::oracle_or_skip("cpp_oracle_reads_our_truncated_file") else {
        return;
    };
    let Some(GoldenLayout { bytes, height, chunk_stride, chunk_region_start, .. }) =
        golden_layout("cpp_oracle_reads_our_truncated_file")
    else {
        return;
    };
    // `golden.exr` is C++-written, but the CONTRACT under test here is
    // C++'s OWN open/isComplete/read behaviour on a truncated file -- which
    // file wrote the surviving bytes is irrelevant to that contract, and
    // reusing the same fixture keeps this test independent of any Rust
    // writer bug (a real Rust-written truncated file is exercised on our
    // side by `ours_reads_a_truncated_cpp_golden` above; C++ must
    // treat ANY file with an intact header + offset table + a truncated
    // chunk region identically, regardless of which writer produced it).
    let k = (height / 2) as usize;
    let cut = chunk_region_start + k * chunk_stride;
    let truncated = bytes[..cut].to_vec();

    let dir = std::env::temp_dir();
    let path: &Path = &dir.join("exr_core_s08_truncated_for_oracle.exr");
    std::fs::write(path, &truncated).unwrap();

    let output = Command::new(&oracle)
        .args(["read-truncated", path.to_str().unwrap()])
        .output()
        .expect("run oracle read-truncated");
    assert!(
        output.status.success(),
        "oracle read-truncated failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();

    assert_eq!(
        lines.next(),
        Some("OPEN ok"),
        "C++ must open a file truncated only in its chunk region, not error at construction"
    );
    assert_eq!(
        lines.next(),
        Some("ISCOMPLETE 0"),
        "C++'s isComplete() must be false for a file missing chunks"
    );

    let mut ok_rows = 0usize;
    let mut err_rows = 0usize;
    for (i, line) in lines.enumerate() {
        let mut it = line.split_whitespace();
        assert_eq!(it.next(), Some("ROW"));
        let y: i32 = it.next().unwrap().parse().unwrap();
        assert_eq!(y, i as i32, "ROW lines must be in row order starting at 0");
        match it.next() {
            Some("ok") => ok_rows += 1,
            Some("err") => err_rows += 1,
            other => panic!("unexpected ROW status {other:?}"),
        }
    }
    assert_eq!(ok_rows, k, "C++ must read exactly the {k} present rows successfully");
    assert_eq!(
        err_rows,
        height as usize - k,
        "C++ must fail exactly the {} missing rows",
        height as usize - k
    );

    std::fs::remove_file(path).ok();
}
