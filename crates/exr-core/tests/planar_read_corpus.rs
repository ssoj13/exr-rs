//! The planar read (`InputFile::read_pixels_planes`) against the single-buffer read over the REAL
//! image set: every single-part scanline file of `openexr-images` and OpenEXR's `src/test` that has
//! only 1x1-sampled channels is decoded twice, once into one interleaved FLOAT buffer
//! (`read_pixels`, the long-established path) and once into one FLOAT plane per channel
//! (`read_pixels_planes`), at 0 and 4 threads. Every plane sample must equal the interleaved one.
//! Covers every codec the corpus holds (NONE, RLE, ZIPS, ZIP, PIZ, PXR24, B44/B44A, DWAA/DWAB,
//! HTJ2K), both line orders, odd widths and off-origin data windows, HALF and FLOAT and UINT
//! channels (all converted to FLOAT at unpack). Needs the image set: `--features test-data`.

use std::path::{Path, PathBuf};

use exr_core::attr::PixelType;
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::scanline::InputFile;
use exr_core::StdIfStream;

fn collect_exr(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_exr(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("exr") {
            out.push(p);
        }
    }
}

fn slice(base: i64, x_stride: usize, y_stride: usize) -> Slice {
    Slice {
        type_: PixelType::Float,
        base_offset: base,
        x_stride,
        y_stride,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    }
}

/// `Some(true)` = compared and equal, `Some(false)` never returned (asserts), `None` = not a file
/// this gate covers (tiled / deep / multipart / subsampled / unreadable header / too big).
fn check(path: &Path, threads: i32) -> Option<()> {
    if std::fs::metadata(path).ok()?.len() > 40 << 20 {
        return None;
    }
    let mut is = StdIfStream::open(path.to_str()?).ok()?;
    let mut inf = InputFile::with_num_threads(&mut is, Some(threads)).ok()?;
    let hdr = inf.header().clone();
    if hdr.has_tile_description() || (hdr.has_type() && hdr.type_().ok()? != "scanlineimage") {
        return None;
    }
    let dw = hdr.data_window().ok()?;
    let chans = hdr.channels().ok()?;
    let names: Vec<String> = chans.iter().map(|(n, _)| n.to_string()).collect();
    if chans.iter().any(|(_, c)| c.x_sampling != 1 || c.y_sampling != 1) || names.is_empty() {
        return None;
    }
    let (w, h) = ((dw.max.x - dw.min.x + 1) as usize, (dw.max.y - dw.min.y + 1) as usize);
    let n = names.len();

    // Interleaved reference: n FLOAT samples per pixel in one buffer.
    let (px, row) = (4 * n, 4 * n * w);
    let origin = dw.min.x as i64 * px as i64 + dw.min.y as i64 * row as i64;
    let mut fb = FrameBuffer::new();
    for (i, name) in names.iter().enumerate() {
        fb.insert(name.as_str(), slice(i as i64 * 4 - origin, px, row)).ok()?;
    }
    let mut inter = vec![0u8; row * h];
    inf.set_frame_buffer(&fb).ok()?;
    inf.read_pixels(dw.min.y, dw.max.y, &mut inter).ok()?;

    // Planar: one FLOAT plane per channel, each its own allocation.
    let (px1, row1) = (4usize, 4 * w);
    let origin1 = dw.min.x as i64 * px1 as i64 + dw.min.y as i64 * row1 as i64;
    let mut fbp = FrameBuffer::new();
    for (i, name) in names.iter().enumerate() {
        fbp.insert_in(name.as_str(), slice(-origin1, px1, row1), i).ok()?;
    }
    let mut planes: Vec<Vec<u8>> = (0..n).map(|_| vec![0xAAu8; row1 * h]).collect();
    let mut is2 = StdIfStream::open(path.to_str()?).ok()?;
    let mut inf2 = InputFile::with_num_threads(&mut is2, Some(threads)).ok()?;
    inf2.set_frame_buffer(&fbp).ok()?;
    let mut refs: Vec<&mut [u8]> = planes.iter_mut().map(|p| &mut p[..]).collect();
    inf2.read_pixels_planes(dw.min.y, dw.max.y, &mut refs)
        .unwrap_or_else(|e| panic!("{}: planar read failed ({threads} threads): {e}", path.display()));

    for y in 0..h {
        for x in 0..w {
            for (i, plane) in planes.iter().enumerate() {
                let a = &inter[y * row + x * px + i * 4..][..4];
                let b = &plane[y * row1 + x * px1..][..4];
                assert_eq!(a, b, "{}: channel {} ({x},{y}) threads={threads}", path.display(), names[i]);
            }
        }
    }
    Some(())
}

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn planar_read_equals_single_buffer_read_over_the_corpus() {
    let mut files = Vec::new();
    collect_exr(&exr_test_data::openexr_images(), &mut files);
    collect_exr(&exr_test_data::openexr_test(), &mut files);
    files.sort();
    let mut compared = 0usize;
    for f in &files {
        for threads in [0, 4] {
            if check(f, threads).is_some() {
                compared += 1;
            }
        }
    }
    assert!(compared >= 100, "only {compared} (file, threads) pairs compared of {} files", files.len());
    eprintln!("planar == single-buffer on {compared} (file, threads) pairs of {} files", files.len());
}
