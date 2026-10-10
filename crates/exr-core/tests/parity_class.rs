//! Pixel-read error CLASS parity against real C++ OpenEXR (`oracle.exe read-class`).
//!
//! For every `.exr` of the vendored C++ test tree, the official image set and the parity
//! goldens, each reader path the OpenEXR tests use (`aces` / `rgba` as in
//! `testMalformedImages.cpp`, plain `input`, `tiled`, `deepscan`, `deeptile`) is run through
//! C++ and through ours in the same steps; the outcome (`OK`, `SKIP`, or the Iex class that
//! the read ended in) must be identical. The message is printed but not compared.
//!
//! Gated `#[ignore]`: needs the oracle (`EXR_ORACLE_EXE`, or `tests/parity/oracle.exe`) and
//! runs from `cargo run -p xtask -- parity-exr`. The summary line is
//! `pixel-class diff: N reads compared over M files ...`.

use std::path::{Path, PathBuf};
use std::process::Command;

use exr_core::attr::PixelType;
use exr_core::deep::{DeepFrameBuffer, DeepScanLineInputFile, DeepTiledInputFile};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::io::StdIfStream;
use exr_core::rgba::{AcesInputFile, Rgba, RgbaInputFile};
use exr_core::scanline::InputFile;
use exr_core::tiled::TiledInputFile;
use exr_core::header::Header;
use exr_core::{ExrError, IexClass, Result};
use imath_rs::Box2i;

const APIS: [&str; 6] = ["aces", "rgba", "input", "tiled", "deepscan", "deeptile"];
/// Must match `RC_MAX_PIXELS` / `RC_MAX_SAMPLES` in `oracle.cpp`.
const MAX_PIXELS: i64 = 1 << 24;
const MAX_SAMPLES: i64 = 1 << 26;

/// The Iex class name `oracle.cpp::exc_class` would print for this error.
fn class_of(e: &ExrError) -> &'static str {
    match e.iex_class() {
        IexClass::Input => "Input",
        IexClass::Io => "Io",
        IexClass::Arg => "Arg",
        IexClass::Logic => "Logic",
        IexClass::Type => "Type",
        IexClass::Errno => "Errno",
        IexClass::Math | IexClass::Overflow | IexClass::Underflow | IexClass::DivZero => "Math",
        IexClass::Base => "Base",
    }
}

/// `w`,`h` of the window, or `None` when it is empty or too big to allocate for the check.
fn window(dw: Box2i) -> Option<(i64, i64)> {
    let w = i64::from(dw.max.x) - i64::from(dw.min.x) + 1;
    let h = i64::from(dw.max.y) - i64::from(dw.min.y) + 1;
    (w > 0 && h > 0 && w * h <= MAX_PIXELS).then_some((w, h))
}

/// The header of `path` as the file type `api` opens it (opening errors propagate).
fn header_of(path: &str, api: &str) -> Result<Header> {
    let mut s = StdIfStream::open(path)?;
    Ok(match api {
        "input" => InputFile::new(&mut s)?.header().clone(),
        "tiled" => TiledInputFile::new(&mut s)?.header().clone(),
        "deepscan" => DeepScanLineInputFile::new(&mut s)?.header().clone(),
        _ => DeepTiledInputFile::new(&mut s)?.header().clone(),
    })
}

/// Our side of `oracle.cpp::rc_read`; `Ok(false)` = SKIP.
fn read_ours(path: &str, api: &str) -> Result<bool> {
    let mut s = StdIfStream::open(path)?;
    match api {
        "aces" => {
            let mut f = AcesInputFile::new(&mut s)?;
            let dw = f.data_window()?;
            let Some((w, h)) = window(dw) else { return Ok(false) };
            let mut p = vec![Rgba::default(); (w * h) as usize];
            f.read_pixels(dw.min.y, dw.max.y, &mut p, 1, w as isize)?;
        }
        "rgba" => {
            let mut f = RgbaInputFile::new(&mut s)?;
            let dw = f.data_window()?;
            let Some((w, h)) = window(dw) else { return Ok(false) };
            let mut p = vec![Rgba::default(); (w * h) as usize];
            f.read_pixels(dw.min.y, dw.max.y, &mut p, 1, w as isize)?;
        }
        "input" | "tiled" => {
            let hdr = header_of(path, api)?;
            let dw = hdr.data_window()?;
            let Some((w, h)) = window(dw) else { return Ok(false) };
            let nch = hdr.channels()?.iter().count() as i64;
            if w * h * nch > MAX_SAMPLES {
                return Ok(false);
            }
            let plane = w * h * 4;
            let mut fb = FrameBuffer::new();
            for (c, (name, ch)) in hdr.channels()?.iter().enumerate() {
                let base = c as i64 * plane - 4 * i64::from(dw.min.x) - 4 * i64::from(dw.min.y) * w;
                let slice =
                    Slice::new(PixelType::Float, base, 4, 4 * w as usize, ch.x_sampling, ch.y_sampling, 0.0, false, false)?;
                fb.insert(name.clone(), slice)?;
            }
            let mut out = vec![0u8; (plane * nch) as usize];
            if api == "input" {
                let mut f = InputFile::new(&mut s)?;
                f.set_frame_buffer(&fb)?;
                f.read_pixels(dw.min.y, dw.max.y, &mut out)?;
            } else {
                let mut f = TiledInputFile::new(&mut s)?;
                f.set_frame_buffer(&fb)?;
                let (nx, ny) = (f.num_x_tiles(0)?, f.num_y_tiles(0)?);
                f.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut out)?;
            }
        }
        _ => {
            let hdr = header_of(path, api)?;
            let dw = hdr.data_window()?;
            let Some((w, h)) = window(dw) else { return Ok(false) };
            let mut fb = DeepFrameBuffer::new();
            let base = -4 * i64::from(dw.min.x) - 4 * i64::from(dw.min.y) * w;
            fb.insert_sample_count_slice(Slice::new(PixelType::Uint, base, 4, 4 * w as usize, 1, 1, 0.0, false, false)?)?;
            let mut counts = vec![0u8; (w * h * 4) as usize];
            if api == "deepscan" {
                let mut f = DeepScanLineInputFile::new(&mut s)?;
                f.set_frame_buffer(&fb)?;
                f.read_pixel_sample_counts(dw.min.y, dw.max.y, &mut counts)?;
            } else {
                let mut f = DeepTiledInputFile::new(&mut s)?;
                f.set_frame_buffer(&fb)?;
                let (nx, ny) = (f.num_x_tiles(0)?, f.num_y_tiles(0)?);
                f.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, 0, 0, &mut counts)?;
            }
        }
    }
    Ok(true)
}

fn ours(path: &str, api: &str) -> String {
    match std::panic::catch_unwind(|| read_ours(path, api)) {
        Ok(Ok(true)) => "OK".into(),
        Ok(Ok(false)) => "SKIP".into(),
        Ok(Err(e)) => format!("{} {e}", class_of(&e)),
        Err(_) => "PANIC".into(),
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|x| x == "exr") {
            out.push(p);
        }
    }
}

/// Goldens whose truncations / corruptions are added to the corpus: every storage kind and a
/// spread of codecs, so the chunk-payload read seams of all four readers are exercised.
/// (file, byte offset of the first chunk's payload-size field from the chunk start, its width):
/// the field the `bigsize` mutant inflates so the payload READ, not the header, is what fails.
const MUTANT_BASES: [(&str, usize, usize); 9] = [
    ("golden.exr", 4, 4),
    ("golden_zip.exr", 4, 4),
    ("golden_piz.exr", 4, 4),
    ("golden_dwab.exr", 4, 4),
    ("golden_htj2k256.exr", 4, 4),
    ("golden_tiled_zip_one.exr", 16, 4),
    ("golden_tiled_piz_one.exr", 16, 4),
    ("golden_deep_scanline_zips.exr", 12, 8),
    ("golden_deep_tiled_zips_one.exr", 24, 8),
];

/// Truncated and byte-flipped copies of `bytes`, named after what was done. The flips hit
/// the middle of the chunk data (well past the header) so a decoder, not the header parser,
/// sees them.
fn mutants(bytes: &[u8], size_at: usize, size_len: usize) -> Vec<(String, Vec<u8>)> {
    let n = bytes.len();
    let mut out = Vec::new();
    for k in 1..=9usize {
        let cut = n * k / 10;
        out.push((format!("cut{k}of10"), bytes[..cut].to_vec()));
    }
    for d in [1usize, 4, 17] {
        out.push((format!("cut_end-{d}"), bytes[..n - d].to_vec()));
    }
    for (i, at) in [n / 2, n * 3 / 4, n - n / 8].into_iter().enumerate() {
        let mut b = bytes.to_vec();
        for x in &mut b[at..(at + 24).min(n)] {
            *x ^= 0xA5;
        }
        out.push((format!("flip{i}"), b));
    }
    // `bigsize`: the first chunk claims far more payload than the file holds.
    if let Some(v) = first_chunk(bytes) {
        let mut b = bytes.to_vec();
        let big = (n as u64 + 1000).to_le_bytes();
        b[v + size_at..v + size_at + size_len].copy_from_slice(&big[..size_len]);
        out.push(("bigsize".into(), b));
    }
    out
}

/// Offset of the first chunk: the value of the first entry of the chunk offset table, found by
/// looking for the table start `t` where `u64 at t == v` and `v - t` is a whole table.
fn first_chunk(bytes: &[u8]) -> Option<usize> {
    let u = |i: usize| bytes.get(i..i + 8).map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")));
    (8..bytes.len().saturating_sub(8)).find_map(|t| {
        let v = u(t)? as usize;
        (v > t && (v - t).is_multiple_of(8) && v - t <= 8 * 4096 && v < bytes.len() && (t..v).step_by(8).all(|i| u(i).is_some_and(|e| e as usize >= v))).then_some(v)
    })
}

#[derive(Default)]
struct Tally {
    n: usize,
    bad: Vec<String>,
    classes: std::collections::BTreeMap<String, usize>,
}

impl Tally {
    /// Runs every API on `path` through C++ and ours and records any class difference.
    fn compare(&mut self, exe: &Path, path: &Path, label: &str) {
        let key = |s: &str| s.split(' ').next().unwrap_or("").to_string();
        let path = path.to_str().expect("utf-8 path");
        for api in APIS {
            let out = Command::new(exe).args(["read-class", path, api]).output().expect("spawn oracle");
            let theirs = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let mine = ours(path, api);
            if label.ends_with(":bigsize") && api != "aces" && api != "rgba" {
                eprintln!("  payload-size mutant {label} [{api}]: C++ {} | ours {}", key(&theirs), key(&mine));
            }
            *self.classes.entry(key(&theirs)).or_default() += 1;
            self.n += 1;
            if theirs.is_empty() || key(&theirs) != key(&mine) {
                self.bad.push(format!("{label} [{api}]\n    C++ : {theirs}\n    ours: {mine}"));
            }
        }
    }
}

#[test]
#[ignore = "needs oracle.exe: run `cargo run -p xtask -- parity-exr`"]
fn pixel_read_class_matches_cpp() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::var_os("EXR_ORACLE_EXE")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("tests/parity/oracle.exe"));
    assert!(exe.is_file(), "oracle not found at {}", exe.display());

    let mut files = Vec::new();
    collect(&manifest.join("tests/parity"), &mut files);
    collect(&exr_test_data::openexr_test(), &mut files);
    collect(&exr_test_data::openexr_images(), &mut files);
    if let Ok(p) = std::env::var("EXR_CORPUS_DIR") {
        collect(Path::new(&p), &mut files);
    }
    files.sort();
    files.dedup();

    let mut t = Tally::default();
    // Dev knob: only the synthetic mutants (the corpus pass is the slow part in a debug build).
    let corpus_len = if std::env::var_os("EXR_CLASS_SYNTH_ONLY").is_some() { 0 } else { files.len() };
    for f in &files[..corpus_len] {
        let label = f.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        t.compare(&exe, f, &label);
    }
    let tmp = std::env::temp_dir().join(format!("exr_class_mutants_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    for (base, size_at, size_len) in MUTANT_BASES {
        let bytes = std::fs::read(manifest.join("tests/parity").join(base)).expect("golden");
        for (what, data) in mutants(&bytes, size_at, size_len) {
            let p = tmp.join(format!("{}.{what}.exr", base.trim_end_matches(".exr")));
            std::fs::write(&p, data).expect("write mutant");
            t.compare(&exe, &p, &format!("{base}:{what}"));
            std::fs::remove_file(&p).expect("remove mutant this test created");
        }
    }
    std::fs::remove_dir(&tmp).expect("remove temp dir this test created");

    eprintln!(
        "pixel-class diff: {} reads compared over {corpus_len} corpus files + {} synthetic mutants, {} differ; \
         C++ outcomes {:?}",
        t.n,
        MUTANT_BASES.len() * 16,
        t.bad.len(),
        t.classes
    );
    for b in &t.bad {
        eprintln!("  {b}");
    }
    assert!(t.bad.is_empty(), "{} reads ended in a different class than C++", t.bad.len());
}
