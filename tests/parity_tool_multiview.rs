//! `exr-rs multiview` against OpenEXR's `exrmultiview` (v3.4.15): exit code, stdout and the output
//! file, byte for byte, over combinations of the goldens and the private image set (differing data
//! windows, two to five views, every codec), for every option placement. Failures must end alike
//! (exit -1). `#[ignore]`d: run by `cargo run -p xtask -- parity-tools`.
//!
//! One documented difference is checked on our side alone: a tiled last input crashes the C++ tool
//! (`makeMultiView.cpp:72`, its header goes to a scanline `OutputFile` with the `tiles` attribute).

mod tool_support;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use exr_core::attr::{Compression, ExrString};
use exr_core::{Image, insert_view_name};
use tool_support::*;

const OUT: &str = "out.exr";

fn golden(name: &str) -> PathBuf {
    goldens().into_iter().find(|p| p.ends_with(name)).unwrap_or_else(|| panic!("golden {name} not found"))
}

/// `view file view file ... out.exr` with the options before them.
fn line(opts: &[&str], views: &[(&str, &Path)]) -> Vec<OsString> {
    line_to(OUT.as_ref(), opts, views)
}

/// [`line`] with the output at `out`.
fn line_to(out: &Path, opts: &[&str], views: &[(&str, &Path)]) -> Vec<OsString> {
    let mut v: Vec<OsString> = opts.iter().map(|o| OsString::from(*o)).collect();
    for (name, file) in views {
        v.push((*name).into());
        v.push((*file).into());
    }
    v.push(out.into());
    v
}

/// The names the views get in the tests: `left` first (the default view), then others.
const NAMES: [&str; 5] = ["left", "right", "center", "up", "down"];

fn named(files: &[PathBuf]) -> Vec<(&'static str, &Path)> {
    files.iter().enumerate().map(|(i, f)| (NAMES[i], f.as_path())).collect()
}

/// Run one combination through both tools; `Ok(true)` when the C++ tool made the file.
fn compare(pair: &Pair, report: &mut Report, opts: &[&str], views: &[(&str, &Path)]) -> bool {
    let verdict = pair.check(line(opts, views), &[OUT]);
    let made = verdict == Ok(true);
    let what = views.iter().map(|(n, f)| format!("{n} {}", f.file_name().map_or_else(String::new, |s| s.to_string_lossy().into_owned()))).collect::<Vec<_>>().join(" ");
    report.check(format!("multiview {} {what}", opts.join(" ")), verdict.map(|_| ()));
    made
}

/// The private image set's single-view images used as views: the display-window tests (different
/// data windows, all in one image size) and a mid-size natural image.
fn corpus_views() -> Vec<PathBuf> {
    let images = exr_test_data::openexr_images();
    let mut v: Vec<PathBuf> = (1..=16).map(|i| images.join(format!("DisplayWindow/t{i:02}.exr"))).collect();
    v.push(images.join("v2/LowResLeftView/Trunks.exr"));
    v.push(images.join("ScanLines/Desk.exr"));
    v
}

/// Pairs and triples of the goldens, of the display-window images, of everything with `golden_zip`.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn output_files_are_identical_over_combinations() {
    let pair = Pair::new("exrmultiview", "multiview");
    let mut report = Report::default();
    let mut made = 0;

    // Every golden as the first and as the second view next to a fixed one.
    let zip = golden("golden_zip.exr");
    for g in goldens() {
        made += usize::from(compare(&pair, &mut report, &[], &[("left", &g), ("right", &zip)]));
        // A tiled last view is the C++ crash, checked on its own below.
        if !g.to_string_lossy().contains("tiled") {
            made += usize::from(compare(&pair, &mut report, &[], &[("left", &zip), ("right", &g)]));
        }
    }

    // Different data windows: consecutive and far-apart display-window tests, and triples.
    let c = corpus_views();
    let dw: Vec<&PathBuf> = c.iter().take(16).collect();
    for w in dw.windows(2) {
        made += usize::from(compare(&pair, &mut report, &[], &[("left", w[0]), ("right", w[1])]));
    }
    for (a, b) in [(0, 15), (4, 10), (2, 13), (7, 8)] {
        made += usize::from(compare(&pair, &mut report, &["-v"], &[("left", dw[a]), ("right", dw[b])]));
    }
    for t in [(0, 5, 10), (1, 9, 15), (3, 4, 12)] {
        made += usize::from(compare(&pair, &mut report, &[], &[("left", dw[t.0]), ("mid", dw[t.1]), ("right", dw[t.2])]));
    }
    // A natural image with another, both orders.
    made += usize::from(compare(&pair, &mut report, &["-z", "zip"], &[("a", &c[16]), ("b", &c[17])]));
    made += usize::from(compare(&pair, &mut report, &["-z", "zip"], &[("b", &c[17]), ("a", &c[16])]));

    // Three to five views from the goldens.
    let pool: Vec<PathBuf> = ["golden_tiled_zip_one.exr", "golden_rgba.exr", "golden_zip.exr", "golden_piz.exr", "golden_b44.exr"].map(golden).into();
    for n in 3..=5 {
        made += usize::from(compare(&pair, &mut report, &[], &named(&pool[..n])));
    }
    let rev: Vec<PathBuf> = pool.iter().rev().cloned().collect();
    made += usize::from(compare(&pair, &mut report, &["-v", "-z", "dwab"], &named(&rev[..4])));

    assert!(made >= 100, "only {made} combinations made a file: the comparison proves little");
    report.finish(&format!("multiview combinations ({made} files made)"));
}

/// Every codec, and the option placements the C++ parser accepts.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn every_option_is_identical() {
    let pair = Pair::new("exrmultiview", "multiview");
    let mut report = Report::default();
    let files: Vec<PathBuf> = ["golden_rgba.exr", "golden_zip.exr", "golden_dwaa.exr"].map(golden).into();
    let views = named(&files);
    let mut made = 0;
    for c in ["no", "none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "htj2k256", "htj2k32", "ZIP", "Piz"] {
        made += usize::from(compare(&pair, &mut report, &["-z", c], &views));
        made += usize::from(compare(&pair, &mut report, &["-z", c], &views[..2]));
    }
    // Options between the views, after the output, repeated.
    let (l, r) = (files[0].to_string_lossy().into_owned(), files[1].to_string_lossy().into_owned());
    let with = |parts: &[&str]| -> Vec<OsString> {
        parts.iter().map(|p| OsString::from(match *p { "@L" => l.as_str(), "@R" => r.as_str(), o => o })).collect()
    };
    for parts in [
        &["-v", "left", "@L", "right", "@R", OUT][..],
        &["left", "@L", "-v", "right", "@R", OUT],
        &["left", "@L", "right", "@R", "-z", "rle", OUT],
        &["left", "@L", "right", "@R", OUT, "-z", "zips", "-v"],
        &["-z", "zip", "-z", "piz", "left", "@L", "right", "@R", OUT],
        // The same view name twice, the same file twice.
        &["v", "@L", "v", "@R", OUT],
        &["left", "@L", "right", "@L", OUT],
    ] {
        let verdict = pair.check(with(parts), &[OUT]);
        made += usize::from(verdict == Ok(true));
        report.check(format!("multiview {parts:?}"), verdict.map(|_| ()));
    }
    assert!(made >= 35, "only {made} files made");
    report.finish(&format!("multiview options ({made} files made)"));
}

/// Bad command lines, bad inputs and unwritable outputs all end with exit code -1 and the same
/// stdout; inputs that are multi-view already, subsampled or deep are refused alike.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn failures_are_identical() {
    let pair = Pair::new("exrmultiview", "multiview");
    let (cdir, odir) = pair.dirs();
    let (zip, rgba) = (golden("golden_zip.exr"), golden("golden_rgba.exr"));
    let mut report = Report::default();
    let mut fails = 0;
    let mut case = |what: &str, args: Vec<OsString>| {
        let verdict = pair.check(&args, &[OUT]);
        fails += usize::from(verdict == Ok(false));
        report.check(what.to_string(), verdict.map(|_| ()));
    };
    let os = |s: &str| OsString::from(s);
    let f = |p: &Path| p.as_os_str().to_owned();
    case("one view", vec![os("left"), f(&zip), os(OUT)]);
    case("no views", vec![os(OUT)]);
    case("no output", vec![os("left"), f(&zip), os("right"), f(&rgba)]);
    case("two outputs", vec![os("left"), f(&zip), os("right"), f(&rgba), os("a.exr"), os("b.exr")]);
    case("missing -z value", vec![os("left"), f(&zip), os("right"), f(&rgba), os(OUT), os("-z")]);
    case("unknown codec", vec![os("-z"), os("lzw"), os("left"), f(&zip), os("right"), f(&rgba), os(OUT)]);
    case("missing input", vec![os("left"), f(&zip), os("right"), os("missing.exr"), os(OUT)]);
    case("unwritable output", vec![os("left"), f(&zip), os("right"), f(&rgba), os("no/such/dir/out.exr")]);
    for d in [&cdir, &odir] {
        std::fs::write(d.join("not_exr.txt"), b"this is not an image").expect("write the text file");
    }
    case("not an EXR file", vec![os("left"), f(&zip), os("right"), os("not_exr.txt"), os(OUT)]);
    for sub in ["golden_yc.exr", "golden_sub_none_x2y2.exr", "golden_sub_zip_x1y2.exr"] {
        case(sub, vec![os("left"), f(&zip), os("right"), f(&golden(sub)), os(OUT)]);
        case(sub, vec![os("left"), f(&golden(sub)), os("right"), f(&zip), os(OUT)]);
    }
    for deep in ["golden_deep_scanline_none.exr", "golden_deep_tiled_zips_one.exr"] {
        case(deep, vec![os("left"), f(&zip), os("right"), f(&golden(deep)), os(OUT)]);
    }
    // An image that already is multi-view (multipart, stereo, multi-view channel layers) is refused.
    let images = exr_test_data::openexr_images();
    for mv in ["v2/Stereo/Balls.exr", "MultiView/Adjuster.exr", "MultiView/Fog.exr"] {
        case(mv, vec![os("left"), f(&zip), os("right"), f(&images.join(mv)), os(OUT)]);
        case(mv, vec![os("left"), f(&images.join(mv)), os("right"), f(&zip), os(OUT)]);
    }
    assert!(fails >= 20, "only {fails} failures compared");
    report.finish(&format!("multiview failures ({fails} failing cases)"));
}

/// The C++ tool's exit codes really are -1 (and `-h` / a view with no file differ only in words).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn exit_codes_are_minus_one() {
    let zip = golden("golden_zip.exr");
    for args in [vec![OsString::from("left"), zip.clone().into_os_string(), OsString::from(OUT)], vec![OsString::from("-z")]] {
        let (c, o) = (run(&cpp("exrmultiview"), &args), run_ours("multiview", &args));
        assert_eq!((c.code, o.code), (Some(-1), Some(-1)), "{args:?}");
    }
    for args in [&["-h"][..], &["--help"], &["a", "1", "-h"]] {
        let (c, o) = (run(&cpp("exrmultiview"), args), run_ours("multiview", args));
        assert_eq!((c.code, o.code), (Some(0), Some(0)), "{args:?}");
        assert!(c.text().contains("-z") && o.text().contains("-z") && o.text().contains("Examples:"), "{args:?}");
    }
}

/// `-v`: the two reading passes and the writing line on stdout.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn verbose_text_is_identical() {
    let pair = Pair::new("exrmultiview", "multiview");
    let mut report = Report::default();
    let files: Vec<PathBuf> = ["golden_rgba.exr", "golden_zip.exr", "golden_piz.exr"].map(golden).into();
    for opts in [&["-v"][..], &["-v", "-z", "zips"]] {
        for n in [2, 3] {
            let ok = compare(&pair, &mut report, opts, &named(&files[..n]));
            assert!(ok);
        }
    }
    let (cdir, _) = pair.dirs();
    let c = run_in(&cdir, &cpp("exrmultiview"), line(&["-v"], &named(&files[..2])));
    let t = c.text();
    assert_eq!(t.matches("reading file").count(), 4, "{t}");
    assert!(t.ends_with("writing file out.exr\n"), "{t}");
    report.finish("multiview -v text");
}

/// `--version`: same text, exit code 0 (alone or after names).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn version_text_is_identical() {
    assert_same_version("exrmultiview", "multiview", &[&["--version"], &["a", "1.exr", "--version"]]);
}

/// The views of `files` put into one window by an independent scatter: the expected channels
/// (name, bytes) of the combined image and its window.
fn expected(files: &[PathBuf]) -> (imath_rs::Box2i, Vec<(String, Vec<u8>)>) {
    let images: Vec<Image> = files.iter().map(|f| Image::read(f).expect("read a view")).collect();
    let mut win = images[0].data_window;
    for i in &images[1..] {
        win.extend_by_box(&i.data_window);
    }
    let names: Vec<ExrString> = NAMES[..files.len()].iter().map(|n| ExrString::from(*n)).collect();
    let (w, h) = ((win.max.x - win.min.x + 1) as usize, (win.max.y - win.min.y + 1) as usize);
    let mut out = Vec::new();
    for (vi, img) in images.iter().enumerate() {
        for (name, data) in img.channels() {
            let src = data.to_ne_bytes();
            let size = src.len() / img.width() as usize / img.height() as usize;
            let mut dst = vec![0u8; w * h * size];
            for y in 0..img.height() as usize {
                for x in 0..img.width() as usize {
                    let (ux, uy) = ((img.data_window.min.x - win.min.x) as usize + x, (img.data_window.min.y - win.min.y) as usize + y);
                    dst[(uy * w + ux) * size..][..size].copy_from_slice(&src[(y * img.width() as usize + x) * size..][..size]);
                }
            }
            let out_name = String::from_utf8(insert_view_name(name.as_bytes(), &names, vi)).expect("utf-8 name");
            out.push((out_name, dst));
        }
    }
    (win, out)
}

/// Whatever the views' windows, the output holds every channel of every view, renamed, in the union
/// window, black outside its view, with the `multiView` list and the asked codec. Checked on our
/// output against an independent placement (and against the C++ output by the tests above).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn views_land_in_the_union_window() {
    let dir = TempDir::new("multiview-union");
    let out = dir.join(OUT);
    let c = corpus_views();
    let sets: Vec<Vec<PathBuf>> = vec![
        vec![c[0].clone(), c[9].clone()],
        vec![c[3].clone(), c[12].clone(), c[5].clone()],
        vec![golden("golden_zip.exr"), golden("golden_rgba.exr")],
    ];
    for files in sets {
        let args = line_to(&out, &["-z", "zips"], &named(&files));
        let o = run_ours("multiview", args);
        assert!(o.ok(), "{}", String::from_utf8_lossy(&o.stderr));
        let got = Image::read(&out).expect("read the output");
        let (win, want) = expected(&files);
        assert_eq!(got.data_window, win, "the union window");
        assert_eq!(got.compression, Compression::Zips);
        assert_eq!(got.channels().count(), want.len(), "channel count");
        for (name, bytes) in &want {
            let data = got.channel(name).unwrap_or_else(|| panic!("no channel {name}"));
            assert_eq!(&data.to_ne_bytes(), bytes, "channel {name} of {files:?}");
        }
    }
}

/// The C++ tool crashes when the last input is tiled; here it combines like any other and the
/// pixels are where they belong.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn a_tiled_last_input_combines() {
    let dir = TempDir::new("multiview-tiled");
    let out = dir.join(OUT);
    for files in [
        vec![golden("golden_zip.exr"), golden("golden_tiled_rgba.exr")],
        vec![golden("golden_rgba.exr"), golden("golden_tiled_zip_one.exr")],
        vec![golden("golden_zip.exr"), golden("golden_tiled_zip_one_randomy.exr")],
        vec![golden("golden_zip.exr"), golden("golden_tiled_zip_mipmap.exr")],
    ] {
        let o = run_ours("multiview", line_to(&out, &[], &named(&files)));
        assert!(o.ok(), "{files:?}: {}", String::from_utf8_lossy(&o.stderr));
        let got = Image::read(&out).expect("read the output");
        let (win, want) = expected(&files);
        assert_eq!(got.data_window, win);
        for (name, bytes) in &want {
            assert_eq!(&got.channel(name).unwrap_or_else(|| panic!("no channel {name}")).to_ne_bytes(), bytes, "channel {name} of {files:?}");
        }
        assert!(!exr_core::scanline::InputFile::open(&out).expect("open").header().has_tile_description(), "a scanline file");
    }
}

/// Views larger than one strip buffer (8192 x 300 float RGBA: 39 MB per view, the union image several
/// strips) with the last view decreasing-Y, so the output is written bottom-up strip by strip; other
/// line orders, windows and codecs around it.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn multi_strip_views_are_identical() {
    use exr_core::attr::{Compression, LineOrder};
    let (pair, dir) = (Pair::new("exrmultiview", "multiview"), TempDir::new("multiview-big"));
    let (inc, dec, off) = (dir.join("inc.exr"), dir.join("dec.exr"), dir.join("off.exr"));
    synth::pattern(&inc, (8192, 300), (0, 0), LineOrder::IncreasingY, Compression::NoCompression);
    synth::pattern(&dec, (8192, 300), (0, 0), LineOrder::DecreasingY, Compression::NoCompression);
    synth::pattern(&off, (8000, 310), (-20, 10), LineOrder::DecreasingY, Compression::Zip);
    let mut report = Report::default();
    let mut made = 0;
    for (opts, views) in [
        (&["-z", "none"][..], [("left", &inc), ("right", &dec)]),
        (&["-z", "none"], [("left", &dec), ("right", &inc)]),
        (&["-z", "zip"], [("left", &inc), ("right", &off)]),
        (&["-z", "piz"], [("left", &off), ("right", &dec)]),
    ] {
        let views: Vec<(&str, &Path)> = views.iter().map(|(n, p)| (*n, p.as_path())).collect();
        made += usize::from(compare(&pair, &mut report, opts, &views));
    }
    assert_eq!(made, 4);
    report.finish(&format!("multiview multi-strip views ({made} files)"));
}

/// The output may be one of the inputs (the C++ closes every input before creating the output):
/// both tools end with the same file, for the first, the last and a repeated input.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn the_output_may_be_an_input() {
    let pair = Pair::new("exrmultiview", "multiview");
    let (cdir, odir) = pair.dirs();
    let (a, b) = (golden("golden_zip.exr"), golden("golden_rgba.exr"));
    for (first, second, out) in [("a.exr", "b.exr", "a.exr"), ("a.exr", "b.exr", "b.exr"), ("a.exr", "a.exr", "a.exr")] {
        for d in [&cdir, &odir] {
            std::fs::copy(&a, d.join("a.exr")).expect("copy a");
            std::fs::copy(&b, d.join("b.exr")).expect("copy b");
        }
        let args: Vec<OsString> = ["left", first, "right", second, out].map(OsString::from).into();
        assert_eq!(pair.check(&args, &[out]), Ok(true), "{first} {second} -> {out}");
    }
}
