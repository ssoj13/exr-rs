//! `exr-rs envmap` against OpenEXR's `exrenvmap` (v3.4.15): exit code, stdout and every output
//! file, byte for byte, over generated lat-long and cube-face maps, the goldens and the private
//! image set, for every option and the combinations that change the output. Failures must end
//! alike. `#[ignore]`d: run by `cargo run -p xtask -- parity-tools`.

mod tool_support;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use exr_core::attr::{Compression, Envmap, LineOrder};
use exr_core::header::Header;
use exr_core::rgba::{Rgba, RgbaChannels, RgbaOutputFile, TiledRgbaOutputFile};
use imath_rs::{Box2i, V2f, V2i};
use tool_support::*;

const OUT: &str = "out.exr";

fn os(s: &str) -> OsString {
    OsString::from(s)
}

/// How a generated map is stored.
#[derive(Clone, Copy)]
enum Layout {
    Scan,
    Tiled(u32, u32),
}

/// An asymmetric, HDR-ish pattern: a misplaced or mirrored pixel shows.
fn pattern(x: i32, y: i32, w: i32, h: i32) -> Rgba {
    let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
    let spot = if (x / 5 + y / 3) % 2 == 0 { 50.0 } else { 0.0 };
    Rgba::from_f32(u * 4.0 + (x % 7) as f32 * 0.1, v * 2.0 + spot, (u * v * 8.0).sin().abs() + 0.25, 0.5 + 0.5 * u)
}

/// Write a `w x h` map whose data window starts at `origin`, tagged `ty` (none: no attribute).
fn write_map(path: &Path, (w, h): (i32, i32), origin: (i32, i32), ty: Option<Envmap>, layout: Layout, comp: Compression, f: impl Fn(i32, i32, i32, i32) -> Rgba) {
    let dw = Box2i { min: V2i { x: origin.0, y: origin.1 }, max: V2i { x: origin.0 + w - 1, y: origin.1 + h - 1 } };
    let mut header = Header::with_windows(dw, dw, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, comp).expect("header");
    if let Some(ty) = ty {
        header.add_envmap(ty).expect("envmap attribute");
    }
    let px: Vec<Rgba> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y, w, h)).collect();
    match layout {
        Layout::Scan => RgbaOutputFile::write(path, &header, RgbaChannels::WRITE_RGBA, &px).expect("write scanline map"),
        Layout::Tiled(tx, ty) => TiledRgbaOutputFile::write(path, &header, RgbaChannels::WRITE_RGBA, tx, ty, &px).expect("write tiled map"),
    }
}

/// A map with NaN, infinities, negatives and denormals among ordinary pixels.
fn odd(x: i32, y: i32, w: i32, h: i32) -> Rgba {
    match (x * 3 + y * 5) % 9 {
        0 => Rgba::from_f32(f32::NAN, 1.0, 2.0, 1.0),
        1 => Rgba::from_f32(f32::INFINITY, -f32::INFINITY, 0.0, 1.0),
        2 => Rgba::from_f32(-5.0, 1e-7, 65000.0, 0.0),
        _ => pattern(x, y, w, h),
    }
}

fn golden(name: &str) -> PathBuf {
    goldens().into_iter().find(|p| p.ends_with(name)).unwrap_or_else(|| panic!("golden {name} not found"))
}

/// `opts`, the input and `out.exr`.
fn line(opts: &[&str], input: &Path) -> Vec<OsString> {
    let mut v: Vec<OsString> = opts.iter().map(|o| os(o)).collect();
    v.push(input.into());
    v.push(OUT.into());
    v
}

/// Run both tools; count the runs that succeeded (output compared) in `made`.
fn sweep(pair: &Pair, report: &mut Report, opts: &[&str], files: &[PathBuf]) -> usize {
    let mut made = 0;
    for f in files {
        let verdict = pair.check(line(opts, f), &[OUT]);
        made += usize::from(verdict == Ok(true));
        report.check(format!("envmap {} {}", opts.join(" "), f.display()), verdict.map(|_| ()));
    }
    made
}

/// The generated inputs: lat-long and cube maps of awkward sizes, both layouts, an offset window.
fn generated(dir: &TempDir) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let mut add = |name: &str, size, origin, ty, layout, comp| {
        let p = dir.join(name);
        write_map(&p, size, origin, ty, layout, comp, pattern);
        v.push(p);
    };
    let (ll, cube) = (Some(Envmap::LatLong), Some(Envmap::Cube));
    add("ll64.exr", (64, 32), (0, 0), ll, Layout::Scan, Compression::Zip);
    add("ll100.exr", (100, 50), (0, 0), ll, Layout::Tiled(16, 16), Compression::Piz);
    add("ll37.exr", (37, 19), (0, 0), ll, Layout::Scan, Compression::NoCompression);
    add("ll_off.exr", (48, 24), (-10, 7), ll, Layout::Scan, Compression::Zips);
    add("ll3.exr", (3, 2), (0, 0), ll, Layout::Scan, Compression::Rle);
    add("ll1.exr", (1, 1), (0, 0), ll, Layout::Scan, Compression::NoCompression);
    add("noattr.exr", (40, 20), (0, 0), None, Layout::Scan, Compression::Zip);
    add("cube16.exr", (16, 96), (0, 0), cube, Layout::Scan, Compression::Zip);
    add("cube20.exr", (20, 120), (0, 0), cube, Layout::Tiled(8, 8), Compression::Zip);
    add("cube5.exr", (5, 30), (0, 0), cube, Layout::Scan, Compression::Zip);
    add("cube1.exr", (1, 6), (0, 0), cube, Layout::Scan, Compression::Zip);
    add("cube_wide.exr", (24, 72), (0, 0), cube, Layout::Scan, Compression::Zip);
    add("cube_off.exr", (12, 72), (3, 5), cube, Layout::Scan, Compression::Zip);
    v
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn output_files_are_identical_over_generated_maps() {
    let (pair, dir) = (Pair::new("exrenvmap", "envmap"), TempDir::new("envmap-in"));
    let files = generated(&dir);
    let mut report = Report::default();
    let mut made = 0;
    for opts in [
        &["-w", "32"][..],
        &["-w", "24", "-l"],
        &["-w", "33", "-m", "-t", "8", "8"],
        &["-w", "20", "-l", "-m", "-u", "-t", "16", "4"],
        &["-w", "16", "-f", "2.5", "3", "-z", "piz"],
        &["-w", "16", "-f", "0", "1", "-l"],
        &["-w", "16", "-ci"],
        &["-w", "16", "-li", "-l"],
        &["-w", "16", "-p", "0.5", "0.25"],
        &["-w", "16", "-p", "1", "0", "-l", "-m"],
        &["-w", "7", "-m", "-z", "b44a"],
        &["-w", "1"],
        &["-w", "2", "-l"],
        &["-w", "3", "-m"],
        &["-w", "64"],
    ] {
        made += sweep(&pair, &mut report, opts, &files);
    }
    assert!(made >= 150, "only {made} conversions: the comparison proves little");
    report.finish(&format!("envmap over generated maps ({made} conversions)"));
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn output_files_are_identical_over_goldens_and_the_image_set() {
    let pair = Pair::new("exrenvmap", "envmap");
    let mut files: Vec<PathBuf> = ["golden_zip.exr", "golden_rgba.exr", "golden_tiled_zip_one_decreasing.exr", "golden_yc.exr", "golden_multipart.exr"].map(golden).into();
    let res = exr_test_data::openexr_images().join("MultiResolution");
    for n in ["WavyLinesLatLong", "WavyLinesCube", "WavyLinesSphere", "KernerEnvCube", "KernerEnvLatLong", "OrientationCube", "OrientationLatLong", "StageEnvCube", "StageEnvLatLong"] {
        files.push(res.join(format!("{n}.exr")));
    }
    let mut report = Report::default();
    let mut made = 0;
    for opts in [&["-w", "48"][..], &["-w", "48", "-l", "-m"], &["-w", "40", "-f", "1.5", "4", "-u", "-m"], &["-w", "32", "-ci", "-l"]] {
        made += sweep(&pair, &mut report, opts, &files);
    }
    assert!(made >= 20, "only {made} conversions");
    report.finish(&format!("envmap over goldens + image set ({made} conversions)"));
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn the_default_width_is_identical() {
    let pair = Pair::new("exrenvmap", "envmap");
    let res = exr_test_data::openexr_images().join("MultiResolution");
    let files = [res.join("WavyLinesLatLong.exr"), res.join("WavyLinesCube.exr")];
    let mut report = Report::default();
    let made = sweep(&pair, &mut report, &[], &files) + sweep(&pair, &mut report, &["-l", "-m"], &files);
    assert_eq!(made, 4);
    report.finish("envmap default width");
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn diffuse_blur_is_identical() {
    let (pair, dir) = (Pair::new("exrenvmap", "envmap"), TempDir::new("envmap-blur"));
    let mut files = generated(&dir);
    files.retain(|f| ["ll64.exr", "ll3.exr", "cube16.exr", "cube1.exr", "ll_off.exr"].iter().any(|n| f.ends_with(n)));
    files.push(exr_test_data::openexr_images().join("MultiResolution/WavyLinesLatLong.exr"));
    files.push(exr_test_data::openexr_images().join("MultiResolution/WavyLinesCube.exr"));
    let mut report = Report::default();
    let mut made = 0;
    for opts in [&["-b", "-w", "24"][..], &["-b", "-w", "16", "-l", "-m"]] {
        made += sweep(&pair, &mut report, opts, &files);
    }
    assert_eq!(made, files.len() * 2);
    report.finish(&format!("envmap -b ({made} conversions)"));
}

/// Six-file input and output, and the round trip through both.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn six_files_are_identical() {
    let pair = Pair::new("exrenvmap", "envmap");
    let (cdir, odir) = pair.dirs();
    for d in [&cdir, &odir] {
        for (i, n) in ["+X", "-X", "+Y", "-Y", "+Z", "-Z"].iter().enumerate() {
            write_map(&d.join(format!("face_{n}.exr")), (12, 12), (0, 0), None, Layout::Scan, Compression::Zip, |x, y, w, h| pattern(x + i as i32 * 3, y, w * 2, h));
        }
        write_map(&d.join("face_+X_bad.exr"), (12, 10), (0, 0), None, Layout::Scan, Compression::Zip, pattern);
        write_map(&d.join("ll.exr"), (64, 32), (0, 0), Some(Envmap::LatLong), Layout::Scan, Compression::Zip, pattern);
    }
    let mut report = Report::default();
    let faces: Vec<String> = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"].iter().map(|n| format!("o_{n}.exr")).collect();
    let outs: Vec<&str> = faces.iter().map(String::as_str).collect();
    let mut made = 0;
    // Six in, one out; one in, six out; six in, six out; six in, lat-long out.
    for args in [
        vec!["-w", "20", "face_%.exr", OUT],
        vec!["-w", "20", "-m", "face_%.exr", OUT],
        vec!["-w", "20", "-l", "face_%.exr", OUT],
        vec!["-w", "20", "-l", "-li", "face_%.exr", OUT],
        vec!["-w", "12", "face_%.exr", OUT],
        vec!["-w", "20", "-b", "face_%.exr", OUT],
    ] {
        let v = pair.check(args.clone(), &[OUT]);
        made += usize::from(v == Ok(true));
        report.check(format!("envmap {}", args.join(" ")), v.map(|_| ()));
    }
    for (args, names) in [
        (vec!["-w", "16", "ll.exr", "o_%.exr"], outs.clone()),
        (vec!["-w", "16", "-t", "5", "7", "-z", "piz", "-v", "ll.exr", "o_%.exr"], outs.clone()),
        (vec!["-w", "16", "-m", "face_%.exr", "o_%.exr"], outs.clone()),
        (vec!["-w", "16", "-l", "ll.exr", "o_%.exr"], vec!["o_%.exr"]),
    ] {
        let v = pair.check(args.clone(), &names);
        made += usize::from(v == Ok(true));
        report.check(format!("envmap {}", args.join(" ")), v.map(|_| ()));
    }
    // Faces that disagree in shape: both fail alike.
    for args in [vec!["face_%_bad.exr", OUT], vec!["missing_%.exr", OUT]] {
        let v = pair.check(args.clone(), &[OUT]);
        assert_eq!(v, Ok(false), "{args:?}");
        report.check(format!("envmap {}", args.join(" ")), v.map(|_| ()));
    }
    report.finish(&format!("envmap six files ({made} conversions)"));
    assert!(made >= 6, "only {made} conversions");
}

/// Bad options, bad inputs and unwritable outputs: same exit code and stdout, and the same files
/// left behind (existence), the input untouched.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn failures_are_identical() {
    let (pair, dir) = (Pair::new("exrenvmap", "envmap"), TempDir::new("envmap-fail"));
    let good = dir.join("ll.exr");
    write_map(&good, (32, 16), (0, 0), Some(Envmap::LatLong), Layout::Scan, Compression::Zip, pattern);
    let (cdir, odir) = pair.dirs();
    for d in [&cdir, &odir] {
        std::fs::write(d.join("not_exr.txt"), b"this is not an image").expect("write text");
    }
    let mut report = Report::default();
    let mut fails = 0;
    let g: OsString = good.clone().into_os_string();
    let mut case = |what: &str, args: Vec<OsString>| {
        let (c, o) = (run_in(&cdir, &cpp("exrenvmap"), &args), run_ours_in(&odir, "envmap", &args));
        let left = (cdir.join(OUT).exists(), odir.join(OUT).exists());
        std::fs::remove_file(cdir.join(OUT)).ok();
        std::fs::remove_file(odir.join(OUT)).ok();
        let verdict = same_run(&c, &o, || Ok(())).and_then(|()| if left.0 == left.1 { Ok(()) } else { Err(format!("output left behind: C++ {}, ours {}", left.0, left.1)) });
        fails += usize::from(!c.ok());
        report.check(what.to_string(), verdict);
    };
    case("missing input", vec![os("missing.exr"), os(OUT)]);
    case("not an EXR", vec![os("not_exr.txt"), os(OUT)]);
    case("unwritable output", vec![g.clone(), os("no/such/dir/out.exr")]);
    case("unknown codec", vec![os("-z"), os("lzw"), g.clone(), os(OUT)]);
    for bad in [
        vec!["-w", "0"],
        vec!["-w", "-3"],
        vec!["-f", "-1", "5"],
        vec!["-f", "1", "0"],
        vec!["-t", "0", "0"],
        vec!["-t", "0", "8"],
        vec!["-p", "0", "-0.5"],
        vec!["-p", "-1", "0"],
    ] {
        let mut a: Vec<OsString> = bad.iter().map(|s| os(s)).collect();
        a.extend([g.clone(), os(OUT)]);
        case(&format!("envmap {}", bad.join(" ")), a);
    }
    // An option missing its values at the end of the line.
    for last in [&["-w"][..], &["-f", "1"], &["-t", "8"], &["-p", "1"], &["-z"]] {
        let mut a = vec![g.clone(), os(OUT)];
        a.extend(last.iter().map(|s| os(s)));
        case(&format!("envmap <files> {}", last.join(" ")), a);
    }
    case("one file name", vec![g.clone()]);
    case("a lat-long map of width 1 (height 0)", vec![os("-w"), os("1"), os("-l"), g.clone(), os(OUT)]);
    case("a chroma input (tiled output has none)", vec![golden("golden_yc.exr").into_os_string(), os(OUT)]);
    for deep in ["golden_deep_scanline_none.exr", "golden_deep_tiled_zips_one.exr"] {
        case(deep, vec![golden(deep).into_os_string(), os(OUT)]);
    }
    assert!(fails >= 15, "only {fails} failing cases");
    report.finish(&format!("envmap failures ({fails} failing cases)"));

    // The input is also the output: the C++ reads it all first, so this works, and so must ours.
    let same = cdir.join("same.exr");
    std::fs::copy(&good, &same).expect("copy");
    let same_o = odir.join("same.exr");
    std::fs::copy(&good, &same_o).expect("copy");
    let (c, o) = (run_in(&cdir, &cpp("exrenvmap"), ["-w", "16", "same.exr", "same.exr"]), run_ours_in(&odir, "envmap", ["-w", "16", "same.exr", "same.exr"]));
    assert_eq!((c.code, o.code), (Some(0), Some(0)), "{}", String::from_utf8_lossy(&o.stderr));
    assert_same_file(&same, &same_o);
}

/// The fuzzer cases of the C++ project: the tools end alike on each.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn fuzzer_cases_end_alike() {
    let pair = Pair::new("exrenvmap", "envmap");
    let dir = exr_test_data::openexr_images().join("Damaged");
    let files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("Damaged dir")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().contains("exrenvmap_fuzzer")))
        .collect();
    assert!(files.len() >= 10, "{} fuzzer cases", files.len());
    let mut report = Report::default();
    sweep(&pair, &mut report, &["-w", "16"], &files);
    sweep(&pair, &mut report, &["-w", "16", "-l", "-m"], &files);
    report.finish(&format!("envmap fuzzer cases ({})", files.len()));
}

/// NaN, infinities, negatives, denormals and huge values must come out of the filters alike.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn extreme_pixels_are_identical() {
    let (pair, dir) = (Pair::new("exrenvmap", "envmap"), TempDir::new("envmap-extreme"));
    let mut files = Vec::new();
    for (n, size, ty) in [("x_ll.exr", (32, 16), Envmap::LatLong), ("x_cube.exr", (8, 48), Envmap::Cube)] {
        let p = dir.join(n);
        write_map(&p, size, (0, 0), Some(ty), Layout::Scan, Compression::NoCompression, odd);
        files.push(p);
    }
    let mut report = Report::default();
    let mut made = 0;
    for opts in [&["-w", "16"][..], &["-w", "16", "-l", "-m"], &["-w", "8", "-b"] ] {
        made += sweep(&pair, &mut report, opts, &files);
    }
    report.finish(&format!("envmap extreme pixels ({made} conversions)"));
    assert!(made >= 4, "only {made} conversions");
}

/// An infinite filter radius makes every lookup direction NaN; the NaN payloads follow the MSVC SSE
/// operand order (`exr_core::sse`), so the output is byte-identical to the C++ in release too.
///
/// KNOWN GAP (pinned): a NaN radius additionally reaches a NaN-to-`int` conversion in the C++
/// (undefined; `INT_MIN` on x86), whose result depends on the compiler (openexr_bug.md T7); ours
/// differs in those bytes. The test fails once they agree, so the entry must then be removed.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn non_finite_filter_radius_is_a_pinned_gap() {
    let (pair, dir) = (Pair::new("exrenvmap", "envmap"), TempDir::new("envmap-gap"));
    let p = dir.join("ll.exr");
    write_map(&p, (32, 16), (0, 0), Some(Envmap::LatLong), Layout::Scan, Compression::Zip, odd);
    let v = pair.check(line(&["-w", "16", "-f", "inf", "3"], &p), &[OUT]);
    assert!(matches!(v, Ok(true)), "-f inf: {v:?}");
    let v = pair.check(line(&["-w", "16", "-f", "nan", "3"], &p), &[OUT]);
    assert!(matches!(&v, Err(m) if m.contains("first difference")), "-f nan: the pinned gap is gone or changed ({v:?}): remove it");
}

/// `-v` text for every kind of run.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn verbose_text_is_identical() {
    let (pair, dir) = (Pair::new("exrenvmap", "envmap"), TempDir::new("envmap-verbose"));
    let files = generated(&dir);
    let mut report = Report::default();
    for f in files.iter().filter(|f| ["ll64.exr", "cube16.exr", "ll3.exr"].iter().any(|n| f.ends_with(n))) {
        for opts in [&["-v", "-w", "16"][..], &["-v", "-w", "16", "-m", "-l"], &["-v", "-b", "-w", "8"], &["-v", "-w", "16", "-p", "0.1", "0.1"]] {
            let v = pair.check(line(opts, f), &[OUT]);
            assert_eq!(v, Ok(true), "{opts:?} {}", f.display());
            report.check(format!("envmap {}", opts.join(" ")), v.map(|_| ()));
        }
    }
    let (cdir, _) = pair.dirs();
    let c = run_in(&cdir, &cpp("exrenvmap"), ["-v", "-m", "-w", "8", files[0].to_str().expect("utf-8"), OUT]);
    assert!(c.text().contains("level 1") && c.text().ends_with("done.\n"), "{}", c.text());
    report.finish("envmap -v text");
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn help_succeeds() {
    for args in [&["-h"][..], &["--help"], &["in.exr", "out.exr", "-h"], &["-m", "--help"]] {
        let (c, o) = (run(&cpp("exrenvmap"), args), run_ours("envmap", args));
        assert_eq!((c.code, o.code), (Some(0), Some(0)), "{args:?}");
        assert!(c.text().contains("-ci") && o.text().contains("-ci") && o.text().contains("Examples:"), "{args:?}");
    }
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn version_text_is_identical() {
    assert_same_version("exrenvmap", "envmap", &[&["--version"], &["in.exr", "out.exr", "--version"]]);
}
