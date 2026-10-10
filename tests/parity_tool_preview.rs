//! `exr-rs preview` against OpenEXR's `exrmakepreview` (v3.4.15): exit code, stdout and the output
//! file, byte for byte, over the goldens and the private image set, for every option (width,
//! exposure, verbose) and the edge values of the curve (clamped exposures, NaN, infinities, a
//! one-pixel preview). Failures must end alike. `#[ignore]`d: run by
//! `cargo run -p xtask -- parity-tools`.

mod tool_support;

use std::path::{Path, PathBuf};

use exr_core::Image;
use tool_support::*;

const OUT: &str = "out.exr";

/// `opts`, the input and `out.exr`, as the tool wants them.
fn line(opts: &[&str], input: &Path) -> Vec<std::ffi::OsString> {
    let mut v: Vec<std::ffi::OsString> = opts.iter().map(|o| std::ffi::OsString::from(*o)).collect();
    v.push(input.into());
    v.push(OUT.into());
    v
}

fn golden(name: &str) -> PathBuf {
    goldens().into_iter().find(|p| p.ends_with(name)).unwrap_or_else(|| panic!("golden {name} not found"))
}

/// Inputs `exrmakepreview` crashes on (access violation, exit -1073741819): the deep images (the
/// RGBA reader's deep path) and the `v2` multipart stereo set (`InputFile` on part 0 of a multipart
/// file, then `copyPixels`). Ours refuses them cleanly, exit 1: a documented
/// difference, pinned here so it is noticed if either side changes.
fn cpp_crashes(f: &Path) -> bool {
    let p = f.to_string_lossy().replace('\\', "/");
    p.contains(".deep.exr") || p.contains("golden_deep_composite") || (p.contains("/v2/") && !p.contains("composited"))
}

/// One input through both tools where the C++ tool crashes: it must crash, we must fail cleanly.
fn crash_case(pair: &Pair, opts: &[&str], f: &Path) -> Result<(), String> {
    let (cdir, odir) = pair.dirs();
    let (c, o) = (run_in(&cdir, &cpp("exrmakepreview"), line(opts, f)), run_ours_in(&odir, "preview", line(opts, f)));
    let crashed = c.timed_out || c.code.is_some_and(|x| !(0..=255).contains(&x) && x != -1);
    // Each side's partial output (both create the file before failing) must not reach the next case.
    std::fs::remove_file(odir.join(OUT)).ok();
    std::fs::remove_file(cdir.join(OUT)).ok();
    let clear = String::from_utf8_lossy(&o.stderr).contains("Cannot make a preview of deep data");
    if !crashed || o.code != Some(1) || !clear {
        return Err(format!("the pinned C++ crash changed: C++ {}, ours {} (message given: {clear})", c.end(), o.end()));
    }
    Ok(())
}

/// `opts` over `files` through both tools; how many previews the C++ tool made.
fn sweep(pair: &Pair, report: &mut Report, opts: &[&str], files: &[PathBuf]) -> usize {
    let mut made = 0;
    for f in files {
        if cpp_crashes(f) {
            report.check(format!("preview {} {} (C++ crashes)", opts.join(" "), f.display()), crash_case(pair, opts, f));
            continue;
        }
        let verdict = pair.check(line(opts, f), &[OUT]);
        made += usize::from(verdict == Ok(true));
        report.check(format!("preview {} {}", opts.join(" "), f.display()), verdict.map(|_| ()));
    }
    made
}

const WIRE_OVERFLOW_STDERR: &[u8] =
    b"error: OpenEXR overflow error: value 2994003008 exceeds the i32 wire-length limit\n";

/// The single 64x48, aspect-f32-bits 0x3a83126f, width-999 fixture requests a 999x749250
/// preview: its serialized 2,994,003,008-byte attribute cannot fit the i32 wire length.
/// Bounded C++ 3.4.15 reproductions exit 1 with bad allocation (or may time out).
/// This explicit oracle resource difference is not rerun as a multi-gigabyte allocation.
/// The real Rust result must be a complete structured refusal and must preserve the output.
fn classify_wire_invalid_preview(
    ours: &Out,
    before: Option<&[u8]>,
    after: Option<&[u8]>,
) -> Result<bool, String> {
    if ours.code != Some(1)
        || ours.timed_out
        || ours.truncated
        || !ours.stdout.is_empty()
        || ours.stderr != WIRE_OVERFLOW_STDERR
        || before != after
    {
        return Err(format!(
            "wire-invalid preview must fail with exact Overflow and preserve output: {} {:?}, unchanged {}",
            ours.end(),
            String::from_utf8_lossy(&ours.stderr),
            before == after
        ));
    }
    Ok(false)
}

fn wire_invalid_case(pair: &Pair, input: &Path) -> Result<bool, String> {
    let (_, odir) = pair.dirs();
    let output = odir.join(OUT);
    std::fs::remove_file(&output).ok();
    let mut verdict = Ok(false);
    // Both absent and preexisting output must survive the structured refusal.
    for seed in [None, Some(SEED)] {
        if let Some(bytes) = seed {
            std::fs::write(&output, bytes).expect("seed output");
        }
        let o = run_ours_in(&odir, "preview", line(&["-w", "999"], input));
        let after = std::fs::read(&output).ok();
        let result = classify_wire_invalid_preview(&o, seed, after.as_deref());
        if result.is_err() {
            verdict = result;
        }
        std::fs::remove_file(&output).ok();
    }
    verdict
}

#[test]
fn wire_invalid_preview_exception_rejects_success_or_changed_output() {
    let out = |code, timed_out, stderr: &[u8]| Out {
        code,
        timed_out,
        stdout: Vec::new(),
        stderr: stderr.to_vec(),
        truncated: false,
    };
    let rust = out(Some(1), false, WIRE_OVERFLOW_STDERR);
    assert_eq!(classify_wire_invalid_preview(&rust, None, None), Ok(false));
    assert_eq!(
        classify_wire_invalid_preview(&rust, Some(SEED), Some(SEED)),
        Ok(false)
    );
    assert!(classify_wire_invalid_preview(&out(Some(0), false, b""), None, None).is_err());
    assert!(classify_wire_invalid_preview(&rust, None, Some(SEED)).is_err());
    assert!(classify_wire_invalid_preview(&rust, Some(SEED), Some(b"changed")).is_err());
    assert!(
        classify_wire_invalid_preview(&out(None, true, WIRE_OVERFLOW_STDERR), None, None).is_err()
    );
    assert!(
        classify_wire_invalid_preview(&out(Some(1), false, b"error: other\n"), None, None).is_err()
    );
    let mut truncated = rust;
    truncated.truncated = true;
    assert!(classify_wire_invalid_preview(&truncated, None, None).is_err());
}

/// Every input with the default, a narrow and brighter, and a wide and darker preview.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn output_files_are_identical_over_every_input() {
    let pair = Pair::new("exrmakepreview", "preview");
    let files = inputs();
    let mut report = Report::default();
    let mut made = 0;
    made += sweep(&pair, &mut report, &[], &files);
    made += sweep(&pair, &mut report, &["-w", "7", "-e", "2.5"], &files);
    made += sweep(&pair, &mut report, &["-w", "300", "-e", "-3"], &files);
    assert!(made >= 300, "only {made} previews: the comparison proves little");
    report.finish(&format!("preview over goldens + image set ({made} previews)"));
}

/// Widths and exposures, including the values the curve clamps and the ones that are not finite.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn every_width_and_exposure_on_representative_inputs() {
    let pair = Pair::new("exrmakepreview", "preview");
    let files: Vec<PathBuf> = ["golden_zip.exr", "golden_rgba.exr", "golden_yc.exr", "golden_tiled_zip_mipmap.exr", "golden_b44.exr", "golden_dwaa.exr"].map(golden).into();
    let mut report = Report::default();
    let mut made = 0;
    for w in ["1", "2", "3", "17", "64", "100", "257", "1000", "0x20", "010"] {
        for e in ["0", "1", "-1", "5.5", "-20", "17.75", "30", "-30", "0.0001", "1e2", "-1e-3", "+2", ".5", "inf", "-inf", "nan"] {
            made += sweep(&pair, &mut report, &["-w", w, "-e", e], &files[..2]);
        }
    }
    // Every file shape, a few settings each.
    for (w, e) in [("1", "0"), ("100", "0"), ("33", "3"), ("500", "-2")] {
        made += sweep(&pair, &mut report, &["-w", w, "-e", e], &files);
    }
    // Option order and repeats: the last value wins; options may follow the files.
    made += sweep(&pair, &mut report, &["-w", "50", "-w", "20", "-e", "1", "-e", "2"], &files);
    made += sweep(&pair, &mut report, &["-v", "-w", "9", "-v"], &files);
    let verdict = pair.check([files[0].as_os_str(), OUT.as_ref(), "-w".as_ref(), "13".as_ref(), "-e".as_ref(), "-1".as_ref()], &[OUT]);
    made += usize::from(verdict == Ok(true));
    report.check("options after the files", verdict.map(|_| ()));
    assert!(made >= 350, "only {made} previews");
    report.finish(&format!("preview widths and exposures ({made} previews)"));
}

/// The preview really is in the header, with the size and the colours the curve gives: a white
/// (1.0) image at exposure 0 previews as 164 (hand-computed: m = 2^2.47393 = 5.556, knee(4.556) =
/// 3.305, (1 + 3.305)^0.4545 * 84.66 = 164.4), and the rows follow the aspect ratio.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn the_preview_is_in_the_header() {
    use exr_core::attr::Compression;
    use exr_core::scanline::InputFile;
    use exr_core::ChannelData;
    use imath_rs::{Box2i, Half, V2i};

    let dir = TempDir::new("preview-header");
    let (input, out) = (dir.join("white.exr"), dir.join(OUT));
    let (w, h) = (40usize, 10usize);
    let window = Box2i { min: V2i { x: -3, y: 5 }, max: V2i { x: -3 + w as i32 - 1, y: 5 + h as i32 - 1 } };
    let plane = |v: f32| ChannelData::Half(vec![Half::from_f32(v); w * h]);
    let img = Image::new(window).with_channel("R", plane(1.0)).with_channel("G", plane(1.0)).with_channel("B", plane(1.0)).with_channel("A", plane(0.5));
    img.write(&input, Compression::Zip).expect("write the input");
    let run_it = |args: &[&str]| {
        let mut a: Vec<std::ffi::OsString> = args.iter().map(|x| std::ffi::OsString::from(*x)).collect();
        a.extend([input.clone().into_os_string(), out.clone().into_os_string()]);
        let o = run_ours("preview", a);
        assert!(o.ok(), "{}", String::from_utf8_lossy(&o.stderr));
        let file = InputFile::open(&out).expect("open the output");
        file.header().preview_image().expect("a preview attribute").clone()
    };
    let p = run_it(&["-w", "20"]);
    // 10 rows x 20 / 40 columns = 5 rows.
    assert_eq!((p.width(), p.height()), (20, 5));
    assert!(p.pixels().iter().all(|px| (px.r, px.g, px.b, px.a) == (164, 164, 164, 128)), "{:?}", &p.pixels()[..3]);
    // One f-stop down: m = 2^1.47393 = 2.778, 1 + knee(1.778) = 2.537, 2.537^0.4545 * 84.66 = 129.
    let p = run_it(&["-w", "8", "-e", "-1"]);
    assert_eq!((p.width(), p.height()), (8, 2));
    assert!(p.pixels().iter().all(|px| (px.r, px.a) == (129, 128)), "{:?}", &p.pixels()[..3]);
    // Two stops up: m = 2^4.47393 = 22.2, 1 + knee(21.2) = 9.62, 9.62^0.4545 * 84.66 = 236.9 -> 236
    // (a knee constant or exposure offset off by a hair moves this one).
    let p = run_it(&["-w", "4", "-e", "2"]);
    assert!(p.pixels().iter().all(|px| (px.r, px.g, px.b) == (236, 236, 236)), "{:?}", &p.pixels()[..2]);
    // The pixels of the file itself are untouched.
    let after = Image::read(&out).expect("read the output");
    for (name, data) in img.channels() {
        assert_eq!(after.channel(name), Some(data), "channel {name}");
    }
}

/// Bad input, bad options and unwritable outputs end with the same exit code and stdout; the input
/// survives a same-file request untouched.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn failures_are_identical() {
    let pair = Pair::new("exrmakepreview", "preview");
    let (cdir, odir) = pair.dirs();
    let good = golden("golden_zip.exr");
    let mut report = Report::default();
    let mut fails = 0;
    let mut case = |what: &str, args: Vec<std::ffi::OsString>| {
        let verdict = pair.check(&args, &[OUT]);
        fails += usize::from(verdict == Ok(false));
        report.check(what.to_string(), verdict.map(|_| ()));
    };
    let g = good.as_os_str();
    let os = |s: &str| std::ffi::OsString::from(s);
    case("missing input", vec![os("missing.exr"), os(OUT)]);
    case("unwritable output", vec![g.into(), os("no/such/dir/out.exr")]);
    case("zero width", vec![os("-w"), os("0"), g.into(), os(OUT)]);
    case("negative width", vec![os("-w"), os("-5"), g.into(), os(OUT)]);
    case("width that is no number", vec![os("-w"), os("wide"), g.into(), os(OUT)]);
    case("missing -w value", vec![g.into(), os(OUT), os("-w")]);
    case("missing -e value", vec![g.into(), os(OUT), os("-e")]);
    case("one file name", vec![g.into()]);
    case("one file name and options", vec![os("-w"), os("9"), g.into()]);
    for deep in ["golden_deep_scanline_none.exr", "golden_deep_tiled_zips_one.exr"] {
        case(deep, vec![golden(deep).into_os_string(), os(OUT)]);
    }
    for d in [&cdir, &odir] {
        std::fs::write(d.join("not_exr.txt"), b"this is not an image").expect("write the text file");
    }
    case("not an EXR file", vec![os("not_exr.txt"), os(OUT)]);
    assert!(fails >= 8, "only {fails} failures compared");
    report.finish(&format!("preview failures ({fails} failing cases)"));

    // The same file twice (even spelled differently) is refused, and the input stays as it was.
    let copy = cdir.join("same.exr");
    std::fs::copy(&good, &copy).expect("copy the input");
    let before = std::fs::read(&copy).expect("read the copy");
    let c = run(&cpp("exrmakepreview"), [&copy, &copy]);
    let o = run_ours("preview", [&copy, &copy]);
    assert_eq!((c.code, o.code), (Some(1), Some(1)));
    let dotted = cdir.join(".").join("same.exr");
    let o = run_ours("preview", [&copy, &dotted]);
    assert_eq!(o.code, Some(1), "two spellings of one path");
    assert_eq!(std::fs::read(&copy).expect("read the copy"), before, "the input changed");
}

/// `-v`: the three lines on stdout, for a scanline and a tiled input.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn verbose_text_is_identical() {
    let pair = Pair::new("exrmakepreview", "preview");
    let mut report = Report::default();
    for file in [golden("golden_zip.exr"), golden("golden_tiled_zip_mipmap.exr")] {
        for opts in [&["-v"][..], &["-v", "-w", "20", "-e", "1"]] {
            let verdict = pair.check(line(opts, &file), &[OUT]);
            assert_eq!(verdict, Ok(true));
            report.check(format!("preview {} {}", opts.join(" "), file.display()), verdict.map(|_| ()));
        }
    }
    let (cdir, _) = pair.dirs();
    let c = run_in(&cdir, &cpp("exrmakepreview"), ["-v", golden("golden_zip.exr").to_str().expect("utf-8"), OUT]);
    assert!(c.text().starts_with("generating preview image\ncopying ") && c.text().ends_with("done.\n"), "{}", c.text());
    report.finish("preview -v text");
}

/// `-h` prints an option reference (the wording differs by design) and succeeds, anywhere.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn help_succeeds() {
    for args in [&["-h"][..], &["--help"], &["in.exr", "out.exr", "-h"], &["-w", "9", "--help"]] {
        let (c, o) = (run(&cpp("exrmakepreview"), args), run_ours("preview", args));
        assert_eq!((c.code, o.code), (Some(0), Some(0)), "{args:?}");
        assert!(c.text().contains("-w") && o.text().contains("-w") && o.text().contains("Examples:"), "{args:?}");
    }
}

/// `--version`: same text, exit code 0 (alone or after file names).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn version_text_is_identical() {
    assert_same_version("exrmakepreview", "preview", &[&["--version"], &["in.exr", "out.exr", "--version"]]);
}

/// A multi-strip-sized input (the RGBA read is one call, but the copy pass sees 39 MB) and the
/// pixel aspect ratios that change the preview height, the ones that make it 1 (`1e-30` overflows
/// the `int`, `1e30` rounds to 0) and the ones the readers refuse (0, negative).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn generated_inputs_and_aspect_ratios_are_identical() {
    use exr_core::attr::{Compression, LineOrder};
    let (pair, dir) = (
        Pair::new("exrmakepreview", "preview"),
        TempDir::new("preview-gen"),
    );
    let mut report = Report::default();
    let big = dir.join("big.exr");
    synth::pattern(
        &big,
        (8192, 300),
        (-4, 9),
        LineOrder::DecreasingY,
        Compression::NoCompression,
    );
    let mut made = 0;
    for opts in [&[][..], &["-w", "640", "-e", "1"]] {
        let verdict = pair.check(line(opts, &big), &[OUT]);
        made += usize::from(verdict == Ok(true));
        report.check(
            format!("preview {} big", opts.join(" ")),
            verdict.map(|_| ()),
        );
    }
    let mut refused = 0;
    for (i, par) in [
        2.5f32, 0.37, 1e-3, 1000.0, 1.0, 1e-30, 1e30, 0.0, -1.0, -1e-30,
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.join(&format!("par{i}.exr"));
        synth::pattern(
            &path,
            (64, 48),
            (0, 0),
            LineOrder::IncreasingY,
            Compression::NoCompression,
        );
        synth::set_aspect(&path, par);
        for opts in [&[][..], &["-w", "37"], &["-w", "1"], &["-w", "999"]] {
            let verdict = if par.to_bits() == (1e-3f32).to_bits() && opts == ["-w", "999"] {
                assert_eq!(par.to_bits(), 0x3a83126f);
                wire_invalid_case(&pair, &path)
            } else {
                pair.check(line(opts, &path), &[OUT])
            };
            made += usize::from(verdict == Ok(true));
            refused += usize::from(verdict == Ok(false));
            report.check(
                format!("preview {} aspect {par}", opts.join(" ")),
                verdict.map(|_| ()),
            );
        }
    }
    assert!(
        made >= 15 && refused >= 15,
        "{made} previews, {refused} refusals"
    );
    report.finish(&format!("preview generated inputs: 41 paired cases + 1 wire-invalid Rust refusal (C++ allocation omitted; {made} previews, {refused} refused)"));
}
