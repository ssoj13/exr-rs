//! `exr-rs tile` against OpenEXR's `exrmaketiled` (v3.4.15): exit code, stdout and the output file,
//! byte for byte, over the goldens and the private image set, for every option and the
//! combinations that change the output (levels, rounding, tile size, extrapolation, unfiltered
//! channels, codec, part). Failures must end alike. `#[ignore]`d: run by
//! `cargo run -p xtask -- parity-tools`.

mod tool_support;

use std::path::{Path, PathBuf};

use exr_core::io::StdIfStream;
use exr_core::MultiPartInputFile;
use tool_support::*;

const OUT: &str = "out.exr";

/// `args` then the input and `out.exr`, as the tool wants them.
fn line(opts: &[&str], input: &Path) -> Vec<std::ffi::OsString> {
    let mut v: Vec<std::ffi::OsString> = opts.iter().map(|o| std::ffi::OsString::from(*o)).collect();
    v.push(input.into());
    v.push(OUT.into());
    v
}

fn golden(name: &str) -> PathBuf {
    goldens().into_iter().find(|p| p.ends_with(name)).unwrap_or_else(|| panic!("golden {name} not found"))
}

fn size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// How many parts the file has (0 when it cannot be opened).
fn parts(p: &Path) -> i32 {
    StdIfStream::open(p.to_str().expect("utf-8 path"))
        .and_then(|mut s| MultiPartInputFile::new(&mut s).map(|f| f.parts()))
        .unwrap_or(0)
}

/// Run `opts` over `files` through both tools, collect the verdicts; how many conversions the
/// C++ tool did.
fn sweep(pair: &Pair, report: &mut Report, opts: &[&str], files: &[PathBuf]) -> usize {
    let mut made = 0;
    for f in files {
        let verdict = pair.check(line(opts, f), &[OUT]);
        if let Ok(true) = verdict {
            made += 1;
        }
        report.check(format!("tile {} {}", opts.join(" "), f.display()), verdict.map(|_| ()));
    }
    made
}

/// Every input through the default, MIP, RIP (small files: there are 100+ levels) and multipart
/// settings.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn output_files_are_identical_over_every_input() {
    let pair = Pair::new("exrmaketiled", "tile");
    let files = inputs();
    let small: Vec<PathBuf> = files.iter().filter(|f| size(f) < 1_500_000).cloned().collect();
    let mut report = Report::default();
    let mut made = 0;
    made += sweep(&pair, &mut report, &[], &files);
    made += sweep(&pair, &mut report, &["-m"], &files);
    made += sweep(&pair, &mut report, &["-r", "-u", "-t", "32", "16"], &small);
    made += sweep(&pair, &mut report, &["-m", "-e", "mirror", "periodic", "-f", "Z", "-z", "piz"], &small);
    made += sweep(&pair, &mut report, &["-p", "1"], &small);
    assert!(made >= 500, "only {made} conversions: the comparison proves little");
    report.finish(&format!("tile over goldens + image set ({made} conversions)"));
}

/// The option matrix on files of every shape: mixed sample types, tiled with RIP levels, a
/// decreasing line order, long attribute lists.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn every_option_on_representative_inputs() {
    let pair = Pair::new("exrmaketiled", "tile");
    let files: Vec<PathBuf> =
        ["golden_zip.exr", "golden_rgba.exr", "golden_tiled_zip_ripmap.exr", "golden_tiled_zip_one_decreasing.exr", "golden_header_attrs.exr", "golden_b44.exr"]
            .map(golden)
            .into();
    let mut sets: Vec<Vec<String>> = Vec::new();
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    // Levels x rounding x tile size.
    for mode in ["-o", "-m", "-r"] {
        for round in ["-d", "-u"] {
            for tile in [None, Some(("16", "16")), Some(("7", "5")), Some(("1", "1")), Some(("200", "3"))] {
                let mut o = s(&[mode, round]);
                if let Some((x, y)) = tile {
                    o.extend(s(&["-t", x, y]));
                }
                sets.push(o);
            }
        }
    }
    // Every extrapolation pair, both ways of making levels.
    let modes = ["black", "clamp", "periodic", "mirror"];
    for x in modes {
        for y in modes {
            sets.push(s(&["-m", "-e", x, y]));
            sets.push(s(&["-r", "-u", "-t", "16", "16", "-e", &x.to_uppercase(), &y.to_uppercase()]));
        }
    }
    // Channels resampled without filtering (one, several, all, one that is not there).
    let unfiltered: [&[&str]; 4] =
        [&["-f", "R"], &["-f", "A", "-f", "U", "-f", "Z"], &["-f", "R", "-f", "G", "-f", "B", "-f", "A", "-f", "U", "-f", "Z"], &["-f", "nothere"]];
    for f in unfiltered {
        for mode in ["-m", "-r"] {
            let mut o = s(&[mode, "-t", "16", "16"]);
            o.extend(s(f));
            sets.push(o);
        }
    }
    // Every codec, with levels.
    for c in ["no", "none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "htj2k256", "htj2k32", "ZIP", "Piz"] {
        sets.push(s(&["-m", "-z", c]));
    }
    // Option order and repeats: the last of competing options wins.
    sets.push(s(&["-m", "-r", "-o", "-u", "-d"]));
    sets.push(s(&["-t", "8", "8", "-m", "-t", "32", "32"]));
    let mut report = Report::default();
    let mut made = 0;
    for o in &sets {
        let opts: Vec<&str> = o.iter().map(String::as_str).collect();
        made += sweep(&pair, &mut report, &opts, &files);
    }
    assert!(made >= sets.len() * files.len() / 2, "only {made} conversions of {}", sets.len() * files.len());
    report.finish(&format!("tile option matrix ({} option sets x {} inputs, {made} conversions)", sets.len(), files.len()));
}

/// Part selection on every multipart input, valid and out of range.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn part_selection_is_identical() {
    let pair = Pair::new("exrmaketiled", "tile");
    let multi: Vec<PathBuf> = inputs().into_iter().filter(|f| size(f) < 12_000_000 && parts(f) > 1).collect();
    assert!(multi.len() >= 3, "few multipart inputs: {multi:?}");
    let mut report = Report::default();
    let mut made = 0;
    for f in &multi {
        let n = parts(f);
        for p in -1..=n {
            let p = p.to_string();
            for opts in [&["-p", &p][..], &["-m", "-p", &p], &["-p", &p, "-z", "zips"]] {
                let verdict = pair.check(line(opts, f), &[OUT]);
                made += usize::from(verdict == Ok(true));
                report.check(format!("tile {} {}", opts.join(" "), f.display()), verdict.map(|_| ()));
            }
        }
    }
    assert!(made >= multi.len() * 3, "only {made} conversions over {} multipart files", multi.len());
    report.finish(&format!("tile -p over {} multipart files ({made} conversions)", multi.len()));
}

/// Bad input, bad options and unwritable outputs end with the same exit code and stdout; the
/// input survives a same-file request untouched.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn failures_are_identical() {
    let pair = Pair::new("exrmaketiled", "tile");
    let (cdir, _) = pair.dirs();
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
    case("unknown codec", vec![os("-z"), os("lzw"), g.into(), os(OUT)]);
    case("unknown extrapolation", vec![os("-m"), os("-e"), os("foo"), os("clamp"), g.into(), os(OUT)]);
    case("capitalised extrapolation", vec![os("-m"), os("-e"), os("Clamp"), os("clamp"), g.into(), os(OUT)]);
    case("zero tile", vec![os("-t"), os("0"), os("8"), g.into(), os(OUT)]);
    case("negative tile", vec![os("-t"), os("8"), os("-8"), g.into(), os(OUT)]);
    case("missing -z value", vec![g.into(), os(OUT), os("-z")]);
    case("missing -e values", vec![g.into(), os(OUT), os("-e"), os("clamp")]);
    case("missing -t values", vec![g.into(), os(OUT), os("-t"), os("8")]);
    case("missing -f value", vec![g.into(), os(OUT), os("-f")]);
    case("missing -p value", vec![g.into(), os(OUT), os("-p")]);
    case("part out of range", vec![os("-p"), os("1"), g.into(), os(OUT)]);
    case("negative part", vec![os("-p"), os("-1"), g.into(), os(OUT)]);
    case("one file name", vec![g.into()]);
    case("one file name and options", vec![os("-m"), g.into()]);
    for deep in ["golden_deep_scanline_none.exr", "golden_deep_tiled_zips_one.exr", "golden_deep_tiled_none_mipmap.exr"] {
        case(deep, vec![golden(deep).into_os_string(), os(OUT)]);
    }
    for sub in ["golden_sub_none_x2y2.exr", "golden_sub_zip_x2y1.exr", "golden_yc.exr"] {
        case(sub, vec![golden(sub).into_os_string(), os(OUT)]);
    }
    let (cd, od) = pair.dirs();
    for d in [cd, od] {
        std::fs::write(d.join("not_exr.txt"), b"this is not an image").expect("write the text file");
    }
    case("not an EXR file", vec![os("not_exr.txt"), os(OUT)]);
    // Environment maps cannot be made multiresolution here.
    for env in inputs().into_iter().filter(|f| f.to_string_lossy().to_lowercase().contains("latlong") || f.to_string_lossy().to_lowercase().contains("cube")) {
        case("envmap -m", vec![os("-m"), env.clone().into_os_string(), os(OUT)]);
        case("envmap -o", vec![env.into_os_string(), os(OUT)]);
    }
    assert!(fails >= 18, "only {fails} failures compared");
    report.finish(&format!("tile failures ({fails} failing cases)"));

    // The same file twice is refused, and the input stays as it was.
    let copy = cdir.join("same.exr");
    std::fs::copy(&good, &copy).expect("copy the input");
    let before = std::fs::read(&copy).expect("read the copy");
    let (c, o) = (run(&cpp("exrmaketiled"), [&copy, &copy]), run_ours("tile", [&copy, &copy]));
    assert_eq!((c.code, o.code), (Some(1), Some(1)));
    assert_eq!(std::fs::read(&copy).expect("read the copy"), before, "the input changed");
}

/// `-v`, `-p`'s echo and the `done.` line, on stdout, for every kind of run.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn verbose_text_is_identical() {
    let pair = Pair::new("exrmaketiled", "tile");
    let mut report = Report::default();
    let multi = inputs().into_iter().find(|f| f.ends_with("golden_multipart.exr")).expect("golden_multipart.exr");
    let zip = golden("golden_zip.exr");
    let runs: [(&[&str], PathBuf); 6] = [
        (&["-v"], zip.clone()),
        (&["-m", "-v"], zip.clone()),
        (&["-r", "-v", "-u", "-t", "16", "16"], zip),
        (&["-v", "-p", "1"], multi.clone()),
        (&["-p", "0", "-m", "-v"], multi.clone()),
        (&["-p", "0x1"], multi),
    ];
    for (opts, file) in runs {
        let verdict = pair.check(line(opts, &file), &[OUT]);
        assert_eq!(verdict, Ok(true), "tile {} {}", opts.join(" "), file.display());
        report.check(format!("tile {}", opts.join(" ")), verdict.map(|_| ()));
    }
    // The text really is there: the C++ prints the levels and `done.`.
    let (cdir, _) = pair.dirs();
    let c = run_in(&cdir, &cpp("exrmaketiled"), ["-m", "-v", golden("golden_zip.exr").to_str().expect("utf-8"), OUT]);
    assert!(c.text().contains("level (1, 1)") && c.text().ends_with("done.\n"), "{}", c.text());
    report.finish("tile -v / -p text");
}

/// `-h` prints an option reference (the wording differs by design) and succeeds, anywhere.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn help_succeeds() {
    for args in [&["-h"][..], &["--help"], &["in.exr", "out.exr", "-h"], &["-m", "--help"]] {
        let (c, o) = (run(&cpp("exrmaketiled"), args), run_ours("tile", args));
        assert_eq!((c.code, o.code), (Some(0), Some(0)), "{args:?}");
        assert!(c.text().contains("-m") && o.text().contains("-m") && o.text().contains("Examples:"), "{args:?}");
    }
}

/// `--version`: same text, exit code 0 (alone or after file names).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn version_text_is_identical() {
    assert_same_version("exrmaketiled", "tile", &[&["--version"], &["in.exr", "out.exr", "--version"]]);
}

/// Inputs larger than one strip buffer (8192 x 300 float RGBA is 39 MB, three 16 MiB strips) in both
/// line orders: the strips must join up, for one level and for levels.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn multi_strip_inputs_are_identical() {
    use exr_core::attr::{Compression, LineOrder};
    let (pair, dir) = (Pair::new("exrmaketiled", "tile"), TempDir::new("tile-big"));
    let mut report = Report::default();
    let mut made = 0;
    for (name, order, comp, origin) in [
        ("inc.exr", LineOrder::IncreasingY, Compression::NoCompression, (0, 0)),
        ("dec.exr", LineOrder::DecreasingY, Compression::NoCompression, (-5, 7)),
        ("zip.exr", LineOrder::IncreasingY, Compression::Zip, (3, -2)),
    ] {
        let path = dir.join(name);
        synth::pattern(&path, (8192, 300), origin, order, comp);
        for opts in [&[][..], &["-m"], &["-r", "-u", "-t", "256", "64"]] {
            let verdict = pair.check(line(opts, &path), &[OUT]);
            made += usize::from(verdict == Ok(true));
            report.check(format!("tile {} {name}", opts.join(" ")), verdict.map(|_| ()));
        }
    }
    assert_eq!(made, 9, "every generated input converts");
    report.finish(&format!("tile multi-strip inputs ({made} conversions)"));
}
