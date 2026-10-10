//! `exr-rs attr` against OpenEXR's `exrstdattr` (v3.4.15): every option, combinations, the `-part`
//! selection, erasing, and the error cases, over one file of each storage kind (scanline, tiled,
//! deep scanline, deep tiled, multipart) plus a pass-through sweep of the whole image set. Each run
//! must end with the same exit code and, on success, the same output file byte for byte.
//! `#[ignore]`d: run by `cargo run -p xtask -- parity-tools`.

mod tool_support;

use std::path::{Path, PathBuf};

use tool_support::*;

/// One file per storage kind, from the committed goldens.
fn samples() -> Vec<(&'static str, PathBuf)> {
    let all = goldens();
    let pick = |name: &str| {
        all.iter().find(|p| p.ends_with(name)).cloned().unwrap_or_else(|| panic!("golden {name} is missing"))
    };
    vec![
        ("scanline", pick("golden_header_attrs.exr")),
        ("tiled", pick("golden_tiled_rgba.exr")),
        ("deep scanline", pick("golden_deep_scanline_zips.exr")),
        ("deep tiled", pick("golden_deep_tiled_zips_one.exr")),
        ("multipart", pick("golden_multipart.exr")),
    ]
}

/// Runs a command line through both tools in scratch directories.
struct Pair {
    cpp_exe: PathBuf,
    cdir: TempDir,
    odir: TempDir,
    /// How many `compare` runs the C++ tool finished successfully.
    cpp_ok: std::cell::Cell<usize>,
}

impl Pair {
    fn new() -> Self {
        Pair {
            cpp_exe: cpp("exrstdattr"),
            cdir: TempDir::new("attr-cpp"),
            odir: TempDir::new("attr-ours"),
            cpp_ok: Default::default(),
        }
    }

    /// `args input out` through both; exit codes must agree and, on success, the files too.
    fn compare(&self, args: &[&str], input: &Path) -> Result<(), String> {
        let (cout, oout) = (self.cdir.join("out.exr"), self.odir.join("out.exr"));
        // A failed run may leave a partial file behind; never let it leak into the next case.
        std::fs::remove_file(&cout).ok();
        std::fs::remove_file(&oout).ok();
        let mut c_args: Vec<&std::ffi::OsStr> = args.iter().map(|a| a.as_ref()).collect();
        let mut o_args = c_args.clone();
        c_args.extend([input.as_os_str(), cout.as_os_str()]);
        o_args.extend([input.as_os_str(), oout.as_os_str()]);
        let (c, o) = (run(&self.cpp_exe, &c_args), run_ours("attr", &o_args));
        self.cpp_ok.set(self.cpp_ok.get() + usize::from(c.ok()));
        same_run(&c, &o, || same_bytes(&cout, &oout))
    }

    /// `args input out` with both tools: both must fail with exit code 1 (the C++ code for every
    /// error of this tool).
    fn both_fail(&self, args: &[&str], input: &Path) -> Result<(), String> {
        self.fail_with(|cout| {
            let mut a: Vec<std::ffi::OsString> = args.iter().map(Into::into).collect();
            a.extend([input.into(), cout.into()]);
            a
        })
    }

    /// `input out args`: the options come last, so a value missing at the end of the command line
    /// is really missing (after the file names there is nothing to take as its value).
    fn both_fail_last(&self, args: &[&str], input: &Path) -> Result<(), String> {
        self.fail_with(|cout| {
            let mut a: Vec<std::ffi::OsString> = vec![input.into(), cout.into()];
            a.extend(args.iter().map(Into::into));
            a
        })
    }

    fn fail_with(&self, argv: impl Fn(&Path) -> Vec<std::ffi::OsString>) -> Result<(), String> {
        let (cout, oout) = (self.cdir.join("out.exr"), self.odir.join("out.exr"));
        let (c, o) = (run(&self.cpp_exe, argv(&cout)), run_ours("attr", argv(&oout)));
        if c.code != Some(1) {
            return Err(format!("the C++ tool did not fail with 1: {:?}", c.code));
        }
        same_run(&c, &o, || Ok(()))
    }
}

/// Every option with representative values, the C++ single-dash spellings.
const OPTIONS: &[&[&str]] = &[
    &["-chromaticities", "0.64", "0.33", "0.3", "0.6", "0.15", "0.06", "0.3127", "0.329"],
    &["-whiteLuminance", "100"],
    &["-whiteLuminance", "0"],
    &["-adoptedNeutral", "0.3127", "0.329"],
    &["-renderingTransform", "rrt"],
    &["-lookModTransform", "look mod"],
    &["-xDensity", "72"],
    &["-owner", "Jane Doe"],
    &["-owner", ""],
    &["-comments", "a longer comment, with punctuation: 1;2"],
    &["-capDate", "2026:09:29 12:30:59"],
    &["-capDate", "2024:02:29 00:00:00"],
    &["-utcOffset", "-7200"],
    &["-utcOffset", "19800.5"],
    &["-longitude", "-122.4194"],
    &["-latitude", "37.7749"],
    &["-altitude", "16"],
    &["-focus", "2.5"],
    &["-focus", "inf"],
    &["-focus", "infinity"],
    &["-expTime", "0.0166667"],
    &["-aperture", "5.6"],
    &["-isoSpeed", "800"],
    &["-envmap", "latlong"],
    &["-envmap", "LATLONG"],
    &["-envmap", "cube"],
    &["-envmap", "CUBE"],
    &["-framesPerSecond", "24000", "1001"],
    &["-framesPerSecond", "25", "1"],
    &["-keyCode", "12", "3", "456789", "42", "7", "4", "64"],
    &["-keyCode", "0", "0", "0", "0", "0", "1", "20"],
    &["-keyCode", "99", "99", "999999", "9999", "119", "15", "120"],
    &["-timeCode", "0x01020304", "0xdeadbeef"],
    &["-timeCode", "01020304", "deadbeef"],
    &["-timeCode", "0", "0"],
    &["-wrapmodes", "clamp"],
    &["-wrapmodes", "periodic,clamp"],
    &["-pixelAspectRatio", "2"],
    &["-pixelAspectRatio", "0.5"],
    &["-screenWindowWidth", "3"],
    &["-screenWindowWidth", "0"],
    &["-screenWindowCenter", "0.5", "-0.5"],
    &["-string", "note", "hello world"],
    &["-string", "owner", "replaces the string"],
    &["-float", "gain", "1.25"],
    &["-float", "tiny", "1e-30"],
    &["-int", "take", "42"],
    &["-int", "hex", "0x10"],
    &["-int", "octal", "010"],
    &["-int", "negative", "-7"],
];

/// One of each option on every storage kind.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn every_option_on_every_storage_kind() {
    let pair = Pair::new();
    let mut report = Report::default();
    for (kind, file) in samples() {
        for opt in OPTIONS {
            report.check(format!("{kind}: {}", opt.join(" ")), pair.compare(opt, &file));
        }
    }
    report.finish("attr options x storage kinds");
}

/// All the options in one command line, in both orders (later options win over earlier ones
/// setting the same name), and repeated options.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn combinations() {
    let pair = Pair::new();
    let all: Vec<&str> = OPTIONS.iter().take(18).flat_map(|o| o.iter().copied()).collect();
    // Whole options reversed (a reversed token list would separate the values from their options).
    let reversed: Vec<&str> = OPTIONS.iter().take(18).rev().flat_map(|o| o.iter().copied()).collect();
    let lines: Vec<(&str, Vec<&str>)> = vec![
        ("the first 18 options", all),
        ("the same, reversed option order", reversed),
        ("the same option twice", vec!["-owner", "first", "-owner", "second", "-int", "n", "1", "-int", "n", "2"]),
        ("different types, one name, erased between", vec!["-string", "t", "s", "-erase", "t", "-int", "t", "1"]),
        ("set then erase the same name", vec!["-owner", "x", "-erase", "owner"]),
        ("erase then set: the erase is applied first", vec!["-erase", "owner", "-owner", "y"]),
        ("a custom name that is a standard one, same type", vec!["-float", "screenWindowWidth", "4"]),
        ("custom int over a standard int", vec!["-int", "parityTag", "9"]),
    ];
    let mut report = Report::default();
    for (kind, file) in samples() {
        for (what, args) in &lines {
            report.check(format!("{kind}: {what}"), pair.compare(args, &file));
        }
    }
    report.finish("attr combinations");
}

/// `-erase` of existing, missing and essential attributes.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn erase() {
    let pair = Pair::new();
    let (_, single) = samples().swap_remove(0);
    let (_, multi) = samples().swap_remove(4);
    let mut report = Report::default();
    for (what, args, file) in [
        ("existing", vec!["-erase", "owner"], &single),
        (
            "several",
            vec!["-erase", "owner", "-erase", "parityTag", "-erase", "myRational", "-erase", "chromaticities"],
            &single,
        ),
        ("missing", vec!["-erase", "no_such_attribute"], &single),
        ("erase and re-insert with another type", vec!["-erase", "owner", "-int", "owner", "5"], &single),
        ("in multipart, all parts", vec!["-erase", "owner"], &multi),
        ("in multipart, one part", vec!["-part", "0", "-erase", "owner"], &multi),
        ("in multipart, another part", vec!["-part", "1", "-erase", "owner"], &multi),
    ] {
        report.check(what, pair.compare(&args, file));
    }
    for (what, args) in [
        ("essential: channels", vec!["-erase", "channels"]),
        ("essential: compression", vec!["-erase", "compression"]),
        ("essential: dataWindow", vec!["-erase", "dataWindow"]),
        ("essential: displayWindow", vec!["-erase", "displayWindow"]),
        ("essential: lineOrder", vec!["-erase", "lineOrder"]),
        ("essential: pixelAspectRatio", vec!["-erase", "pixelAspectRatio"]),
        ("essential: screenWindowCenter", vec!["-erase", "screenWindowCenter"]),
        ("essential: screenWindowWidth", vec!["-erase", "screenWindowWidth"]),
        ("an empty name", vec!["-erase", ""]),
    ] {
        report.check(what, pair.both_fail(&args, &single));
    }
    report.finish("attr erase");
}

/// `-part`: a number, `any`, -1, several in one line, out-of-range numbers (errors), and a part
/// number that is never used (no error).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn part_selection() {
    let pair = Pair::new();
    let (_, multi) = samples().swap_remove(4);
    let (_, single) = samples().swap_remove(0);
    let mut report = Report::default();
    for (what, args, file) in [
        ("part 0", vec!["-part", "0", "-owner", "p0"], &multi),
        ("part 1", vec!["-part", "1", "-owner", "p1"], &multi),
        ("part 2", vec!["-part", "2", "-owner", "p2"], &multi),
        ("any", vec!["-part", "any", "-owner", "all"], &multi),
        ("-1", vec!["-part", "-1", "-owner", "all"], &multi),
        (
            "before the first -part all headers are current",
            vec!["-focus", "3", "-part", "1", "-aperture", "8", "-expTime", "0.01", "-part", "any", "-owner", "luke"],
            &multi,
        ),
        (
            "the help text's example",
            vec!["-focus", "3", "-part", "2", "-aperture", "8", "-expTime", "0.01", "-part", "any", "-owner", "luke"],
            &multi,
        ),
        (
            "each part its own",
            vec!["-part", "0", "-int", "k", "0", "-part", "1", "-int", "k", "1", "-part", "2", "-int", "k", "2"],
            &multi,
        ),
        ("hex and octal part numbers", vec!["-part", "0x1", "-owner", "hex"], &multi),
        ("part 0 on a single-part file", vec!["-part", "0", "-owner", "x"], &single),
        ("-part alone selects nothing", vec!["-part", "5"], &single),
        ("-part that no option uses", vec!["-owner", "x", "-part", "9"], &single),
    ] {
        report.check(what, pair.compare(&args, file));
    }
    for (what, args, file) in [
        ("part past the end", vec!["-part", "3", "-owner", "x"], &multi),
        ("part 1 of a single-part file", vec!["-part", "1", "-owner", "x"], &single),
        ("negative part", vec!["-part", "-2", "-owner", "x"], &multi),
        ("out-of-range part erase", vec!["-part", "7", "-erase", "owner"], &multi),
        ("erase of the required multipart name", vec!["-part", "0", "-erase", "name"], &multi),
    ] {
        report.check(what, pair.both_fail(&args, file));
    }
    report.finish("attr -part");
}

/// Invalid values and malformed command lines end with exit code 1 in both tools.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn errors() {
    let pair = Pair::new();
    let (_, file) = samples().swap_remove(0);
    let mut report = Report::default();
    for args in [
        vec!["-xDensity", "0"],
        vec!["-xDensity", "-1"],
        vec!["-expTime", "0"],
        vec!["-aperture", "-2"],
        vec!["-isoSpeed", "0"],
        vec!["-pixelAspectRatio", "0"],
        vec!["-screenWindowWidth", "-0.5"],
        vec!["-focus", "0"],
        vec!["-focus", "-3"],
        vec!["-envmap", "Cube"],
        vec!["-envmap", "sphere"],
        vec!["-capDate", "2026-09-29 12:30:59"],
        vec!["-capDate", "2026:13:01 00:00:00"],
        vec!["-capDate", "2026:02:30 00:00:00"],
        vec!["-capDate", "2025:02:29 00:00:00"],
        vec!["-capDate", "2026:01:01 24:00:00"],
        vec!["-capDate", "2026:01:01 00:60:00"],
        vec!["-capDate", "2026:01:01 00:00:60"],
        vec!["-capDate", "short"],
        vec!["-keyCode", "100", "0", "0", "0", "0", "4", "64"],
        vec!["-keyCode", "0", "0", "0", "0", "0", "0", "64"],
        vec!["-keyCode", "0", "0", "0", "0", "0", "4", "19"],
        vec!["-keyCode", "0", "0", "0", "0", "120", "4", "64"],
        // Type clashes with the file's own attributes.
        vec!["-float", "owner", "1"],
        vec!["-int", "owner", "1"],
        vec!["-string", "compression", "zip"],
        vec!["-string", "channels", "x"],
        vec!["-int", "chromaticities", "1"],
        vec!["-float", "", "1"],
        vec!["-string", "", "x"],
    ] {
        report.check(args.join(" "), pair.both_fail(&args, &file));
    }
    // Too few values: the option is the last word of the command line.
    for args in [
        vec!["-owner"],
        vec!["-chromaticities", "1", "2", "3"],
        vec!["-adoptedNeutral", "0.3"],
        vec!["-int", "x"],
        vec!["-string", "x"],
        vec!["-float", "x"],
        vec!["-framesPerSecond", "24"],
        vec!["-keyCode", "1", "2", "3"],
        vec!["-timeCode", "1"],
        vec!["-xDensity"],
        vec!["-focus"],
        vec!["-envmap"],
        vec!["-part"],
        vec!["-erase"],
    ] {
        report.check(format!("{} (last)", args.join(" ")), pair.both_fail_last(&args, &file));
    }
    report.finish("attr errors");

    // Missing files and file-name problems (no options involved).
    let dir = TempDir::new("attr-files");
    let (cpp_exe, out) = (cpp("exrstdattr"), dir.join("out.exr"));
    let (missing, same) = (dir.join("missing.exr"), file.clone());
    let mut report = Report::default();
    let cases: Vec<(&str, Vec<std::ffi::OsString>)> = vec![
        ("missing input", vec![missing.clone().into_os_string(), out.clone().into_os_string()]),
        ("input and output the same file", vec![same.clone().into_os_string(), same.into_os_string()]),
        ("unwritable output", vec![file.clone().into_os_string(), dir.join("no/such/dir/out.exr").into_os_string()]),
        ("one file name only", vec![file.clone().into_os_string()]),
        (
            "three file names",
            vec![file.clone().into_os_string(), out.clone().into_os_string(), out.clone().into_os_string()],
        ),
    ];
    for (what, args) in cases {
        let (c, o) = (run(&cpp_exe, &args), run_ours("attr", &args));
        report.check(what, same_run(&c, &o, || Err("a file-name problem succeeded".into())));
        assert_eq!(c.code, Some(1), "{what}: the C++ tool's own code");
    }
    report.finish("attr file-name errors");
}

/// A plain pass-through (`-int pass 1`, one attribute so the header really changes) of every
/// input: the C++ tool's failures must be ours, its files ours, over the whole image set. Plus
/// the same with `-part any` and an erase.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn pass_through_over_every_input() {
    let pair = Pair::new();
    let mut report = Report::default();
    let mut total = 0;
    for file in inputs() {
        total += 2;
        for args in [vec!["-int", "pass", "1"], vec!["-part", "any", "-owner", "o", "-erase", "comments"]] {
            let ok = pair.compare(&args, &file);
            report.check(format!("{} {}", args.join(" "), file.display()), ok);
        }
    }
    let copied = pair.cpp_ok.get();
    assert!(copied * 10 >= total * 8, "only {copied} of {total} runs copied: the comparison proves little");
    report.finish(&format!("attr pass-through ({copied} of {total} runs copied)"));
}

/// The input is never the output: both tools refuse, and (what the exit code alone cannot show)
/// the input file is byte-unchanged afterwards. Works on temp copies of a golden, never the
/// committed file. Ours also refuses a second spelling of the same path (the C++ compares the
/// strings only and would truncate the input, so it is not run on that case).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn same_file_is_refused_and_the_input_untouched() {
    let (_, golden) = samples().swap_remove(0);
    let original = std::fs::read(&golden).unwrap();
    let dir = TempDir::new("attr-same");
    let cpp_exe = cpp("exrstdattr");
    let (for_cpp, for_ours, for_dotted) = (dir.join("c.exr"), dir.join("o.exr"), dir.join("d.exr"));
    for f in [&for_cpp, &for_ours, &for_dotted] {
        std::fs::write(f, &original).unwrap();
    }
    let c = run(&cpp_exe, ["-owner".as_ref(), "x".as_ref(), for_cpp.as_os_str(), for_cpp.as_os_str()]);
    let o = run_ours("attr", ["-owner".as_ref(), "x".as_ref(), for_ours.as_os_str(), for_ours.as_os_str()]);
    let dotted = dir.join(".").join("d.exr");
    let d = run_ours("attr", ["-owner".as_ref(), "x".as_ref(), for_dotted.as_os_str(), dotted.as_os_str()]);
    assert_eq!((c.code, o.code, d.code), (Some(1), Some(1), Some(1)));
    for f in [&for_cpp, &for_ours, &for_dotted] {
        assert!(std::fs::read(f).unwrap() == original, "{} was modified", f.display());
    }
}

/// `nan` parses to a different NaN payload than MSVC's `strtod` gives (`opts.rs` module docs): the
/// files are equal except for the four value bytes, which are a NaN in both.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn nan_attribute_values_differ_only_in_payload() {
    let pair = Pair::new();
    let (_, file) = samples().swap_remove(0);
    for args in [&["-float", "f", "nan"][..], &["-xDensity", "nan"], &["-float", "f", "-nan"]] {
        let (cout, oout) = (pair.cdir.join("out.exr"), pair.odir.join("out.exr"));
        let mut a: Vec<&std::ffi::OsStr> = args.iter().map(|s| s.as_ref()).collect();
        a.extend([file.as_os_str(), cout.as_os_str()]);
        let c = run(&pair.cpp_exe, &a);
        a.truncate(args.len());
        a.extend([file.as_os_str(), oout.as_os_str()]);
        let o = run_ours("attr", &a);
        assert!(c.ok() && o.ok(), "{args:?}: {:?} / {:?}", c.code, o.code);
        let (x, y) = (std::fs::read(&cout).unwrap(), std::fs::read(&oout).unwrap());
        assert_eq!(x.len(), y.len(), "{args:?}");
        let diffs: Vec<usize> = (0..x.len()).filter(|&i| x[i] != y[i]).collect();
        let (lo, hi) = (*diffs.first().expect("the NaN payloads differ"), *diffs.last().unwrap());
        assert!(hi - lo < 4, "{args:?}: differences span more than one float: {diffs:?}");
        let neg = args.contains(&"-nan");
        let (cw, ow) = (&x[lo..lo + 4], &y[lo..lo + 4]);
        // Ours: the canonical quiet NaN with the sign as typed; MSVC: all payload bits set.
        assert_eq!(ow, if neg { [0, 0, 0xc0, 0xff] } else { [0, 0, 0xc0, 0x7f] }, "{args:?} ours");
        assert_eq!(cw, if neg { [0xff; 4] } else { [0xff, 0xff, 0xff, 0x7f] }, "{args:?} C++");
    }
}

/// `--version`: same text, exit code 0 (alone or after other arguments).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn version_text_is_identical() {
    assert_same_version("exrstdattr", "attr", &[&["--version"], &["-owner", "x", "in.exr", "out.exr", "--version"]]);
}

/// Owner decision (`opts.rs` module docs): a negative `-whiteLuminance` is refused. The C++ tool
/// accepts it and writes the file; we exit 1 and write nothing. The input (a temp copy) is untouched.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn white_luminance_negative_is_refused() {
    let (_, golden) = samples().swap_remove(0);
    let original = std::fs::read(&golden).unwrap();
    let dir = TempDir::new("attr-whitelum");
    let input = dir.join("in.exr");
    std::fs::write(&input, &original).unwrap();
    let (cout, oout) = (dir.join("c.exr"), dir.join("o.exr"));
    let args = |out: &std::path::Path| {
        vec!["-whiteLuminance".into(), "-1".into(), input.clone().into_os_string(), out.as_os_str().to_owned()]
    };
    let c = run(&cpp("exrstdattr"), args(&cout));
    let o = run_ours("attr", args(&oout));
    assert!(c.ok() && cout.is_file(), "the C++ tool should accept and write: {:?}", c.code);
    assert_eq!(o.code, Some(1), "ours must refuse");
    assert!(String::from_utf8_lossy(&o.stderr).contains("whiteLuminance"), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!oout.exists(), "nothing may be written");
    assert!(std::fs::read(&input).unwrap() == original, "the input was modified");
    // Zero and positive values are still identical to the C++ (covered by `every_option_*`).
}
