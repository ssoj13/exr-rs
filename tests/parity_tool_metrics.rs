//! `exr-rs metrics` against OpenEXR's `exrmetrics` (v3.4.15): exit code, stdout and any output file,
//! over the goldens and the private image set, for every option. `#[ignore]`d: run by
//! `cargo run -p xtask -- parity-tools`.
//!
//! Times are measured, so they differ between any two runs: [`norm`] replaces every number on a time
//! line of the JSON report (and in a time column of the CSV report) by `#`, which still compares the
//! keys, the number of samples per time and the statistics' layout. Everything else, sizes and pixel,
//! channel and tile counts above all, is compared exactly, as are the files written with `-o`.
//!
//! **Single-part tiled files.** `exrmetrics` fails on those that carry no `type` attribute (nearly
//! all of them) with `Cannot find image attribute "type"`; we measure them. Such an input is run
//! again after `exrmultipart -combine` rewrote it with the attribute, which both tools must report
//! alike.

mod tool_support;

use std::ffi::OsString;
use std::path::{MAIN_SEPARATOR, Path, PathBuf};

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LevelMode, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemOStream, StdIfStream};
use exr_core::multipart::{MultiPartInputFile, MultiPartOutputFile};
use exr_core::tiled::misc::data_window_for_level;
use exr_core::StringAttribute;
use imath_rs::{Box2i, Half, V2f, V2i};
use tool_support::*;

// ----- normalizing the times ----------------------------------------------------------------

/// The JSON keys whose value is a time.
const TIME_KEYS: [&str; 5] = ["\"count read time\": ", "\"read time\": ", "\"write time\": ", "\"count re-read time\": ", "\"re-read time\": "];

/// `s` with every number, `nan` and `inf` replaced by `#` (MSVC spells a NaN `-nan(ind)`).
fn numbers_to_hash(s: &str) -> String {
    let (mut out, mut rest) = (String::new(), s);
    while let Some(c) = rest.chars().next() {
        let unsigned = rest.strip_prefix('-').unwrap_or(rest);
        let word = ["nan(ind)", "nan", "inf"].into_iter().find(|w| unsigned.starts_with(w));
        let digits = unsigned.bytes().next().is_some_and(|b| b.is_ascii_digit());
        let len = if let Some(w) = word {
            rest.len() - unsigned.len() + w.len()
        } else if digits {
            let n = unsigned.bytes().take_while(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-')).count();
            rest.len() - unsigned.len() + n
        } else {
            0
        };
        if len > 0 {
            out.push('#');
            rest = &rest[len..];
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// A report with its times made comparable.
fn norm(text: &str) -> String {
    if text.starts_with("file name,") {
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap_or_default().split(',').collect();
        let mut out = String::from(header.join(",").as_str());
        for row in lines {
            out.push('\n');
            let cells: Vec<String> = row
                .split(',')
                .enumerate()
                .map(|(i, c)| if header.get(i).is_some_and(|h| h.contains("time")) && c != "---" { numbers_to_hash(c) } else { c.to_string() })
                .collect();
            out.push_str(&cells.join(","));
        }
        out.push('\n');
        return out;
    }
    let mut out = String::new();
    for line in text.lines() {
        match TIME_KEYS.iter().find(|k| line.trim_start().starts_with(**k)) {
            Some(_) => {
                let cut = line.find(": ").map_or(0, |c| c + 2);
                out.push_str(&line[..cut]);
                // The C++ prints nothing for no samples, we `null`: the same here.
                let rest = &line[cut..];
                let rest = if rest.trim_end_matches(',') == "null" { rest.replace("null", "") } else { numbers_to_hash(rest) };
                out.push_str(&rest);
            }
            None => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

/// What the C++ gets wrong in the report of these inputs and we do not (`tile count` of a mipmap,
/// the compression, part type and tile count of a deep tiled part): those lines are not compared.
#[derive(Default, Clone, Copy)]
struct Skip {
    tiles: bool,
    deep_tiled: bool,
}

/// [`Skip`] for the input files among `args`.
fn skips(args: &[OsString]) -> Skip {
    let mut skip = Skip::default();
    for a in args.iter().map(Path::new).filter(|p| p.is_file()) {
        let Ok(mut stream) = StdIfStream::open(utf8(a)) else { continue };
        let Ok(file) = MultiPartInputFile::new(&mut stream) else { continue };
        for p in 0..file.parts() {
            let Ok(h) = file.header(p) else { continue };
            match h.type_() {
                Ok("deeptile") => skip.deep_tiled = true,
                Ok("tiledimage") => skip.tiles |= h.tile_description().is_ok_and(|t| t.mode != LevelMode::OneLevel),
                _ => {}
            }
        }
    }
    skip
}

/// [`norm`] with the lines of `skip` hidden.
fn norm_for(text: &str, skip: Skip) -> String {
    let text = norm(text);
    if !skip.tiles && !skip.deep_tiled {
        return text;
    }
    if text.starts_with("file name,") {
        let header: Vec<String> = text.lines().next().unwrap_or_default().split(',').map(String::from).collect();
        let cells = |row: &str| -> String {
            row.split(',').enumerate().map(|(i, c)| if header.get(i).is_some_and(|h| h == "tile count") { "?".to_string() } else { c.to_string() }).collect::<Vec<_>>().join(",")
        };
        return text.lines().enumerate().map(|(i, l)| if i == 0 { l.to_string() } else { cells(l) }).collect::<Vec<_>>().join("\n") + "\n";
    }
    // The size section of a file's entry is what precedes its `"metrics":` line.
    let mut in_size = false;
    let mut out = String::new();
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("\"file\":") {
            in_size = true;
        } else if t.starts_with("\"metrics\":") {
            in_size = false;
        }
        let hidden = in_size
            && ((skip.tiles || skip.deep_tiled) && t.starts_with("\"tile count\":")
                || skip.deep_tiled && (t.starts_with("\"compression\":") || t.starts_with("\"part type\":")));
        match (hidden, t.split_once(':')) {
            (true, Some((key, _))) => out.push_str(&format!("{}{key}: ?", &line[..line.len() - t.len()])),
            _ => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

// ----- the two tools ------------------------------------------------------------------------

/// What comparing one run of each tool found.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Verdict {
    /// Both succeeded with the same report and files.
    Same,
    /// Both failed (exit code 1, nothing on stdout).
    FailedAlike,
    /// The C++ tool failed on a tiled file without a `type` attribute; ours measured it.
    TypelessTiled,
}

struct Env {
    cpp: PathBuf,
    cdir: TempDir,
    odir: TempDir,
    /// Scratch for `typed_copy`: its input copies, its working directory, its outputs.
    typed: (TempDir, TempDir, TempDir),
}

fn os(words: &[&str]) -> Vec<OsString> {
    words.iter().map(OsString::from).collect()
}

impl Env {
    fn new() -> Self {
        Env {
            cpp: cpp("exrmetrics"),
            cdir: TempDir::new("metrics-cpp"),
            odir: TempDir::new("metrics-ours"),
            typed: (TempDir::new("mm-typed-in"), TempDir::new("mm-typed-work"), TempDir::new("mm-typed-out")),
        }
    }

    /// Run both tools with `args`; compare the reports (times normalized) and the `outputs` files.
    fn run(&self, args: &[OsString], outputs: &[&str]) -> Result<Verdict, String> {
        let (cd, od) = (self.cdir.join(""), self.odir.join(""));
        seed_outputs(&[&cd, &od], outputs);
        let before = snapshot(&od, outputs);
        let (c, o) = (run_in(&cd, &self.cpp, args), run_ours_in(&od, "metrics", args));
        let verdict = self.compare(&c, &o, outputs, (&cd, &od), &before, args);
        for name in outputs {
            std::fs::remove_file(cd.join(name)).ok();
            std::fs::remove_file(od.join(name)).ok();
        }
        verdict
    }

    fn compare(&self, c: &Out, o: &Out, outputs: &[&str], (cd, od): (&Path, &Path), before: &[Option<Vec<u8>>], args: &[OsString]) -> Result<Verdict, String> {
        let shown = args.iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>().join(" ");
        if c.end() != o.end() {
            if !c.ok() && o.ok() && String::from_utf8_lossy(&c.stderr).contains("Cannot find image attribute \"type\"") {
                return Ok(Verdict::TypelessTiled);
            }
            return Err(format!(
                "{shown}: runs ended differently: C++ {}, ours {}\n    C++ stderr: {}\n    our stderr: {}",
                c.end(),
                o.end(),
                String::from_utf8_lossy(&c.stderr).trim(),
                String::from_utf8_lossy(&o.stderr).trim()
            ));
        }
        if !c.ok() {
            same_failure_state(od, outputs, before).map_err(|m| format!("{shown}: {m}"))?;
            return match (c.stdout.is_empty(), o.stdout.is_empty()) {
                (true, true) => Ok(Verdict::FailedAlike),
                _ => Err(format!("{shown}: both failed, stdout differs:\n--- C++\n{}\n--- ours\n{}", c.text(), o.text())),
            };
        }
        let skip = skips(args);
        let (a, b) = (norm_for(&c.text(), skip), norm_for(&o.text(), skip));
        if let Some(m) = first_diff(a.as_bytes(), b.as_bytes()) {
            return Err(format!("{shown}: reports differ: {m}\n--- C++\n{a}\n--- ours\n{b}"));
        }
        for name in outputs {
            // A name every run may write: a run that did not has neither.
            if cd.join(name).exists() || od.join(name).exists() {
                same_bytes(&cd.join(name), &od.join(name)).map_err(|m| format!("{shown}: {name}: {m}"))?;
            }
        }
        Ok(Verdict::Same)
    }

    /// `src` rewritten by the C++ `exrmultipart -combine`, which writes the `type` attribute.
    fn typed_copy(&self, src: &Path) -> PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = |t: &TempDir| t.join("").file_name().expect("a directory name").to_string_lossy().into_owned();
        let (inp, out) = (format!("{n}.exr"), format!("typed_{n}.exr"));
        std::fs::copy(src, self.typed.0.join(&inp)).unwrap_or_else(|e| panic!("cannot copy {}: {e}", src.display()));
        let rel = |d: &TempDir, f: &str| format!("..{MAIN_SEPARATOR}{}{MAIN_SEPARATOR}{f}", dir(d));
        let r = run_in(&self.typed.1.join(""), &cpp("exrmultipart"), os(&["-combine", "-i", &rel(&self.typed.0, &inp), "-o", &rel(&self.typed.2, &out)]));
        assert!(r.ok(), "exrmultipart could not rewrite {}: {}", src.display(), String::from_utf8_lossy(&r.stderr));
        self.typed.2.join(&out)
    }

    /// `opts` then `file`, through both tools; a typeless tiled input is checked on its typed copy.
    /// Returns what the comparison found (for the typed copy, that of the copy).
    fn file(&self, report: &mut Report, opts: &[&str], file: &Path, outputs: &[&str]) -> Verdict {
        let mut args = os(opts);
        args.push(file.into());
        let mut verdict = self.run(&args, outputs);
        if verdict == Ok(Verdict::TypelessTiled) {
            let typed = self.typed_copy(file);
            let mut again = os(opts);
            again.push(typed.into());
            verdict = self.run(&again, outputs).and_then(|v| if v == Verdict::Same { Ok(Verdict::TypelessTiled) } else { Err(format!("typed copy of {}: {v:?}", file.display())) });
        }
        let shown = verdict.clone().unwrap_or(Verdict::FailedAlike);
        report.check(format!("metrics {} {}", opts.join(" "), file.display()), verdict.map(|_| ()));
        shown
    }
}

fn golden(name: &str) -> PathBuf {
    goldens().into_iter().find(|p| p.ends_with(name)).unwrap_or_else(|| panic!("golden {name} not found"))
}

fn size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// Inputs both tools can measure quickly and safely: the damaged files claim data windows of any
/// size, and the tool holds the whole image in memory.
fn measurable() -> Vec<PathBuf> {
    inputs().into_iter().filter(|f| size(f) < 2_000_000 && !f.to_string_lossy().contains("Damaged")).collect()
}

/// Deep tiled buffers are sized by the input's channel types in the C++, so any change of sample type
/// writes past them: its output is undefined and cannot be compared (`differences_and_fixed_c_bugs`).
fn writes_past_the_buffers(opts: &[&str], file: &Path) -> bool {
    let changes_types = opts.windows(2).any(|w| matches!(w[0], "--pixelmode" | "-16") && !matches!(w[1], "orig" | "")) || opts.contains(&"--bench");
    changes_types && file.ends_with("golden_deep_tiled_zips_one.exr")
}

/// Files of every storage type, sample type and layout the option sweeps run on.
fn samples() -> Vec<PathBuf> {
    ["golden_zip.exr", "golden_rgba.exr", "golden_tiled_zip_one.exr", "golden_tiled_zip_mipmap.exr", "golden_tiled_zip_ripmap.exr", "golden_deep_scanline_zips.exr", "golden_deep_tiled_zips_one.exr", "golden_multipart.exr", "golden_sub_zip_x2y2.exr", "golden_header_attrs.exr"]
        .map(golden)
        .into()
}

// ----- tests --------------------------------------------------------------------------------

/// Every input through the default report, and a subset through the CSV report with several passes.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn default_reports_over_every_input() {
    let env = Env::new();
    let files = measurable();
    let mut report = Report::default();
    let (mut same, mut failed, mut typeless) = (0, 0, 0);
    let mut tally = |v: Verdict| match v {
        Verdict::Same => same += 1,
        Verdict::FailedAlike => failed += 1,
        Verdict::TypelessTiled => typeless += 1,
    };
    for f in &files {
        tally(env.file(&mut report, &[], f, &[]));
    }
    for f in files.iter().step_by(4) {
        tally(env.file(&mut report, &["--csv", "--passes", "3"], f, &[]));
    }
    report.finish(&format!("metrics default and csv reports over {} inputs ({same} identical, {typeless} typeless tiled, {failed} failing alike)", files.len()));
    assert!(same >= 100 && typeless >= 3, "{same} identical, {typeless} typeless tiled: the comparison proves little");
}

/// Every option and their combinations on files of every shape.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn every_option() {
    let env = Env::new();
    let files = samples();
    let mut sets: Vec<Vec<&str>> = Vec::new();
    // Compression lists: one codec, a list, all, with the original method among them.
    for c in ["no", "none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "htj2k256", "htj2k32", "orig", "ZIP", "Piz"] {
        sets.push(vec!["-z", c]);
    }
    sets.extend([
        vec!["-z", "zip,piz,orig"],
        vec!["--compression", "rle,zips"],
        vec!["-z", "all"],
        vec!["-z", "orig,zip"],
        vec!["-z", "zip,"],
        vec!["-z", ""],
        vec!["-z", "zip", "-z", "piz"],
    ]);
    // Pixel modes.
    for m in ["half", "float", "mixed", "rgba", "orig", "half,float", "orig,mixed,half", ""] {
        sets.push(vec!["--pixelmode", m]);
    }
    sets.extend([vec!["-16", "all"], vec!["-16", "rgba"], vec!["-16", "all", "-16", "rgba"], vec!["-16", "rgba", "--pixelmode", "float"]]);
    // What is timed and what is reported.
    for t in ["read", "write", "reread", "read,write", "write,reread", "read,reread", "read,write,reread", "none", ""] {
        sets.push(vec!["--time", t]);
        sets.push(vec!["--time", t, "--csv"]);
    }
    sets.extend([
        vec!["--no-size"],
        vec!["--no-size", "--time", "reread"],
        vec!["--no-size", "--time", "read,reread"],
        vec!["--no-size", "--time", "write"],
        vec!["--no-size", "--csv"],
        vec!["--csv"],
        vec!["--csv", "--json"],
        vec!["--json", "--csv"],
        vec!["--part-disk-size"],
        vec!["--part-disk-size", "--csv"],
        vec!["--part-disk-size", "--time", "none"],
        vec!["--part-disk-size", "--no-size"],
    ]);
    // Passes and threads.
    for p in ["0", "1", "2", "5"] {
        sets.push(vec!["--passes", p]);
    }
    sets.extend([vec!["--passes", "3", "--csv"], vec!["-m"], vec!["-t", "0"], vec!["-t", "3"], vec!["-m", "-z", "piz,zip", "--passes", "2"]]);
    // Compression levels: where they apply and where they do not.
    // Levels out of the codecs' ranges too (ZIP above 9 is the default level, DWA is unbounded).
    for l in ["0", "1", "4", "9", "10", "12", "45", "4.7", "9.99", "0x1", "0x9", "0xa", "1e0", "inf", "nan", " 3", ".5e1", "-0"] {
        sets.push(vec!["-l", l, "-z", "zip"]);
    }
    for l in ["0", "30", "45", "60", "100", "150", "200", "1000", "33.3", "0x1e", "99.9", "inf", "nan"] {
        sets.push(vec!["-l", l, "-z", "dwaa"]);
        sets.push(vec!["-l", l, "-z", "dwab,dwaa"]);
    }
    sets.extend([vec!["-l", "5"], vec!["-l", "5", "-z", "piz"], vec!["-l", "5", "-z", "orig"], vec!["-l", "5", "-z", "zip,piz"]]);
    // Shorthands and their order.
    sets.extend([
        vec!["--convert"],
        vec!["--convert", "--pixelmode", "half"],
        vec!["--pixelmode", "half", "--convert"],
        vec!["--convert", "-z", "zips"],
        vec!["--convert", "--csv"],
        vec!["--convert", "--time", "write"],
        vec!["--convert", "--no-size"],
        vec!["--bench"],
        vec!["--bench", "--json"],
        vec!["--bench", "-z", "zip,piz"],
        vec!["--bench", "--passes", "2", "--time", "read"],
    ]);
    // Parts, repeated and odd options.
    sets.extend([vec!["-p", "all"], vec!["-p", "0"], vec!["-p", "1"], vec!["-p", "2"], vec!["-p", "3"], vec!["-p", "-1"], vec!["-p", "-2"], vec!["-p", "1", "--csv"]]);
    sets.extend([vec!["-v"], vec!["--json", "-v", "--csv"]]);

    // `EXRMETRICS_SETS=<text>` keeps the option sets that contain the text (a focused run).
    if let Ok(only) = std::env::var("EXRMETRICS_SETS") {
        sets.retain(|o| o.join(" ").contains(&only));
    }
    let mut report = Report::default();
    let (mut made, mut failed) = (0, 0);
    for opts in &sets {
        for f in &files {
            if writes_past_the_buffers(opts, f) {
                continue;
            }
            match env.file(&mut report, opts, f, &[]) {
                Verdict::Same | Verdict::TypelessTiled => made += 1,
                Verdict::FailedAlike => failed += 1,
            }
        }
    }
    report.finish(&format!("metrics option matrix ({} option sets x {} inputs: {made} measured, {failed} failing alike)", sets.len(), files.len()));
    assert!(made >= sets.len() * files.len() / 2 || std::env::var_os("EXRMETRICS_SETS").is_some(), "only {made} measured of {}", sets.len() * files.len());
}

/// Several inputs in one run, in any spelling (`-i`, bare words, the same file twice), with lists.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn several_inputs() {
    let env = Env::new();
    let (z, r, d, m) = (golden("golden_zip.exr"), golden("golden_rgba.exr"), golden("golden_deep_scanline_zips.exr"), golden("golden_multipart.exr"));
    let mut report = Report::default();
    let mut made = 0;
    let sets: Vec<Vec<OsString>> = vec![
        vec![z.clone().into(), r.clone().into()],
        vec![z.clone().into(), z.clone().into()],
        vec![z.clone().into(), d.clone().into(), m.clone().into()],
        vec!["-i".into(), z.clone().into(), "-i".into(), r.clone().into()],
        vec!["--csv".into(), z.clone().into(), d.clone().into(), r.clone().into()],
        vec!["-z".into(), "zips,zip,piz".into(), z.clone().into(), d.clone().into()],
        vec!["-z".into(), "orig,zips,zip".into(), z.clone().into(), d.clone().into()],
        vec!["-z".into(), "all".into(), "--pixelmode".into(), "half,float".into(), "--csv".into(), r.clone().into(), d.clone().into()],
        vec!["--part-disk-size".into(), m.clone().into(), z.clone().into()],
        vec![z.clone().into(), "missing.exr".into()],
        vec!["missing.exr".into(), z.clone().into()],
    ];
    for args in sets {
        let v = env.run(&args, &[]);
        made += usize::from(v == Ok(Verdict::Same));
        report.check(args.iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>().join(" "), v.map(|_| ()));
    }
    report.finish(&format!("metrics several inputs ({made} measured)"));
    assert!(made >= 8);
}

/// `-o`: the file written is the same bytes as the C++ writes, for every codec, pixel mode and part,
/// and several passes overwrite it.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn output_files_are_identical() {
    let env = Env::new();
    let mut report = Report::default();
    let mut made = 0;
    let mut sets: Vec<Vec<&str>> = vec![vec![], vec!["--passes", "3"], vec!["--convert"], vec!["-l", "2", "-z", "zip"], vec!["-l", "70", "-z", "dwab"]];
    for c in ["no", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "htj2k256", "htj2k32"] {
        sets.push(vec!["-z", c, "-o", "out.exr"]);
        sets.push(vec!["-z", c, "--pixelmode", "half", "-o", "out.exr"]);
        sets.push(vec!["-z", c, "--pixelmode", "float", "-o", "out.exr"]);
    }
    sets.extend([vec!["--pixelmode", "mixed", "-o", "out.exr"], vec!["-p", "1", "-o", "out.exr"], vec!["-p", "0", "-z", "zips", "-o", "out.exr"]]);
    for f in samples() {
        for opts in &sets {
            if writes_past_the_buffers(opts, &f) {
                continue;
            }
            let mut o = opts.clone();
            if !o.contains(&"-o") {
                o.extend(["-o", "out.exr"]);
            }
            // A deep tiled tile stored uncompressed holds its count table padded to the whole tile:
            // the C++ pads it with stale bytes of the previous tile, exr-core with zeros. The report
            // (with the size of the file) must still agree.
            let padded = f.ends_with("golden_deep_tiled_zips_one.exr") && o.windows(2).any(|w| w[0] == "-z" && matches!(w[1], "no" | "none"));
            let outputs: &[&str] = if padded { &[] } else { &["out.exr"] };
            made += usize::from(env.file(&mut report, &o, &f, outputs) != Verdict::FailedAlike);
        }
    }
    report.finish(&format!("metrics -o files ({made} runs measured)"));
    assert!(made >= 150, "only {made}");
}

/// Bad options and bad inputs end alike; an output file and `--part-disk-size` exclude each other.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn failures_are_identical() {
    let env = Env::new();
    let (cd, od) = (env.cdir.join(""), env.odir.join(""));
    for d in [&cd, &od] {
        std::fs::write(d.join("not_exr.txt"), b"this is not an image").expect("write the text file");
    }
    let z = golden("golden_zip.exr");
    let z = z.to_str().expect("utf-8 path");
    let mut report = Report::default();
    let mut failed = 0;
    let cases: Vec<Vec<&str>> = vec![
        vec!["-t"],
        vec![z, "-t"],
        vec!["-t", "-1", z],
        vec![z, "--passes"],
        vec!["--passes", "-1", z],
        vec![z, "-z"],
        vec!["-z", "lzw", z],
        vec!["-z", "zip,lzw", z],
        vec!["-z", "zip,,piz", z],
        vec![z, "-p"],
        vec!["-p", "-2", z],
        vec!["-p", "1", z],
        vec!["-p", "9", z],
        vec![z, "-l"],
        vec!["-l", "-1", z],
        vec!["-l", "5", "-z", "piz", z],
        vec![z, "-o"],
        vec!["-o", "a.exr", "-o", "b.exr", z],
        vec!["--part-disk-size", "-o", "a.exr", z],
        vec!["-o", "a.exr", "--part-disk-size", z],
        vec![z, "-16"],
        vec!["-16", "half", z],
        vec![z, "--pixelmode"],
        vec!["--pixelmode", "double", z],
        vec![z, "--time"],
        vec!["--time", "all", z],
        vec![z, "-i"],
        vec!["--csv"],
        vec!["--no-size", "--time", "none", z],
        vec!["missing.exr"],
        vec!["not_exr.txt"],
        vec![z, "not_exr.txt"],
        vec!["-x", z],
        vec!["--unknown", z],
        vec!["--no-size", "--time", "none", "-o", "no/such/dir/out.exr", z],
        vec!["-o", "no/such/dir/out.exr", z],
    ];
    for args in &cases {
        let v = env.run(&os(args), &["a.exr", "b.exr", "out.exr"]);
        failed += usize::from(v == Ok(Verdict::FailedAlike));
        report.check(args.join(" "), v.map(|_| ()));
    }
    report.finish(&format!("metrics failures ({failed} failing alike of {})", cases.len()));
    assert!(failed >= 30, "only {failed} failures compared");
}

/// `-h` and `--version` print their text; the C++ then prints an empty report (`[`, `]`) after it
/// (its `parse` returns 0 and `main` only tests for non-zero), which we do not.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn help_and_version() {
    for args in [&["-h"][..], &["--help"], &["a.exr", "-h"], &["-z", "zip", "--help", "a.exr"]] {
        let (c, o) = (run(&cpp("exrmetrics"), args), run_ours("metrics", args));
        assert_eq!((c.code, o.code), (Some(0), Some(0)), "{args:?}");
        assert!(c.text().ends_with("\n[\n]\n"), "the C++ report after -h: {}", c.text());
        assert!(c.text().contains("--part-disk-size") && o.text().contains("--part-disk-size") && o.text().contains("Examples:"), "{args:?}");
        assert!(!o.text().contains("\n[\n]\n"), "an empty report after -h: {}", o.text());
    }
    for args in [&["--version"][..], &["a.exr", "--version"]] {
        let (c, o) = (run(&cpp("exrmetrics"), args), run_ours("metrics", args));
        assert_eq!((c.code, o.code), (Some(0), Some(0)), "{args:?}");
        assert_eq!(c.text(), format!("{}[\n]\n", o.text()), "{args:?}");
        assert!(o.text().starts_with("exrmetrics (OpenEXR) "), "{}", o.text());
    }
    // The first error still wins over a later -h.
    let (c, o) = (run(&cpp("exrmetrics"), ["-t", "-1", "-h"]), run_ours("metrics", ["-t", "-1", "-h"]));
    assert_eq!((c.code, o.code), (Some(1), Some(1)));
}

// ----- where we deliberately differ, and the C++ quirks we keep -----------------------------

/// Every `"key": N` of a JSON report, in order.
fn json_nums(text: &str, key: &str) -> Vec<u64> {
    let prefix = format!("\"{key}\": ");
    text.lines().filter_map(|l| l.trim_start().strip_prefix(&prefix)).filter_map(|v| v.trim_end_matches(',').parse().ok()).collect()
}

/// Our report for `args`, which must succeed.
fn ours_ok(args: &[OsString]) -> String {
    let o = run_ours("metrics", args);
    assert!(o.ok(), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    o.text()
}

/// The channels of part 0 of `path`, name and type, from its header.
fn channels_of(path: &Path) -> Vec<(String, PixelType)> {
    let mut stream = StdIfStream::open(path.to_str().expect("utf-8 path")).expect("open the image");
    let file = MultiPartInputFile::new(&mut stream).expect("read the image");
    file.header(0).and_then(Header::channels).expect("channels").iter().map(|(n, c)| (String::from_utf8_lossy(n.as_bytes()).into_owned(), c.type_)).collect()
}

/// A scanline image with the given channels, for the pixel-mode tests.
fn write_image(path: &Path, (w, h): (i32, i32), chans: &[(&str, PixelType)]) {
    let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w - 1, y: h - 1 } };
    let mut head = Header::with_windows(dw, dw, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Zip).expect("header");
    let mut list = ChannelList::new();
    for (n, t) in chans {
        list.insert(*n, Channel::new(*t, 1, 1, false)).expect("channel");
    }
    head.insert("channels", ChannelListAttribute::new(list)).expect("channels");
    head.insert("type", StringAttribute::new("scanlineimage".into())).expect("type");
    let (mut fb, mut buf) = (FrameBuffer::new(), Vec::new());
    for (c, (n, t)) in chans.iter().enumerate() {
        let (start, size) = (buf.len(), if *t == PixelType::Half { 2 } else { 4 });
        for i in 0..(w * h) as usize {
            let v = ((i * 7 + c * 131) % 2039) as u32;
            match t {
                PixelType::Half => buf.extend_from_slice(&Half::from_f32(v as f32 / 8.0).bits().to_ne_bytes()),
                PixelType::Float => buf.extend_from_slice(&(v as f32 / 8.0).to_ne_bytes()),
                _ => buf.extend_from_slice(&v.to_ne_bytes()),
            }
        }
        fb.insert(*n, Slice::new(*t, start as i64, size, w as usize * size, 1, 1, 0.0, false, false).expect("slice")).expect("insert");
    }
    let mut out = MemOStream::new();
    {
        let mut file = MultiPartOutputFile::new(&mut out, &[head]).expect("file");
        let mut part = file.output_part(0).expect("part");
        part.set_frame_buffer(&fb).expect("frame buffer");
        part.write_pixels(&buf, h).expect("pixels");
    }
    std::fs::write(path, out.into_data()).expect("write the image");
}

fn utf8(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn differences_and_fixed_c_bugs() {
    let env = Env::new();
    let cpp_exe = cpp("exrmetrics");
    let work = TempDir::new("mm-quirks");

    // A single-part tiled file without `type`: the C++ fails, we measure it.
    let tiled = golden("golden_tiled_zip_one.exr");
    let c = run(&cpp_exe, [tiled.as_os_str()]);
    assert!(!c.ok() && String::from_utf8_lossy(&c.stderr).contains("Cannot find image attribute \"type\""), "{}", String::from_utf8_lossy(&c.stderr));
    let t = ours_ok(&[tiled.clone().into()]);
    assert!(t.contains("\"part type\": \"tiledimage\"") && json_nums(&t, "tile count") == [6], "{t}");

    // `tile count` of a mipmap: the C++ adds the tiles of every PAIR of levels (`exrmetrics.cpp:244`, the
    // addition sits outside the `if`), we the tiles of the levels there are.
    let mip = golden("golden_tiled_zip_mipmap.exr");
    let mut stream = StdIfStream::open(utf8(&mip)).expect("open the mipmap");
    let mut file = MultiPartInputFile::new(&mut stream).expect("read the mipmap");
    let part = file.tiled_input_part_wrapper(0).expect("the tiled part");
    let (mut all, mut diagonal) = (0u64, 0u64);
    for xl in 0..part.num_x_levels() {
        for yl in 0..part.num_y_levels() {
            let n = part.num_x_tiles(xl).expect("tiles") as u64 * part.num_y_tiles(yl).expect("tiles") as u64;
            all += n;
            diagonal += if xl == yl { n } else { 0 };
        }
    }
    assert!(all > diagonal, "a mipmap with more than one level");
    // The pixels and raw size count the levels a mipmap has: worked out from the level windows.
    let header = part.header().clone();
    let (td, dw) = (header.tile_description().expect("tiles"), header.data_window().expect("data window"));
    let level_pixels: u64 = (0..part.num_x_levels())
        .map(|l| {
            let w = data_window_for_level(&td, dw.min.x, dw.max.x, dw.min.y, dw.max.y, l, l).expect("level window");
            (w.max.x - w.min.x + 1) as u64 * (w.max.y - w.min.y + 1) as u64
        })
        .sum();
    let bps: u64 = channels_of(&mip).iter().map(|(_, t)| if *t == PixelType::Half { 2 } else { 4 }).sum();
    let typed = env.typed_copy(&mip);
    let (c, o) = (run(&cpp_exe, [typed.as_os_str()]).text(), ours_ok(&[typed.into()]));
    for text in [&c, &o] {
        assert_eq!((json_nums(text, "pixels"), json_nums(text, "total raw size")), (vec![level_pixels], vec![level_pixels * bps]), "pixels and raw size of the levels");
    }
    assert_eq!((json_nums(&c, "tile count"), json_nums(&o, "tile count")), (vec![all], vec![diagonal]), "C++ adds every pair ({all}), we the levels ({diagonal})");

    // A deep tiled part: the C++ forgets to record its compression, part type and tile count (and says
    // `mixed` / `mixed` / 0); we record them.
    let deep = golden("golden_deep_tiled_zips_one.exr");
    let d = ours_ok(&[deep.clone().into()]);
    let c = run(&cpp_exe, [deep.as_os_str()]).text();
    for line in ["\"compression\": \"mixed\",", "\"part type\": \"mixed\",", "\"tile count\": 0,"] {
        assert!(c.contains(line), "C++: {line} in {c}");
    }
    let mut stream = StdIfStream::open(utf8(&deep)).expect("open the deep tiled file");
    let file = MultiPartInputFile::new(&mut stream).expect("read the deep tiled file");
    let (td, dw) = (file.header(0).and_then(Header::tile_description).expect("tiles"), file.header(0).and_then(Header::data_window).expect("window"));
    let tiles = (((dw.max.x - dw.min.x) as u32 / td.x_size) + 1) * (((dw.max.y - dw.min.y) as u32 / td.y_size) + 1);
    for line in ["\"compression\": \"zips\",".to_string(), "\"part type\": \"deeptile\",".to_string(), format!("\"tile count\": {tiles},")] {
        assert!(d.contains(&line), "ours: {line} in {d}");
    }
    // `--pixelmode` on it: the sample buffers follow the OUTPUT types (the C++ sizes them by the input's
    // and writes past them). Raw size = samples x bytes per sample + 4 per pixel, worked out independently.
    let chans = channels_of(&deep);
    let orig_bps: u64 = chans.iter().map(|(_, t)| if *t == PixelType::Half { 2 } else { 4 }).sum();
    let raw = |mode: &str| json_nums(&ours_ok(&os(&["--pixelmode", mode, "--time", "none", utf8(&deep)])), "total raw size")[0];
    let pixels = json_nums(&d, "pixels")[0];
    let samples = (raw("orig") - 4 * pixels) / orig_bps;
    assert!(samples > 0 && (raw("orig") - 4 * pixels).is_multiple_of(orig_bps));
    let rgba = |n: &str| matches!(n, "R" | "G" | "B" | "A");
    let mixed: u64 = chans.iter().map(|(n, _)| if rgba(n) { 2 } else { 4 }).sum();
    for (mode, bps) in [("float", 4 * chans.len() as u64), ("half", 2 * chans.len() as u64), ("mixed", mixed)] {
        assert_eq!(raw(mode), samples * bps + 4 * pixels, "--pixelmode {mode} on a deep tiled file");
        let out = work.join("deep.exr");
        let o = run_ours("metrics", os(&["--pixelmode", mode, "--time", "none", "-o", utf8(&out), utf8(&deep)]));
        assert!(o.ok(), "{}", String::from_utf8_lossy(&o.stderr));
        let want = |n: &str| match mode {
            "float" => PixelType::Float,
            "half" => PixelType::Half,
            _ if rgba(n) => PixelType::Half,
            _ => PixelType::Float,
        };
        assert!(channels_of(&out).iter().all(|(n, t)| *t == want(n)), "{mode}: {:?}", channels_of(&out));
    }

    // `--pixelmode mixed` keeps half for R, G, B, A: the C++ finds the suffix after the last lower case `r`
    // (`exrmetrics.cpp:1079`, where its comment says 'R' in 'layer.R'), so `diffuse.R` and `layer.B` stay
    // float and `colorR` becomes half; we take the suffix after the last `.`.
    let layered = work.join("layered.exr");
    let f = PixelType::Float;
    write_image(&layered, (40, 30), &[("R", f), ("G", f), ("B", f), ("A", f), ("diffuse.R", f), ("colorR", f), ("Z", f), ("layer.B", f)]);
    let types_of = |out: &Path| -> std::collections::BTreeMap<String, PixelType> { channels_of(out).into_iter().collect() };
    let (cout, oout) = (work.join("c_mixed.exr"), work.join("o_mixed.exr"));
    let c = run(&cpp_exe, os(&["--pixelmode", "mixed", "--time", "none", "-o", utf8(&cout), utf8(&layered)]));
    assert!(c.ok(), "{}", String::from_utf8_lossy(&c.stderr));
    ours_ok(&os(&["--pixelmode", "mixed", "--time", "none", "-o", utf8(&oout), utf8(&layered)]));
    let (ct, ot) = (types_of(&cout), types_of(&oout));
    let h = PixelType::Half;
    for (name, c_want, o_want) in [("R", h, h), ("A", h, h), ("Z", f, f), ("colorR", h, f), ("diffuse.R", f, h), ("layer.B", f, h)] {
        assert_eq!((ct[name], ot[name]), (c_want, o_want), "{name}: C++ / ours");
    }

    // A part of a type no reader knows: the C++ reports zeros for it, we refuse.
    let mut bytes = std::fs::read(golden("golden_multipart.exr")).expect("read the golden");
    let at = bytes.windows(14).position(|w| w == b"scanlineimage\0").expect("part 0's type");
    bytes[at + 12] = b'X';
    let unknown = work.join("unknown.exr");
    std::fs::write(&unknown, &bytes).expect("write the patched file");
    let c = run(&cpp_exe, [unknown.as_os_str()]);
    assert!(c.ok(), "{}", String::from_utf8_lossy(&c.stderr));
    let o = run_ours("metrics", [unknown.as_os_str()]);
    assert_eq!(o.code, Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("unsupported type \"scanlineimagX\""), "{}", String::from_utf8_lossy(&o.stderr));

    // No passes: the C++ CSV report crashes taking the median of nothing, ours has `---`.
    let zip = golden("golden_zip.exr");
    let c = run(&cpp_exe, ["--csv", "--passes", "0", utf8(&zip)]);
    assert!(!c.ok(), "the C++ survived an empty median: {}", c.text());
    let o = ours_ok(&os(&["--csv", "--passes", "0", utf8(&zip)]));
    let row: Vec<&str> = o.lines().nth(1).expect("a row").split(',').collect();
    assert_eq!(&row[row.len() - 3..], ["---"; 3], "no write, count re-read or re-read samples: {o}");

    // The JSON report has no value for a time without samples in the C++ (`"write time": ,`, not JSON): `null`.
    let c = run(&cpp_exe, ["--passes", "0", utf8(&zip)]).text();
    let o = ours_ok(&os(&["--passes", "0", utf8(&zip)]));
    assert!(c.contains("\"write time\": ,
") && c.contains("\"re-read time\": 
"), "{c}");
    assert!(o.contains("\"write time\": null,
") && o.contains("\"re-read time\": null
"), "{o}");
}

/// A number that is not a number is an error here; the C++ takes the digits it starts with, else 0.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn numbers_must_be_numbers() {
    let z = golden("golden_zip.exr");
    let z = utf8(&z);
    for args in [vec!["-t", "x", z], vec!["--passes", "1x", z], vec!["-p", "0abc", z], vec!["-l", "abc", "-z", "zip", z], vec!["-l", "5x", "-z", "zip", z], vec!["-t", "1.5", z]] {
        let (c, o) = (run(&cpp("exrmetrics"), &args), run_ours("metrics", &args));
        assert!(c.ok(), "{args:?}: the C++ tool takes it: {}", String::from_utf8_lossy(&c.stderr));
        assert_eq!(o.code, Some(1), "{args:?}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("Expected a"), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    }
    // The forms both accept: hex and octal integers, hex floats.
    for args in [vec!["-t", "0x2", z], vec!["-p", "00", z], vec!["-l", "0x1p2", "-z", "zip", z], vec!["-l", "1e0", "-z", "zip", z]] {
        let (c, o) = (run(&cpp("exrmetrics"), &args), run_ours("metrics", &args));
        assert!(c.ok() && o.ok(), "{args:?}");
        assert_eq!(output_size(&c.text()), output_size(&o.text()), "{args:?}");
    }
}

/// The `output size` of the report of one run.
fn output_size(text: &str) -> u64 {
    json_nums(text, "output size")[0]
}

// ----- the comparison itself ----------------------------------------------------------------

/// The normalizer hides times and nothing else: a different size, key or number of samples shows.
#[test]
fn norm_hides_only_the_times() {
    let a = "  \"output size\": 8049,\n  \"read time\": 0.0001547,\n  \"write time\": {\"values\": [ 1e-05 , 0.5 ] , \"min\": 1e-05, \"max\": 0.5, \"mean\": 0.25, \"median\": 0.25, \"std dev\": -nan(ind)},\n";
    let b = "  \"output size\": 8049,\n  \"read time\": 9.3e-05,\n  \"write time\": {\"values\": [ 0.25 , 2 ] , \"min\": 0.25, \"max\": 2, \"mean\": 1, \"median\": 1, \"std dev\": nan},\n";
    assert_eq!(norm(a), norm(b));
    assert!(norm(a).contains("\"output size\": 8049") && norm(a).contains("\"read time\": #,\n"), "{}", norm(a));
    // Another size, another key, another sample count, a missing time: all differ.
    assert_ne!(norm(a), norm(&b.replace("8049", "8050")));
    assert_ne!(norm(a), norm(&b.replace("\"read time\"", "\"re-read time\"")));
    assert_ne!(norm(a), norm(&b.replace("[ 0.25 , 2 ]", "[ 0.25 , 2 , 3 ]")));
    assert_ne!(norm(a), norm(&b.replace("\"read time\": 9.3e-05,", "\"read time\": ,")));
    // CSV: only the `time` columns.
    let csv = |size: &str, t: &str| format!("file name,input size,read time,write time\nx.exr,{size},{t},---\n");
    assert_eq!(norm(&csv("10", "0.5")), norm(&csv("10", "1.25e-05")));
    assert_ne!(norm(&csv("10", "0.5")), norm(&csv("11", "0.5")));
    assert_ne!(norm(&csv("10", "0.5")), norm(&csv("10", "---")));
}
