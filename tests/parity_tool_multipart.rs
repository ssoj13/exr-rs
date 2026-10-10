//! `exr-rs multipart` against OpenEXR's `exrmultipart` (v3.4.15): exit code, stdout and the output
//! files over the goldens and the private image set, for every mode and option: `-combine` (every
//! file alone, pairs and triples of every storage type, part selection, forced names, views,
//! `-override`), `-separate`, `-convert`, the argument grammar and the failures. `#[ignore]`d: run
//! by `cargo run -p xtask -- parity-tools`.
//!
//! Every input is addressed by a relative path from the tools' working directory: the C++ takes a
//! `:` for the start of a part number, so a Windows drive letter (`C:/...`) cannot name a file.
//!
//! **`-combine` is compared up to one C++ defect.** `combine ()` builds a `MultiPartOutputFile temp`
//! "to bail out early" and then a second one on the same file. `temp` keeps the prologue in its
//! stream buffer, and its destructor (after the real file is finished) writes it over the real
//! file: the chunk offset table of the last part (of the only part, for one input) is left all
//! zeros. Readers rebuild such a table by scanning the file, so it still opens, but it is a damaged
//! file. We write the table. [`same_output`] accepts exactly that difference: every byte outside the
//! offset tables equal, and each table entry equal or zero in the C++ file.

mod tool_support;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{MAIN_SEPARATOR, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemOStream, StdIfStream};
use exr_core::multipart::{MultiPartInputFile, MultiPartOutputFile};
use exr_core::StringAttribute;
use imath_rs::{Box2i, Half, V2f, V2i};
use tool_support::*;

const OUT: &str = "out.exr";

/// Table entries the C++ left zeroed, over the whole run of one test binary.
static ZEROED: AtomicUsize = AtomicUsize::new(0);

// ----- the offset tables -------------------------------------------------------------------

/// A NUL-terminated string at `at`, and the position after its terminator.
fn cstr(b: &[u8], at: usize) -> Option<(&[u8], usize)> {
    let end = at + b.get(at..)?.iter().position(|&c| c == 0)?;
    Some((&b[at..end], end + 1))
}

/// Skip the attributes of one header (name, type, size, value); the position after its empty-name
/// terminator.
fn skip_header(b: &[u8], mut at: usize) -> Option<usize> {
    loop {
        let (name, next) = cstr(b, at)?;
        if name.is_empty() {
            return Some(next);
        }
        let (_, next) = cstr(b, next)?;
        let size = u32::from_le_bytes(b.get(next..next + 4)?.try_into().ok()?) as usize;
        at = next + 4 + size;
    }
}

/// Where the chunk offset tables of a file written by exr-core start, and how many entries they
/// hold: the headers end at the tables, and the first chunk (the smallest offset in them) follows
/// the last entry.
fn offset_tables(b: &[u8]) -> Option<(usize, usize)> {
    let version = u32::from_le_bytes(b.get(4..8)?.try_into().ok()?);
    let multipart = version & 0x1000 != 0;
    let mut at = skip_header(b, 8)?;
    if multipart {
        // A header list ends with one more empty name.
        while *b.get(at)? != 0 {
            at = skip_header(b, at)?;
        }
        at += 1;
    }
    let entry = |i: usize| -> Option<u64> { Some(u64::from_le_bytes(b.get(at + 8 * i..at + 8 * i + 8)?.try_into().ok()?)) };
    let mut least = u64::MAX;
    for n in 1.. {
        least = least.min(entry(n - 1)?);
        if least == (at + 8 * n) as u64 {
            return Some((at, n));
        }
    }
    None
}

/// `Ok(zeroed)` when `cpp` is `ours` except that some offset table entries are zero in `cpp`; the
/// number of those entries.
fn same_but_zeroed(cpp: &[u8], ours: &[u8]) -> Result<usize, String> {
    if cpp == ours {
        return Ok(0);
    }
    let (start, n) = offset_tables(ours).ok_or_else(|| "our file has no readable offset tables".to_string())?;
    if cpp.len() != ours.len() {
        return Err(first_diff(cpp, ours).unwrap_or_default());
    }
    let end = start + 8 * n;
    let outside = |a: &[u8], b: &[u8]| first_diff(a, b).map(|m| format!("outside the offset tables: {m}"));
    // Bytes before and after the tables must agree.
    if let Some(m) = outside(&cpp[..start], &ours[..start]).or_else(|| outside(&cpp[end..], &ours[end..])) {
        return Err(m);
    }
    let word = |b: &[u8], i: usize| u64::from_le_bytes(b[start + 8 * i..start + 8 * i + 8].try_into().expect("8 bytes"));
    let mut zeroed = 0;
    for i in 0..n {
        let (c, o) = (word(cpp, i), word(ours, i));
        if c == o {
            continue;
        }
        if c != 0 {
            return Err(format!("offset table entry {i} differs: C++ {c}, ours {o}"));
        }
        if o < end as u64 || o >= ours.len() as u64 {
            return Err(format!("our offset table entry {i} ({o}) does not point into the chunks"));
        }
        zeroed += 1;
    }
    Ok(zeroed)
}

/// Files written by the two tools agree up to [`same_but_zeroed`]; counted in [`ZEROED`].
fn same_output(cpp: &Path, ours: &Path) -> Result<(), String> {
    // A name every run may create, a run that does not: neither has the file.
    if !cpp.exists() && !ours.exists() {
        return Ok(());
    }
    let read = |p: &Path| std::fs::read(p).map_err(|e| format!("cannot read {}: {e}", p.display()));
    ZEROED.fetch_add(same_but_zeroed(&read(cpp)?, &read(ours)?)?, Ordering::Relaxed);
    Ok(())
}

// ----- the two tools on one set of inputs ---------------------------------------------------

/// A pair of scratch directories for the tools and a third with copies of the inputs, next to them:
/// every input is named `..\<dir>\<file>` from both working directories.
struct Env {
    pair: Pair,
    inputs: TempDir,
    dir: String,
    /// Source path -> name of its copy.
    copies: RefCell<BTreeMap<PathBuf, String>>,
}

impl Env {
    fn new() -> Self {
        let inputs = TempDir::new("mp-in");
        let dir = inputs.join("").file_name().expect("a directory name").to_string_lossy().into_owned();
        Env { pair: Pair::new("exrmultipart", "multipart"), inputs, dir, copies: RefCell::default() }
    }

    /// The command line word that names `src` (copied into the shared input directory on first use).
    fn arg(&self, src: &Path) -> String {
        let mut copies = self.copies.borrow_mut();
        let name = copies.entry(src.to_path_buf()).or_insert_with(|| {
            let file = src.file_name().expect("a file name").to_string_lossy().into_owned();
            let name = if file.starts_with("golden") {
                file
            } else {
                let parent = src.parent().and_then(Path::file_name).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
                format!("{parent}_{file}")
            };
            std::fs::copy(src, self.inputs.join(&name)).unwrap_or_else(|e| panic!("cannot copy {}: {e}", src.display()));
            name
        });
        format!("..{MAIN_SEPARATOR}{}{MAIN_SEPARATOR}{name}", self.dir)
    }

    /// Run both tools; record the verdict (`Ok(true)`: both succeeded and the outputs agree,
    /// `Ok(false)`: both failed alike).
    fn go(&self, report: &mut Report, args: &[String], outputs: &[&str]) -> bool {
        let verdict = self.pair.check_with(args, outputs, &same_output);
        let made = verdict == Ok(true);
        report.check(args.join(" "), verdict.map(|_| ()));
        made
    }

    /// The inputs are as they were: nothing wrote into the shared directory.
    fn assert_inputs_untouched(&self) {
        for (src, name) in self.copies.borrow().iter() {
            let (a, b) = (std::fs::read(src).expect("read the source"), std::fs::read(self.inputs.join(name)).expect("read the copy"));
            assert!(a == b, "{} was modified", src.display());
        }
    }
}

fn golden(name: &str) -> PathBuf {
    goldens().into_iter().find(|p| p.ends_with(name)).unwrap_or_else(|| panic!("golden {name} not found"))
}

fn size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn s(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

/// Files of every storage type and sampling the pair/triple sweeps combine.
fn samples() -> Vec<PathBuf> {
    [
        "golden_zip.exr",
        "golden_piz.exr",
        "golden_rgba.exr",
        "golden_tiled_zip_one.exr",
        "golden_tiled_zip_mipmap.exr",
        "golden_tiled_rle_one.exr",
        "golden_deep_scanline_zips.exr",
        "golden_deep_tiled_zips_one.exr",
        "golden_multipart.exr",
        "golden_header_attrs.exr",
        "golden_b44.exr",
        "golden_dwaa.exr",
        "golden_htj2k256.exr",
        "golden_sub_zip_x2y2.exr",
        "golden_nonutf8_names.exr",
        "golden_yc.exr",
    ]
    .map(golden)
    .into()
}

// ----- combine ------------------------------------------------------------------------------

/// Every input alone, through `-combine` (one header: a single-part output) and `-separate`.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn combine_and_separate_every_input_alone() {
    let env = Env::new();
    let files: Vec<PathBuf> = inputs().into_iter().filter(|f| size(f) < 1_500_000).collect();
    let mut report = Report::default();
    let (mut combined, mut separated) = (0, 0);
    let names: Vec<String> = (1..=8).map(|p| format!("sep.{p}.exr")).collect();
    let sep_outputs: Vec<&str> = names.iter().map(String::as_str).collect();
    for f in &files {
        let a = env.arg(f);
        combined += usize::from(env.go(&mut report, &[s(&["-combine", "-i"]), vec![a.clone()], s(&["-o", OUT])].concat(), &[OUT]));
        separated += usize::from(env.go(&mut report, &[s(&["-separate", "-i"]), vec![a], s(&["-o", "sep"])].concat(), &sep_outputs));
    }
    env.assert_inputs_untouched();
    report.finish(&format!(
        "multipart: {} files alone: {combined} combined, {separated} separated; C++ zeroed {} table entries",
        files.len(),
        ZEROED.load(Ordering::Relaxed)
    ));
    assert!(combined >= 100 && separated >= 100, "only {combined} combines and {separated} separations of {} files", files.len());
}

/// Every ordered pair and every triple with a repeat of the sample files: part names from file names,
/// clashing names made unique, shared attributes that agree or conflict.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn combine_pairs_and_triples() {
    let env = Env::new();
    let files = samples();
    let mut report = Report::default();
    let mut made = 0;
    // Shared attributes (the display window above all) rarely agree between the goldens, so every
    // combination also runs with `-override 1`, which copies part 0's.
    let line = |fs: &[&PathBuf], over: bool| -> Vec<String> {
        let mut v = s(&["-combine", "-i"]);
        v.extend(fs.iter().map(|f| env.arg(f)));
        v.extend(s(&["-o", OUT]));
        if over {
            v.extend(s(&["-override", "1"]));
        }
        v
    };
    for over in [false, true] {
        for a in &files {
            for b in &files {
                made += usize::from(env.go(&mut report, &line(&[a, b], over), &[OUT]));
            }
            // Three of a kind: the second and third name are made from the first's.
            made += usize::from(env.go(&mut report, &line(&[a, a, a], over), &[OUT]));
        }
        let (z, p, r) = (golden("golden_zip.exr"), golden("golden_piz.exr"), golden("golden_rgba.exr"));
        for set in [[&z, &p, &r], [&r, &z, &p], [&p, &p, &z], [&z, &r, &r]] {
            made += usize::from(env.go(&mut report, &line(&set, over), &[OUT]));
        }
    }
    env.assert_inputs_untouched();
    report.finish(&format!("multipart -combine pairs and triples ({made} combined; C++ zeroed {} table entries)", ZEROED.load(Ordering::Relaxed)));
    assert!(made >= 300, "only {made} combinations: the comparison proves little");
}

/// `:<partnum>`, `::<partname>`, both, several of one file, out of range, malformed, the default name
/// taken from the file name (frame number, `.exr`), and views.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn combine_part_selection_names_and_views() {
    let env = Env::new();
    let (m, z, t) = (env.arg(&golden("golden_multipart.exr")), env.arg(&golden("golden_zip.exr")), env.arg(&golden("golden_tiled_zip_one.exr")));
    let mut report = Report::default();
    let (mut made, mut failed) = (0, 0);
    // Each line runs as it is and with `-override 1` (files of different sizes disagree on the display
    // window, and only the second run then combines them).
    let mut case = |words: Vec<String>| {
        for over in [false, true] {
            let mut argv = s(&["-combine", "-i"]);
            argv.extend(words.clone());
            argv.extend(s(&["-o", OUT]));
            if over {
                argv.extend(s(&["-override", "1"]));
            }
            let ok = env.go(&mut report, &argv, &[OUT]);
            made += usize::from(ok);
            failed += usize::from(!ok);
        }
    };
    let with = |f: &str, suffix: &str| format!("{f}{suffix}");
    // One chosen part, every part number and beyond.
    for n in 0..=4 {
        case(vec![with(&m, &format!(":{n}"))]);
    }
    // Chosen parts of the same file, in any order, a part twice.
    case(vec![with(&m, ":2"), with(&m, ":0")]);
    case(vec![with(&m, ":0"), with(&m, ":1"), with(&m, ":2")]);
    case(vec![with(&m, ":1"), with(&m, ":1")]);
    case(vec![m.clone(), with(&m, ":1")]);
    // Forced names: whole file, chosen part, several files, clashes, empty, odd characters.
    case(vec![with(&z, "::alpha")]);
    case(vec![with(&m, "::all")]);
    case(vec![with(&m, ":1::only")]);
    case(vec![with(&z, "::same"), with(&t, "::same")]);
    case(vec![with(&z, "::a"), with(&t, "::a_1_1"), with(&m, ":1::a")]);
    case(vec![with(&z, "::a"), with(&t, "::a_2_1"), with(&m, ":0::a")]);
    case(vec![with(&z, "::")]);
    case(vec![with(&z, "::with space"), with(&t, "::we:ird")]);
    case(vec![with(&z, "::x"), with(&z, "::x"), with(&z, "::x"), with(&z, "::x")]);
    // Malformed.
    for bad in [":x", ":", ":-1", ":1x", "::x:3"] {
        case(vec![with(&z, bad)]);
    }
    case(vec![with(&z, ":0:1")]);
    // Default names from file names: extra dots, frame numbers, no `.exr`.
    let copies = [("shot.0012.exr", &z), ("shot.exr", &t), ("noext", &z), ("a.b.c.0001.exr", &t), ("UPPER.EXR", &z), ("x.exr.exr", &t), (".exr", &z), ("1234.exr", &z), ("v1.2.exr", &t)];
    let (cdir, odir) = env.pair.dirs();
    let mut named = Vec::new();
    for (name, src) in copies {
        // The copies sit in the tools' own directories so the name is what the tool sees.
        let source = env.inputs.join(src.rsplit(MAIN_SEPARATOR).next().expect("a file name"));
        for d in [&cdir, &odir] {
            std::fs::copy(&source, d.join(name)).unwrap_or_else(|e| panic!("cannot copy to {name}: {e}"));
        }
        named.push(name.to_string());
    }
    for name in &named {
        case(vec![name.clone()]);
        case(vec![name.clone(), z.clone()]);
    }
    case(named.clone());
    // Views: each file, sticky across files, with part selection and names, before and after -i.
    let v = |w: &[&str]| s(w);
    case(v(&["-view", "left"]).into_iter().chain([z.clone(), t.clone()]).collect());
    case([v(&["-view", "left"]), vec![z.clone()], v(&["-view", "right"]), vec![t.clone()]].concat());
    case([vec![z.clone()], v(&["-view", "right"]), vec![t.clone(), m.clone()]].concat());
    case([v(&["-view", "v"]), vec![with(&m, ":1::named")]].concat());
    case([v(&["-view", "a"]), vec![z.clone()], v(&["-i"]), vec![t.clone()]].concat());
    case([v(&["-view", "a"]), vec![z.clone()], v(&["-view", "b"]), vec![t.clone()], v(&["-view", "a"]), vec![m.clone()]].concat());
    case([vec![z.clone()], v(&["-view"])].concat());
    // The last `-view` value is not a file, the next option word still ends it.
    case([v(&["-view", "x", "-i"]), vec![z.clone()]].concat());
    env.assert_inputs_untouched();
    report.finish(&format!("multipart -combine selection/names/views ({made} combined, {failed} failing alike)"));
    assert!(made >= 60 && failed >= 10, "{made} combined, {failed} failed: the comparison proves little");
}

/// `-override` with every value, on inputs whose shared attributes (display window, aspect ratio,
/// chromaticities, time code) conflict and on inputs whose do not.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn combine_override_and_conflicts() {
    let env = Env::new();
    let dw = exr_test_data::openexr_images().join("DisplayWindow");
    let chroma = exr_test_data::openexr_images().join("Chromaticities");
    let mut files: Vec<PathBuf> = ["t01.exr", "t02.exr", "t05.exr", "t09.exr", "t16.exr"].iter().map(|n| dw.join(n)).collect();
    files.extend(["Rec709.exr", "XYZ.exr"].iter().map(|n| chroma.join(n)));
    files.extend(["golden_zip.exr", "golden_header_attrs.exr", "golden_yc.exr"].map(golden));
    let mut report = Report::default();
    let mut made = 0;
    for a in &files {
        for b in &files {
            for over in [None, Some("0"), Some("1"), Some("2"), Some("-1"), Some("007")] {
                let mut argv = s(&["-combine", "-i", &env.arg(a), &env.arg(b), "-o", OUT]);
                if let Some(o) = over {
                    argv.extend(s(&["-override", o]));
                }
                made += usize::from(env.go(&mut report, &argv, &[OUT]));
            }
        }
    }
    // `-override` before the files, and its value followed by stray words.
    let (a, b) = (env.arg(&files[0]), env.arg(&files[1]));
    for argv in [
        s(&["-combine", "-override", "1", "-i", &a, &b, "-o", OUT]),
        s(&["-combine", "-i", &a, &b, "-override", "1", "-o", OUT]),
        s(&["-combine", "-i", &a, &b, "-o", OUT, "-override", "1", "0"]),
        s(&["-combine", "-i", &a, &b, "-o", OUT, "-override", "0", "1"]),
        s(&["-combine", "-i", &a, &b, "-o", OUT, "-override"]),
    ] {
        made += usize::from(env.go(&mut report, &argv, &[OUT]));
    }
    env.assert_inputs_untouched();
    report.finish(&format!("multipart -override ({made} combined; C++ zeroed {} table entries)", ZEROED.load(Ordering::Relaxed)));
    assert!(made >= 200, "only {made} combinations");
}

// ----- separate and convert -----------------------------------------------------------------

/// Multipart inputs into single-part files: every multipart file in the set, and files the tool
/// itself combined; the output base name with and without a directory.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn separate_multipart_inputs() {
    let env = Env::new();
    let (cdir, odir) = env.pair.dirs();
    let cpp_tool = cpp("exrmultipart");
    let mut report = Report::default();
    let names: Vec<String> = (1..=6).map(|p| format!("sep.{p}.exr")).collect();
    let outputs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut made = 0;
    let mut inputs: Vec<PathBuf> = vec![golden("golden_multipart.exr")];
    inputs.extend(self::inputs().into_iter().filter(|f| size(f) < 4_000_000));
    // Files made by combining: three storage types, the same file thrice, views. The C++ makes them
    // (from its own working directory, the inputs being named relative to it).
    let s3 = [golden("golden_zip.exr"), golden("golden_tiled_zip_one.exr"), golden("golden_deep_scanline_zips.exr")];
    let work = TempDir::new("mp-made");
    let combos: [(&str, Vec<&PathBuf>); 3] = [("three.exr", s3.iter().collect()), ("same3.exr", vec![&s3[0], &s3[0], &s3[0]]), ("two.exr", vec![&s3[0], &s3[1]])];
    for (name, parts) in combos {
        let mut argv = s(&["-combine", "-i"]);
        argv.extend(parts.iter().map(|p| env.arg(p)));
        // The goldens differ in display window: `-override 1` takes the first one's.
        argv.extend(s(&["-o", name, "-override", "1"]));
        let c = run_in(&work.join(""), &cpp_tool, &argv);
        assert!(c.ok(), "{}", String::from_utf8_lossy(&c.stderr));
        inputs.push(work.join(name));
    }
    for f in &inputs {
        let a = env.arg(f);
        made += usize::from(env.go(&mut report, &s(&["-separate", "-i", &a, "-o", "sep"]), &outputs));
    }
    // A base name with a directory, an existing output, several inputs, none.
    let z = env.arg(&golden("golden_multipart.exr"));
    for d in [&cdir, &odir] {
        std::fs::create_dir_all(d.join("sub")).expect("create a directory");
    }
    made += usize::from(env.go(&mut report, &s(&["-separate", "-i", &z, "-o", "sub/deep"]), &["sub/deep.1.exr", "sub/deep.2.exr", "sub/deep.3.exr"]));
    made += usize::from(env.go(&mut report, &s(&["-separate", "-i", &z, "-o", "sep.0.exr"]), &["sep.0.exr.1.exr", "sep.0.exr.2.exr", "sep.0.exr.3.exr"]));
    env.go(&mut report, &s(&["-separate", "-i", &z, &z, "-o", "sep"]), &outputs);
    env.go(&mut report, &s(&["-separate", "-i", &z, "-o", "sep", "-override", "1"]), &outputs);
    env.assert_inputs_untouched();
    report.finish(&format!("multipart -separate ({made} separations over {} inputs)", inputs.len() + 4));
    assert!(made >= 100, "only {made} separations");
}

/// `-convert` over the multi-view images and every other input: multi-view single-part files become
/// one part per layer and view, others become one part; multipart, deep, tiled and subsampled
/// inputs fail alike.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn convert_every_input() {
    let env = Env::new();
    let mut report = Report::default();
    let (mut made, mut failed) = (0, 0);
    let mut multiview = 0;
    // The damaged files claim data windows of any size: `convert` holds the whole image in memory.
    let files: Vec<PathBuf> = inputs()
        .into_iter()
        .filter(|f| {
            let p = f.to_string_lossy();
            !p.contains("Damaged") && (size(f) < 3_000_000 || (p.contains("MultiView") && size(f) < 7_000_000))
        })
        .collect();
    for f in &files {
        let ok = env.go(&mut report, &s(&["-convert", "-i", &env.arg(f), "-o", OUT]), &[OUT]);
        made += usize::from(ok);
        failed += usize::from(!ok);
        multiview += usize::from(ok && f.to_string_lossy().contains("MultiView"));
    }
    // The same files with an unused `-view`, and `-override` (convert ignores both).
    for f in files.iter().filter(|f| f.to_string_lossy().contains("MultiView")) {
        let a = env.arg(f);
        env.go(&mut report, &s(&["-convert", "-i", "-view", "left", &a, "-o", OUT]), &[OUT]);
        env.go(&mut report, &s(&["-convert", "-i", &a, "-o", OUT, "-override", "1"]), &[OUT]);
    }
    env.go(&mut report, &s(&["-convert", "-i", &env.arg(&files[0]), &env.arg(&files[1]), "-o", OUT]), &[OUT]);
    env.assert_inputs_untouched();
    report.finish(&format!("multipart -convert over {} inputs ({made} converted, {multiview} multi-view, {failed} failing alike)", files.len()));
    assert!(multiview >= 4 && made >= 60 && failed >= 10, "{made} converted ({multiview} multi-view), {failed} failed: the comparison proves little");
}

/// One scanline image with the given channels (name, type) and, when `views` is not empty, a
/// `multiView` attribute; every sample is a function of its position and channel, so a channel
/// sent to the wrong part or a misplaced row shows in the bytes.
fn write_multiview(path: &Path, dw: Box2i, order: LineOrder, comp: Compression, chans: &[(&str, PixelType)], views: &[&str], name: Option<&str>) {
    let (w, h) = ((dw.max.x - dw.min.x + 1) as usize, (dw.max.y - dw.min.y + 1) as usize);
    let mut head = Header::with_windows(dw, dw, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, order, comp).expect("header");
    let mut list = ChannelList::new();
    for (n, t) in chans {
        list.insert(*n, Channel::new(*t, 1, 1, false)).expect("channel");
    }
    head.insert("channels", ChannelListAttribute::new(list)).expect("channels");
    head.insert("type", StringAttribute::new("scanlineimage".into())).expect("type");
    if let Some(n) = name {
        head.insert("name", StringAttribute::new(n.into())).expect("name");
    }
    if !views.is_empty() {
        head.add_multi_view(views.iter().map(|v| (*v).into()).collect()).expect("multiView");
    }
    let size = |t: PixelType| if t == PixelType::Half { 2 } else { 4 };
    let (mut fb, mut buf) = (FrameBuffer::new(), Vec::new());
    for (c, (n, t)) in chans.iter().enumerate() {
        let (start, sz) = (buf.len(), size(*t));
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 31 + y * 17 + c * 1009) % 4093) as u32;
                match t {
                    PixelType::Half => buf.extend_from_slice(&Half::from_f32(v as f32 / 8.0).bits().to_ne_bytes()),
                    PixelType::Float => buf.extend_from_slice(&(v as f32 / 8.0).to_ne_bytes()),
                    _ => buf.extend_from_slice(&(v * 3).to_ne_bytes()),
                }
            }
        }
        // The plane holds the window from its first pixel; slices address by absolute coordinates.
        let base = start as i64 - (dw.min.y as i64 * w as i64 + dw.min.x as i64) * sz as i64;
        fb.insert(*n, Slice::new(*t, base, sz, w * sz, 1, 1, 0.0, false, false).expect("slice")).expect("insert");
    }
    let mut out = MemOStream::new();
    {
        let mut file = MultiPartOutputFile::new(&mut out, &[head]).expect("file");
        let mut part = file.output_part(0).expect("part");
        part.set_frame_buffer(&fb).expect("frame buffer");
        part.write_pixels(&buf, h as i32).expect("pixels");
    }
    std::fs::write(path, out.into_data()).expect("write the image");
}

/// One generated image of [`convert_generated_multiview_images`].
struct Gen {
    name: &'static str,
    dw: Box2i,
    order: LineOrder,
    comp: Compression,
    chans: Vec<(String, PixelType)>,
    views: Vec<&'static str>,
    part: Option<&'static str>,
    /// How many parts `SplitChannels` must make of it: (layer, view) pairs, worked out by hand.
    parts: i32,
}

/// `-convert` on generated images that pin `SplitChannels`: views with and without layers, the hero
/// view's unadorned names, channels shared by every view, layers without `multiView`, every sample
/// type, windows off the origin, both line orders, a header that already has a part name.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn convert_generated_multiview_images() {
    use PixelType::{Float as F, Half as H, Uint as U};
    let env = Env::new();
    let work = TempDir::new("mp-gen");
    let win = |x0, y0, w: i32, h: i32| Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x0 + w - 1, y: y0 + h - 1 } };
    let rgb = |v: &str| -> Vec<(String, PixelType)> { ["R", "G", "B"].map(|c| (if v.is_empty() { c.to_string() } else { format!("{v}.{c}") }, H)).into() };
    let lr = || [rgb(""), rgb("right")].concat();
    let img = |name, dw, order, comp, chans, views: &[&'static str], part, parts| Gen { name, dw, order, comp, chans, views: views.to_vec(), part, parts };
    let (inc, dec) = (LineOrder::IncreasingY, LineOrder::DecreasingY);
    let mut cases = vec![
        img("stereo.exr", win(0, 0, 67, 43), inc, Compression::Zip, lr(), &["left", "right"], None, 2),
        img("stereo_dec.exr", win(-9, 5, 40, 33), dec, Compression::Piz, lr(), &["left", "right"], None, 2),
        // The hero is the first view: its channels keep plain names, here the second view's.
        img("hero_right.exr", win(3, -4, 31, 20), inc, Compression::NoCompression, [rgb("left"), rgb("")].concat(), &["right", "left"], None, 2),
    ];
    let mut three = [lr(), rgb("center")].concat();
    three.extend([("A".to_string(), F), ("Z".to_string(), F)]);
    cases.push(img("three_views.exr", win(0, 0, 50, 37), inc, Compression::Zips, three, &["left", "right", "center"], None, 3));
    let mut layered = lr();
    for layer in ["diffuse", "spec"] {
        for v in ["", "right"] {
            let name = if v.is_empty() { format!("{layer}.R") } else { format!("{layer}.{v}.R") };
            layered.push((name, if layer == "diffuse" { F } else { U }));
        }
    }
    cases.push(img("layered.exr", win(0, 0, 64, 64), inc, Compression::Rle, layered.clone(), &["left", "right"], None, 6));
    cases.push(img("layered_named.exr", win(0, 0, 64, 64), inc, Compression::Zip, layered, &["left", "right"], Some("beauty"), 6));
    // No `multiView`: parts follow the layers alone, and a single layer stays one part.
    let mut plain = rgb("");
    plain.extend([("fx.R".to_string(), F), ("fx.G".to_string(), F), ("depth.Z".to_string(), F), ("depth.id".to_string(), U)]);
    cases.push(img("layers_only.exr", win(0, 0, 33, 35), inc, Compression::Zip, plain, &[], None, 3));
    cases.push(img("one_layer.exr", win(0, 0, 33, 35), inc, Compression::Zip, rgb(""), &[], None, 1));
    cases.push(img("named_single.exr", win(0, 0, 10, 10), inc, Compression::Zip, rgb(""), &[], Some("only"), 1));
    // A view in the attribute that no channel uses, and channels of a view that is not listed.
    cases.push(img("unused_view.exr", win(0, 0, 20, 20), inc, Compression::Zip, lr(), &["left", "right", "top"], None, 2));
    cases.push(img("unlisted_view.exr", win(0, 0, 20, 20), inc, Compression::Zip, lr(), &["left"], None, 2));
    cases.push(img("one_row.exr", win(0, 0, 100, 1), inc, Compression::Zip, lr(), &["left", "right"], None, 2));
    cases.push(img("one_column.exr", win(0, 0, 1, 100), inc, Compression::Zip, lr(), &["left", "right"], None, 2));

    let mut report = Report::default();
    let (mut made, n) = (0, cases.len());
    let mut shape = Vec::new();
    for g in cases {
        let path = work.join(g.name);
        let chans: Vec<(&str, PixelType)> = g.chans.iter().map(|(n, t)| (n.as_str(), *t)).collect();
        write_multiview(&path, g.dw, g.order, g.comp, &chans, &g.views, g.part);
        let a = env.arg(&path);
        made += usize::from(env.go(&mut report, &s(&["-convert", "-i", &a, "-o", OUT]), &[OUT]));
        shape.push((g.name, a, g.parts));
    }
    report.finish(&format!("multipart -convert generated images ({made} of {n} converted)"));
    // The parity above would also hold if both tools made one part of everything: count ours.
    let (_, odir) = env.pair.dirs();
    for (name, a, parts) in shape {
        let o = run_ours_in(&odir, "multipart", s(&["-convert", "-i", &a, "-o", "shape.exr"]));
        assert!(o.ok(), "{name}: {}", String::from_utf8_lossy(&o.stderr));
        let mut stream = StdIfStream::open(odir.join("shape.exr").to_str().expect("utf-8 path")).expect("open the output");
        let got = MultiPartInputFile::new(&mut stream).expect("read the output").parts();
        assert_eq!(got, parts, "{name}: parts of the converted file");
    }
    env.assert_inputs_untouched();
    assert!(made >= n - 2, "only {made} of {n} generated images converted");
}

// ----- the command line and the failures ----------------------------------------------------

/// Grammar corners of `main`: modes, option order, stray words, missing pieces, unknown modes.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn grammar_and_failures() {
    let env = Env::new();
    let (cdir, odir) = env.pair.dirs();
    let z = env.arg(&golden("golden_zip.exr"));
    let t = env.arg(&golden("golden_tiled_zip_one.exr"));
    let multi = env.arg(&golden("golden_multipart.exr"));
    for d in [&cdir, &odir] {
        std::fs::write(d.join("not_exr.txt"), b"this is not an image").expect("write the text file");
        std::fs::copy(golden("golden_zip.exr"), d.join("local.exr")).expect("copy an input");
    }
    let mut report = Report::default();
    let (mut ok, mut failed) = (0, 0);
    let mut case = |argv: Vec<String>| {
        let made = env.go(&mut report, &argv, &[OUT, "sep.1.exr"]);
        ok += usize::from(made);
        failed += usize::from(!made);
    };
    let c = |w: &[&str]| s(w);
    // Order and stray words.
    case(c(&["-combine", "-o", OUT, "-i", &z]));
    case(c(&["-combine", "stray", "-i", &z, "-o", OUT, "-i", &t]));
    case(c(&["-combine", "-i", &z, "-i", &t, "-o", OUT]));
    case(c(&["-i", &z, "-o", OUT, "-combine"]));
    case(c(&["-combine", "-i", &z, "-o", "other.exr", "-o", OUT]));
    case(c(&["-combine", "-i", &z, "-o", "a.exr", OUT]));
    case(c(&["-combine", "-i", &z, "-o", OUT, "extra", "words"]));
    // Missing pieces.
    case(c(&["-combine"]));
    case(c(&["-combine", "-o", OUT]));
    case(c(&["-combine", "-i"]));
    case(c(&["-combine", "-i", "-o", OUT]));
    case(c(&["-combine", "-i", &z]));
    case(c(&["-combine", "-i", &z, "-o"]));
    case(c(&["-separate", "-i", &z]));
    case(c(&["-convert", "-i", &z]));
    case(c(&["-combine", "-view", "x", "-i", &z, "-o", OUT]));
    case(c(&["-combine", "-i", &z, "-o", OUT, "-view", "x"]));
    // Unknown modes: the C++ prints the usage and exits 0 without writing.
    case(c(&["-foo", "-i", &z, "-o", OUT]));
    case(c(&["combine", "-i", &z, "-o", OUT]));
    case(c(&["-COMBINE", "-i", &z, "-o", OUT]));
    case(c(&["-i", &z, "-o", OUT]));
    // Each mode against inputs it does not fit.
    case(c(&["-separate", "-i", &z, &t, "-o", "sep"]));
    case(c(&["-convert", "-i", &z, &t, "-o", OUT]));
    case(c(&["-convert", "-i", &multi, "-o", OUT]));
    case(c(&["-convert", "-i", &t, "-o", OUT]));
    // Files: missing, not EXR, unwritable, the same as an input.
    case(c(&["-combine", "-i", "missing.exr", "-o", OUT]));
    case(c(&["-combine", "-i", &z, "missing.exr", "-o", OUT]));
    case(c(&["-combine", "-i", "not_exr.txt", "-o", OUT]));
    case(c(&["-separate", "-i", "missing.exr", "-o", "sep"]));
    case(c(&["-separate", "-i", "not_exr.txt", "-o", "sep"]));
    case(c(&["-convert", "-i", "missing.exr", "-o", OUT]));
    case(c(&["-convert", "-i", "not_exr.txt", "-o", OUT]));
    case(c(&["-combine", "-i", &z, "-o", "no/such/dir/out.exr"]));
    case(c(&["-separate", "-i", &multi, "-o", "no/such/dir/out"]));
    case(c(&["-convert", "-i", &z, "-o", "no/such/dir/out.exr"]));
    case(c(&["-combine", "-i", "local.exr", "-o", "local.exr"]));
    case(c(&["-combine", "-i", "local.exr", &z, "-o", "local.exr"]));
    case(c(&["-separate", "-i", "local.exr", "-o", "local.exr"]));
    case(c(&["-combine", "-i", "local.exr::x", "-o", "local.exr"]));
    case(c(&["-combine", "-i", "local.exr:0", "-o", "local.exr"]));
    // A file named by more than a drive letter can hold: the C++ cannot name this one either.
    case(c(&["-combine", "-i", "C:/x.exr", "-o", OUT]));
    env.assert_inputs_untouched();
    report.finish(&format!("multipart grammar and failures ({ok} succeeded, {failed} failing alike)"));
    assert!(ok >= 4 && failed >= 25, "{ok} succeeded, {failed} failed");
}

/// `-h` prints an option reference and succeeds wherever it stands; `--version` is the C++'s text.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn help_and_version() {
    for args in [&["-h"][..], &["--help"], &["-combine", "-i", "a.exr", "-h"], &["-combine", "-o", "--help"]] {
        let (c, o) = (run(&cpp("exrmultipart"), args), run_ours("multipart", args));
        assert_eq!((c.code, o.code), (Some(0), Some(0)), "{args:?}");
        assert!(c.text().contains("-override") && o.text().contains("-override") && o.text().contains("Examples:"), "{args:?}");
    }
    assert_same_version("exrmultipart", "multipart", &[&["--version"], &["-combine", "-i", "a.exr", "--version"]]);
    // `-view` outside `-i` stops the run before a later `-h`.
    let (c, o) = (run(&cpp("exrmultipart"), ["-view", "v", "-h"]), run_ours("multipart", ["-view", "v", "-h"]));
    assert_eq!((c.code, o.code), (Some(1), Some(1)));
}

/// Where we deliberately do better: an output that is an input by another spelling is refused (the C++
/// empties the input first), and a part of no known type is an error (the C++ writes it without
/// pixels).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn deliberate_differences() {
    let dir = TempDir::new("mp-diff");
    let src = golden("golden_zip.exr");
    let original = std::fs::read(&src).expect("read the golden");
    for mode in [&["-combine", "-i", "a.exr", "-o", ".\\a.exr"][..], &["-combine", "-i", "a.exr::n", "-o", "./a.exr"], &["-convert", "-i", "a.exr", "-o", ".\\a.exr"]] {
        let file = dir.join("a.exr");
        std::fs::write(&file, &original).expect("write the input");
        let o = run_ours_in(&dir.join(""), "multipart", mode);
        assert_eq!(o.code, Some(1), "{mode:?}: {}", o.text());
        assert_eq!(std::fs::read(&file).expect("read the input"), original, "{mode:?}: the input changed");
        assert!(String::from_utf8_lossy(&o.stderr).contains("cannot be the same"), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // A part of a type no reader knows: the C++ reports success and writes the part without a
    // single chunk (so the output cannot be read back whole); we refuse, naming the type.
    let mut bytes = std::fs::read(golden("golden_multipart.exr")).expect("read the golden");
    let at = bytes.windows(14).position(|w| w == b"scanlineimage\0").expect("part 0's type");
    bytes[at + 12] = b'X';
    std::fs::write(dir.join("unknown.exr"), &bytes).expect("write the patched file");
    let d = dir.join("");
    let c = run_in(&d, &cpp("exrmultipart"), ["-combine", "-i", "unknown.exr", "-o", "c_unknown.exr"]);
    assert!(c.ok() && c.text().ends_with("Combine Success\n") && !c.text().contains("part 0:"), "{}", c.text());
    let c = run_in(&d, &cpp("exrmultipart"), ["-separate", "-i", "unknown.exr", "-o", "c_sep"]);
    assert!(c.ok() && c.text().ends_with("Separate Success\n"), "{}", c.text());
    for args in [&["-combine", "-i", "unknown.exr", "-o", "o_unknown.exr"][..], &["-separate", "-i", "unknown.exr", "-o", "o_sep"]] {
        let o = run_ours_in(&d, "multipart", args);
        assert_eq!(o.code, Some(1), "{args:?}: {}", o.text());
        assert!(String::from_utf8_lossy(&o.stderr).contains("unsupported type \"scanlineimagX\""), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // `-override`'s value must be a number: the C++ `atoi` makes anything else 0 (exrmultipart.cpp:700).
    let file = dir.join("n.exr");
    std::fs::write(&file, &original).expect("write the input");
    for bad in ["x", "1x", "1.5"] {
        let args = ["-combine", "-i", "n.exr", "-o", "n_out.exr", "-override", bad];
        let (c, o) = (run_in(&dir.join(""), &cpp("exrmultipart"), args), run_ours_in(&dir.join(""), "multipart", args));
        assert!(c.ok(), "{bad}: the C++ tool takes it: {}", String::from_utf8_lossy(&c.stderr));
        assert_eq!(o.code, Some(1), "{bad}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("Expected an integer"), "{}", String::from_utf8_lossy(&o.stderr));
    }
    // A refused set of headers leaves an existing output as it was (the C++ checks before it opens it).
    let (other, out) = (golden("golden_tiled_zip_one.exr"), dir.join("keep.exr"));
    std::fs::copy(&other, dir.join("t.exr")).expect("copy the tiled golden");
    std::fs::write(&out, b"precious").expect("write the output");
    let args = ["-combine", "-i", "n.exr", "t.exr", "-o", "keep.exr"];
    let (c, o) = (run_in(&dir.join(""), &cpp("exrmultipart"), args), run_ours_in(&dir.join(""), "multipart", args));
    assert_eq!((c.code, o.code), (Some(1), Some(1)), "conflicting display windows");
    assert_eq!(std::fs::read(&out).expect("read the output"), b"precious");
    // The spelling the C++ does catch is still caught.
    let file = dir.join("b.exr");
    std::fs::write(&file, &original).expect("write the input");
    let (c, o) = (run_in(&dir.join(""), &cpp("exrmultipart"), ["-combine", "-i", "b.exr", "-o", "b.exr"]), run_ours_in(&dir.join(""), "multipart", ["-combine", "-i", "b.exr", "-o", "b.exr"]));
    assert_eq!((c.code, o.code), (Some(1), Some(1)));
    assert_eq!(std::fs::read(&file).expect("read the input"), original);
}

// ----- the comparison itself ----------------------------------------------------------------

/// The comparator must bite: it accepts exactly "offset table entries zeroed", nothing else.
#[test]
fn zeroed_tables_are_the_only_accepted_difference() {
    let ours = std::fs::read(golden("golden_multipart.exr")).expect("read the golden");
    let (start, n) = offset_tables(&ours).expect("the golden has tables");
    assert!(n >= 3, "{n} entries");
    assert_eq!(same_but_zeroed(&ours, &ours), Ok(0));

    // A zeroed table, whole or in part, passes.
    let mut cpp = ours.clone();
    cpp[start..start + 8 * n].fill(0);
    assert_eq!(same_but_zeroed(&cpp, &ours), Ok(n));
    let mut cpp = ours.clone();
    cpp[start + 8 * (n - 1)..start + 8 * n].fill(0);
    assert_eq!(same_but_zeroed(&cpp, &ours), Ok(1));

    // Any other byte, a wrong non-zero entry, a longer file, or zeroes outside the tables, fails.
    for at in [0, 9, start - 1, start + 8 * n, ours.len() - 1] {
        let mut cpp = ours.clone();
        cpp[at] ^= 1;
        assert!(same_but_zeroed(&cpp, &ours).is_err(), "a change at {at} passed");
    }
    let mut cpp = ours.clone();
    cpp[start] ^= 1;
    assert!(same_but_zeroed(&cpp, &ours).is_err(), "a wrong entry passed");
    let mut cpp = ours.clone();
    cpp.push(0);
    assert!(same_but_zeroed(&cpp, &ours).is_err(), "a longer file passed");
    // Ours is zero where the C++ has data: not the C++ defect.
    let mut damaged = ours.clone();
    damaged[start..start + 8].fill(0);
    assert!(same_but_zeroed(&ours, &damaged).is_err(), "our zeroed entry passed");
}

#[test]
fn tables_are_found_in_single_and_multipart_files() {
    for (name, parts) in [("golden_zip.exr", 1), ("golden_multipart.exr", 3), ("golden_tiled_zip_one.exr", 1)] {
        let bytes = std::fs::read(golden(name)).expect("read the golden");
        let (start, n) = offset_tables(&bytes).unwrap_or_else(|| panic!("{name}: no tables"));
        let first = u64::from_le_bytes(bytes[start..start + 8].try_into().expect("8 bytes"));
        assert!(n >= parts, "{name}: {n} entries");
        assert_eq!(first, (start + 8 * n) as u64, "{name}: the first chunk follows the tables");
    }
}
