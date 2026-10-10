//! `exr-rs check` against OpenEXR's `exrcheck` (the C++ tool built by `cargo xtask parity-tools`):
//! the same files through the same options must give the same exit code and the same stdout, for
//! the committed goldens and for the whole private image set, fuzz reproducers included.
//!
//! Run through `cargo run -p xtask -- parity-tools` (sets `EXR_TOOLS_DIR`); `#[ignore]`d otherwise.

mod tool_support;

use std::path::{Path, PathBuf};
use tool_support::{cpp, ours, run};

/// Option sets of exrcheck: default, each flag alone, and combinations.
const MODES: &[&[&str]] = &[
    &[],
    &["-m"],
    &["-t"],
    &["-s"],
    &["-c"],
    &["-m", "-t"],
    &["-m", "-c"],
    &["-t", "-c"],
    &["-s", "-m", "-t"],
    &["-s", "-c"],
];

/// Every file under `dir` that is not obviously not an image (the fuzz reproducers have no or odd
/// extensions, so this excludes instead of including).
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if !p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.')) {
                walk(&p, out);
            }
            continue;
        }
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        let doc = matches!(ext, "md" | "rst" | "txt" | "json" | "jpg" | "f32" | "py" | "yml" | "cmake")
            || name.starts_with('.')
            || ["LICENSE", "COPYING", "README"].iter().any(|d| name.starts_with(d));
        if !doc {
            out.push(p);
        }
    }
}

/// Run both tools on `files` for every mode; returns the mismatches as printable lines.
fn compare(files: &[PathBuf], modes: &[&[&str]]) -> (usize, Vec<String>) {
    let (n, diffs, _) = compare_hangs(files, modes, &[]);
    (n, diffs)
}

/// [`compare`], allowing only the exact known C++ timeout pairs in [`KNOWN_CPP_HANGS`].
/// Ours must finish with `bad`; a C++ run that finishes is compared normally. Resource-dependent
/// C++ runs can finish or time out, so the observed timeout set need not include every allowed pair.
fn compare_hangs(
    files: &[PathBuf],
    modes: &[&[&str]],
    known: &[(&str, &[&str])],
) -> (usize, Vec<String>, Vec<(String, String)>) {
    let (cpp_exe, mut n, mut diffs, mut hung) = (cpp("exrcheck"), 0usize, Vec::new(), Vec::new());
    for mode in modes {
        for f in files {
            let args: Vec<&std::ffi::OsStr> = mode
                .iter()
                .map(std::ffi::OsStr::new)
                .chain(std::iter::once(f.as_os_str()))
                .collect();
            let (theirs, mine) = (
                run(&cpp_exe, &args),
                run(
                    ours(),
                    std::iter::once("check".as_ref()).chain(args.iter().copied()),
                ),
            );
            n += 1;
            let name = f
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let is_known = known.iter().any(|(k, m)| *k == name && m == mode);
            let bad_verdict = format!(" file {} bad\n", f.display());
            match classify_timeout(is_known, &theirs, &mine, &bad_verdict) {
                Ok(true) => {
                    hung.push((name, mode.join(" ")));
                    continue;
                }
                Err(why) => {
                    diffs.push(format!("{mode:?} {}: {why}", f.display()));
                    continue;
                }
                Ok(false) => {}
            }
            if is_flatten_bug(f, mode, &theirs, &mine) {
                continue;
            }
            if theirs.code != mine.code || theirs.text() != mine.text() {
                diffs.push(format!(
                    "{mode:?} {}: cpp {} {:?} / ours {} {:?}",
                    f.display(),
                    theirs.end(),
                    theirs.text().trim(),
                    mine.end(),
                    mine.text().trim()
                ));
            }
        }
    }
    (n, diffs, hung)
}

/// Only a pinned C++ timeout paired with a complete Rust rejection bypasses normal parity.
/// A hung reference can overflow the bounded diagnostic capture; that is not a completed
/// verdict to compare. Completed truncated output and every Rust truncation still fail.
fn classify_timeout(
    known: bool,
    theirs: &tool_support::Out,
    mine: &tool_support::Out,
    bad_verdict: &str,
) -> Result<bool, String> {
    if mine.timed_out
        || !matches!(mine.code, Some(0) | Some(1))
        || (!theirs.timed_out && !matches!(theirs.code, Some(0) | Some(1)))
        || mine.truncated
        || (!theirs.timed_out && theirs.truncated)
    {
        return Err(format!(
            "invalid comparison outcome: \
             cpp {{ code: {:?}, timed_out: {}, truncated: {}, stdout_len: {}, stderr_len: {}, stdout_prefix: {:?}, stderr_prefix: {:?} }}; \
             rust {{ code: {:?}, timed_out: {}, truncated: {}, stdout_len: {}, stderr_len: {}, stdout_prefix: {:?}, stderr_prefix: {:?} }}",
            theirs.code, theirs.timed_out, theirs.truncated, theirs.stdout.len(), theirs.stderr.len(),
            String::from_utf8_lossy(&theirs.stdout[..theirs.stdout.len().min(512)]),
            String::from_utf8_lossy(&theirs.stderr[..theirs.stderr.len().min(512)]),
            mine.code, mine.timed_out, mine.truncated, mine.stdout.len(), mine.stderr.len(),
            String::from_utf8_lossy(&mine.stdout[..mine.stdout.len().min(512)]),
            String::from_utf8_lossy(&mine.stderr[..mine.stderr.len().min(512)]),
        ));
    }
    if !theirs.timed_out {
        if known
            && !(theirs.code == Some(1)
                && mine.code == Some(1)
                && theirs.text() == bad_verdict
                && mine.text() == bad_verdict)
        {
            return Err("a finished pinned pair must agree on the bad verdict".into());
        }
        return Ok(false);
    }
    if known && mine.code == Some(1) && mine.text() == bad_verdict && mine.stderr.is_empty() {
        Ok(true)
    } else {
        Err(format!(
            "unexpected timeout pair: C++ {}, ours {} {:?}",
            theirs.end(),
            mine.end(),
            mine.text()
        ))
    }
}

#[test]
fn timeout_allowance_requires_a_pinned_cpp_hang_and_clean_rust_rejection() {
    let out = |code, timed_out, stdout: &[u8]| tool_support::Out {
        code,
        timed_out,
        stdout: stdout.to_vec(),
        stderr: Vec::new(),
        truncated: false,
    };
    let classify = |known, theirs: &tool_support::Out, mine: &tool_support::Out| classify_timeout(known, theirs, mine, " file damaged bad\n");
    let crash = out(None, false, b"");
    assert!(classify(false, &crash, &crash).is_err());
    let windows_crash = out(Some(-1073741819), false, b"");
    assert!(classify(false, &windows_crash, &windows_crash).is_err());
    let other_exit = out(Some(2), false, b"");
    assert!(classify(false, &other_exit, &other_exit).is_err());
    let hang = out(None, true, b" file damaged ");
    let bad = out(Some(1), false, b" file damaged bad\n");
    assert_eq!(classify(true, &hang, &bad), Ok(true));
    assert!(classify(false, &hang, &bad).is_err());
    assert!(classify(true, &hang, &out(Some(0), false, b" file damaged OK\n")).is_err());
    assert!(classify(true, &hang, &hang).is_err());
    assert!(classify(true, &hang, &out(Some(1), false, b"bad")).is_err());
    assert_eq!(
        classify(true, &bad, &bad),
        Ok(false),
        "finished runs still require normal parity"
    );
    let unexpected_success = out(Some(0), false, b" file damaged OK\n");
    assert!(classify(true, &unexpected_success, &unexpected_success).is_err());
    assert!(classify(true, &bad, &out(Some(1), false, b" file other bad\n")).is_err());
    assert!(classify(true, &hang, &out(Some(1), false, b" file other bad\n")).is_err());
    let mut cut = bad;
    cut.truncated = true;
    assert!(classify(true, &hang, &cut).is_err());
    let mut noisy_hang = hang;
    noisy_hang.truncated = true;
    noisy_hang.stderr = b"repeated chunk-table error".to_vec();
    assert_eq!(classify(true, &noisy_hang, &out(Some(1), false, b" file damaged bad\n")), Ok(true));
    assert!(classify(false, &noisy_hang, &out(Some(1), false, b" file damaged bad\n")).is_err());
    assert!(classify(true, &cut, &out(Some(1), false, b" file damaged bad\n")).is_err());
    assert!(classify(true, &noisy_hang, &cut).is_err());
    assert!(classify(true, &noisy_hang, &noisy_hang).is_err());
    for incomplete in [
        out(None, false, b""),
        out(Some(-1073741819), false, b""),
        out(Some(2), false, b" file damaged bad\n"),
        out(Some(0), false, b" file damaged OK\n"),
        out(Some(1), false, b" file other bad\n"),
        out(Some(1), false, b" file damaged bad\nextra\n"),
    ] {
        assert!(classify(true, &noisy_hang, &incomplete).is_err());
    }
    let mut noisy_rust = out(Some(1), false, b" file damaged bad\n");
    noisy_rust.stderr = b"unexpected diagnostic".to_vec();
    assert!(classify(true, &noisy_hang, &noisy_rust).is_err());
}

/// Whether some part of `f` is a deep scanline part without a `Z` or an `A` channel: the flat readers of
/// `exrcheck` cannot composite such a part and it reports the valid file as bad (`ImfCheckFile.cpp:898`,
/// `:1176`, `:1189`; `ImfCompositeDeepScanLine.cpp:103-123`), which `exr-rs check` does not.
fn has_unflattenable_deep_part(f: &Path) -> bool {
    let Some(name) = f.to_str() else { return false };
    let Ok(stream) = exr_core::io::StdIfStream::open(name) else { return false };
    let Ok(multi) = exr_core::multipart::MultiPartInputFile::new_owned(Box::new(stream)) else { return false };
    (0..multi.parts()).filter_map(|i| multi.header(i).ok()).any(|h| {
        h.type_().is_ok_and(|t| t == exr_core::header::DEEP_SCANLINE)
            && h.channels().is_ok_and(|c| c.find_channel("Z").is_none() || c.find_channel("A").is_none())
    })
}

/// The known divergence of the flat readers' exception: `exrcheck` bad (exit 1), ours OK (exit 0), same
/// text but the verdict word, on a file with an unflattenable deep scanline part, in a mode without `-c`.
/// Anything else that differs is a mismatch.
fn is_flatten_bug(f: &Path, mode: &[&str], theirs: &tool_support::Out, mine: &tool_support::Out) -> bool {
    !mode.contains(&"-c")
        && theirs.code == Some(1)
        && mine.code == Some(0)
        && theirs.text().replace(" bad\n", " OK\n") == mine.text()
        && has_unflattenable_deep_part(f)
}

/// Better than the C++: files of the fuzz corpus on which `exrcheck` does not finish (it is killed by
/// the harness timeout; `memory_DOS_*` and the `memory_allocation_error_2` reproducer ask it for
/// gigabytes of memory or hours of work) while `exr-rs check` answers `bad` in bounded time. Only
/// these eight `(file, mode)` pairs may time out. Whether a resource-heavy C++ run finishes
/// depends on available resources; a finished pair still requires the normal parity comparison.
const KNOWN_CPP_HANGS: &[(&str, &[&str])] = &[
    ("memory_DOS_1", &[]),
    ("memory_DOS_1", &["-s"]),
    ("memory_DOS_1", &["-c"]),
    ("memory_DOS_1", &["-s", "-c"]),
    ("openexr_2.2.0_memory_allocation_error_2_exr", &[]),
    ("openexr_2.2.0_memory_allocation_error_2_exr", &["-s"]),
    ("openexr_2.2.0_memory_allocation_error_2_exr", &["-c"]),
    ("openexr_2.2.0_memory_allocation_error_2_exr", &["-s", "-c"]),
];

fn assert_same(what: &str, (n, diffs): (usize, Vec<String>)) {
    eprintln!("{what}: {n} comparisons, {} mismatches", diffs.len());
    assert!(diffs.is_empty(), "{what}: {} of {n} differ:\n{}", diffs.len(), diffs.join("\n"));
}

#[test]
#[ignore = "needs the C++ tools (cargo xtask parity-tools)"]
fn goldens_match_exrcheck_in_every_mode() {
    let mut files = Vec::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity"), &mut files);
    files.retain(|p| p.extension().is_some_and(|e| e == "exr"));
    assert!(files.len() > 100, "{} goldens", files.len());
    assert_same("goldens", compare(&files, MODES));
}

/// Goldens with 20 bytes flipped in the compressed pixel data, truncated at several points, and with
/// the magic broken: the damaged-input verdicts.
#[test]
#[ignore = "needs the C++ tools (cargo xtask parity-tools)"]
fn damaged_goldens_match_exrcheck_in_every_mode() {
    let dir = tool_support::TempDir::new("check-damaged");
    let parity = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity");
    let mut files = Vec::new();
    for name in [
        "golden", "golden_piz", "golden_rle", "golden_pxr24", "golden_b44", "golden_dwab", "golden_htj2k256",
        "golden_multipart", "golden_deep_scanline_zips", "golden_deep_tiled_zips_one", "golden_deep_tiled_zips_mipmap",
        "golden_deep_composite_a", "golden_sub_dwaa_x2y2", "golden_rgba",
    ] {
        let good = std::fs::read(parity.join(format!("{name}.exr"))).expect("golden");
        let n = good.len();
        let mut variants: Vec<(String, Vec<u8>)> = Vec::new();
        let mut flipped = good.clone();
        flipped[n - 120..n - 100].iter_mut().for_each(|b| *b ^= 0xff);
        variants.push(("flip".into(), flipped));
        let mut magic = good.clone();
        magic[0] ^= 0xff;
        variants.push(("magic".into(), magic));
        for k in 1..8 {
            variants.push((format!("cut{k}"), good[..n * k / 8].to_vec()));
        }
        for (tag, data) in variants {
            let path = dir.join(&format!("{name}_{tag}.exr"));
            std::fs::write(&path, data).expect("write");
            files.push(path);
        }
    }
    assert_same("damaged goldens", compare(&files, MODES));
}

#[test]
#[ignore = "needs the C++ tools (cargo xtask parity-tools) and the private image set"]
fn image_set_matches_exrcheck_in_every_mode() {
    let mut files = Vec::new();
    walk(&exr_test_data::root(), &mut files);
    assert!(files.len() > 300, "{} files", files.len());
    for (name, mode) in KNOWN_CPP_HANGS {
        assert!(
            files
                .iter()
                .any(|f| f.file_name().is_some_and(|n| n == *name))
                && MODES.contains(mode),
            "allowlisted pair was not visited: {name} {mode:?}"
        );
    }
    let (n, diffs, hung) = compare_hangs(&files, MODES, KNOWN_CPP_HANGS);
    assert_same("image set", (n, diffs));
    eprintln!(
        "C++ timed out on {} of {} allowed file/mode pairs",
        hung.len(),
        KNOWN_CPP_HANGS.len()
    );
}

/// The command-line behaviour around the check: output format, exit codes, `--version`, a missing file.
#[test]
#[ignore = "needs the C++ tools (cargo xtask parity-tools)"]
fn command_line_matches_exrcheck() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity/golden.exr");
    let bad = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity/golden_multipart.exr");
    let missing = Path::new("definitely/not/here.exr");
    let cpp_exe = cpp("exrcheck");
    for args in [
        vec![golden.as_os_str()],
        vec![golden.as_os_str(), golden.as_os_str()],
        vec![bad.as_os_str()],
        vec![golden.as_os_str(), bad.as_os_str(), golden.as_os_str()],
        vec![missing.as_os_str()],
        vec![golden.as_os_str(), missing.as_os_str(), bad.as_os_str()],
        vec!["--version".as_ref()],
    ] {
        let theirs = run(&cpp_exe, &args);
        let mine = run(ours(), std::iter::once("check".as_ref()).chain(args.iter().copied()));
        assert_eq!((theirs.code, theirs.text()), (mine.code, mine.text()), "{args:?}");
        assert_eq!(String::from_utf8_lossy(&theirs.stderr).replace("\r\n", "\n"), String::from_utf8_lossy(&mine.stderr), "{args:?} stderr");
    }
}

/// Deep scanline files `exr-rs check` calls OK and `exrcheck` calls bad: the second known, deliberate
/// divergence. The library path of `exrcheck` reads every deep scanline part through the flat readers, which
/// composite it and throw "missing a Z channel" / "missing an alpha channel" unless the part has channels
/// named `Z` and `A` (`ImfCompositeDeepScanLine.cpp:103-123`); `ImfCheckFile.cpp:898`, `:1176`, `:1189`
/// take that exception for damage ("only DeepTiled parts are expected to throw"). Valid deep data without
/// `Z` or `A` is reported bad; its core path (`-c`) and ours say OK. Files with both
/// channels must agree in every mode.
#[test]
#[ignore = "needs the C++ tools (cargo xtask parity-tools)"]
fn deep_scanline_without_z_or_alpha_is_the_only_flatten_divergence_from_exrcheck() {
    use exr_core::attr::{Compression, LevelMode, LevelRoundingMode, PixelType};
    use exr_util::deep_image::DeepImage;
    use exr_util::deep_image_io::save_deep_image;
    use exr_util::image_data_window::DataWindowSource;
    use imath_rs::{Box2i, V2i};
    use PixelType::{Float, Half, Uint};

    let dir = tool_support::TempDir::new("check-deep-scan-az");
    // (name, channels, diverges)
    type Channels = &'static [(&'static str, PixelType)];
    let cases: [(&str, Channels, bool); 6] = [
        ("a_z_r", &[("A", Half), ("Z", Float), ("R", Half)], false),
        ("a_z_float", &[("A", Float), ("Z", Float)], false),
        ("a_z_zback_id", &[("A", Half), ("Z", Float), ("ZBack", Float), ("id", Uint)], false),
        ("z_r", &[("Z", Float), ("R", Half)], true),
        ("a_r", &[("A", Half), ("R", Half)], true),
        ("r_only", &[("R", Half)], true),
    ];
    let mut files = Vec::new();
    for (name, channels, diverges) in cases {
        let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 9, y: 5 } };
        let mut img = DeepImage::new(&dw, LevelMode::OneLevel, LevelRoundingMode::RoundDown).unwrap();
        for (n, t) in channels {
            img.insert_type(*n, *t).unwrap();
        }
        img.level_mut(0).unwrap().edit_counts(|c| c.iter_mut().enumerate().for_each(|(i, x)| *x = (i % 3) as u32)).unwrap();
        let mut hdr = exr_core::header::Header::new();
        *hdr.compression_mut().unwrap() = Compression::Zips;
        let path = dir.join(&format!("{name}.exr"));
        save_deep_image(&path, Some(&hdr), &img, DataWindowSource::Image).unwrap();
        files.push((path, diverges));
    }

    let cpp_exe = cpp("exrcheck");
    for (f, flagged) in &files {
        let mut default_diverges = false;
        for mode in MODES {
            let args: Vec<&std::ffi::OsStr> = mode.iter().map(std::ffi::OsStr::new).chain(std::iter::once(f.as_os_str())).collect();
            let (theirs, mine) = (run(&cpp_exe, &args), run(ours(), std::iter::once("check".as_ref()).chain(args.iter().copied())));
            let known = *flagged && is_flatten_bug(f, mode, &theirs, &mine);
            assert!(
                known || (theirs.code == mine.code && theirs.text() == mine.text()),
                "{} {mode:?}: cpp {:?} {:?} / ours {:?} {:?}",
                f.display(),
                theirs.code,
                theirs.text().trim(),
                mine.code,
                mine.text().trim()
            );
            default_diverges |= known && mode.is_empty();
        }
        assert_eq!(*flagged, default_diverges, "{}", f.display());
    }
}

/// Deep tiled files `exr-rs check` calls OK and `exrcheck` calls bad: the one known, deliberate
/// divergence. `readDeepTile` (`ImfCheckFile.cpp:698`) passes `(x, y, x, y)` for `(dx1, dx2, dy1, dy2)` and
/// so asks for tiles that do not exist whenever `max (x, y)` reaches the number of tile columns or rows of
/// a level; `exrcheck`'s library path rejects such valid files, its core path (`-c`) and ours do not.
/// Every other verdict must be equal.
#[test]
#[ignore = "needs the C++ tools (cargo xtask parity-tools)"]
fn deep_tile_request_bug_is_the_only_divergence_from_exrcheck() {
    use exr_core::attr::{Compression, LevelMode, LevelRoundingMode, PixelType, TileDescription};
    use exr_util::deep_image::DeepImage;
    use exr_util::deep_image_io::save_deep_image;
    use exr_util::image_data_window::DataWindowSource;
    use imath_rs::{Box2i, V2i};

    let dir = tool_support::TempDir::new("check-deep-tiles");
    // (name, size, level mode, diverges) with 8 x 4 tiles
    let cases = [
        ("wide", (40, 8), LevelMode::OneLevel, true),     // 5 x 2 tiles
        ("tall", (16, 28), LevelMode::OneLevel, true),    // 2 x 7 tiles
        ("rip", (22, 12), LevelMode::RipmapLevels, true), // levels with 1 x 3 tiles
        ("square", (24, 12), LevelMode::OneLevel, false), // 3 x 3 tiles
        ("mip", (22, 12), LevelMode::MipmapLevels, false),
    ];
    let mut files = Vec::new();
    for (name, (w, h), mode, diverges) in cases {
        let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w - 1, y: h - 1 } };
        let mut img = DeepImage::new(&dw, mode, LevelRoundingMode::RoundDown).unwrap();
        img.insert_type("Z", PixelType::Float).unwrap();
        for ly in 0..img.num_y_levels() {
            for lx in 0..img.num_x_levels() {
                if let Ok(lv) = img.level_at_mut(lx, ly) {
                    lv.edit_counts(|c| c.fill(1)).unwrap();
                }
            }
        }
        let mut hdr = exr_core::header::Header::new();
        *hdr.compression_mut().unwrap() = Compression::Zips;
        hdr.set_tile_description(TileDescription { x_size: 8, y_size: 4, ..Default::default() }).unwrap();
        let path = dir.join(&format!("{name}.exr"));
        save_deep_image(&path, Some(&hdr), &img, DataWindowSource::Image).unwrap();
        files.push((path, diverges));
    }

    let cpp_exe = cpp("exrcheck");
    for (f, flagged) in &files {
        let mut default_diverges = false;
        for mode in MODES {
            let args: Vec<&std::ffi::OsStr> = mode.iter().map(std::ffi::OsStr::new).chain(std::iter::once(f.as_os_str())).collect();
            let (theirs, mine) = (run(&cpp_exe, &args), run(ours(), std::iter::once("check".as_ref()).chain(args.iter().copied())));
            let (t_bad, m_bad) = (theirs.text().trim_end().ends_with("bad"), mine.text().trim_end().ends_with("bad"));
            // the only allowed difference: exrcheck bad, we OK, on a flagged file, without `-c`
            let known = *flagged && !mode.contains(&"-c") && t_bad && !m_bad;
            assert!(
                known || (theirs.code == mine.code && theirs.text() == mine.text()),
                "{} {mode:?}: cpp {:?} {:?} / ours {:?} {:?}",
                f.display(),
                theirs.code,
                theirs.text().trim(),
                mine.code,
                mine.text().trim()
            );
            default_diverges |= known && mode.is_empty();
        }
        // a flagged file really diverges in the default mode, an unflagged one never does
        assert_eq!(*flagged, default_diverges, "{}", f.display());
    }
}

/// Byte range of the value of attribute `name` in the first header of an EXR file.
fn attr_value(d: &[u8], name: &str) -> std::ops::Range<usize> {
    let le = |i: usize| i32::from_le_bytes(d[i..i + 4].try_into().unwrap()) as usize;
    let mut i = 8; // after magic and version word
    while d[i] != 0 {
        let n = d[i..].iter().position(|&b| b == 0).unwrap();
        let t = d[i + n + 1..].iter().position(|&b| b == 0).unwrap();
        let size_at = i + n + 1 + t + 1;
        let (start, len) = (size_at + 4, le(size_at));
        if &d[i..i + n] == name.as_bytes() {
            return start..start + len;
        }
        i = start + len;
    }
    panic!("no attribute {name}");
}

/// Damage only the `type` and `version` attributes of goldens: a `type` that is not UTF-8 or names no
/// known part type (OpenEXRCore takes any byte string, an unknown one is an unknown storage), and a deep
/// part with `version != 1` (`exr_decoding_initialize` refuses it: ignored by the core scanline check
/// `ImfCheckFile.cpp:1385-1386`, flagged by the tiled one). Every verdict must equal `exrcheck`'s.
#[test]
#[ignore = "needs the C++ tools (cargo xtask parity-tools)"]
fn type_and_version_damage_matches_exrcheck() {
    let parity = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity");
    let dir = tool_support::TempDir::new("check-type-version");
    let mut files = Vec::new();
    for name in [
        "golden", "golden_tiled_zip_one", "golden_deep_scanline_zips", "golden_deep_scanline_none", "golden_deep_tiled_zips_one",
        "golden_deep_tiled_none_mipmap", "golden_deep_composite_a", "golden_multipart",
    ] {
        let good = std::fs::read(parity.join(format!("{name}.exr"))).expect("golden");
        for (tag, attr, damage) in [("type-ff", "type", 0xffu8), ("type-x", "type", b'x'), ("version-2", "version", 2u8)] {
            let mut d = good.clone();
            // files without the attribute (single-part flat ones carry no `version`) are skipped
            let found = std::panic::catch_unwind(|| attr_value(&good, attr));
            let Ok(range) = found else { continue };
            d[range.start] = damage;
            let path = dir.join(&format!("{name}_{tag}.exr"));
            std::fs::write(&path, d).expect("write");
            files.push(path);
        }
    }
    assert!(files.len() >= 14, "{} damaged files", files.len());
    assert_same("type / version damage", compare(&files, MODES));
}

