//! `exr-rs aces` against OpenEXR's `exr2aces` (v3.4.15): the exit code and the output file, byte
//! for byte, over the goldens and the private image set (inputs the C++ tool rejects must be
//! rejected alike), plus the verbose text. `#[ignore]`d: run by `cargo run -p xtask -- parity-tools`.

mod tool_support;

use tool_support::*;

/// Every input through both tools into separate scratch directories, the same relative output
/// name in each; both must end alike and, when they succeed, write the same bytes.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn output_files_are_identical_over_every_input() {
    let cpp_exe = cpp("exr2aces");
    let (cdir, odir) = (TempDir::new("aces-cpp"), TempDir::new("aces-ours"));
    let (mut report, mut converted) = (Report::default(), 0);
    for file in inputs() {
        let c = run_in(&cdir.join(""), &cpp_exe, [file.as_os_str(), "out.aces".as_ref()]);
        let o = run_ours_in(&odir.join(""), "aces", [file.as_os_str(), "out.aces".as_ref()]);
        let (cout, oout) = (cdir.join("out.aces"), odir.join("out.aces"));
        converted += usize::from(c.ok());
        let verdict = same_run(&c, &o, || same_bytes(&cout, &oout));
        report.check(file.display(), verdict);
        // The next input must not see this one's output, in particular after a failure.
        std::fs::remove_file(&cout).ok();
        std::fs::remove_file(&oout).ok();
    }
    // Both tools refusing every file would pass the loop above; most inputs convert.
    assert!(converted >= 50, "only {converted} inputs converted: the comparison proves little");
    report.finish(&format!("aces over goldens + image set ({converted} converted)"));
}

/// `-v`: `Reading file ..` / `Writing file ..` on stdout, file names echoed as typed (so both run
/// in their own directory with the same relative names).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn verbose_text_is_identical() {
    let cpp_exe = cpp("exr2aces");
    let (cdir, odir) = (TempDir::new("aces-v-cpp"), TempDir::new("aces-v-ours"));
    let mut report = Report::default();
    for file in goldens().into_iter().filter(|p| p.to_string_lossy().contains("golden_aces")) {
        for flag_first in [true, false] {
            let (mut c_args, mut o_args) = (vec![], vec![]);
            for args in [&mut c_args, &mut o_args] {
                // The C++ flag is accepted anywhere; so is our global one.
                let names = [file.as_os_str().to_owned(), "out.aces".into()];
                if flag_first {
                    args.push("-v".into());
                    args.extend(names);
                } else {
                    args.extend(names);
                    args.push("-v".into());
                }
            }
            let c = run_in(&cdir.join(""), &cpp_exe, &c_args);
            let o = run_ours_in(&odir.join(""), "aces", &o_args);
            report.check(
                format!("{} (-v {})", file.display(), if flag_first { "first" } else { "last" }),
                same_run(&c, &o, || match first_diff(c.text().as_bytes(), o.text().as_bytes()) {
                    Some(m) => Err(format!("stdout: {m}\n--- C++\n{}\n--- ours\n{}", c.text(), o.text())),
                    None => Ok(()),
                }),
            );
            assert!(c.text().starts_with("Reading file "), "{}", c.text());
        }
    }
    // Without -v the tools print nothing.
    let file = goldens().into_iter().next().expect("goldens exist");
    let (c, o) = (
        run_in(&cdir.join(""), &cpp_exe, [file.as_os_str(), "q.aces".as_ref()]),
        run_ours_in(&odir.join(""), "aces", [file.as_os_str(), "q.aces".as_ref()]),
    );
    assert!(c.stdout.is_empty() && o.stdout.is_empty(), "{:?} / {:?}", c.text(), o.text());
    report.finish("aces -v");
}

/// A missing input and an unwritable output end with exit code 1 in both, and the missing input
/// leaves no output file behind.
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn failures_are_identical() {
    let cpp_exe = cpp("exr2aces");
    let dir = TempDir::new("aces-fail");
    let good = goldens().into_iter().find(|p| p.ends_with("golden_aces.exr")).expect("golden_aces.exr");
    let missing = dir.join("missing.exr");
    let no_dir = dir.join("no/such/dir/out.aces");
    let mut report = Report::default();
    for (what, args) in [
        ("missing input", vec![missing.as_os_str().to_owned(), dir.join("m.aces").into_os_string()]),
        ("unwritable output", vec![good.as_os_str().to_owned(), no_dir.into_os_string()]),
    ] {
        let (c, o) = (run(&cpp_exe, &args), run_ours("aces", &args));
        assert_eq!(c.code, Some(1), "{what}: the C++ tool's own code");
        report.check(what, same_run(&c, &o, || Ok(())));
    }
    assert!(!dir.join("m.aces").exists());
    report.finish("aces failures");
}

/// `--version`: same text, exit code 0 (alone or after file names).
#[test]
#[ignore = "needs the C++ tools: cargo run -p xtask -- parity-tools"]
fn version_text_is_identical() {
    assert_same_version("exr2aces", "aces", &[&["--version"], &["in.exr", "out.aces", "--version"]]);
}
