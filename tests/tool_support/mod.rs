//! Shared helpers of the tool parity tests (`tests/parity_tool_*.rs`, one per `exr-rs` subcommand
//! that ports a C++ OpenEXR tool). Each test runs the C++ tool and the Rust subcommand on the same
//! inputs and compares what they leave behind: exit code, stdout, output files byte for byte.
//!
//! The C++ tools are built by `cargo xtask parity-tools`, which points `EXR_TOOLS_DIR` at them and
//! then runs every `parity_tool_*` test (they are `#[ignore]`d for a plain `cargo test`).
//!
//! Include with `mod tool_support;` (each test crate compiles its own copy, so unused helpers
//! are allowed here).
#![allow(dead_code)]

use std::ffi::OsStr;
use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Wall-clock limit of one tool run. The private set's `Damaged/memory_DOS_*` files make the C++
/// tools run for minutes, writing gigabytes; a debug build of ours needs up to a minute for the
/// largest honest inputs, so two minutes separates "slow" from "never ends".
pub const RUN_TIMEOUT: Duration = Duration::from_secs(120);

/// Bytes of each output stream kept per run (the rest is read and dropped, `Out::truncated`).
pub const OUTPUT_CAP: usize = 64 << 20;

/// The `exr-rs` binary under test.
pub fn ours() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_exr-rs"))
}

/// The C++ tool `name` (`exrstdattr`, ...) that `cargo xtask parity-tools` built.
pub fn cpp(name: &str) -> PathBuf {
    let dir = std::env::var_os("EXR_TOOLS_DIR")
        .expect("EXR_TOOLS_DIR is not set: run these tests through `cargo run -p xtask -- parity-tools`");
    let exe = Path::new(&dir).join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    assert!(exe.is_file(), "the C++ tool {} was not built", exe.display());
    exe
}

/// What a finished program left on its standard streams.
pub struct Out {
    /// `None` when the process was killed by a signal or by the timeout.
    pub code: Option<i32>,
    /// The first [`OUTPUT_CAP`] bytes of stdout.
    pub stdout: Vec<u8>,
    /// The first [`OUTPUT_CAP`] bytes of stderr.
    pub stderr: Vec<u8>,
    /// The run hit [`RUN_TIMEOUT`] and was killed.
    pub timed_out: bool,
    /// A stream had more than [`OUTPUT_CAP`] bytes.
    pub truncated: bool,
}

impl Out {
    /// The stdout as text with `\r\n` folded to `\n`: the MSVC C++ runtime writes `cout` in text
    /// mode (CRLF on Windows), Rust writes `\n`; nothing else is normalized.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).replace("\r\n", "\n")
    }

    pub fn ok(&self) -> bool {
        self.code == Some(0) && !self.timed_out
    }

    /// How the run ended, for comparing two runs: the exit code, or the timeout.
    pub fn end(&self) -> String {
        if self.timed_out { "timed out".into() } else { format!("{:?}", self.code) }
    }
}

/// Run `prog args...` to completion and capture it.
pub fn run<I, S>(prog: &Path, args: I) -> Out
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    finish(Command::new(prog).args(args), prog)
}

/// [`run`] with `dir` as the working directory: a tool that echoes its (relative) file names
/// prints the same text whichever scratch directory it ran in.
pub fn run_in<I, S>(dir: &Path, prog: &Path, args: I) -> Out
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    finish(Command::new(prog).current_dir(dir).args(args), prog)
}

/// Read `r` to its end, keeping the first [`OUTPUT_CAP`] bytes; whether more came.
fn drain(r: impl Read) -> (Vec<u8>, bool) {
    drain_cap(r, OUTPUT_CAP)
}

fn drain_cap(mut r: impl Read, cap: usize) -> (Vec<u8>, bool) {
    let (mut kept, mut chunk, mut truncated) = (Vec::new(), vec![0u8; 1 << 16], false);
    // A read error is the pipe closing under a killed child: what was read so far is the output.
    while let Ok(n) = r.read(&mut chunk) {
        if n == 0 {
            break;
        }
        let room = cap - kept.len();
        kept.extend_from_slice(&chunk[..n.min(room)]);
        truncated |= n > room;
    }
    (kept, truncated)
}

/// Run `cmd` with bounded output and a [`RUN_TIMEOUT`]: the streams are read on threads (a child
/// that writes gigabytes neither fills the pipe nor our memory), a child still running at the
/// limit is killed.
fn finish(cmd: &mut Command, prog: &Path) -> Out {
    finish_within(cmd, prog, RUN_TIMEOUT)
}

fn finish_within(cmd: &mut Command, prog: &Path, limit: Duration) -> Out {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", prog.display()));
    let (so, se) = (child.stdout.take().expect("piped stdout"), child.stderr.take().expect("piped stderr"));
    let (t_out, t_err) = (std::thread::spawn(move || drain(so)), std::thread::spawn(move || drain(se)));
    let start = Instant::now();
    let (mut status, mut timed_out) = (None, false);
    while status.is_none() && !timed_out {
        match child.try_wait().unwrap_or_else(|e| panic!("cannot wait for {}: {e}", prog.display())) {
            Some(s) => status = Some(s),
            None if start.elapsed() >= limit => {
                // The child may have exited between the check and the kill: then it is simply done.
                child.kill().ok();
                timed_out = true;
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    let status = match status {
        Some(s) => Some(s),
        None => child.wait().ok(),
    };
    let ((stdout, t1), (stderr, t2)) = (t_out.join().expect("stdout reader"), t_err.join().expect("stderr reader"));
    Out { code: status.and_then(|s| s.code()).filter(|_| !timed_out), stdout, stderr, timed_out, truncated: t1 || t2 }
}

/// Run `exr-rs <sub> args...`.
pub fn run_ours<I, S>(sub: &str, args: I) -> Out
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run(ours(), sub_args(sub, args))
}

/// [`run_ours`] in the working directory `dir`.
pub fn run_ours_in<I, S>(dir: &Path, sub: &str, args: I) -> Out
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run_in(dir, ours(), sub_args(sub, args))
}

fn sub_args<I, S>(sub: &str, args: I) -> Vec<std::ffi::OsString>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut argv: Vec<std::ffi::OsString> = vec![sub.into()];
    argv.extend(args.into_iter().map(|a| a.as_ref().to_os_string()));
    argv
}

/// A scratch directory unique to this process and call, removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join("exr-rs-parity").join(format!(
            "{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        TempDir(dir)
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Best effort: a leftover scratch directory must not fail a passing test.
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// Panic unless `a` and `b` hold the same bytes, naming the first differing offset.
pub fn assert_same_file(a: &Path, b: &Path) {
    let read = |p: &Path| std::fs::read(p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    let (x, y) = (read(a), read(b));
    if let Some(msg) = first_diff(&x, &y) {
        panic!("{} vs {}: {msg}", a.display(), b.display());
    }
}

/// The first difference of two byte strings, described; `None` when they are equal.
pub fn first_diff(x: &[u8], y: &[u8]) -> Option<String> {
    let common = x.len().min(y.len());
    match x[..common].iter().zip(&y[..common]).position(|(p, q)| p != q) {
        Some(at) => Some(format!(
            "first difference at byte {at} ({:#04x} vs {:#04x}), sizes {} / {}",
            x[at],
            y[at],
            x.len(),
            y.len()
        )),
        None if x.len() != y.len() => {
            Some(format!("equal for {common} bytes, then sizes differ: {} / {}", x.len(), y.len()))
        }
        None => None,
    }
}

/// Every `.exr` file under `dir`, recursively, sorted (the order is part of the failure report).
pub fn exr_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        let entries = std::fs::read_dir(&d).unwrap_or_else(|e| panic!("cannot list {}: {e}", d.display()));
        for entry in entries {
            let path = entry.unwrap_or_else(|e| panic!("cannot list {}: {e}", d.display())).path();
            if path.is_dir() {
                todo.push(path);
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr")) {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// The committed goldens, `crates/exr-core/tests/parity/golden*.exr` (always available).
pub fn goldens() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity");
    exr_files(&dir)
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("golden")))
        .collect()
}

/// The private image set (OpenEXR's `src/test` images and openexr-images); panics with where it
/// looked when `exr-rs-tests` is not checked out (`EXR_TEST_DATA`).
pub fn corpus() -> Vec<PathBuf> {
    let mut all = exr_files(&exr_test_data::openexr_test());
    all.extend(exr_files(&exr_test_data::openexr_images()));
    all
}

/// [`goldens`] plus [`corpus`]: every input a tool parity test runs over.
pub fn inputs() -> Vec<PathBuf> {
    let mut all = goldens();
    all.extend(corpus());
    all
}

/// Collects mismatches over many inputs and fails once at the end with all of them, so one bad
/// file does not hide the rest.
#[derive(Default)]
pub struct Report {
    compared: usize,
    failures: String,
}

impl Report {
    /// Count one comparison; `Err(why)` records a failure for `what`.
    pub fn check(&mut self, what: impl std::fmt::Display, result: Result<(), String>) {
        self.compared += 1;
        if let Err(why) = result {
            writeln!(self.failures, "  {what}: {why}").expect("writing to a String");
        }
    }

    /// Fail when anything mismatched; otherwise report how many comparisons passed.
    pub fn finish(self, label: &str) {
        assert!(self.failures.is_empty(), "{label}: mismatches over {} comparisons:\n{}", self.compared, self.failures);
        println!("{label}: {} comparisons, all identical", self.compared);
    }
}

/// `Ok` when both runs ended alike (exit code) and, when both succeeded, `same_output` holds.
pub fn same_run(cpp: &Out, ours: &Out, same_output: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    if cpp.end() != ours.end() {
        return Err(format!(
            "runs ended differently: C++ {}, ours {}\n    C++ stderr: {}\n    our stderr: {}",
            cpp.end(),
            ours.end(),
            String::from_utf8_lossy(&cpp.stderr).trim(),
            String::from_utf8_lossy(&ours.stderr).trim()
        ));
    }
    if cpp.ok() { same_output() } else { Ok(()) }
}

/// `--version` of the C++ tool `tool` and of our subcommand `sub`: same text, same exit code,
/// for each argument list in `arg_lists` (C++ acts on `--version` wherever it appears).
pub fn assert_same_version(tool: &str, sub: &str, arg_lists: &[&[&str]]) {
    let cpp_exe = cpp(tool);
    let mut report = Report::default();
    for args in arg_lists {
        let (c, o) = (run(&cpp_exe, *args), run_ours(sub, *args));
        assert!(c.text().starts_with(&format!("{tool} (OpenEXR) ")), "{}", c.text());
        report.check(
            args.join(" "),
            same_run(&c, &o, || first_diff(c.text().as_bytes(), o.text().as_bytes()).map_or(Ok(()), Err)),
        );
    }
    report.finish(&format!("{sub} --version"));
}

/// The first difference of two runs' stdout, `None` when equal. When either was cut at
/// [`OUTPUT_CAP`] only the common prefix can be compared (both timing out mid-stream ends there).
pub fn same_stdout(c: &Out, o: &Out) -> Option<String> {
    let (a, b) = (c.text().into_bytes(), o.text().into_bytes());
    if c.truncated || o.truncated || (c.timed_out && o.timed_out) {
        let n = a.len().min(b.len());
        return first_diff(&a[..n], &b[..n]);
    }
    first_diff(&a, &b)
}

/// `Ok` when the two files hold the same bytes.
pub fn same_bytes(a: &Path, b: &Path) -> Result<(), String> {
    let read = |p: &Path| std::fs::read(p).map_err(|e| format!("cannot read {}: {e}", p.display()));
    match first_diff(&read(a)?, &read(b)?) {
        Some(msg) => Err(msg),
        None => Ok(()),
    }
}

/// One C++ tool and its `exr-rs` subcommand, each run in its own scratch directory: a relative
/// output name, and the file names the tool echoes, are then the same on both sides.
pub struct Pair {
    cpp: PathBuf,
    sub: String,
    cdir: TempDir,
    odir: TempDir,
}

impl Pair {
    /// The C++ tool `tool` (`exrmaketiled`) against the subcommand `sub` (`tile`).
    pub fn new(tool: &str, sub: &str) -> Self {
        Pair {
            cpp: cpp(tool),
            sub: sub.into(),
            cdir: TempDir::new(&format!("{sub}-cpp")),
            odir: TempDir::new(&format!("{sub}-ours")),
        }
    }

    /// Where the C++ tool / our subcommand ran (for inputs a test creates next to the outputs).
    pub fn dirs(&self) -> (PathBuf, PathBuf) {
        (self.cdir.join(""), self.odir.join(""))
    }

    /// Run both with `args`. `Ok(true)` when both succeeded with equal stdout and identical bytes in
    /// every file of `outputs` (names relative to the working directory); `Ok(false)` when both
    /// failed alike (same exit code, same stdout; the messages differ by design). Anything else is
    /// the `Err` describing the first difference. The outputs are removed afterwards, so the next
    /// call starts clean, in particular after a failure.
    pub fn check<I, S>(&self, args: I, outputs: &[&str]) -> Result<bool, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.check_with(args, outputs, &same_bytes)
    }

    /// [`check`](Self::check) with `same` (C++ file, our file) deciding whether two output files
    /// agree: for a tool whose C++ output is deliberately not reproduced byte for byte.
    pub fn check_with<I, S>(
        &self,
        args: I,
        outputs: &[&str],
        same: &dyn Fn(&Path, &Path) -> Result<(), String>,
    ) -> Result<bool, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let args: Vec<std::ffi::OsString> = args.into_iter().map(|a| a.as_ref().to_os_string()).collect();
        let (cdir, odir) = self.dirs();
        seed_outputs(&[&cdir, &odir], outputs);
        let before = snapshot(&odir, outputs);
        let (c, o) = (run_in(&cdir, &self.cpp, &args), run_ours_in(&odir, &self.sub, &args));
        let verdict = self.compare(&c, &o, outputs, (&cdir, &odir), &before, same);
        for name in outputs {
            // The file may not exist (a failed run); a leftover is what matters.
            std::fs::remove_file(cdir.join(name)).ok();
            std::fs::remove_file(odir.join(name)).ok();
        }
        verdict
    }

    fn compare(
        &self,
        c: &Out,
        o: &Out,
        outputs: &[&str],
        (cdir, odir): (&Path, &Path),
        before: &[Option<Vec<u8>>],
        same: &dyn Fn(&Path, &Path) -> Result<(), String>,
    ) -> Result<bool, String> {
        if c.end() != o.end() {
            return Err(format!(
                "runs ended differently: C++ {}, ours {}\n    C++ stderr: {}\n    our stderr: {}",
                c.end(),
                o.end(),
                String::from_utf8_lossy(&c.stderr).trim(),
                String::from_utf8_lossy(&o.stderr).trim()
            ));
        }
        if let Some(m) = same_stdout(c, o) {
            return Err(format!("stdout: {m}\n--- C++\n{}\n--- ours\n{}", c.text(), o.text()));
        }
        if !c.ok() {
            // A failed run of ours leaves every output as it was (the seeded file, or none).
            same_failure_state(odir, outputs, before)?;
            return Ok(false);
        }
        for name in outputs {
            same(&cdir.join(name), &odir.join(name)).map_err(|m| format!("{name}: {m}"))?;
        }
        Ok(true)
    }
}

/// What every output name holds before a run (see [`seed_outputs`]).
pub const SEED: &[u8] = b"seed: not an image, written by the harness";

/// Put [`SEED`] under every output name of a run in each directory, so that what a failing tool
/// does to an output that already exists shows. A name whose directory is missing is skipped.
pub fn seed_outputs(dirs: &[&Path], outputs: &[&str]) {
    for d in dirs {
        for name in outputs {
            // An existing file is an input that is also named as the output (a test of that): not ours
            // to overwrite. A name in a directory that does not exist stays unseeded.
            if !d.join(name).exists() {
                std::fs::write(d.join(name), SEED).ok();
            }
        }
    }
}

/// What the output names hold in `dir` now (`None`: no file), to compare after a run.
pub fn snapshot(dir: &Path, outputs: &[&str]) -> Vec<Option<Vec<u8>>> {
    outputs.iter().map(|n| std::fs::read(dir.join(n)).ok()).collect()
}

/// After a failed run: every output of OUR tool is exactly as `before` (absent, or the old file byte
/// for byte): files are written through `exr_core::io::write_file`, which leaves the target alone
/// on any failure. The C++ tool writes in place and may have created or truncated it; that is the
/// deliberate difference and nothing is asserted about it.
pub fn same_failure_state(odir: &Path, outputs: &[&str], before: &[Option<Vec<u8>>]) -> Result<(), String> {
    for (name, was) in outputs.iter().zip(before) {
        let now = std::fs::read(odir.join(name)).ok();
        if &now != was {
            let show = |v: &Option<Vec<u8>>| v.as_ref().map_or("absent".to_string(), |b| format!("{} bytes", b.len()));
            let why = format!("after the failure {name} is {} for ours, was {}", show(&now), show(was));
            // A test that counts successes before it reports would hide this verdict.
            eprintln!("{why}");
            return Err(why);
        }
    }
    Ok(())
}

/// The harness itself: a child that never ends is killed and reported as a timeout, output beyond
/// the cap is dropped and flagged, and two timeouts agree. (Each parity test crate runs these.)
#[cfg(test)]
mod harness_tests {
    use super::*;

    #[test]
    fn output_is_capped_and_flagged() {
        let data = vec![7u8; 1000];
        let (kept, cut) = drain_cap(&data[..], 400);
        assert_eq!((kept.len(), cut), (400, true));
        let (kept, cut) = drain_cap(&data[..], 1000);
        assert_eq!((kept.len(), cut), (1000, false), "exactly the cap is not truncation");
    }

    #[cfg(windows)]
    #[test]
    fn a_child_that_never_ends_is_killed() {
        let sleeper = |secs: u32| {
            // A direct child: killing a `cmd` would leave its `ping` holding the pipes open.
            let mut c = Command::new("ping");
            c.args(["-n", &(secs + 1).to_string(), "127.0.0.1"]);
            c
        };
        let t = Instant::now();
        let slow = finish_within(&mut sleeper(30), Path::new("ping"), Duration::from_millis(400));
        assert!(slow.timed_out && slow.code.is_none() && !slow.ok(), "{:?}", slow.code);
        assert!(t.elapsed() < Duration::from_secs(10), "the kill did not take effect");
        let done = finish_within(&mut sleeper(0), Path::new("ping"), Duration::from_secs(30));
        assert!(done.ok() && !done.timed_out);
        // Two timeouts agree, a timeout against an exit does not.
        assert!(same_run(&slow, &slow, || Err("not compared".into())).is_ok());
        assert!(same_run(&slow, &done, || Ok(())).is_err());
    }
}

/// Generated inputs: images of a size and shape the committed files do not have.
pub mod synth {
    use std::path::Path;

    use exr_core::attr::{Compression, LineOrder};
    use exr_core::{ChannelData, Image};
    use imath_rs::{Box2i, V2i};

    /// Write a `w x h` image with float `R G B A` channels, data window origin `(x0, y0)`, in the
    /// given line order and codec; every sample is a distinct-looking function of its position, so a
    /// misplaced row or column shows. Returns the image as written.
    pub fn pattern(
        path: &Path,
        (w, h): (i32, i32),
        (x0, y0): (i32, i32),
        order: LineOrder,
        comp: Compression,
    ) -> Image {
        let window = Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x0 + w - 1, y: y0 + h - 1 } };
        let mut img = Image::new(window);
        for (c, name) in ["R", "G", "B", "A"].into_iter().enumerate() {
            let data: Vec<f32> = (0..h)
                .flat_map(|y| (0..w).map(move |x| ((x * 7 + y * 13 + c as i32 * 101) % 1000) as f32 / 10.0))
                .collect();
            img.insert_channel(name, ChannelData::Float(data)).expect("insert a channel");
        }
        img.line_order = order;
        img.write(path, comp).expect("write the generated image");
        img
    }

    /// Overwrite the `pixelAspectRatio` value in the header of `path` (the API refuses to write the
    /// values a test wants the readers to meet).
    pub fn set_aspect(path: &Path, value: f32) {
        let mut bytes = std::fs::read(path).expect("read the image");
        let key = b"pixelAspectRatio\0float\0";
        let at = bytes.windows(key.len()).position(|w| w == key).expect("a pixelAspectRatio attribute") + key.len();
        // `at` is the attribute's size field (4), then the float.
        bytes[at + 4..at + 8].copy_from_slice(&value.to_le_bytes());
        std::fs::write(path, bytes).expect("write the image");
    }
}
