//! xtask — developer automation for exr-rs.
//!
//! Hosts the three C/C++ **parity oracles** (Imath, libdeflate, OpenEXR). Each
//! one follows the same shape, so the shape lives in one place:
//!
//! 1. assemble the MSVC environment as a value ([`env_setup::Toolchain`]) —
//!    never by mutating this process's env;
//! 2. resolve the vcpkg tree that actually has the headers ([`vcpkg::Vcpkg`]);
//! 3. build the oracle with `cl` ([`build_oracle`]);
//! 4. run it to (re)generate the committed goldens ([`Oracle::run`], driven by
//!    a table of jobs for the OpenEXR oracle);
//! 5. run the Rust parity test against the fresh goldens ([`cargo_test`]).

mod env_setup;
mod vcpkg;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use env_setup::Toolchain;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
#[command(name = "xtask", about = "exr-rs developer automation (native MSVC parity infra)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Apply the native MSVC toolchain and prove `cl`/`cmake` are reachable.
    Setup,
    /// Build & run the C++ Imath golden-parity oracle, regenerate `golden.txt`,
    /// and run the `parity_golden` Rust test against it (Windows/vcpkg only).
    Parity,
    /// Build & run the C libdeflate golden-parity oracle, regenerate deflate-rs's
    /// `golden.txt`, and run the `parity_libdeflate` Rust test against it
    /// (Windows/vcpkg only). Proves deflate-rs's zlib compressor is byte-identical
    /// to the real C libdeflate.
    ParityDeflate,
    /// Build & run the C++ OpenEXR oracle for the read/write parity gate,
    /// regenerate exr-core's committed `golden*.exr`, and run the `parity_exr`
    /// Rust test against them in BOTH directions (Windows/vcpkg only). Proves
    /// exr-core's reader/writer interoperate pixel/byte-exactly with real C++
    /// OpenEXR.
    ParityExr,
    /// Build OpenEXR's own `exrheader` from the reference source and run the
    /// ignored exr-core test that compares its output attribute by attribute
    /// with `exr_core::api::describe` (the formatter behind `exr-rs info`)
    /// over every committed golden and corpus file (Windows/vcpkg only).
    ParityHeader,
    /// Build OpenEXR's command-line tools (exrcheck, exrmaketiled, exrmakepreview,
    /// exrstdattr, exrmultipart, exrmultiview, exrenvmap, exrmetrics, exrmanifest,
    /// exr2aces) from the unmodified reference sources into `target/parity-tools/`, then
    /// run every ignored `tests/parity_tool_*.rs` of the root crate, which compare each
    /// `exr-rs` subcommand with its C++ tool (Windows/vcpkg only).
    ParityTools {
        /// Build the C++ tools without running the Rust comparison tests.
        #[arg(short = 'b', long)]
        build_only: bool,
    },
}

/// The libdeflate release the byte-parity gate is actually verified against.
///
/// deflate-rs ports the **v1.26** sources (bumped from v1.25 — v1.26's only
/// behavioural delta over v1.25 is `compression_level == -1` aliasing to the
/// default level 6; everything else is ARM/build-guard/cosmetic). Pinning the
/// verified version here means a silent vcpkg bump can no longer change the
/// meaning of "byte-identical to C libdeflate" without anyone noticing (the
/// golden header used to drift unchecked).
const VERIFIED_LIBDEFLATE: &str = "1.26";

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Setup => setup(),
        Cmd::Parity => parity(),
        Cmd::ParityDeflate => parity_deflate(),
        Cmd::ParityExr => parity_exr(),
        Cmd::ParityHeader => parity_header(),
        Cmd::ParityTools { build_only } => parity_tools(build_only),
    }
}

// ---------------------------------------------------------------------------
// Tool probing
// ---------------------------------------------------------------------------

/// A required build tool and how to prove it is present and working.
///
/// Different tools have different "healthy" signatures: `cl` prints its banner
/// to stderr and exits non-zero when given no inputs, whereas `cmake --version`
/// prints to stdout and exits zero. The probe captures that per-tool so
/// [`probe_tool`] can fail closed without special-casing at the call site.
struct ToolProbe {
    /// Program name resolved on `PATH`.
    program: &'static str,
    /// Arguments that elicit a version banner (empty for `cl`).
    args: &'static [&'static str],
    /// `cl` writes its banner to stderr; most tools use stdout.
    banner_on_stderr: bool,
    /// Whether a zero exit status is part of a healthy probe. `cl` legitimately
    /// exits non-zero with no inputs, so it opts out and is judged by its
    /// banner alone.
    require_zero_exit: bool,
}

/// The `cl` probe — the only tool every parity command genuinely runs.
const CL_PROBE: ToolProbe = ToolProbe {
    program: "cl",
    args: &[],
    banner_on_stderr: true,
    require_zero_exit: false,
};

/// The `cmake` probe. `setup` reports it because a developer environment is
/// expected to have it; no parity command invokes it.
const CMAKE_PROBE: ToolProbe = ToolProbe {
    program: "cmake",
    args: &["--version"],
    banner_on_stderr: false,
    require_zero_exit: true,
};

/// Spawn a required build tool and return the first line of its version banner.
///
/// Fails closed: a spawn failure (tool missing), an empty banner, or — when
/// `require_zero_exit` is set — a non-zero exit all return `Err`. This is the
/// single source of truth for "is this tool actually usable", so `setup` cannot
/// report success against an absent or broken toolchain.
fn probe_tool(probe: &ToolProbe, tc: &Toolchain) -> Result<String> {
    let mut cmd = Command::new(probe.program);
    tc.apply(&mut cmd);
    let out = cmd
        .args(probe.args)
        .output()
        .with_context(|| format!("`{}` not found on PATH", probe.program))?;
    let stream = if probe.banner_on_stderr { &out.stderr } else { &out.stdout };
    let banner = String::from_utf8_lossy(stream);
    let line = banner.lines().next().unwrap_or("").trim().to_string();
    if line.is_empty() {
        anyhow::bail!("`{}` ran but produced no version banner", probe.program);
    }
    if probe.require_zero_exit && !out.status.success() {
        anyhow::bail!("`{}` exited with failure status ({})", probe.program, out.status);
    }
    Ok(line)
}

/// Apply the MSVC toolchain env, then prove `cl` and `cmake` are wired (no
/// manual vcvars step required).
///
/// Fail-closed: BOTH tools must spawn and satisfy their [`ToolProbe`] or this
/// returns `Err`. A missing, non-zero-exit (for `cmake`), or empty-banner probe
/// is a failure — `setup` must never report success against a broken toolchain.
fn setup() -> Result<()> {
    if !cfg!(windows) {
        println!("native MSVC setup is Windows-only");
        return Ok(());
    }
    let tc = env_setup::prepare_native_toolchain()?;
    for probe in [&CL_PROBE, &CMAKE_PROBE] {
        println!("{}: {}", probe.program, probe_tool(probe, &tc)?);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The shared oracle machinery
// ---------------------------------------------------------------------------

/// How to build one oracle: everything that differs between the three.
struct OracleSpec<'a> {
    /// Crate whose `tests/parity` dir holds the source and the goldens.
    krate: &'a str,
    /// Source file name (`oracle.c` / `oracle.cpp`).
    source: &'a str,
    /// A header path relative to `include/` that identifies the vcpkg tree.
    vcpkg_probe: &'a str,
}

/// A built oracle plus the environment every run of it needs.
struct Oracle {
    exe: PathBuf,
    /// Working directory for each run (the parity dir).
    dir: PathBuf,
    /// `PATH` for each run (vcpkg `bin/` first on a dynamic triplet).
    path: String,
}

impl Oracle {
    /// Run the oracle once and return its stdout.
    ///
    /// The oracles report progress on stderr and signal failure with a non-zero
    /// exit; both are handled here so no caller re-implements it (the three
    /// commands used to carry nine near-identical copies of this block).
    fn run<I, S>(&self, args: I) -> Result<Vec<u8>>
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
        let out = Command::new(&self.exe)
            .current_dir(&self.dir)
            .env("PATH", &self.path)
            .args(&args)
            .output()
            .with_context(|| format!("failed to run {}", self.exe.display()))?;
        let note = String::from_utf8_lossy(&out.stderr);
        if !note.trim().is_empty() {
            println!("{}", note.trim());
        }
        if !out.status.success() {
            anyhow::bail!(
                "oracle {:?} exited non-zero ({})",
                args.iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>(),
                out.status
            );
        }
        Ok(out.stdout)
    }

    /// Run the oracle to (re)write `file`, then report the resulting size.
    fn write_golden<I, S>(&self, label: &str, file: &Path, args: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        println!("\n--- running oracle {label} ---");
        let mut argv: Vec<OsString> = Vec::new();
        for a in args {
            argv.push(a.into());
        }
        // The output path always follows the mode argument.
        argv.insert(1, file.as_os_str().to_os_string());
        self.run(argv)?;
        let len = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
        println!("wrote {} ({len} bytes)", file.display());
        Ok(())
    }
}

/// Compile one oracle with `cl` and return it ready to run.
///
/// `extra_cl` are flags inserted before the source file (include dirs, C++
/// standard); `libs` are the archives/import libs to link.
fn build_oracle(
    spec: &OracleSpec<'_>,
    tc: &Toolchain,
    vcpkg: &vcpkg::Vcpkg,
    extra_cl: &[String],
    libs: &[String],
) -> Result<Oracle> {
    let dir = repo_root().join("crates").join(spec.krate).join("tests").join("parity");
    let source = dir.join(spec.source);
    anyhow::ensure!(source.exists(), "oracle source not found at {}", source.display());
    let exe = dir.join("oracle.exe");

    println!("\n--- compiling {} with cl ---", spec.source);
    let status = vcpkg
        .cl(&dir, tc)
        .args(extra_cl)
        .arg(spec.source)
        .arg(format!("/Fe:{}", exe.display()))
        .arg("/link")
        .arg(vcpkg.libpath_arg())
        .args(libs)
        .status()
        .context("failed to launch `cl` (MSVC compiler)")?;
    if !status.success() {
        anyhow::bail!("cl failed to compile {} (exit {status})", spec.source);
    }
    Ok(Oracle { exe, dir, path: vcpkg.run_path(tc) })
}

/// Common preamble for every parity command: toolchain + `cl` probe + vcpkg.
///
/// Requires only what the command actually runs — `cl`. (It used to demand
/// `git` and `cmake`, neither of which any oracle invokes.)
fn prepare(spec: &OracleSpec<'_>) -> Result<(Toolchain, vcpkg::Vcpkg)> {
    let tc = env_setup::prepare_native_toolchain()?;
    probe_tool(&CL_PROBE, &tc).context("the C/C++ parity oracles need MSVC `cl`")?;
    let vcpkg = vcpkg::Vcpkg::resolve(spec.vcpkg_probe)?;
    Ok((tc, vcpkg))
}

/// Run the oracle once with no arguments and make its stdout the golden text.
fn stdout_golden(oracle: &Oracle, golden_txt: &Path) -> Result<Vec<u8>> {
    println!("\n--- running oracle.exe -> golden.txt ---");
    let stdout = oracle.run(Vec::<OsString>::new())?;
    // Normalize CRLF -> LF before writing: the oracle's stdout is a Windows
    // console handle opened in TEXT mode by default, so every "\n" it prints
    // becomes "\r\n" on the wire. Left as-is, every `cargo xtask parity*` run
    // rewrites the committed golden with 1 extra \r per line, leaving the
    // tree dirty after a gate that changed nothing semantically (caught by
    // team-lead's review: `git status --short` was non-empty right after a
    // clean `parity-deflate` run). Single point for every oracle that writes
    // a golden this way (Imath's `parity` golden shares this function).
    let stdout = normalize_crlf(&stdout);
    std::fs::write(golden_txt, &stdout)
        .with_context(|| format!("failed to write {}", golden_txt.display()))?;
    let lines = stdout.iter().filter(|&&b| b == b'\n').count();
    println!("wrote {} ({lines} lines)", golden_txt.display());
    Ok(stdout)
}

/// Replace every `\r\n` with `\n`; leaves lone `\r` (none expected from a
/// C `printf`-based oracle) untouched rather than guessing at intent.
fn normalize_crlf(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
            i += 1; // skip the \r, the \n is copied on the next iteration
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

/// Run one Rust parity test against the freshly regenerated goldens.
///
/// `extra_env` carries whatever the test needs to reach the oracle itself
/// (`EXR_ORACLE_EXE`, `PATH`). `extra_test_args` are extra test-binary
/// arguments after `-- --nocapture` (e.g. `&["--ignored"]` for a test gated
/// `#[ignore]` because it spawns a live subprocess per case rather than
/// reading a committed golden); pass `&[]` when there are none.
fn cargo_test(
    krate: &str,
    test: &str,
    extra_env: &[(&str, OsString)],
    extra_test_args: &[&str],
) -> Result<()> {
    println!("\n--- cargo test -p {krate} --test {test} {} ---", extra_test_args.join(" "));
    let mut cmd = Command::new("cargo");
    cmd.current_dir(repo_root());
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let status = cmd
        .args(["test", "-p", krate, "--test", test, "--", "--nocapture"])
        .args(extra_test_args)
        .status()
        .context("failed to launch `cargo test`")?;
    if !status.success() {
        anyhow::bail!("{test} test FAILED (exit {status})");
    }
    Ok(())
}

/// Shared "this oracle is Windows/vcpkg-only" early-out.
fn windows_only(cmd: &str, krate: &str) -> bool {
    if cfg!(windows) {
        return false;
    }
    println!("cargo xtask {cmd}: the oracle is Windows/vcpkg-only; skipping.");
    println!("(The committed goldens still let `cargo test -p {krate}` run standalone.)");
    true
}

// ---------------------------------------------------------------------------
// The three parity commands
// ---------------------------------------------------------------------------

/// Build the C++ Imath oracle, regenerate `golden.txt`, and run the
/// `parity_golden` Rust test (bit-exact / tight-ULP agreement with C++ Imath).
fn parity() -> Result<()> {
    if windows_only("parity", "imath-rs") {
        return Ok(());
    }
    let spec = OracleSpec { krate: "imath-rs", source: "oracle.cpp", vcpkg_probe: "Imath/half.h" };
    let (tc, vcpkg) = prepare(&spec)?;

    // `/DIMATH_DLL` (dynamic triplets only): a shared Imath exports
    // `imath_half_to_float_table` etc., which the consumer must see as
    // `__declspec(dllimport)` or linking fails with LNK2019 (ImathExport.h).
    let mut cflags = vec!["/EHsc".into(), "/std:c++17".into()];
    if !vcpkg.is_static {
        cflags.push("/DIMATH_DLL".into());
    }
    let oracle = build_oracle(&spec, &tc, &vcpkg, &cflags, &[vcpkg.lib("Imath-")?])?;

    stdout_golden(&oracle, &oracle.dir.join("golden.txt"))?;
    cargo_test("imath-rs", "parity_golden", &[], &[])?;
    println!("\nparity: OK");
    Ok(())
}

/// Build the C libdeflate oracle, regenerate deflate-rs's `golden.txt`, assert
/// the libdeflate release it was produced with, and run `parity_libdeflate`
/// (BYTE-IDENTICAL output vs the real C libdeflate).
fn parity_deflate() -> Result<()> {
    if windows_only("parity-deflate", "deflate-rs") {
        return Ok(());
    }
    let spec = OracleSpec { krate: "deflate-rs", source: "oracle.c", vcpkg_probe: "libdeflate.h" };
    let (tc, vcpkg) = prepare(&spec)?;
    let oracle = build_oracle(&spec, &tc, &vcpkg, &[], &[vcpkg.deflate_lib()?])?;

    let stdout = stdout_golden(&oracle, &oracle.dir.join("golden.txt"))?;
    assert_libdeflate_version(&stdout)?;
    cargo_test("deflate-rs", "parity_libdeflate", &[], &[])?;

    // V1/V2 (plan4): live-oracle differential tests. These are `#[ignore]`d
    // in the crate (they spawn a subprocess per case, not a committed
    // golden), so `cargo test` alone -- what `bootstrap.py t` runs -- never
    // exercises them; wire them into the gate explicitly here, exactly like
    // parity-exr already does for its own oracle, so V1/V2 actually catch a
    // regression instead of only running when someone remembers `--ignored`.
    let oracle_env: [(&str, OsString); 2] = [
        ("PATH", OsString::from(&oracle.path)),
        ("EXR_DEFLATE_ORACLE_EXE", oracle.exe.clone().into_os_string()),
    ];
    cargo_test("deflate-rs", "parity_libdeflate_large", &oracle_env, &["--ignored"])?;
    cargo_test("deflate-rs", "parity_libdeflate_malformed", &oracle_env, &["--ignored"])?;

    println!("\nparity-deflate: OK (vs libdeflate {VERIFIED_LIBDEFLATE})");
    Ok(())
}

/// Fail when the oracle was linked against a libdeflate other than the one this
/// gate claims to verify against.
///
/// The oracle's first golden line is
/// `# ... (libdeflate <version>)` (it prints `LIBDEFLATE_VERSION_STRING`).
/// Without this check a vcpkg upgrade silently redefined what the byte-parity
/// claim means.
fn assert_libdeflate_version(golden: &[u8]) -> Result<()> {
    let head = String::from_utf8_lossy(&golden[..golden.len().min(512)]);
    let line = head.lines().next().unwrap_or_default();
    let found = line
        .rsplit_once("libdeflate ")
        .and_then(|(_, rest)| rest.split(')').next())
        .map(str::trim)
        .with_context(|| format!("golden header does not name a libdeflate version: {line:?}"))?;
    anyhow::ensure!(
        found == VERIFIED_LIBDEFLATE,
        "libdeflate version drift: the oracle links libdeflate {found}, but this gate \
         is pinned to {VERIFIED_LIBDEFLATE}. Either install the pinned vcpkg port or \
         update VERIFIED_LIBDEFLATE (and deflate-rs's docs) after re-verifying."
    );
    println!("oracle links libdeflate {found} (pinned)");
    Ok(())
}

/// One golden EXR the C++ oracle writes: the file name and the oracle argv
/// *without* the output path (which [`Oracle::write_golden`] splices in).
struct ExrGolden {
    file: String,
    args: Vec<String>,
}

fn exr_golden(file: impl Into<String>, args: &[&str]) -> ExrGolden {
    ExrGolden { file: file.into(), args: args.iter().map(|s| (*s).to_string()).collect() }
}

/// Every committed `golden*.exr`, as a table.
///
/// This replaced six hand-copied `Command::new(&oracle_exe)…output()` blocks;
/// adding a codec / level mode / part shape is now one row.
fn exr_goldens() -> Vec<ExrGolden> {
    let mut jobs = vec![exr_golden("golden.exr", &["write"])];

    // W2.7 "header wire bytes": the KNOWN image plus chromaticities/
    // rational/keyCode attributes, NO_COMPRESSION so the whole file
    // (including the header) is deterministic and byte-exact-comparable.
    jobs.push(exr_golden("golden_header_attrs.exr", &["write-header-attrs"]));

    // C-08: byte-exact Name — a non-UTF-8 channel name, a non-UTF-8 attribute
    // name, and a 255-byte (Name::MAX_LENGTH) attribute name.
    jobs.push(exr_golden("golden_nonutf8_names.exr", &["write-nonutf8-names"]));

    // Scanline: the KNOWN image once per codec.
    for codec in [
        "rle", "zip", "zips", "pxr24", "piz", "b44", "b44a", "dwaa", "dwab", "htj2k256", "htj2k32",
    ] {
        jobs.push(exr_golden(format!("golden_{codec}.exr"), &["write", codec]));
    }

    // K-06: ZIP/ZIPS at a non-default `Header::zipCompressionLevel()` (the
    // EXR_DEFAULT_ZLIB_COMPRESS_LEVEL both our writer and C++ otherwise use is
    // 4). Level 1 (fastest) and 9 (best) bracket the range and both differ
    // from 4, so a byte-exact match at either level is real level-plumbing
    // evidence, not an accidental match at the shared default.
    for codec in ["zip", "zips"] {
        for level in [1, 9] {
            jobs.push(exr_golden(
                format!("golden_{codec}_level{level}.exr"),
                &["write-zip-level", codec, &level.to_string()],
            ));
        }
    }

    // Tiled: KNOWN tiled image (20x15, tile 8x8) per codec x level mode.
    // ONE_LEVEL for every codec; MIPMAP for the cheap lossless ones; one
    // OFF-ORIGIN ONE_LEVEL/NONE variant (min = 11,7) for the tile-rect seam.
    for codec in
        ["none", "rle", "zip", "zips", "piz", "b44", "b44a", "pxr24", "dwaa", "dwab"]
    {
        jobs.push(exr_golden(
            format!("golden_tiled_{codec}_one.exr"),
            &["write-tiled", codec, "one"],
        ));
    }
    for codec in ["none", "zip", "piz"] {
        jobs.push(exr_golden(
            format!("golden_tiled_{codec}_mipmap.exr"),
            &["write-tiled", codec, "mipmap"],
        ));
    }
    jobs.push(exr_golden(
        "golden_tiled_none_one_offset.exr",
        &["write-tiled", "none", "one", "11", "7"],
    ));
    // Header-variant tiled goldens (r3_tests.md §8: these paths had no
    // C++-verified test at all -- ROUND_UP changed the level count, RANDOM_Y
    // and DECREASING_Y changed the on-disk tile order, and only the attribute
    // BYTE was ever checked).
    jobs.push(exr_golden(
        "golden_tiled_zip_mipmap_roundup.exr",
        &["write-tiled", "zip", "mipmap", "round=up"],
    ));
    jobs.push(exr_golden(
        "golden_tiled_zip_one_randomy.exr",
        &["write-tiled", "zip", "one", "order=random"],
    ));
    jobs.push(exr_golden(
        "golden_tiled_zip_one_decreasing.exr",
        &["write-tiled", "zip", "one", "order=decreasing"],
    ));
    // W2.7: RIPMAP -- independent x/y level grids (numXLevels != numYLevels
    // for the non-square 68x44 KNOWN image), never exercised before this.
    jobs.push(exr_golden(
        "golden_tiled_zip_ripmap.exr",
        &["write-tiled", "zip", "ripmap"],
    ));

    // Deep scanline: the valid DEEP codecs only (deep is never ZIP/PIZ/B44/
    // PXR24/DWA), plus one off-origin (min = -3,7) data-window variant.
    for codec in ["none", "rle", "zips"] {
        jobs.push(exr_golden(
            format!("golden_deep_scanline_{codec}.exr"),
            &["write-deep-scanline", codec],
        ));
    }
    jobs.push(exr_golden(
        "golden_deep_scanline_none_offset.exr",
        &["write-deep-scanline", "none", "-3", "7"],
    ));

    // D-12: two deep scanline "sources" for the CompositeDeepScanLine oracle
    // gate (see the `composite-deep-scanline` special-case dump job below,
    // after `DECODE_DUMPS`).
    jobs.push(exr_golden("golden_deep_composite_a.exr", &["write-deep-composite", "a"]));
    jobs.push(exr_golden("golden_deep_composite_b.exr", &["write-deep-composite", "b"]));

    // Deep tiled: ONE_LEVEL + MIPMAP (all levels). For MIPMAP each level is
    // addressed in our whole-(level-0)-window flat-heap order with counts only
    // in the level's corner (see parity_exr.rs `deep_count_buffer_mip` and the
    // "MIPMAP gotcha" in src/deep/CLAUDE.md).
    for (codec, mode) in
        [("none", "one"), ("rle", "one"), ("zips", "one"), ("none", "mipmap"), ("zips", "mipmap")]
    {
        jobs.push(exr_golden(
            format!("golden_deep_tiled_{codec}_{mode}.exr"),
            &["write-deep-tiled", codec, mode],
        ));
    }

    // Luminance-chroma pair: the subsampled Y/RY/BY fixture and its
    // full-resolution RGBA twin (the reference for chroma reconstruction).
    jobs.push(exr_golden("golden_yc.exr", &["write-yc"]));
    jobs.push(exr_golden("golden_rgba.exr", &["write-rgba"]));
    // Non-default setYCRounding (default is roundY=7, roundC=5), for a
    // bit-exact check of `RgbaOutputFile::set_yc_rounding`.
    jobs.push(exr_golden("golden_yc_rounded.exr", &["write-yc-rounded", "3", "2"]));

    // Tiled RGBA pair (plain + luminance-only), 20x13 with 8x8 tiles (a
    // deliberate partial edge tile) -- for exr_core::rgba::{TiledRgbaOutputFile,
    // TiledRgbaInputFile}. Neither shape filters/subsamples, so the file
    // itself is the bit-exact reference (no separate decode dump needed).
    jobs.push(exr_golden("golden_tiled_rgba.exr", &["write-tiled-rgba", "rgba"]));
    jobs.push(exr_golden("golden_tiled_rgba_grey.exr", &["write-tiled-rgba", "grey"]));

    // ACES fixtures: `golden_aces.exr` (AcesOutputFile, ACES chromaticities
    // already stamped -- byte-exact check of our AcesOutputFile) and
    // `golden_aces_rec709.exr` (a plain Rec.709-chromaticities RGBA file --
    // the read-side fixture for AcesInputFile's Bradford color conversion,
    // decode dump wired in DECODE_DUMPS below).
    jobs.push(exr_golden("golden_aces.exr", &["write-aces", "rgba"]));
    jobs.push(exr_golden("golden_aces_rec709.exr", &["write-aces", "rec709"]));

    // Multipart: ONE file with THREE mixed-type parts (scanline/NONE,
    // tiled/ZIP, deep-scanline/ZIPS), all lossless -> bit-exact both directions.
    jobs.push(exr_golden("golden_multipart.exr", &["write-multipart"]));

    // W-06 (`.bughunt/r4_audit_dwa_htj2k.md`): DWA with sub-sampled channels —
    // a CSC group (R/G/B, unsampled) alongside a standalone LOSSY_DCT channel
    // (BY, 2x2), an RLE channel (A, 2x1), and an UNKNOWN/zlib channel (Z,
    // 1x2). One golden per scan-line-batch size (DWAA=32, DWAB=256) so both
    // AC-entropy-coder choices (DEFLATE is scanline-DWAA-never; both DWAA/DWAB
    // scanline paths use static Huffman, but this at least covers the DWAB
    // 256-line batching against the 64-row fixture).
    jobs.push(exr_golden("golden_dwaa_sub.exr", &["write-dwa-sub", "dwaa"]));
    jobs.push(exr_golden("golden_dwab_sub.exr", &["write-dwa-sub", "dwab"]));

    // Full per-codec x per-sampling scanline matrix (team-lead follow-up to the
    // HTJ2K y-subsampling bug, `.bughunt/progress_htj2k_ysub.md`): the bug slipped
    // through because no test exercised channel sub-sampling for 11 of the 12
    // codecs. Two channels, "A" (unsampled reference) + "S" (the sampling under
    // test) -- `write-sub`/`dump-sub` in `oracle.cpp`. (1,1) is included as the
    // baseline/regression cell even though it duplicates the per-codec KNOWN-image
    // coverage above, so the matrix is uniform and complete.
    for codec in [
        "none", "rle", "zip", "zips", "piz", "b44", "b44a", "pxr24", "dwaa", "dwab",
        "htj2k256", "htj2k32",
    ] {
        for (xs, ys) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            jobs.push(exr_golden(
                format!("golden_sub_{codec}_x{xs}y{ys}.exr"),
                &["write-sub", codec, &xs.to_string(), &ys.to_string()],
            ));
        }
    }

    jobs
}

/// The goldens whose C++ DECODE is committed next to them as `<name>.dump`.
///
/// Only the lossy codecs need it: a lossless golden is checked against the
/// KNOWN formula, which is a stronger and completely independent oracle. For a
/// lossy one there is no such formula, so without a committed decode the
/// "our decode == C++ decode" tests can only run when the oracle binary is
/// present -- which is why 11 of them silently degraded to a pixel COUNT in a
/// plain `cargo test` (.bughunt/r3_tests.md 2.2).
///
/// `(golden file, oracle mode)`.
const DECODE_DUMPS: &[(&str, &str)] = &[
    ("golden_b44.exr", "dump"),
    ("golden_b44a.exr", "dump"),
    ("golden_dwaa.exr", "dump"),
    ("golden_dwab.exr", "dump"),
    ("golden_pxr24.exr", "dump"),
    ("golden_tiled_b44_one.exr", "dump-tiled"),
    ("golden_tiled_b44a_one.exr", "dump-tiled"),
    ("golden_tiled_dwaa_one.exr", "dump-tiled"),
    ("golden_tiled_dwab_one.exr", "dump-tiled"),
    ("golden_tiled_pxr24_one.exr", "dump-tiled"),
    // `rgba-read <golden> <out>` takes the SAME `mode golden out` shape as
    // `dump`, so it reuses this loop unchanged. golden_yc.exr's chroma
    // subsampling is lossy (no independent formula either), and even
    // golden_rgba.exr's lossless decode is worth pinning bit-exact: this is
    // `exr_core::rgba::RgbaInputFile`'s only committed-fixture bit-exact
    // check (`rgba_yc.rs` only asserts closeness for the lossy YC path).
    ("golden_yc.exr", "rgba-read"),
    ("golden_rgba.exr", "rgba-read"),
    // AcesInputFile's Bradford color conversion has no independent formula
    // (like the other lossy/transform paths above), so the C++ decode is
    // committed next to the golden.
    ("golden_aces_rec709.exr", "aces-read"),
    // W-06: BY is LOSSY_DCT (subsampled), so its committed C++ decode is the
    // only oracle-free exact check, same rationale as the other DWA goldens.
    ("golden_dwaa_sub.exr", "dump-dwa-sub"),
    ("golden_dwab_sub.exr", "dump-dwa-sub"),
    // W3.4 oracle gate: the real, GENERIC `Imf::InputFile` flattening a
    // single deep-scanline source through its internal `CompositeDeepScanLine`
    // -- reuses the existing `golden_deep_composite_a.exr` fixture (already
    // built above for the D-12 two-source compositing gate) as a single
    // source; `read-deep-via-inputfile` takes the same `mode golden out`
    // shape as `dump`/`rgba-read`, so it fits this table directly (unlike
    // `composite-deep-scanline`, which is 2-input and lives in its own block
    // below). Produces `golden_deep_composite_a.exr.dump`, distinct from the
    // 2-source `golden_deep_composite.dump` the special-case block below
    // writes.
    ("golden_deep_composite_a.exr", "read-deep-via-inputfile"),

    // Lossy cells of the per-codec x per-sampling matrix (`golden_sub_*`,
    // `exr_goldens()` above): PXR24/B44/B44A/DWAA/DWAB have no independent
    // formula (same rationale as the rest of this table), so their sampled
    // "S" channel needs a committed C++ decode. The lossless cells (none/
    // rle/zip/zips/piz/htj2k*) are checked directly against the KNOWN pixel
    // formulas instead -- no dump needed for those.
    ("golden_sub_pxr24_x1y1.exr", "dump-sub"),
    ("golden_sub_pxr24_x2y1.exr", "dump-sub"),
    ("golden_sub_pxr24_x1y2.exr", "dump-sub"),
    ("golden_sub_pxr24_x2y2.exr", "dump-sub"),
    ("golden_sub_b44_x1y1.exr", "dump-sub"),
    ("golden_sub_b44_x2y1.exr", "dump-sub"),
    ("golden_sub_b44_x1y2.exr", "dump-sub"),
    ("golden_sub_b44_x2y2.exr", "dump-sub"),
    ("golden_sub_b44a_x1y1.exr", "dump-sub"),
    ("golden_sub_b44a_x2y1.exr", "dump-sub"),
    ("golden_sub_b44a_x1y2.exr", "dump-sub"),
    ("golden_sub_b44a_x2y2.exr", "dump-sub"),
    ("golden_sub_dwaa_x1y1.exr", "dump-sub"),
    ("golden_sub_dwaa_x2y1.exr", "dump-sub"),
    ("golden_sub_dwaa_x1y2.exr", "dump-sub"),
    ("golden_sub_dwaa_x2y2.exr", "dump-sub"),
    ("golden_sub_dwab_x1y1.exr", "dump-sub"),
    ("golden_sub_dwab_x2y1.exr", "dump-sub"),
    ("golden_sub_dwab_x1y2.exr", "dump-sub"),
    ("golden_sub_dwab_x2y2.exr", "dump-sub"),
];

/// The `cl` flags and link libraries every OpenEXR-linked C++ program in this
/// repo needs (`oracle.cpp`, `exrheader`): one place, so they cannot drift.
fn openexr_build(vcpkg: &vcpkg::Vcpkg) -> Result<(Vec<String>, Vec<String>)> {
    // vcpkg lays the EXR headers under include/OpenEXR and include/Imath;
    // OpenEXR's own headers `#include <ImathBox.h>` / `<half.h>` etc. with no
    // subdir prefix, so both subdirs must be on the include path (the bare
    // `include` dir alone is not enough).
    let include = vcpkg.include().display().to_string();
    let cflags = vec![
        "/EHsc".to_string(),
        "/std:c++17".to_string(),
        format!("/I{include}/OpenEXR"),
        format!("/I{include}/Imath"),
    ];
    // A dynamic triplet links the import libraries (DLLs from `bin/` at run
    // time); a static triplet must also link what the OpenEXR archive itself
    // depends on: OpenEXRCore, OpenJPH (HTJ2K) and libdeflate.
    let mut libs = vec![
        vcpkg.lib("OpenEXR-")?,
        vcpkg.lib("Iex-")?,
        vcpkg.lib("IlmThread-")?,
        vcpkg.lib("Imath-")?,
    ];
    if vcpkg.is_static {
        libs.extend([vcpkg.lib("OpenEXRCore-")?, vcpkg.lib("openjph")?, vcpkg.deflate_lib()?]);
    }
    Ok((cflags, libs))
}

/// Build the C++ OpenEXR oracle, regenerate every committed golden EXR, then
/// run `parity_exr`, which (a) reads the C++-written goldens with OUR reader
/// and checks every pixel + header attr, and (b) has the C++ oracle read back
/// what OUR writer produced. `EXR_ORACLE_EXE` + the vcpkg `bin/` are injected
/// into the test process so direction (b) actually runs.
fn parity_exr() -> Result<()> {
    if windows_only("parity-exr", "exr-core") {
        return Ok(());
    }
    let spec = OracleSpec {
        krate: "exr-core",
        source: "oracle.cpp",
        vcpkg_probe: "OpenEXR/ImfOutputFile.h",
    };
    let (tc, vcpkg) = prepare(&spec)?;

    let (cflags, libs) = openexr_build(&vcpkg)?;
    let oracle = build_oracle(&spec, &tc, &vcpkg, &cflags, &libs)?;

    let jobs = exr_goldens();
    let n = jobs.len();
    for job in jobs {
        let file = oracle.dir.join(&job.file);
        let label = format!("{} <- {}", job.file, job.args.join(" "));
        oracle.write_golden(&label, &file, job.args)?;
    }
    println!("\nregenerated {n} golden EXR files");

    for (file, mode) in DECODE_DUMPS {
        let golden = oracle.dir.join(file);
        let dump = oracle.dir.join(format!("{file}.dump"));
        println!("\n--- running oracle {mode} {file} -> {file}.dump ---");
        oracle.run([
            OsString::from(*mode),
            golden.into_os_string(),
            dump.clone().into_os_string(),
        ])?;
        let len = std::fs::metadata(&dump).map(|m| m.len()).unwrap_or(0);
        println!("wrote {} ({len} bytes)", dump.display());
    }
    println!("committed {} C++ decode dumps", DECODE_DUMPS.len());

    // Not tied to any golden EXR (no image involved): `round12log`/
    // `roundNBit(4)` applied to EVERY half bit pattern, for a bit-exact check
    // of exr_core::rgba::{round12_log, RoundNBit} (`ImfLut.cpp`). Kept
    // outside `exr_goldens()`'s table since that table's own test
    // (`every_exr_golden_job_is_unique_and_named`) requires every entry to be
    // a `golden*.exr` file, which this deliberately is not.
    {
        let lut_dump = oracle.dir.join("lut_dump.bin");
        println!("\n--- running oracle lut-dump {} 4 ---", lut_dump.display());
        oracle.run([OsString::from("lut-dump"), lut_dump.clone().into_os_string(), OsString::from("4")])?;
        let len = std::fs::metadata(&lut_dump).map(|m| m.len()).unwrap_or(0);
        println!("wrote {} ({len} bytes)", lut_dump.display());
    }

    // D-12: a real C++ `CompositeDeepScanLine` compositing BOTH deep sources
    // above (not a decode of one golden -- `DECODE_DUMPS`'s `mode golden out`
    // shape doesn't fit a 2-input job, same reason `lut_dump` above lives
    // outside that table).
    {
        let a = oracle.dir.join("golden_deep_composite_a.exr");
        let b = oracle.dir.join("golden_deep_composite_b.exr");
        let dump = oracle.dir.join("golden_deep_composite.dump");
        println!("\n--- running oracle composite-deep-scanline {} {} -> {} ---",
            a.display(), b.display(), dump.display());
        oracle.run([
            OsString::from("composite-deep-scanline"),
            a.into_os_string(),
            b.into_os_string(),
            dump.clone().into_os_string(),
        ])?;
        let len = std::fs::metadata(&dump).map(|m| m.len()).unwrap_or(0);
        println!("wrote {} ({len} bytes)", dump.display());
    }

    cargo_test(
        "exr-core",
        "parity_exr",
        &[
            ("PATH", OsString::from(&oracle.path)),
            ("EXR_ORACLE_EXE", oracle.exe.clone().into_os_string()),
        ],
        &[],
    )?;
    // Pixel-read exception class over every corpus/golden file, via `oracle read-class`.
    cargo_test(
        "exr-core",
        "parity_class",
        &[
            ("PATH", OsString::from(&oracle.path)),
            ("EXR_ORACLE_EXE", oracle.exe.clone().into_os_string()),
        ],
        &["--ignored"],
    )?;
    println!("\nparity-exr: OK");
    Ok(())
}

/// Build OpenEXR's `exrheader` (from the reference `main.cpp`, unmodified) next
/// to the oracle, then run the ignored `describe` test that diffs its output
/// with `exr_core::api::describe` on every golden/corpus file.
fn parity_header() -> Result<()> {
    if windows_only("parity-header", "exr-core") {
        return Ok(());
    }
    let spec = OracleSpec {
        krate: "exr-core",
        source: "",
        vcpkg_probe: "OpenEXR/ImfOutputFile.h",
    };
    let (tc, vcpkg) = prepare(&spec)?;
    let (cflags, libs) = openexr_build(&vcpkg)?;

    let dir = repo_root().join("crates").join("exr-core").join("tests").join("parity");
    let source = openexr_src()?.join("src/bin/exrheader/main.cpp");
    anyhow::ensure!(source.exists(), "exrheader source not found at {}", source.display());
    let exe = dir.join("exrheader.exe");
    println!("
--- compiling exrheader main.cpp with cl ---");
    let status = vcpkg
        .cl(&dir, &tc)
        .args(&cflags)
        .arg(&source)
        .arg(format!("/Fo:{}", dir.join("exrheader.obj").display()))
        .arg(format!("/Fe:{}", exe.display()))
        .arg("/link")
        .arg(vcpkg.libpath_arg())
        .args(&libs)
        .status()
        .context("failed to launch `cl` (MSVC compiler)")?;
    anyhow::ensure!(status.success(), "cl failed to compile exrheader (exit {status})");

    println!("
--- cargo test -p exr-core --lib describe (exrheader diff) ---");
    let status = Command::new("cargo")
        .current_dir(repo_root())
        .env("PATH", vcpkg.run_path(&tc))
        .env("EXR_EXRHEADER_EXE", &exe)
        .args([
            "test", "-p", "exr-core", "--lib", "api::describe::tests::matches_exrheader",
            "--", "--ignored", "--nocapture",
        ])
        .status()
        .context("failed to launch `cargo test`")?;
    anyhow::ensure!(status.success(), "exrheader comparison FAILED (exit {status})");
    println!("
parity-header: OK");
    Ok(())
}

/// OpenEXR's command-line tools: each tool's directory under the pinned sources' `src/bin`
/// ([`openexr_src`]) and its source files, compiled unmodified.
const TOOLS: &[(&str, &[&str])] = &[
    ("exrcheck", &["main.cpp"]),
    ("exrmaketiled", &["main.cpp", "Image.cpp", "makeTiled.cpp"]),
    ("exrmakepreview", &["main.cpp", "makePreview.cpp"]),
    ("exrstdattr", &["main.cpp"]),
    ("exrmultipart", &["exrmultipart.cpp"]),
    ("exrmultiview", &["main.cpp", "Image.cpp", "makeMultiView.cpp"]),
    (
        "exrenvmap",
        &[
            "main.cpp",
            "blurImage.cpp",
            "EnvmapImage.cpp",
            "makeCubeMap.cpp",
            "makeLatLongMap.cpp",
            "readInputImage.cpp",
            "resizeImage.cpp",
        ],
    ),
    ("exrmetrics", &["main.cpp", "exrmetrics.cpp"]),
    ("exrmanifest", &["main.cpp"]),
    ("exr2aces", &["main.cpp"]),
];

/// Build every tool of [`TOOLS`] into `target/parity-tools/`, then run the root crate's
/// ignored `parity_tool_*` tests with `EXR_TOOLS_DIR` and the vcpkg run path.
fn parity_tools(build_only: bool) -> Result<()> {
    if windows_only("parity-tools", "exr-rs") {
        return Ok(());
    }
    let spec = OracleSpec { krate: "exr-rs", source: "", vcpkg_probe: "OpenEXR/ImfOutputFile.h" };
    let (tc, vcpkg) = prepare(&spec)?;
    let (cflags, mut libs) = openexr_build(&vcpkg)?;
    // exrcheck links OpenEXRUtil (ImfCheckFile), exrmetrics / exrcheck call OpenEXRCore directly.
    libs.push(vcpkg.lib("OpenEXRUtil-")?);
    if !vcpkg.is_static {
        libs.push(vcpkg.lib("OpenEXRCore-")?);
    }
    let out = repo_root().join("target").join("parity-tools");
    let upstream = openexr_src()?;
    for (tool, sources) in TOOLS {
        let src_dir = upstream.join("src/bin").join(tool);
        anyhow::ensure!(src_dir.is_dir(), "{} not found", src_dir.display());
        let obj_dir = out.join("obj").join(tool);
        std::fs::create_dir_all(&obj_dir).with_context(|| format!("create {}", obj_dir.display()))?;
        let exe = out.join(format!("{tool}.exe"));
        println!("\n--- compiling {tool} ---");
        let status = vcpkg
            .cl(&obj_dir, &tc)
            .args(&cflags)
            .arg(format!("/I{}", src_dir.display()))
            // Last: private headers vcpkg does not install (exrmetrics' ImfTiledMisc.h); the
            // installed headers above still win for everything they provide.
            .arg(format!("/I{}", upstream.join("src/lib/OpenEXR").display()))
            .args(sources.iter().map(|s| src_dir.join(s)))
            .arg(format!("/Fo:{}\\", obj_dir.display()))
            .arg(format!("/Fe:{}", exe.display()))
            .arg("/link")
            .arg(vcpkg.libpath_arg())
            .args(&libs)
            .status()
            .context("failed to launch `cl` (MSVC compiler)")?;
        anyhow::ensure!(status.success(), "cl failed to compile {tool} (exit {status})");
    }

    if build_only {
        println!("\nparity-tools: built C++ tools in {} (comparisons not run)", out.display());
        return Ok(());
    }

    println!("\n--- cargo test -p exr-rs --test 'parity_tool_*' -- --ignored ---");
    let status = Command::new("cargo")
        .current_dir(repo_root())
        .env("PATH", vcpkg.run_path(&tc))
        .env("EXR_TOOLS_DIR", &out)
        .args(["test", "-p", "exr-rs", "--test", "parity_tool_*", "--", "--ignored", "--nocapture"])
        .status()
        .context("failed to launch `cargo test`")?;
    anyhow::ensure!(status.success(), "a tool comparison FAILED (exit {status})");
    println!("\nparity-tools: OK");
    Ok(())
}

/// The OpenEXR release whose tool sources the tool oracles compile: the version vcpkg links
/// (`xtask/vcpkg.json`), so each tool's sources match the libraries under it.
const OPENEXR_TAG: &str = "v3.4.15";
/// The commit [`OPENEXR_TAG`] names; a fetched tree is checked against it, so a moved tag fails
/// loudly instead of building the oracles from other sources.
const OPENEXR_COMMIT: &str = "e71cdd5d30a146dcb56c5e4c576d9e9d3c45f4fb";

/// The pinned OpenEXR source tree, shallow-cloned once into `target/ref/openexr-<tag>`. Only the
/// tool oracles need C++ sources (the libraries come from vcpkg), so nothing is vendored.
fn openexr_src() -> Result<PathBuf> {
    let dir = repo_root().join("target").join("ref").join(format!("openexr-{OPENEXR_TAG}"));
    if !dir.join(".git").is_dir() {
        println!("\n--- fetching OpenEXR {OPENEXR_TAG} sources into {} ---", dir.display());
        let status = Command::new("git")
            .args(["clone", "--quiet", "--depth", "1", "--branch", OPENEXR_TAG])
            .arg("https://github.com/AcademySoftwareFoundation/openexr.git")
            .arg(&dir)
            .status()
            .context("failed to launch `git`")?;
        anyhow::ensure!(status.success(), "git clone of OpenEXR {OPENEXR_TAG} failed (exit {status})");
    }
    let head = Command::new("git")
        .arg("-C")
        .arg(&dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .context("failed to launch `git`")?;
    let head = String::from_utf8_lossy(&head.stdout);
    anyhow::ensure!(
        head.trim() == OPENEXR_COMMIT,
        "{} is at {}, not OpenEXR {OPENEXR_TAG} ({OPENEXR_COMMIT}); delete it to fetch again",
        dir.display(),
        head.trim()
    );
    Ok(dir)
}

/// Repo root = the directory containing this `xtask` crate's parent (the workspace
/// root). `CARGO_MANIFEST_DIR` points at `.../exr-rs/xtask`, so its parent is the
/// workspace root regardless of the invoking cwd.
fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The probe spawns real processes, so the tests drive a trivial always-present
    // shell instead of `cl`/`cmake` (which need not be installed to run the suite).
    // The invocation to print a banner / force a non-zero exit differs per OS.
    #[cfg(windows)]
    const SHELL: &str = "cmd";
    #[cfg(windows)]
    const ECHO_OK: &[&str] = &["/C", "echo hi"];
    #[cfg(windows)]
    const ECHO_THEN_FAIL: &[&str] = &["/C", "echo hi & exit 1"];
    #[cfg(windows)]
    const SILENT_OK: &[&str] = &["/C", "exit 0"];

    #[cfg(unix)]
    const SHELL: &str = "sh";
    #[cfg(unix)]
    const ECHO_OK: &[&str] = &["-c", "echo hi"];
    #[cfg(unix)]
    const ECHO_THEN_FAIL: &[&str] = &["-c", "echo hi; exit 1"];
    #[cfg(unix)]
    const SILENT_OK: &[&str] = &["-c", "exit 0"];

    fn probe(program: &'static str, args: &'static [&'static str], require_zero_exit: bool) -> ToolProbe {
        ToolProbe { program, args, banner_on_stderr: false, require_zero_exit }
    }

    #[test]
    fn missing_tool_is_err() {
        // Spawn failure (tool absent) must fail closed.
        let p = probe("exr_rs_definitely_no_such_tool_zzz", &[], true);
        assert!(probe_tool(&p, &Toolchain::default()).is_err());
    }

    #[test]
    fn valid_tool_returns_banner() {
        let p = probe(SHELL, ECHO_OK, true);
        assert_eq!(probe_tool(&p, &Toolchain::default()).unwrap(), "hi");
    }

    #[test]
    fn empty_banner_is_err() {
        // Tool spawns and exits zero but prints nothing: not proof of a toolchain.
        let p = probe(SHELL, SILENT_OK, true);
        assert!(probe_tool(&p, &Toolchain::default()).is_err());
    }

    #[test]
    fn nonzero_exit_is_err_when_required() {
        // Non-empty banner but a failing exit status fails the required-exit probe.
        let p = probe(SHELL, ECHO_THEN_FAIL, true);
        assert!(probe_tool(&p, &Toolchain::default()).is_err());
    }

    #[test]
    fn libdeflate_version_is_asserted() {
        let ok = format!("# golden parity (libdeflate {VERIFIED_LIBDEFLATE})\nkey = 00\n");
        assert!(assert_libdeflate_version(ok.as_bytes()).is_ok());
        // A drifted oracle must fail, not be accepted silently.
        assert!(assert_libdeflate_version(b"# golden parity (libdeflate 9.99)\n").is_err());
        // A golden with no version at all is not proof of anything.
        assert!(assert_libdeflate_version(b"# golden parity\n").is_err());
    }

    #[test]
    fn every_exr_golden_job_is_unique_and_named() {
        let jobs = exr_goldens();
        assert!(jobs.len() >= 34, "the golden table shrank: {} jobs", jobs.len());
        let mut files: Vec<&str> = jobs.iter().map(|j| j.file.as_str()).collect();
        files.sort_unstable();
        let n = files.len();
        files.dedup();
        assert_eq!(files.len(), n, "two jobs write the same golden file");
        assert!(jobs.iter().all(|j| j.file.starts_with("golden") && j.file.ends_with(".exr")));
        assert!(jobs.iter().all(|j| !j.args.is_empty()));
    }
}
