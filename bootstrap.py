#!/usr/bin/env python3
"""bootstrap.py — exr-rs build system with short directives.
Cross-platform, Python 3, stdlib only. Windows-friendly (subprocess, no
bash-isms). On Windows the native MSVC toolchain for the C++ parity oracles
(Imath / OpenEXR / libdeflate) is applied inside `xtask` via vcv-rs — no vcvars
in this script (see xtask/src/env_setup.rs).
Short directives (one/two letters) drive the common loop; full names also work.
  b   build         cargo build (exr-rs only)      (release by default · -d debug)
  b p build-python  maturin build the `OpenEXR` Python module (cross-platform · -d debug)
  t   test          THE GATE: cargo tests + parity + parity-deflate + parity-exr
                    + pytest, including the tests on the private image set
                    (EXR_TEST_DATA, else ../exr-rs-tests; `test-data` features).
                    Select stages: `t cargo`, `t exr python`, …
                    (-n nocapture · [pattern] = cargo stage only)
  i   install       cargo install exr-rs onto PATH (~/.cargo/bin; release · -d debug)
  i p install-python  build the `OpenEXR` wheel and pip install it into this Python
  c   check         cargo clippy on in-scope crates (excludes jph-rs debt)
  p   parity        cargo run -p xtask -- parity          (native Imath oracle)
  pe  parity-exr    cargo run -p xtask -- parity-exr      (real C++ OpenEXR 3.4)
  pd  parity-deflate cargo run -p xtask -- parity-deflate (C libdeflate byte-exact)
  s   setup         cargo run -p xtask -- setup          (prove cl/cmake on PATH)
  v   view          cargo run -p exr-rs -- view [--auto] [file|dir|frame]  (GPU viewer; --auto = scan sequence)
  ci                clippy + tests + release build + portable ZIP in dist/ (ci.py · --target TRIPLE)
  release           release build + portable ZIP in dist/, no checks (ci.py · --target TRIPLE)
  cl  clean         cargo clean
  h   help          this help
Examples:
  python bootstrap.py b            # build exr-rs (release — matches gui.cmd)
  python bootstrap.py b -d         # build debug
  python bootstrap.py b p          # build the OpenEXR python module (maturin wheel)
  python bootstrap.py i            # install the exr-rs CLI + viewer onto PATH
  python bootstrap.py i p          # install the OpenEXR python module into this Python
  python bootstrap.py t            # the full gate (every stage, non-zero on any failure)
  python bootstrap.py t cargo      # only the cargo workspace tests
  python bootstrap.py t -n tiled   # cargo tests, show output, filter "tiled"
  python bootstrap.py c            # clippy
  python bootstrap.py pe           # C++ OpenEXR parity gate
  python bootstrap.py v Desk.exr   # open the viewer on a file
"""

from __future__ import annotations
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
IS_WINDOWS = platform.system() == "Windows"
# Crates that pass clippy cleanly. jph-rs is excluded on purpose: it carries
# pre-existing 1:1-port clippy debt.
CLIPPY_CRATES = [
	"exr-rs",
	"imath-rs",
	"deflate-rs",
	"exr-core",
	"cryptomatte",
	"exr-view",
	"exr-gpu",
	"egui-timeline",
	"egui-statusbar",
	"xtask",
]


class C:
	"""ANSI color codes (enabled on Windows via `os.system('')`)."""

	RST = "\033[0m"
	RED = "\033[91m"
	GRN = "\033[92m"
	YLW = "\033[93m"
	CYN = "\033[96m"

	@classmethod
	def init(cls) -> None:
		if IS_WINDOWS:
			os.system("")


def fmt_time(ms: float) -> str:
	return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def header(text: str) -> None:
	line = "=" * 60
	print(f"\n{C.CYN}{line}\n{text}\n{line}{C.RST}")


def ok(text: str) -> None:
	print(f"  {C.GRN}{text}{C.RST}")


def fail(text: str) -> None:
	print(f"  {C.RED}{text}{C.RST}")


def run(args: list[str]) -> tuple[int, float]:
	"""Run a command from the repo root, streaming output. Returns (code, ms)."""
	start = time.perf_counter()
	result = subprocess.run(args, cwd=ROOT_DIR)
	return result.returncode, (time.perf_counter() - start) * 1000


def _report(label: str, code: int, ms: float) -> int:
	"""Print a green/red one-liner for a finished step; pass the code through."""
	if code == 0:
		ok(f"{label} OK ({fmt_time(ms)})")
	else:
		fail(f"{label} FAILED (exit {code})")
	return code


# ── commands (each takes the trailing argv slice) ──────────────────────────────
def cmd_build(rest: list[str]) -> int:
	# `b p` / `b py` / `b python` → build the Python extension module via maturin
	# instead of the cargo workspace (the crate is excluded from the workspace).
	py_tokens = {"p", "py", "python"}
	if any(a in py_tokens for a in rest):
		return cmd_build_python([a for a in rest if a not in py_tokens])
	header("BUILD")
	# Build ONLY the product binary: the default workspace member `exr-rs`
	# (the viewer is linked in via its `viewer` feature). NOT `--workspace`,
	# which would also build the dev/bench bins (xtask, ojph_bench, vcv, …).
	cmd = ["cargo", "build"]
	# Release is the DEFAULT: the only run path (gui.cmd) launches
	# target/release/exr-rs.exe, so a debug-default `b` silently ran stale code.
	# Opt into debug with -d/--debug; -r/--release still works (and is a no-op).
	debug = "-d" in rest or "--debug" in rest
	if not debug:
		cmd.append("--release")
	return _report("Build", *run(cmd))


def cmd_build_python(rest: list[str]) -> int:
	"""Build the `OpenEXR` Python module (PyO3 + maturin). Cross-platform: maturin
	emits `OpenEXR.pyd` on Windows / `OpenEXR.so` on Linux/macOS from one source."""
	header("BUILD PYTHON (OpenEXR module via maturin)")
	manifest = ROOT_DIR / "crates" / "openexr-py" / "Cargo.toml"
	cmd = ["maturin", "build", "-m", str(manifest)]
	if not ("-d" in rest or "--debug" in rest):
		cmd.append("--release")
	try:
		code, ms = run(cmd)
	except FileNotFoundError:
		fail("maturin not found on PATH — cannot build the Python module.")
		print(
			"\n  maturin is the build tool that compiles a Rust PyO3 crate into a\n"
			"  Python wheel (it produces OpenEXR.pyd on Windows / OpenEXR.so on\n"
			"  Linux/macOS). It is a normal Python package.\n\n"
			"  Install it (cross-platform), then re-run `python bootstrap.py b p`:\n"
			f"      {C.CYN}cargo install maturin{C.RST}   (it is on crates.io — no pip needed)\n"
			"  or via Python:\n"
			f"      {C.CYN}pip install maturin{C.RST}   /   {C.CYN}pipx install maturin{C.RST}\n\n"
			"  Docs: https://www.maturin.rs\n"
		)
		return 127
	if code == 0:
		ok(
			f"wheel written to {_openexr_py_target_dir() / 'wheels'} "
			"(pip install it, or `maturin develop`)"
		)
	return _report("Build Python", code, ms)


# The private test image set (the `exr-rs-tests` checkout). The gate always runs the tests that
# read it: the crates' `test-data` feature for cargo, EXR_TEST_DATA for both cargo and pytest.
TEST_DATA_ENV = "EXR_TEST_DATA"
TEST_DATA_FEATURES = "exr-core/test-data,exr-view/test-data,cryptomatte/test-data"


def _enable_test_data() -> None:
	"""Point every stage at the image set: EXR_TEST_DATA, else a sibling exr-rs-tests."""
	os.environ.setdefault(TEST_DATA_ENV, str(ROOT_DIR.parent / "exr-rs-tests"))


def _cargo_tests(rest: list[str]) -> int:
	"""The cargo workspace test stage (with the optional libtest filter)."""
	# --no-fail-fast: a gate must report EVERY failing target, not stop at the
	# first one (cargo skips the remaining test binaries otherwise).
	cmd = ["cargo", "test", "--workspace", "--no-fail-fast", "--features", TEST_DATA_FEATURES]
	pattern = next((a for a in rest if not a.startswith("-") and a not in STAGES), None)
	if pattern:
		cmd.append(pattern)
	if "-n" in rest or "--nocapture" in rest:
		cmd += ["--", "--nocapture"]
	return run(cmd)[0]


# ── the Python stage ──────────────────────────────────────────────────────────
# Not under target/: CI's rust-cache saves and prunes target/, and a pruned venv restored
# from it (python.exe and maturin's __main__ gone) broke the gate.
VENV_DIR = ROOT_DIR / ".venv-pytest"
PY_MANIFEST = ROOT_DIR / "crates" / "openexr-py" / "Cargo.toml"
PY_TESTS = ROOT_DIR / "crates" / "openexr-py" / "tests"


def _venv_python() -> Path:
	return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def _ensure_venv() -> int:
	"""Create the pytest venv and install its tooling once. Re-runs are a no-op."""
	py = _venv_python()
	if not py.exists():
		print(f"  creating pytest venv in {VENV_DIR.name} …")
		code, _ = run([sys.executable, "-m", "venv", "--clear", str(VENV_DIR)])
		if code != 0:
			return code
	# Probe what the stage runs (`python -m maturin`), not only that the names import.
	probe = subprocess.run(
		[str(py), "-c", "import pytest, numpy, importlib.util as u; "
		 "raise SystemExit(u.find_spec('maturin.__main__') is None)"],
		cwd=ROOT_DIR,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)
	if probe.returncode == 0:
		return 0
	print("  installing pytest / numpy / maturin into the venv …")
	return run([str(py), "-m", "pip", "install", "-q", "pytest", "numpy", "maturin"])[0]


def _openexr_py_target_dir() -> Path:
	"""Resolve the Cargo target directory maturin actually writes into for
	`crates/openexr-py`.

	`openexr-py` is deliberately excluded from the workspace (see its
	`Cargo.toml`'s comment: a PyO3 cdylib built ONLY by maturin, NOT by
	`[workspace]`) — Cargo therefore does NOT discover the repo-root
	`Cargo.toml` as its workspace when built via `--manifest-path`; it is its
	own single-package "workspace" and defaults its target directory to
	`crates/openexr-py/target`, not the root `target/`. Assuming the root
	`target/` (or `CARGO_TARGET_DIR`, which is honoured but usually unset)
	made this stage report a false `maturin produced no wheel` FAIL even
	though the wheel built fine.
	`cargo metadata` is the single source of truth for where Cargo/maturin
	actually put it, so ask it instead of guessing a path."""
	result = subprocess.run(
		["cargo", "metadata", "--no-deps", "--format-version", "1", "--manifest-path", str(PY_MANIFEST)],
		cwd=ROOT_DIR,
		capture_output=True,
		text=True,
	)
	if result.returncode == 0:
		try:
			return Path(json.loads(result.stdout)["target_directory"])
		except (json.JSONDecodeError, KeyError):
			pass
	# Fall back to Cargo's own default for an unbundled package if `cargo
	# metadata` is unavailable or its output is unparseable.
	return Path(os.environ.get("CARGO_TARGET_DIR") or (PY_MANIFEST.parent / "target"))


def _newest_wheel() -> Path | None:
	"""The wheel maturin wrote last for `crates/openexr-py`, if any."""
	wheels = _openexr_py_target_dir() / "wheels"
	built = sorted(wheels.glob("*.whl"), key=lambda w: w.stat().st_mtime) if wheels.is_dir() else []
	return built[-1] if built else None


def _pytest_stage(_rest: list[str]) -> int:
	"""Build the `OpenEXR` wheel with maturin and run the Python tests against it.

	The wheel goes into an isolated venv on purpose: the tests must exercise the
	module this tree builds, not whatever `OpenEXR` happens to be installed
	system-wide (which is how the cryptomatte goldens once ended up generated by
	our own wheel)."""
	code = _ensure_venv()
	if code != 0:
		return code
	py = _venv_python()
	code, _ = run([str(py), "-m", "maturin", "build", "--release", "-m", str(PY_MANIFEST)])
	if code != 0:
		return code
	fresh = _newest_wheel()
	if fresh is None:
		fail(f"maturin produced no wheel in {_openexr_py_target_dir() / 'wheels'}")
		return 1
	code, _ = run([str(py), "-m", "pip", "install", "-q", "--force-reinstall", str(fresh)])
	if code != 0:
		return code
	return run([str(py), "-m", "pytest", str(PY_TESTS), "-q"])[0]


# ── the gate ──────────────────────────────────────────────────────────────────
# name → (human label, runner). `t` runs them all, in this order; naming any
# subset on the command line runs only those.
STAGES: dict[str, tuple[str, object]] = {
	"cargo": ("cargo test --workspace", _cargo_tests),
	# `t` used to never run clippy at all (it was only the
	# separate `c` command) -- that let openexr-py's own clippy findings (it
	# is excluded from the workspace, so `cargo clippy --workspace` never
	# touched it) go unnoticed for a long time. `cmd_check` now covers both
	# the workspace crates AND openexr-py; wiring it into the gate closes
	# that hole for good.
	"clippy": ("cargo clippy (workspace + openexr-py)", lambda _r: cmd_check([])),
	"parity": ("xtask parity (C++ Imath)", lambda _r: cmd_parity([])),
	"deflate": ("xtask parity-deflate (C libdeflate)", lambda _r: cmd_parity_deflate([])),
	"exr": ("xtask parity-exr (C++ OpenEXR)", lambda _r: cmd_parity_exr([])),
	"python": ("pytest crates/openexr-py/tests", _pytest_stage),
}


def cmd_test(rest: list[str]) -> int:
	"""THE gate. Every stage runs; the exit code is non-zero if ANY stage failed.

	A green `cargo test` alone never proved the C++-parity claims (nothing ran
	the oracles, and nothing at all ran the Python tests) — so `t` now means all
	of it. Stages are selectable by name to keep the loop fast while iterating.
	"""
	_enable_test_data()
	selected = [a for a in rest if a in STAGES]
	if not selected:
		# A bare positional that is NOT a stage name is a libtest filter, which
		# only makes sense for the cargo stage.
		filtered = any(not a.startswith("-") for a in rest)
		selected = ["cargo"] if filtered else list(STAGES)
	results: list[tuple[str, int, float]] = []
	for name in selected:
		label, fn = STAGES[name]
		header(f"TEST [{name}] — {label}")
		start = time.perf_counter()
		try:
			code = fn(rest)
		except FileNotFoundError as e:
			fail(f"{label}: {e}")
			code = 127
		ms = (time.perf_counter() - start) * 1000
		_report(label, code, ms)
		results.append((name, code, ms))

	header("GATE SUMMARY")
	for name, code, ms in results:
		mark = f"{C.GRN}PASS{C.RST}" if code == 0 else f"{C.RED}FAIL({code}){C.RST}"
		print(f"  {mark:<22} {name:<9} {fmt_time(ms)}")
	failed = [n for n, c, _ in results if c != 0]
	if failed:
		fail(f"GATE FAILED: {', '.join(failed)}")
		return 1
	ok("GATE OK")
	return 0


def cmd_install(rest: list[str]) -> int:
	"""`i`: cargo install the `exr-rs` binary (default features: viewer + OpenFX) onto PATH.
	`i p`: build the `OpenEXR` wheel and pip install it into the Python running this script."""
	py_tokens = {"p", "py", "python"}
	if any(a in py_tokens for a in rest):
		return cmd_install_python([a for a in rest if a not in py_tokens])
	header("INSTALL (exr-rs onto PATH)")
	# --locked: install exactly the dependency versions the tree was tested with (Cargo.lock).
	cmd = ["cargo", "install", "--path", str(ROOT_DIR), "--locked", "--force"]
	if "-d" in rest or "--debug" in rest:
		cmd.append("--debug")
	return _report("Install", *run(cmd))


def cmd_install_python(rest: list[str]) -> int:
	code = cmd_build_python(rest)
	if code != 0:
		return code
	header(f"INSTALL PYTHON (OpenEXR into {sys.executable})")
	wheel = _newest_wheel()
	if wheel is None:
		fail(f"no wheel in {_openexr_py_target_dir() / 'wheels'}")
		return 1
	return _report("Install Python", *run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)]))


def cmd_check(_rest: list[str]) -> int:
	header("CHECK (clippy, in-scope crates)")
	cmd = ["cargo", "clippy"]
	for crate in CLIPPY_CRATES:
		cmd += ["-p", crate]
	cmd += ["--all-targets", "--", "-D", "warnings"]
	code, ms = run(cmd)
	_report("Clippy", code, ms)
	# openexr-py is excluded from the workspace (its own Cargo.toml/lockfile,
	# maturin-only build), so it needs its own clippy invocation; without it
	# this crate's clippy findings were never checked at all.
	py_code, py_ms = run([
		"cargo", "clippy",
		"--manifest-path", "crates/openexr-py/Cargo.toml",
		"--all-targets", "--", "-D", "warnings",
	])
	_report("Clippy (openexr-py)", py_code, py_ms)
	return code or py_code


def cmd_parity(_rest: list[str]) -> int:
	header("PARITY (native Imath oracle)")
	return run(["cargo", "run", "-p", "xtask", "--", "parity"])[0]


def cmd_parity_exr(_rest: list[str]) -> int:
	header("PARITY-EXR (real C++ OpenEXR 3.4 oracle)")
	return run(["cargo", "run", "-p", "xtask", "--", "parity-exr"])[0]


def cmd_parity_deflate(_rest: list[str]) -> int:
	header("PARITY-DEFLATE (C libdeflate byte-exact oracle)")
	return run(["cargo", "run", "-p", "xtask", "--", "parity-deflate"])[0]


def cmd_setup(_rest: list[str]) -> int:
	header("SETUP (native MSVC toolchain)")
	return run(["cargo", "run", "-p", "xtask", "--", "setup"])[0]


def cmd_view(rest: list[str]) -> int:
	header("VIEW (GPU EXR viewer)")
	# Everything after `v`/`view` is forwarded to the viewer (e.g. a file path).
	return run(["cargo", "run", "-p", "exr-rs", "--", "view", *rest])[0]


def _portable(command: str, rest: list[str]) -> int:
	"""CI/release orchestration lives in ci.py; keep it behind this one entry point."""
	header(f"{command.upper()} (portable release ZIP)")
	return run([sys.executable, str(ROOT_DIR / "ci.py"), command, *rest])[0]


def cmd_ci(rest: list[str]) -> int:
	return _portable("ci", rest)


def cmd_release(rest: list[str]) -> int:
	return _portable("release", rest)


def cmd_clean(_rest: list[str]) -> int:
	header("CLEAN")
	return _report("cargo clean", *run(["cargo", "clean"]))


def cmd_help(_rest: list[str]) -> int:
	print(__doc__)
	return 0


# ── dispatch: short alias → canonical, canonical → fn ─────────────────────────
DISPATCH = {
	"build": cmd_build,
	"install": cmd_install,
	"test": cmd_test,
	"check": cmd_check,
	"parity": cmd_parity,
	"parity-exr": cmd_parity_exr,
	"parity-deflate": cmd_parity_deflate,
	"setup": cmd_setup,
	"view": cmd_view,
	"ci": cmd_ci,
	"release": cmd_release,
	"clean": cmd_clean,
	"help": cmd_help,
}
ALIASES = {
	"b": "build",
	"i": "install",
	"t": "test",
	"c": "check",
	"p": "parity",
	"pe": "parity-exr",
	"pd": "parity-deflate",
	"s": "setup",
	"v": "view",
	"cl": "clean",
	"h": "help",
}


def main() -> int:
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(line_buffering=True)
	C.init()
	argv = sys.argv[1:]
	if not argv or argv[0] in ("-h", "--help"):
		return cmd_help([])
	raw, rest = argv[0], argv[1:]
	canonical = ALIASES.get(raw, raw)
	fn = DISPATCH.get(canonical)
	if fn is None:
		fail(f"unknown command: {raw!r}\n")
		return cmd_help([]) or 2
	return fn(rest)


if __name__ == "__main__":
	sys.exit(main())
