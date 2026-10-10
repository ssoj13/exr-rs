# Installation

## Prebuilt binaries

Each [GitHub release](https://github.com/ssoj13/exr-rs/releases) carries a ZIP per
platform (`windows-x86_64`, `linux-x86_64` for glibc 2.35 or newer,
`macos-aarch64`) with the `exr-rs` binary, its license and
`THIRD-PARTY-LICENSES.html`, and the `OpenEXR` Python wheels (CPython 3.8+, abi3)
for Windows x64, Linux x86_64 / aarch64 and macOS x86_64 / arm64:

```sh
pip install openexr-0.1.0-cp38-abi3-<platform>.whl
```

## Building from source: requirements

- A current stable Rust toolchain (edition 2024; declared MSRV 1.96). Install via
  [rustup](https://rustup.rs/).
- SSH access to GitHub: several dependencies (from the `box-rs`, `codec-simd-rs`, `curves-rs`,
  `egui-phosphor`, `egui-widgets-rs`, `gpu-info-rs`, `jph-rs`, `lcms-rs`, `murmur3`, `ofx-rs`,
  `oiio-rs`, `render-rs`, `scancache-rs`, `scanseq-rs`, `slate-rs` and `vcv-rs` repositories) are
  fetched as git dependencies from `ssh://git@github.com/ssoj13/...`.
- For the viewer / GPU upload: a GPU with a working Vulkan, D3D12, or Metal backend
  (wgpu 30 picks one automatically).

The core libraries (`exr-core` and below) have no system dependencies.

## Build

```sh
git clone https://github.com/ssoj13/exr-rs
cd exr-rs
cargo build --locked --release
```

The reference C++ sources and the test-image set are **not** vendored in the repo.
Neither is needed to build or run the tools. The parity gates link the C++
libraries through vcpkg and fetch the few upstream sources they compile
themselves; the "Reference sources" table in the repo-root `README.md` pins the
exact upstream commit of each port.

The thin `bootstrap.py` wrapper provides shorthands:

```sh
python bootstrap.py b      # cargo build --release of the exr-rs binary (-d for debug)
python bootstrap.py b p    # build the OpenEXR Python wheel via maturin
python bootstrap.py i      # cargo install exr-rs onto PATH (i p: pip install the wheel)
python bootstrap.py t      # the full gate; see Parity and testing
python bootstrap.py c      # clippy -D warnings on the workspace crates and openexr-py
python bootstrap.py v <f>  # run the viewer on a file
```

For a CPU-only CLI without GUI/OpenFX dependencies:

```sh
cargo build --locked --release --no-default-features
```

With the viewer but without OpenFX effects (GPU display remains required):

```sh
cargo build --locked --release --no-default-features --features viewer
```

Default builds enable the viewer and OpenFX. Diagnose git dependency failures against
the checked-in lockfile. Native reference libraries are needed for parity, not for
the Rust image-format runtime.

## The binary

Everything ships in one binary, `exr-rs`: the [command-line tool](cli.md) with its
inspection, conversion and whole-image tools, and the [viewer](viewer.md) as its
`view` (`v`) subcommand. `exr-view` is the library crate behind that subcommand, not
a separate program. A `--no-default-features` build drops the `viewer` feature (and its
winit/egui/wgpu dependencies), leaving the CPU image tools.
