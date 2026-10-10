//! One resolver for the vcpkg install tree used by every C/C++ parity oracle.
//!
//! The oracles used to hard-code `C:/vcpkg/installed/x64-windows` and its DLL
//! layout (import libs + `bin/` on PATH + `/MD` + `/DIMATH_DLL`). A machine with
//! only a static triplet (e.g. `x64-windows-static-release`) then failed with a
//! missing-header or LNK2019 error. This module picks the triplet that actually
//! has the headers and derives CRT flag, defines and library names from its
//! linkage, so the three oracle builders share one codepath.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::env_setup::Toolchain;

/// A resolved vcpkg `installed/<triplet>` directory.
pub struct Vcpkg {
    root: PathBuf,
    /// Static triplet (`*-static*`): link `.lib` archives, no DLLs, no `bin/` on PATH.
    pub is_static: bool,
    /// Static triplet with the static CRT (`*-static` but not `*-static-md*`) → `/MT`.
    static_crt: bool,
}

impl Vcpkg {
    /// Resolve the tree containing `probe` (a path relative to `include/`).
    ///
    /// `VCPKG_INSTALLED` (a full `installed/<triplet>` path) wins when set. Otherwise
    /// the triplets under `$VCPKG_ROOT/installed` (default `C:/vcpkg/installed`) are
    /// searched, preferring the dynamic `x64-windows`, then any other `x64-windows*`.
    pub fn resolve(probe: &str) -> Result<Self> {
        if let Ok(dir) = std::env::var("VCPKG_INSTALLED") {
            let root = PathBuf::from(dir);
            anyhow::ensure!(
                root.join("include").join(probe).exists(),
                "VCPKG_INSTALLED={} has no include/{probe}",
                root.display()
            );
            return Ok(Self::from_root(root));
        }
        let installed = std::env::var("VCPKG_ROOT")
            .map(|r| PathBuf::from(r).join("installed"))
            .unwrap_or_else(|_| PathBuf::from("C:/vcpkg/installed"));
        let mut triplets: Vec<PathBuf> = std::fs::read_dir(&installed)
            .with_context(|| format!("cannot list vcpkg triplets in {}", installed.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("x64-windows"))
            })
            .collect();
        // `x64-windows` sorts first; the rest alphabetically for a deterministic pick.
        triplets.sort();
        triplets
            .into_iter()
            .find(|t| t.join("include").join(probe).exists())
            .map(Self::from_root)
            .with_context(|| {
                format!(
                    "no x64-windows* triplet under {} has include/{probe} \
                     (install the port or set VCPKG_INSTALLED)",
                    installed.display()
                )
            })
    }

    fn from_root(root: PathBuf) -> Self {
        let name = root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let is_static = name.contains("-static");
        let static_crt = is_static && !name.contains("-static-md");
        Self { root, is_static, static_crt }
    }

    pub fn include(&self) -> PathBuf {
        self.root.join("include")
    }

    /// The first library in `lib/` whose file name starts with `prefix` — keeps the
    /// version suffixes (`Imath-3_2`, `OpenEXR-3_4`, `openjph.0.31`) out of the code.
    pub fn lib(&self, prefix: &str) -> Result<String> {
        let dir = self.root.join("lib");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .with_context(|| format!("cannot list {}", dir.display()))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.starts_with(prefix) && n.ends_with(".lib"))
            .collect();
        names.sort();
        names
            .into_iter()
            .next()
            .with_context(|| format!("no {prefix}*.lib in {}", dir.display()))
    }

    /// The libdeflate archive: vcpkg names the static one `deflatestatic.lib`.
    pub fn deflate_lib(&self) -> Result<String> {
        self.lib(if self.is_static { "deflatestatic" } else { "deflate." })
    }

    /// A `cl` invocation in `dir` with the CRT matching the triplet and `include/`.
    ///
    /// `tc` carries the MSVC environment (PATH/INCLUDE/LIB/...) explicitly —
    /// xtask never mutates its own process environment.
    pub fn cl(&self, dir: &Path, tc: &Toolchain) -> Command {
        let mut cmd = Command::new("cl");
        tc.apply(&mut cmd);
        cmd.current_dir(dir)
            .args(["/nologo", "/O2", if self.static_crt { "/MT" } else { "/MD" }])
            .arg(format!("/I{}", self.include().display()));
        cmd
    }

    /// `/LIBPATH:` argument for the linker section of a `cl` command.
    pub fn libpath_arg(&self) -> String {
        format!("/LIBPATH:{}", self.root.join("lib").display())
    }

    /// PATH for running an oracle: dynamic triplets need `bin/` (the DLLs)
    /// first. Derived from the assembled [`Toolchain`], NOT from this process's
    /// environment — the toolchain is the single source of truth for what a
    /// child sees.
    pub fn run_path(&self, tc: &Toolchain) -> String {
        let path = tc.path();
        if self.is_static {
            path
        } else {
            format!("{};{path}", self.root.join("bin").display())
        }
    }
}
