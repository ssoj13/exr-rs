//! MSVC / toolchain bootstrap for the C/C++ parity oracles (Windows).
//!
//! The parity harnesses build small C/C++ oracles with the native MSVC
//! toolchain; this module assembles the environment `cl` needs (via `vcv-rs`)
//! so the user never has to run `vcvars64.bat` first.
//!
//! **The environment is a value, not a process mutation.** `std::env::set_var`
//! is `unsafe` in edition 2024 (it races every other thread in the process),
//! and the project's rule is that `unsafe` exists only in SIMD hot paths behind
//! a safe scalar source of truth. Nothing here is a hot path, so
//! [`prepare_native_toolchain`] returns a [`Toolchain`] that each child process
//! receives through [`Toolchain::apply`] (`Command::envs`). That also makes the
//! toolchain explicit at every spawn site instead of ambient.

use anyhow::Result;
#[cfg(windows)]
use anyhow::Context;

#[cfg(windows)]
use std::path::PathBuf;

#[cfg(windows)]
use vcv_rs::Arch;
#[cfg(windows)]
use vcv_rs::detect::{detect_sdk, detect_ucrt, detect_vs_range};
#[cfg(windows)]
use vcv_rs::env::build_env;

/// The environment a child process needs to run `cl` / the built oracles.
///
/// Built once from the detected VS / SDK / UCRT and passed to every
/// `Command`; the xtask process's own environment is never modified.
#[derive(Debug, Default, Clone)]
pub struct Toolchain {
    /// Variables to set on each child (already merged with this process's
    /// values for the list-valued ones: PATH, INCLUDE, LIB, LIBPATH).
    vars: Vec<(String, String)>,
    /// Variables to *unset* on each child.
    remove: Vec<&'static str>,
}

impl Toolchain {
    /// Apply this toolchain to a child process.
    pub fn apply<'c>(&self, cmd: &'c mut std::process::Command) -> &'c mut std::process::Command {
        for (k, v) in &self.vars {
            cmd.env(k, v);
        }
        for k in &self.remove {
            cmd.env_remove(k);
        }
        cmd
    }

    /// The child `PATH` (toolchain entries first, then the inherited PATH).
    /// The oracle runners prepend vcpkg's `bin/` to this for dynamic triplets.
    pub fn path(&self) -> String {
        self.get("PATH").unwrap_or_default()
    }

    fn get(&self, key: &str) -> Option<String> {
        self.vars
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.clone())
            .or_else(|| std::env::var(key).ok())
    }

    #[cfg(windows)]
    fn set(&mut self, key: impl Into<String>, val: impl Into<String>) {
        self.vars.push((key.into(), val.into()));
    }

    /// Set `key` to `prefix` followed by the value this process inherited.
    #[cfg(windows)]
    fn prepend(&mut self, key: &'static str, extra: &[PathBuf], sep: char) {
        if extra.is_empty() {
            return;
        }
        let prefix = extra.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(";");
        let merged = match self.get(key) {
            Some(rest) if !rest.is_empty() => format!("{prefix}{sep}{rest}"),
            _ => prefix,
        };
        self.set(key, merged);
    }
}

/// Assemble the platform toolchain environment (MSVC on Windows via `vcv-rs`).
pub fn prepare_native_toolchain() -> Result<Toolchain> {
    let mut tc = Toolchain::default();
    drop_broken_libclang(&mut tc);
    #[cfg(windows)]
    windows_msvc_paths(&mut tc)?;
    Ok(tc)
}

/// Drop `LIBCLANG_PATH` for children when it points at ESP-IDF / Xtensa clang
/// (that clang breaks bindgen under MSVC).
fn drop_broken_libclang(tc: &mut Toolchain) {
    let Ok(libclang) = std::env::var("LIBCLANG_PATH") else {
        return;
    };
    let lower = libclang.to_lowercase();
    if lower.contains("esp") || lower.contains("xtensa") {
        eprintln!("xtask: clearing LIBCLANG_PATH for child processes (ESP/Xtensa clang breaks bindgen/MSVC)");
        tc.remove.push("LIBCLANG_PATH");
    }
}

#[cfg(windows)]
fn windows_msvc_paths(tc: &mut Toolchain) -> Result<()> {
    // exr-rs has no CUDA constraint, so pick the newest available Visual Studio.
    let vs = detect_vs_range(None, None).with_context(|| {
        let installed = vcv_rs::detect::list_vs_versions();
        if installed.is_empty() {
            "Visual Studio not found. Install VS 2022 from https://visualstudio.microsoft.com/".to_string()
        } else {
            format!(
                "No usable Visual Studio found.\n\
                 Installed: {installed:?}\n\
                 Install VS 2022: https://visualstudio.microsoft.com/vs/older-downloads/"
            )
        }
    })?;
    eprintln!("xtask: detected VS {} (toolset {})", vs.version, vs.tools_ver);
    let sdk = detect_sdk().context("Windows SDK (10.x) not found")?;
    let ucrt = detect_ucrt().context("Universal CRT (Windows Kits 10.x) not found")?;

    let assembled = build_env(&vs, Some(&sdk), Some(&ucrt), Arch::X64, Arch::X64);

    for (k, v) in &assembled.vars {
        tc.set(k.clone(), v.clone());
    }
    tc.prepend("PATH", &assembled.path, ';');
    tc.prepend("INCLUDE", &assembled.include, ';');
    tc.prepend("LIB", &assembled.lib, ';');
    tc.prepend("LIBPATH", &assembled.libpath, ';');

    eprintln!("xtask: MSVC toolchain assembled (vcv-rs)");
    Ok(())
}
