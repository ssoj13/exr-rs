//! A closed stdout (`exr-rs info f.exr | head -0`) ends a command quietly with the exit code its
//! own result decides, like the C++ tools whose `std::cout` just goes bad: no panic (`println!`
//! panics on a broken pipe), no message, same code as with a reader.
//!
//! The read end is dropped right after the spawn, before the child has produced anything, so its
//! first write meets a closed pipe.

use std::path::Path;
use std::process::{Command, Stdio};

/// Run `exr-rs args..` with stdout closed on the reader's side: (exit code, stderr).
fn run_closed(args: &[&str]) -> (Option<i32>, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_exr-rs"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn exr-rs");
    drop(child.stdout.take());
    let out = child.wait_with_output().expect("wait for exr-rs");
    (out.status.code(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn closed_stdout_ends_quietly() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/exr-core/tests/parity/golden.exr");
    let golden = golden.to_str().expect("utf8 path");
    for cmd in ["info", "manifest", "check"] {
        let (code, err) = run_closed(&[cmd, golden]);
        assert_eq!(code, Some(0), "{cmd}: stderr {err:?}");
        assert!(err.is_empty(), "{cmd} wrote to stderr: {err:?}");
    }
}
