//! The one writer every command prints through (`out!`, `outln!`, `errln!`, `err!`, [`Out`]).
//!
//! `println!` panics when stdout is gone (`exr-rs info f.exr | head`). The C++ tools write through
//! `std::cout`, which just goes into a failed state: the tool keeps running, nothing more is
//! written, and the exit code is whatever the tool would have returned anyway (on Windows there is
//! no SIGPIPE either). The same here: the first failed write closes the stream for the rest of the
//! process, later writes are dropped, and the command's own result decides the exit code. A closed
//! pipe is the normal way to stop reading and stays silent; any other stdout failure (a full disk
//! behind `> file`) is logged once, so output that was lost is not lost unnoticed.
//!
//! `clippy::print_stdout` / `print_stderr` are denied crate-wide in `main.rs`, so a raw `println!`
//! cannot come back.

use std::fmt::Arguments;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

static OUT_OPEN: AtomicBool = AtomicBool::new(true);
static ERR_OPEN: AtomicBool = AtomicBool::new(true);

/// Run `op` on a stream unless an earlier failure closed it; a failure closes it for good.
/// `report` logs failures other than a closed pipe (stderr cannot report its own failure).
fn guard(open: &AtomicBool, report: bool, op: impl FnOnce() -> io::Result<()>) {
    if !open.load(Ordering::Relaxed) {
        return;
    }
    if let Err(e) = op() {
        open.store(false, Ordering::Relaxed);
        if report && e.kind() != io::ErrorKind::BrokenPipe {
            log::warn!("writing to stdout failed, the rest of the output is dropped: {e}");
        }
    }
}

/// Stdout as an [`io::Write`] that never fails (see the module docs): for code that wants a
/// buffered writer (`manifest`).
pub struct Out;

impl Write for Out {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        guard(&OUT_OPEN, true, || io::stdout().lock().write_all(buf));
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        guard(&OUT_OPEN, true, || io::stdout().lock().flush());
        Ok(())
    }
}

/// Stderr, same contract as [`Out`].
struct ErrOut;

impl Write for ErrOut {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        guard(&ERR_OPEN, false, || io::stderr().lock().write_all(buf));
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        guard(&ERR_OPEN, false, || io::stderr().lock().flush());
        Ok(())
    }
}

/// Format `args` into `w`; the streams above never fail, so an error here is a `Display` impl
/// returning `Err`, which `println!` turns into a panic and we log instead.
fn put(mut w: impl Write, args: Arguments<'_>) {
    if let Err(e) = w.write_fmt(args) {
        log::error!("a value could not be formatted for output: {e}");
    }
}

/// Backend of `out!` / `outln!`.
pub fn out(args: Arguments<'_>) {
    put(Out, args);
}

/// Backend of `err!` / `errln!`.
pub fn err(args: Arguments<'_>) {
    put(ErrOut, args);
}

/// Flush stdout (a prompt written without a newline, the end of the run).
pub fn flush() {
    guard(&OUT_OPEN, true, || io::stdout().lock().flush());
}

macro_rules! out {
    ($($t:tt)*) => { $crate::cli::stdio::out(format_args!($($t)*)) };
}
macro_rules! outln {
    () => { $crate::cli::stdio::out(format_args!("\n")) };
    ($($t:tt)*) => { $crate::cli::stdio::out(format_args!("{}\n", format_args!($($t)*))) };
}
macro_rules! err {
    ($($t:tt)*) => { $crate::cli::stdio::err(format_args!($($t)*)) };
}
macro_rules! errln {
    () => { $crate::cli::stdio::err(format_args!("\n")) };
    ($($t:tt)*) => { $crate::cli::stdio::err(format_args!("{}\n", format_args!($($t)*))) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failed write closes the stream, silently for a closed pipe, and every later write is
    /// dropped without calling the sink again.
    #[test]
    fn a_failed_write_closes_the_stream() {
        let open = AtomicBool::new(true);
        let mut calls = 0;
        guard(&open, false, || {
            calls += 1;
            Err(io::ErrorKind::BrokenPipe.into())
        });
        guard(&open, false, || {
            calls += 1;
            Ok(())
        });
        assert_eq!(calls, 1);
        assert!(!open.load(Ordering::Relaxed));
    }

    #[test]
    fn a_working_stream_stays_open() {
        let open = AtomicBool::new(true);
        let mut calls = 0;
        for _ in 0..3 {
            guard(&open, false, || {
                calls += 1;
                Ok(())
            });
        }
        assert_eq!(calls, 3);
        assert!(open.load(Ordering::Relaxed));
    }
}
