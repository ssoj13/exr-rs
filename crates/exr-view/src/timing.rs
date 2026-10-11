//! Frame-switch timing at the display seams, logged as `scrub-timing:` debug
//! lines (`RUST_LOG=exr_view=debug`). A sequence step's cost is spread over
//! materialize, the resident upload, the reduce dispatch, the scopes' upload and
//! the UI frame; one stable prefix lets a log of a scrub be filtered and summed.
//! Only seams that did work log, so an idle or unchanged frame prints nothing.
use std::time::Instant;

/// A started measurement; logs `scrub-timing: <what> <ms> ms [<MB> MB]` on drop.
pub(crate) struct Lap {
    what: &'static str,
    start: Instant,
    bytes: u64,
}

impl Lap {
    pub(crate) fn start(what: &'static str) -> Self {
        Self { what, start: Instant::now(), bytes: 0 }
    }

    /// Record how many bytes the measured step touched.
    pub(crate) fn bytes(&mut self, bytes: u64) {
        self.bytes = bytes;
    }

    /// Milliseconds since the start, without ending the lap.
    pub(crate) fn ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }
}

// Laps finished on this thread, for the scrub bench (`tiles::scrub_bench`) to
// attribute one frame switch to its stages without re-implementing the path.
#[cfg(test)]
thread_local! {
    static DONE: std::cell::RefCell<Vec<(&'static str, f64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Take the `(what, ms, bytes)` laps finished on this thread since the last call.
#[cfg(test)]
pub(crate) fn take() -> Vec<(&'static str, f64, u64)> {
    DONE.with(|done| std::mem::take(&mut *done.borrow_mut()))
}

impl Drop for Lap {
    fn drop(&mut self) {
        #[cfg(test)]
        DONE.with(|done| done.borrow_mut().push((self.what, self.ms(), self.bytes)));
        if self.bytes == 0 {
            log::debug!("scrub-timing: {} {:.2} ms", self.what, self.ms());
        } else {
            log::debug!(
                "scrub-timing: {} {:.2} ms {:.1} MB",
                self.what,
                self.ms(),
                self.bytes as f64 / 1e6
            );
        }
    }
}
