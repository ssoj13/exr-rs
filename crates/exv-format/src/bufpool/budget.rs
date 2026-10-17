//! [`Budget`]: the shared byte cap every pool and window planner charges against.
//!
//! Only the HANDLE lives here. How many bytes a machine may spend (OS RAM/VRAM queries, shares, reserves) is policy
//! that belongs to the host (e.g. `gpu-info-rs::budget`), so this crate stays std-only and OS-agnostic.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A byte budget. Clones SHARE the cap (live-resizable by [`Budget::set`]), so a [`crate::TilePool`], a
/// [`crate::PlanePool`], a window planner and the UI all see the same number.
///
/// Used by: [`crate::TilePool::new`] (resident + leased + in-flight bytes), [`crate::PlanePool::new`] (idle bytes).
#[derive(Clone, Debug)]
pub struct Budget(Arc<AtomicU64>);

impl Budget {
    /// A budget of `bytes`.
    pub fn new(bytes: u64) -> Self {
        Self(Arc::new(AtomicU64::new(bytes)))
    }

    /// Current cap in bytes.
    pub fn cap(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Resize live; every clone sees it. A pool applies a shrink at its next reservation / give-back (it never evicts
    /// asynchronously).
    pub fn set(&self, bytes: u64) {
        self.0.store(bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_cap() {
        let a = Budget::new(10);
        let b = a.clone();
        a.set(99);
        assert_eq!(b.cap(), 99);
    }
}
