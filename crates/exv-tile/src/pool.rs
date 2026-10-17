//! The tile engine's view of `bufpool`: [`TileKey`] (the cache-key convention of tile payloads) and the pool types bound
//! to [`TileError`].
//!
//! # Why a thin adapter
//! The byte-budget LRU ([`TilePool`]: `Lease` pins, single-flight loads, reserve-before-decode, prefetch, cancel) and the
//! plane recycler ([`PlanePool`]/[`PlaneBuf`]) live in the shared crate `bufpool` (ssoj13/bufpool-rs) so other repos
//! reuse them; their contract and tests are there. This module only fixes the error type to [`TileError`] (via
//! `bufpool::PoolErr`, implemented in `source.rs`) so every exv host keeps `exv_tile::{TilePool, Loader}` and loaders
//! keep returning [`crate::TileResult`]. bufpool's feature `rayon` is ON (exv-tile `Cargo.toml`): lookups run on rayon
//! workers, which must never block on another thread's flight.

pub use exv_format::bufpool::{Cancel, Job, Lease, PlaneBuf, PlanePool, PlaneStats, PoolStats};

use crate::source::TileError;

/// [`exv_format::bufpool::TilePool`] whose loaders return [`TileError`]: the RAM / VRAM payload cache of every exv host.
pub type TilePool<K, V> = exv_format::bufpool::TilePool<K, V, TileError>;

/// A prefetch loader of a [`TilePool`] ([`exv_format::bufpool::Loader`] bound to [`TileError`]).
pub type Loader<K, V> = exv_format::bufpool::Loader<K, V, TileError>;

/// Cache key of one decoded unit. `tx`/`ty` (not a flat tile index) because `tiles_x*tiles_y` overflows u32 at
/// 65536^2 with 1-px tiles. The pool is generic over its key; this is the convention for tile payloads.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct TileKey {
    pub frame: u32,
    pub level: u32,
    /// Channel, or [`TileKey::ALL`] for the whole tile (the HTJ2K unit).
    pub chan: u32,
    pub tx: u32,
    pub ty: u32,
}

impl TileKey {
    pub const ALL: u32 = u32::MAX;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Budget, TileBuf, TileResult};

    /// The exv binding: a loader's `TileError` passes through untouched, the pool's own failures arrive as the matching
    /// `TileError` variants, and `cancel.check()?` works inside a `TileResult` loader.
    #[test]
    fn tile_error_round_trips_through_the_pool() {
        let p: TilePool<TileKey, TileBuf> = TilePool::new(Budget::new(100));
        let k = TileKey { frame: 0, level: 0, chan: TileKey::ALL, tx: 0, ty: 0 };
        let io = p.get_or_load(k, 10, &Cancel::new(), |_| Err(TileError::Io("boom".into())));
        assert_eq!(io.err(), Some(TileError::Io("boom".into())));
        let over = p.get_or_load(k, 101, &Cancel::new(), |_| Ok(TileBuf::default()));
        assert!(matches!(over, Err(TileError::OverBudget { need: 101, cap: 100, .. })));
        let c = Cancel::new();
        let cancelled = p.get_or_load(k, 10, &c, |c| -> TileResult<TileBuf> {
            c.cancel();
            c.check()?;
            Ok(TileBuf::default())
        });
        assert_eq!(cancelled.err(), Some(TileError::Cancelled));
        assert_eq!(p.used(), 0);
    }
}
