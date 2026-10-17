//! `bufpool`: budgeted memory for out-of-core pipelines (images, video, tiles: anything that decodes more data
//! than fits in RAM and must never OOM).
//!
//! | Module | What | Why |
//! |---|---|---|
//! | [`pool`] | [`TilePool`]: byte-budget LRU, [`Lease`] pins, single-flight loads, reserve-before-decode, prefetch | a HARD memory bound under N concurrent decoders |
//! | [`planes`] | [`PlanePool`] / [`PlaneBuf`]: recycler of whole-plane buffers matched by exact length | fresh memory costs its first touch (page faults); reuse does not |
//! | [`aligned`] | [`AlignedBuf`]: 64-byte (cache line) aligned byte buffer, no `unsafe` | no false sharing between decode lanes, no split SIMD accesses |
//! | [`budget`] | [`Budget`]: a shared, live-resizable byte cap | pool, window planner and UI see one number |
//! | [`limits`] | [`alloc_zeroed`], [`limits::cap`], [`AllocError`] | sizes read from untrusted files must fail, never abort |
//! | [`track`] | [`TrackingAlloc`]: counting `#[global_allocator]` wrapper | an exact, sampling-free peak-heap signal for OOM gates |
//!
//! Dependencies: `std` + `bytemuck` (+ optional `rayon`, feature `rayon`). The only `unsafe` is the
//! `GlobalAlloc` impl in [`track`] (the trait requires it); every other module is `#![forbid(unsafe_code)]`.

pub mod aligned;
pub mod budget;
pub mod limits;
pub mod planes;
pub mod pool;
pub mod track;

pub use aligned::{AlignedBuf, LINE};
pub use budget::Budget;
pub use limits::{AllocError, alloc_zeroed};
pub use planes::{PlaneBuf, PlanePool, PlaneStats};
pub use pool::{Cancel, Job, Lease, Loader, PoolErr, PoolError, PoolStats, TilePool, Weigh};
pub use track::TrackingAlloc;
