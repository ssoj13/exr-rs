//! Memory budgets: the ONE place that turns "how much RAM/VRAM may this run use" into numbers.
//!
//! # Why
//! Six places used to invent their own formula (cli `--mem`, convert window, container window, viewer cache,
//! C-ABI cache, `DecodeLimits::from_ram`) and the tile window was a tile COUNT, so a 16384^2 F32 tile times 64
//! was 64 GiB. Now:
//! - RAM/VRAM come from the OS through `gpu-info-rs` ONLY ([`usable_ram`], [`BudgetExt::auto`], [`BudgetExt::vram`]);
//!   when the OS cannot answer and the user gave no override the result is [`BudgetError::Unknown`], never an
//!   invented default.
//! - Every "how many may run at once / how many tiles per window" question is [`par_for`] / [`window_tiles`]
//!   over a [`Budget`] and the real byte cost of one unit.

use std::fmt;

use exv_format::DecodeLimits;

/// Smallest accepted budget: below one decoded frame every window is clamped to one unit anyway, so a tiny
/// `--mem` would only claim a bound it cannot keep.
pub const MIN_BYTES: u64 = 64 << 20;
/// Share of the installed RAM the decoded-frame cache may take (the After-Effects "decode once" model).
pub const RAM_SHARE: f64 = 0.5;
/// RAM kept for the OS + GPU driver + the application before decode transients (frame cache formula).
pub const OS_RESERVE: u64 = 2 << 30;
/// Share of the free VRAM a resident path may plan on: the rest stays for the compositor, other applications and
/// driver growth. A unified-memory GPU (Apple Silicon, iGPU: its "VRAM" IS system RAM, which the OS and the CPU
/// decode path need too) gets the smaller share `gpu_info::os::GpuMemInfo::unified` recommends.
pub const VRAM_SHARE: f64 = 0.66;
pub const VRAM_SHARE_UNIFIED: f64 = 0.25;
/// The entropy-decode pool of a resident GPU decoder (peak VRAM of ONE tile decode) is a `1 / POOL_DIV` slice of the
/// planned VRAM, clamped to `[POOL_MIN, POOL_MAX]`: a tile decode needs a working floor, and past the cap more pool
/// adds nothing.
pub const POOL_DIV: u64 = 16;
pub const POOL_MIN: u64 = 64 << 20;
pub const POOL_MAX: u64 = 256 << 20;
/// Raw-tile-sized working sets each rayon thread keeps beside a tile window (HTJ2K per-worker codestream pool,
/// rANS/wavelet scratch, the producer's tile); measured well under it by the `exv mem --stream` gate.
pub const SLACK_SETS: u64 = 8;

/// The memory size could not be determined and no explicit override was given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// `gpu_info::os::sys_mem` failed: pass an explicit budget (`--mem`, viewer `mem_gb`).
    UnknownRam,
    /// `gpu_info::os::query` knows no GPU memory.
    UnknownVram,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnknownRam => "cannot determine system memory; pass an explicit memory limit (--mem <GB>)",
            Self::UnknownVram => "cannot determine GPU memory (neither the driver nor the OS reports it)",
        })
    }
}

impl std::error::Error for BudgetError {}

/// What a resident GPU path may spend of VRAM, in bytes ([`plan_vram`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramPlan {
    /// Resident-tile atlas LRU budget.
    pub atlas: u64,
    /// Entropy-decode pool of one tile decode.
    pub decode: u64,
}

/// [`plan_vram`] could not plan: the GPU memory is unknown, or the fixed targets leave no room for one tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VramError {
    /// Neither the driver nor the OS reports the GPU memory (never an invented default).
    Unknown,
    /// The fixed targets + decode pool exceed the plannable share of the headroom, or leave less than one tile.
    TooSmall {
        /// Targets + decode pool the path must hold regardless of the atlas.
        fixed: u64,
        /// The plannable share of the headroom ([`vram_share`]).
        planned: u64,
        /// The headroom the driver reported.
        headroom: u64,
        /// Unified-memory GPU (its "VRAM" is system RAM).
        unified: bool,
    },
}

impl fmt::Display for VramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("VRAM size unknown (neither the driver nor the OS reports it)"),
            Self::TooSmall { fixed, planned, headroom, unified } => write!(
                f,
                "needs {} MiB of VRAM for its targets + decode pool but only {} MiB of the {} MiB free may be used{}",
                fixed >> 20,
                planned >> 20,
                headroom >> 20,
                if *unified { " (unified memory)" } else { "" },
            ),
        }
    }
}

impl std::error::Error for VramError {}

/// THE VRAM plan of a resident GPU path: [`vram_share`] of the `headroom` the driver reports (budget minus what is
/// committed; `None` = unknown), minus `fixed` bytes the path always holds (full-size targets) and the decode pool
/// (`planned / POOL_DIV` clamped to `POOL_MIN..=POOL_MAX`). What is left is the atlas LRU budget; it must hold at
/// least `one_tile` bytes. (The atlas LRU-evicts only tiles that are neither visible nor loaded this frame, so the
/// visible set always stays resident: the budget bounds the cache of panned-away tiles, and the atlas hard-checks
/// the visible set against it.)
///
/// # Errors
/// [`VramError::Unknown`] without a headroom, [`VramError::TooSmall`] when the atlas would hold less than `one_tile`.
pub fn plan_vram(headroom: Option<u64>, unified: bool, fixed: u64, one_tile: u64) -> Result<VramPlan, VramError> {
    let headroom = headroom.ok_or(VramError::Unknown)?;
    let planned = (headroom as f64 * vram_share(unified)) as u64;
    let decode = (planned / POOL_DIV).clamp(POOL_MIN, POOL_MAX);
    let fixed = fixed.saturating_add(decode);
    let atlas = planned.saturating_sub(fixed);
    if atlas < one_tile {
        return Err(VramError::TooSmall { fixed, planned, headroom, unified });
    }
    Ok(VramPlan { atlas, decode })
}

/// Heap a windowed tile pipeline may hold ABOVE its byte budget: [`SLACK_SETS`] raw-tile working sets per rayon
/// thread (`threads`), the tile index (`index_entry` bytes per tile, twice: parsed entries + the writer's copy) and
/// the writer's 1 MiB buffer. Derived, not tuned: the allowance of the `exv mem --stream` peak gate.
pub fn heap_slack(threads: usize, tile_bytes: u64, tiles: u64, index_entry: u64) -> u64 {
    (threads as u64)
        .saturating_mul(SLACK_SETS)
        .saturating_mul(tile_bytes)
        .saturating_add(tiles.saturating_mul(index_entry).saturating_mul(2))
        .saturating_add(1 << 20)
}

/// The exv host-RAM rule: half of the physical RAM, but never more than what is available right now (a hostile
/// file or a huge job must not push the host into swap). The formula itself is owned by `gpu-info-rs`
/// (`ram_budget_from`, `Policy::Min`, [`RAM_SHARE`], no reserve): this only fixes the exv parameters, so the viewer's
/// Settings > Memory rule and this default cannot drift. Pure so it is testable; the OS query is [`ram`].
pub fn usable_ram(total: u64, avail: u64) -> u64 {
    gpu_info::budget::ram_budget_from(total, avail, gpu_info::budget::Policy::Min, RAM_SHARE, 0)
}

/// Installed + available physical RAM from the OS.
///
/// # Errors
/// [`BudgetError::UnknownRam`] when the OS query fails.
pub fn ram() -> Result<(u64, u64), BudgetError> {
    let m = gpu_info::os::sys_mem().ok_or(BudgetError::UnknownRam)?;
    Ok((m.total_bytes, m.available_bytes))
}

/// The share of VRAM headroom to plan on ([`VRAM_SHARE`] / [`VRAM_SHARE_UNIFIED`]).
pub fn vram_share(unified: bool) -> f64 {
    if unified { VRAM_SHARE_UNIFIED } else { VRAM_SHARE }
}

/// Decoded-frame cache budget for a machine with `total` bytes of RAM and frames of `frame_bytes`: half of the
/// installed RAM, capped so the OS reserve PLUS the in-flight decode transients (`threads` frames being built
/// outside a frame cache that accounts only at insert) stay free. Never below two frames (a smaller cache cannot
/// double-buffer) and never above installed RAM.
pub fn frame_cache(total: u64, frame_bytes: u64, threads: usize) -> u64 {
    let share = (total as f64 * RAM_SHARE) as u64;
    let reserve = OS_RESERVE.saturating_add(frame_bytes.saturating_mul(threads as u64));
    let cap = share.min(total.saturating_sub(reserve)).max(1);
    cap.max(frame_bytes.saturating_mul(2)).min(total)
}

/// Concurrency allowed by `budget` bytes for tasks that each hold `unit` bytes: `budget / unit` clamped to
/// `1..=max` (a unit bigger than the budget still runs, alone).
pub fn par_for(budget: u64, unit: u64, max: usize) -> usize {
    usize::try_from(budget / unit.max(1)).unwrap_or(usize::MAX).clamp(1, max.max(1))
}

/// Tiles per window when each in-flight tile holds `tile_bytes`: `budget / tile_bytes` clamped to `1..=max`.
/// This replaces the old fixed tile-count window (a count ignores what a tile weighs).
pub fn window_tiles(budget: u64, tile_bytes: u64, max: u64) -> u64 {
    (budget / tile_bytes.max(1)).clamp(1, max.max(1))
}

/// Largest single decoded frame a host may hold out of `usable` bytes: half, so at least two fit (double
/// buffering: the frame being shown and the one being decoded).
pub const fn frame_share(usable: u64) -> u64 {
    usable / 2
}

/// Bytes a windowed operation may transiently hold ON TOP of `resident` bytes it already holds (the planes of a
/// whole-frame decode, the input planes of an in-RAM encode): half of them. Parallel windows buy speed with memory;
/// this caps the price relative to what the operation costs anyway, so a small image does not double its peak just
/// to keep many cores busy (`exv-codec/tests/mem_gate.rs` pins the whole-image decode peak at <= 2x raw). A decode of a
/// channel SUBSET passes the bytes of the whole frame as `resident` (lane count is speed and must not follow the subset;
/// the caller's hard cap still applies, see `FrameReader::decode_all_into`).
pub const fn window_spare(resident: u64) -> u64 {
    resident / 2
}

/// What the stage working on a frame keeps beside the decoded frame, as a multiple of its bytes. Measured (not
/// guessed) by `exv-cli/tests/mem_peak.rs`: peak heap / budget stays below 1 with these charges.
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    /// Encode into an `.exv`: decoded planes + encoder scratch + the coded tile bytes of the window.
    Encode,
    /// Per-frame output (`.exr` / display formats): decoded planes + the widened channel / RGBA8 + tonemap copy.
    Write,
}

impl Stage {
    /// Frame-equivalents charged per in-flight frame.
    pub const fn cost(self) -> u64 {
        match self {
            Stage::Encode => 5,
            Stage::Write => 3,
        }
    }
}

/// The shared byte cap (`bufpool::Budget`: clones SHARE it, [`Budget::set`] resizes every holder live), so a pool, a
/// window planner and the UI see the same number.
pub use exv_format::bufpool::Budget;

/// The exv memory policy on a [`Budget`]: the OS-derived constructors and the stage/decode-limit views. An extension
/// trait because the handle itself lives in `bufpool` (std-only, no OS queries, no exv types); bring it into scope
/// (`use exv_tile::BudgetExt`) to call `Budget::auto(..)`, `budget.limits()`, `budget.share(..)` and friends.
pub trait BudgetExt: Sized {
    /// The explicit `bytes` override, else [`usable_ram`] of this machine; floor [`MIN_BYTES`].
    ///
    /// # Errors
    /// [`BudgetError::UnknownRam`] when there is no override and the OS cannot answer.
    fn auto(over: Option<u64>) -> Result<Self, BudgetError>;

    /// VRAM budget from the OS-level query (no GPU context): the free (else dedicated) VRAM times
    /// [`vram_share`]. A live per-device number (DXGI budget) is the GPU crate's job and feeds `Budget::new`.
    ///
    /// # Errors
    /// [`BudgetError::UnknownVram`] when the OS reports no GPU memory.
    fn vram() -> Result<Self, BudgetError>;

    /// The cap as the decode ceiling a `FrameReader`/`VideoIndex` enforces.
    fn limits(&self) -> DecodeLimits;

    /// Bytes of DECODED frames that may be in flight for `stage` once `resident` bytes are held for the whole run
    /// (e.g. a cached frame 0). The `budget_bytes` the windowed writers/readers of `exv-video` take.
    fn share(&self, resident: u64, stage: Stage) -> u64;

    /// Frames that may be worked on in parallel (a thread-pool bound, >= 1) for `stage`.
    fn tasks(&self, frame_bytes: u64, stage: Stage) -> usize;

    /// Refuse a frame of `bytes` that cannot fit the budget once `stage` has charged its working copies (`what`
    /// names it): a typed error up front instead of an OOM kill minutes later.
    ///
    /// # Errors
    /// A human-readable message naming the need and the budget.
    fn fits(&self, what: &str, bytes: u64, stage: Stage) -> Result<(), String>;
}

impl BudgetExt for Budget {
    fn auto(over: Option<u64>) -> Result<Self, BudgetError> {
        let bytes = match over {
            Some(b) => b,
            None => {
                let (total, avail) = ram()?;
                usable_ram(total, avail)
            }
        };
        Ok(Self::new(bytes.max(MIN_BYTES)))
    }

    fn vram() -> Result<Self, BudgetError> {
        let g = gpu_info::os::query().ok_or(BudgetError::UnknownVram)?;
        let headroom = if g.free_vram > 0 { g.free_vram } else { g.dedicated_vram };
        if headroom == 0 {
            return Err(BudgetError::UnknownVram);
        }
        Ok(Self::new((headroom as f64 * vram_share(g.unified)) as u64))
    }

    fn limits(&self) -> DecodeLimits {
        DecodeLimits::new(self.cap())
    }

    fn share(&self, resident: u64, stage: Stage) -> u64 {
        self.cap().saturating_sub(resident) / stage.cost()
    }

    fn tasks(&self, frame_bytes: u64, stage: Stage) -> usize {
        par_for(self.share(0, stage), frame_bytes, usize::MAX)
    }

    fn fits(&self, what: &str, bytes: u64, stage: Stage) -> Result<(), String> {
        let need = bytes.saturating_mul(stage.cost());
        if need > self.cap() {
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            return Err(format!(
                "{what} needs {:.1} GiB, over the {:.1} GiB memory budget (raise --mem or use a smaller size)",
                gib(need),
                gib(self.cap())
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn usable_is_half_ram_capped_by_available() {
        assert_eq!(usable_ram(16 * GIB, 12 * GIB), 8 * GIB);
        assert_eq!(usable_ram(16 * GIB, 3 * GIB), 3 * GIB);
        assert_eq!(usable_ram(0, 5), 0);
        // Odd totals floor like the integer `total / 2`; the rule is the gpu-info one, not a copy.
        assert_eq!(usable_ram(5, 100), 2);    }

    #[test]
    fn override_wins_with_floor_and_os_query_answers() {
        assert_eq!(Budget::auto(Some(2 * GIB)).map(|b| b.cap()), Ok(2 * GIB));
        assert_eq!(Budget::auto(Some(1)).map(|b| b.cap()), Ok(MIN_BYTES));
        let (total, avail) = ram().expect("OS memory query");
        let b = Budget::auto(None).expect("auto");
        assert!(b.cap() <= (total / 2).max(MIN_BYTES) && b.cap() <= avail.max(MIN_BYTES));
    }

    #[test]
    fn frame_cache_is_half_ram_minus_reserve_and_transients() {
        // 64 GiB box, 100 MiB frames: half-RAM (32) wins over (64 - 2 - 8*0.1).
        assert_eq!(frame_cache(64 * GIB, 100 << 20, 8), 32 * GIB);
        // 16 GiB box, 1 GiB frames, 8 threads: reserve = 2 + 8 = 10 GiB -> 6 GiB < half (8).
        assert_eq!(frame_cache(16 * GIB, GIB, 8), 6 * GIB);
    }

    #[test]
    fn frame_cache_holds_two_frames_but_never_exceeds_ram() {
        assert_eq!(frame_cache(2 * GIB, GIB, 8), 2 * GIB);
    }

    #[test]
    fn par_and_window_are_byte_based_and_clamped() {
        assert_eq!(par_for(10 * GIB, GIB, 4), 4);
        assert_eq!(par_for(GIB, 4 * GIB, 8), 1, "a unit over the budget still runs, alone");
        assert_eq!(par_for(0, 0, 0), 1);
        // The old ENCODE_WINDOW=64 would hold 64 x 1 GiB tiles; the byte window holds what the budget allows.
        assert_eq!(window_tiles(4 * GIB, GIB, 64), 4);
        assert_eq!(window_tiles(GIB, 256 * 256 * 16, 64), 64);
    }

    #[test]
    fn window_spare_is_half_the_resident_bytes() {
        assert_eq!(frame_share(GIB), GIB / 2);
        assert_eq!(window_spare(GIB), GIB / 2);
        assert_eq!(window_spare(1), 0);
    }

    #[test]
    fn stage_charges_and_fits() {
        let b = Budget::new(GIB);
        assert_eq!(b.share(0, Stage::Write), GIB / Stage::Write.cost());
        assert_eq!(b.share(GIB / 2, Stage::Encode), (GIB / 2) / Stage::Encode.cost());
        assert_eq!(b.share(2 * GIB, Stage::Encode), 0, "resident above the budget saturates");
        assert_eq!(b.tasks(1 << 40, Stage::Write), 1);
        assert!(b.fits("x", 1 << 20, Stage::Encode).is_ok());
        assert!(b.fits("x", GIB / 2, Stage::Encode).is_err(), "the stage cost counts");
    }

    #[test]
    fn vram_share_is_smaller_when_unified() {
        assert!(vram_share(true) < vram_share(false));
    }

    /// 12 GiB budget, 2 GiB committed: the discrete share of the 10 GiB headroom minus targets and the (capped) pool.
    #[test]
    fn plan_is_a_share_of_the_headroom_after_targets_and_pool() {
        let fixed = 8192u64 * 8192 * 24;
        let got = plan_vram(Some(10 * GIB), false, fixed, 256 * 256 * 16).expect("fits");
        assert_eq!(got.decode, POOL_MAX);
        assert_eq!(got.atlas, ((10 * GIB) as f64 * VRAM_SHARE) as u64 - fixed - POOL_MAX);
        // A unified-memory GPU plans on the smaller share.
        let d = plan_vram(Some(16 * GIB), false, 0, 1).expect("fits");
        let u = plan_vram(Some(16 * GIB), true, 0, 1).expect("fits");
        assert!(u.atlas < d.atlas);
    }

    /// The pool shrinks with a tiny budget but never below its floor; unknown / too-small VRAM is a typed error.
    #[test]
    fn plan_clamps_the_pool_and_refuses_unknown_or_too_small() {
        let p = plan_vram(Some(GIB), false, 0, 1).expect("fits");
        assert!((POOL_MIN..=POOL_MAX).contains(&p.decode));
        assert_eq!(plan_vram(None, false, 0, 1), Err(VramError::Unknown));
        // 16k x 16k targets alone are 6 GiB: a 4 GiB budget cannot hold them.
        let err = plan_vram(Some(4 * GIB), false, 16384u64 * 16384 * 24, 1).unwrap_err();
        assert!(matches!(err, VramError::TooSmall { unified: false, .. }) && err.to_string().contains("MiB"), "{err}");
        // Targets fit but not even one tile of atlas is left.
        let fixed = 1024u64 * 1024 * 24;
        let headroom = ((fixed + POOL_MIN + 1) as f64 / VRAM_SHARE) as u64;
        assert!(plan_vram(Some(headroom), false, fixed, 256 * 256 * 16).is_err());
    }

    #[test]
    fn heap_slack_counts_sets_per_thread_and_the_tile_index() {
        assert_eq!(heap_slack(4, 100, 10, 13), 4 * SLACK_SETS * 100 + 10 * 13 * 2 + (1 << 20));
    }
}
