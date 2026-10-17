//! `consts.rs` — the single source of truth for every number the host Rust code and the WGSL kernels
//! must agree on (workgroup sizes, table capacities, rANS state bounds, descriptor widths, dtype codes,
//! the lossy-log constants).
//!
//! # Why
//! These numbers used to be hand-copied into ~25 WGSL literals (`64u`, `8388608u`, `array<u32, 256>`, ...)
//! next to their Rust twins, so a change on one side silently desynchronised the CPU<->GPU contract.
//! Now each value is a Rust `const` here; [`prelude`] renders them ONCE into WGSL `const` declarations that
//! every shader sees (prepended by [`wgsl`] at pipeline build), and the shader sources use the NAMES.
//! `shaders_use_named_consts` (kernels.rs tests) fails when a bare literal duplicate reappears.
//!
//! # Where used
//! [`wgsl`] is applied by `context::make_pipe`; exv-gpu's compose shader and the viewer take
//! [`prelude`] / [`dtype_wgsl`] / [`WG2`] from here instead of re-declaring numbers.

use crate::prelude::*;
use std::sync::LazyLock;

/// Threads per workgroup of every 1-D compute kernel (`@workgroup_size(WG)`); host dispatches divide by it.
pub const WG: u32 = 64;

/// Edge of the square workgroup of the 2-D texture passes (blit/gather/compose): `@workgroup_size(WG2, WG2)`.
pub const WG2: u32 = 8;

const _: () = assert!(WG2 * WG2 <= WG, "a 2-D display workgroup must fit the 1-D invocation budget checked by from_wgpu");

/// Largest `table_log` the batched GPU rANS kernel can decode. The cleanup kernel stages the
/// symbol-lookup table in `var<workgroup> sh_slot: array<u32, SLOT_MAX>` (= 2^12 entries); a stream
/// with `table_log > 12` has `M = 2^table_log > 4096` and would write past that workgroup array
/// (silent corruption, no error). `rans::parse` accepts `table_log` 8..=15, so a forged/custom
/// `.exv` can carry one — such tiles must fall back to the CPU path, not the GPU kernel. The
/// default encoder uses `table_log = 12`, so normal files are always GPU-eligible. Shared with
/// `exv-gpu`'s dispatch layer (the CPU-fallback gate), which imports it from this crate.
pub const MAX_GPU_TABLE_LOG: u8 = 12;

/// Workgroup `sh_slot` capacity of the batched rANS kernels (`M = 2^MAX_GPU_TABLE_LOG` entries): the shader
/// array size, the host `table_log` gate and [`RANS_WG_BYTES`] are all derived from this ONE value.
pub(crate) const SLOT_MAX: usize = 1 << MAX_GPU_TABLE_LOG;

/// rANS alphabet size (byte symbols): length of the `freq`/`cum` tables (`rans-rs` fixes symbol_bits = 8).
pub const ALPHA: usize = 256;

/// Workgroup storage the batched rANS kernels declare (`sh_slot` + `sh_freq` + `sh_cum`, u32 each) — the
/// minimum `max_compute_workgroup_storage_size` a device needs to build the pipelines (18432 B).
pub const RANS_WG_BYTES: u32 = ((SLOT_MAX + 2 * ALPHA) * 4) as u32;

/// Largest EXV-RT block edge the GPU decodes (`sh_coeff` workgroup scratch holds `RT_MAX_BLOCK^2` i32).
pub const RT_MAX_BLOCK: u32 = 64;

/// Largest block edge of the cooperative/fast EXV-RT inverse. The per-thread `exvrt_inv_fast` scratch is `RT_FAST_BLOCK^2` i32
/// = 1 KiB; the shipped shared inverse holds [`RT_INV_BLOCKS`] padded blocks ([`RT_INV_SH`] i32, 17 KiB).
pub const RT_FAST_BLOCK: u32 = 16;

/// Blocks per workgroup of the cooperative EXV-RT inverse (`exvrt_inv_shared`): one thread per block LINE, so the
/// workgroup has [`RT_INV_WG`] threads (256 = the WebGPU default `max_compute_invocations_per_workgroup`).
pub const RT_INV_BLOCKS: u32 = 16;

/// Threads of the cooperative inverse workgroup: [`RT_INV_BLOCKS`] blocks x [`RT_FAST_BLOCK`] lines.
pub const RT_INV_WG: u32 = RT_INV_BLOCKS * RT_FAST_BLOCK;

/// Padded row stride (i32) of the inverse's workgroup scratch: odd, so threads walking down rows (the row lift:
/// thread stride = row stride) hit distinct banks.
pub const RT_INV_STRIDE: u32 = RT_FAST_BLOCK + 1;

/// i32 of the inverse's workgroup scratch (`sh_blk`): [`RT_INV_BLOCKS`] blocks of [`RT_FAST_BLOCK`] padded rows (17 KiB).
pub const RT_INV_SH: u32 = RT_INV_BLOCKS * RT_FAST_BLOCK * RT_INV_STRIDE;

const _: () = assert!(RT_INV_WG == RT_FAST_BLOCK * RT_FAST_BLOCK, "the inverse regroup/emit gives each thread one coefficient of a block per iteration");

/// Workgroup storage bytes of `exvrt_inv_shared`: `sh_blk` ([`RT_INV_SH`] i32) + the staged `sh_d` ([`RT_INV_BLOCKS`] x
/// [`RT_DESC_WORDS`] u32). `check_device` gates on it, so the limit check cannot drift from the WGSL array sizes.
pub const RT_INV_WG_BYTES: u32 = (RT_INV_SH + RT_INV_BLOCKS * RT_DESC_WORDS as u32) * 4;

/// Largest element size in bytes of any container dtype (`Dtype::U32`/`F32`): the plane count of a reorder+delta
/// tile, i.e. the number of independent serial prefix-sum threads a LosslessFast unit needs (`undelta_b`).
pub const MAX_ESZ: u32 = 4;

/// `u32` words per rANS `Desc` (WGSL `DESC_STRUCT`); hosts build descriptor arrays in this stride.
pub const DESC_WORDS: usize = 16;

/// `u32` words per EXV-RT `RtDesc` (WGSL `RT_DESC_STRUCT`).
pub const RT_DESC_WORDS: usize = 11;

/// `u32` words per EXV-RT per-TILE record (WGSL `TILE_DESC_STRUCT`): what the `rt_prep` kernel expands into the
/// per-block `Desc` + `RtDesc` rows on the GPU instead of the host uploading ~82 MB of them per 8k frame.
pub const TILE_DESC_WORDS: usize = 22;

/// `log2` of the lossy-log fixed scale `exv_codec::hdr::DEFAULT_SCALE` (256 codes per stop): the fractional
/// part of a code indexes the [`EXP2_N`]-entry table. Pinned against `hdr` by `exv-gpu`'s
/// `hdr_consts_match_codec` (this crate may not depend on exv-codec).
pub const HDR_SHIFT: u32 = 8;

/// Entries of the lossy-log decode table (`T[r] = 2^(r/256)`), `= 1 << HDR_SHIFT`.
pub const EXP2_N: usize = 1 << HDR_SHIFT;

/// `hdr::MAX_CODE`: `|code|` saturates here (`EXP_SPAN 64 stops * 256`); a decoded `k` is biased by half of it.
pub const HDR_MAX_CODE: i32 = 64 << HDR_SHIFT;

/// Named dtype codes for WGSL, derived from [`Dtype::code`] (the container's numbering — the ONLY one).
/// A consumer that needs the codes in its own shader (the viewer's display decode) prepends [`dtype_wgsl`]
/// and branches on `DT_*`, never on literals.
const DTYPES: [(&str, Dtype); 6] = [
    ("DT_F16", Dtype::F16),
    ("DT_F32", Dtype::F32),
    ("DT_BF16", Dtype::Bf16),
    ("DT_U8", Dtype::U8),
    ("DT_U16", Dtype::U16),
    ("DT_U32", Dtype::U32),
];

/// WGSL `const DT_<dtype>: u32 = <Dtype::code>u;` declarations (see [`DTYPES`]).
pub fn dtype_wgsl() -> String {
    DTYPES.iter().map(|(n, d)| format!("const {n}: u32 = {}u;\n", d.code())).collect()
}

/// The WGSL constants prelude: every constant of this module as a module-scope `const`. Cached; identical
/// bytes for every shader of the crate.
pub fn prelude() -> &'static str {
    static P: LazyLock<String> = LazyLock::new(|| {
        format!(
            "// --- exv constants prelude (generated from consts.rs; never edit literals in shaders) ---\n\
             const WG: u32 = {WG}u;\n\
             const WG2: u32 = {WG2}u;\n\
             const RANS_L: u32 = {}u;\n\
             const RENORM_GUARD: u32 = {}u;\n\
             const SLOT_MAX: u32 = {SLOT_MAX}u;\n\
             const ALPHA: u32 = {ALPHA}u;\n\
             const MAX_COEFF: u32 = {}u;\n\
             const FAST_COEFF: u32 = {}u;\n\
             const FAST_BLOCK: u32 = {RT_FAST_BLOCK}u;\n\
             const INV_BLOCKS: u32 = {RT_INV_BLOCKS}u;\n\
             const INV_WG: u32 = {RT_INV_WG}u;\n\
             const INV_STRIDE: u32 = {RT_INV_STRIDE}u;\n\
             const INV_SH: u32 = {RT_INV_SH}u;\n\
             const MAX_ESZ: u32 = {MAX_ESZ}u;\n\
             const EXP2_N: u32 = {EXP2_N}u;\n\
             const HDR_SHIFT: u32 = {HDR_SHIFT}u;\n\
             const HDR_MAX_CODE: i32 = {HDR_MAX_CODE};\n\
             const HDR_BIAS: i32 = {};\n\
             const HDR_MASK: i32 = {};\n\
             {}// --- end prelude ---\n",
            rans::RANS_L,
            rans::RENORM_GUARD,
            RT_MAX_BLOCK * RT_MAX_BLOCK,
            RT_FAST_BLOCK * RT_FAST_BLOCK,
            HDR_MAX_CODE / 2,
            EXP2_N - 1,
            dtype_wgsl(),
        )
    });
    &P
}

/// A complete WGSL module source: [`prelude`] + `src`. The ONE place a shader gets its constants.
pub fn wgsl(src: &str) -> String {
    format!("{}{src}", prelude())
}

/// `T[r] = f32(2^(r/256))`, `r in 0..256`, as f32 bit patterns — the decode table of the lossy EXV-RT
/// log transform at the fixed `DEFAULT_SCALE = 256`. **Determinism contract:** the hardware/WGSL `exp2` is
/// only specified to a few ULP (and differs per driver), so the shader never calls it. It does one table
/// fetch + one exact exponent add + one multiply by a power of two, exactly like the CPU
/// `exv_codec::hdr::decode_value`; the table itself is generated with the same fixed-order f64 Horner series
/// (only `+ - * /`, IEEE-exact everywhere) and rounded once to f32, evaluated at compile time. It is a copy
/// of `hdr::exp2_table(256)` (this crate may not depend on exv-codec); `exv-gpu`'s
/// `exp2_table_matches_hdr` test pins the two bit-for-bit.
pub const EXP2_TAB_256: [u32; EXP2_N] = {
    const LN2: f64 = std::f64::consts::LN_2; // == hdr's literal 0.693_147_180_559_945_3 (same f64)
    const TERMS: usize = 18;
    let mut inv = [1.0f64; TERMS + 1]; // 1/k!
    let mut k = 1;
    while k <= TERMS {
        inv[k] = inv[k - 1] / (k as f64);
        k += 1;
    }
    let mut out = [0u32; EXP2_N];
    let mut r = 0;
    while r < EXP2_N {
        let y = (r as f64 / EXP2_N as f64) * LN2;
        let mut acc = inv[TERMS];
        let mut j = TERMS;
        while j > 0 {
            j -= 1;
            acc = acc * y + inv[j];
        }
        out[r] = (acc as f32).to_bits();
        r += 1;
    }
    out
};
