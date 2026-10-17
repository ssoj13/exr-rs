//! `kernels.rs` — raw WGSL tile/block decoders and their `const SHADER` source strings.
//!
//! Holds the GPU compute kernels (rANS expand, repack, inverse 5/3 wavelet, EXV-RT block codec,
//! reorder/undelta) as WGSL string consts plus the Rust host wrappers that build buffers, dispatch,
//! and (for the `*_vram` variants) leave the result resident in VRAM: [`decode_block_gpu`],
//! [`inverse_wavelet_gpu`], [`decode_tile_gpu_wave`]/`_vram`, [`decode_tile_exvrt_vram`],
//! [`prepare_tile_exvrt_vram`]/[`ExvrtWarmTile`], [`decode_tile_gpu_fast`]/`_vram`. NO jph, NO atlas.
//! Shared device/buffer helpers live in [`crate::context`] (a submodule of this crate root).
//!
//! This file is the **lib root** of the `exv-gpu-kernels` crate (`[lib] path = "src/kernels.rs"`): it
//! declares the `context` submodule and the crate-internal `prelude`, and re-exports the whole context
//! surface so the pure-decode API is available at the crate root. Extracted from `exv-gpu` unchanged.

mod consts;
mod context;
mod rt;
pub use consts::*;
pub use context::*;
pub use rt::*;

/// Crate-internal prelude: re-exported so each module can `use crate::prelude::*` (a glob, hence
/// warning-exempt) instead of repeating these imports. Mirrors exv-gpu's prelude (kernels-only subset).
pub(crate) mod prelude {
    pub(crate) use gpu_info::wgpu;
    pub(crate) use exv_format::{BlobKind, Dtype, bufpool, classify_blob};
    pub(crate) use rans::Prepared;
}

use crate::prelude::*;

/// The rANS stream-reading primitives shared by EVERY rANS kernel (single-stream, batched, EXV-RT thread-dense,
/// fused), spliced in for the `//@RANS_COMMON@` token. The including module declares the `streams` (u32 words,
/// LE bytes) and `lane_off` bindings.
///
/// **Lane bounds (CPU==GPU contract):** `lane_off` is `[start0, end0, start1, end1, ...]` — ABSOLUTE byte
/// offsets into `streams`, written by [`push_lane`]. Every byte read is bounds-checked against its lane's `end`
/// and yields 0 at/after it, exactly like the CPU `byte_coder::Decoder` (which zero-feeds past the slice). A
/// stream that over-reads therefore decodes to the SAME deterministic bytes on both sides and can never see the
/// neighbouring lane's bytes. What the GPU does NOT replicate is the CPU's post-decode `Decoder::done()`
/// integrity verdict (state == RANS_L && pos == len): output of such a stream is defined but not rejected.
/// `RENORM_GUARD` mirrors `rans::byte_coder::RENORM_GUARD` (4) — the single CPU<->WGSL renorm bound.
pub(crate) const RANS_COMMON: &str = r#"
fn lane_start(i: u32) -> u32 { return lane_off[2u * i]; }
fn lane_end(i: u32) -> u32 { return lane_off[2u * i + 1u]; }
fn rd_byte(b: u32, end: u32) -> u32 {
    if (b >= end) { return 0u; }
    let w = streams[b >> 2u];
    return (w >> ((b & 3u) * 8u)) & 0xffu;
}
fn rd_state(p: u32, end: u32) -> u32 {
    return rd_byte(p, end) | (rd_byte(p + 1u, end) << 8u) | (rd_byte(p + 2u, end) << 16u) | (rd_byte(p + 3u, end) << 24u);
}
// Renormalise `x` by pulling bytes from `pos` (at most RENORM_GUARD of them: valid data needs <= 3, and a
// corrupt state of 0 would otherwise loop forever = GPU hang). Returns (state, pos).
fn renorm(x_in: u32, pos_in: u32, end: u32) -> vec2<u32> {
    var x = x_in;
    var pos = pos_in;
    var guard = 0u;
    loop {
        if (x >= RANS_L || guard >= RENORM_GUARD) { break; }
        x = (x << 8u) | rd_byte(pos, end);
        pos = pos + 1u;
        guard = guard + 1u;
    }
    return vec2<u32>(x, pos);
}
"#;

/// Append one lane's `[start, end)` byte span to a `lane_off` buffer in the layout [`RANS_COMMON`] reads
/// (the ONE writer of that layout — single-stream, wave batch and EXV-RT hosts all use it).
pub fn push_lane(lane_off: &mut Vec<u32>, start: usize, end: usize) -> GpuResult<()> {
    let w = |v: usize| u32::try_from(v).map_err(|_| GpuError::Limit(format!("lane offset {v} exceeds the u32 shader range")));
    lane_off.push(w(start)?);
    lane_off.push(w(end)?);
    Ok(())
}

/// Signed-i32 inverse 5/3 lift of ONE line (`base` first index, `stride` element step, `n` length) as WGSL,
/// transliterated bit-for-bit from `exv_codec::lossy::lift_inv`: update (even i) `a[i] -= (l + r + 2) >> 2`, then
/// predict (odd i) `a[i] += (l + r) >> 1`, with the edge reflection chosen by `select` BEFORE the load (the
/// `i - 1u` wrap at i==0 is never used as an address); i32 `>>` is arithmetic per the WGSL spec.
///
/// WGSL has no generics over storage classes, so ONE template is instantiated per scratch array: `name` is the fn,
/// `params` extra leading parameters (e.g. a `ptr<function, ..>`), `arr` the indexable array expression.
fn lift_inv_src(name: &str, params: &str, arr: &str) -> String {
    format!(
        r#"
fn {name}({params}base: u32, stride: u32, n: u32) {{
    if (n < 2u) {{ return; }}
    var i = 0u;
    loop {{
        if (i >= n) {{ break; }}
        let im1 = select(1u, i - 1u, i >= 1u);
        let ip1 = select(i - 1u, i + 1u, i + 1u < n);
        let l = {arr}[base + im1 * stride];
        let r = {arr}[base + ip1 * stride];
        {arr}[base + i * stride] = {arr}[base + i * stride] - ((l + r + 2) >> 2u);
        i = i + 2u;
    }}
    i = 1u;
    loop {{
        if (i >= n) {{ break; }}
        let ip1 = select(i - 1u, i + 1u, i + 1u < n);
        let l = {arr}[base + (i - 1u) * stride];
        let r = {arr}[base + ip1 * stride];
        {arr}[base + i * stride] = {arr}[base + i * stride] + ((l + r) >> 1u);
        i = i + 2u;
    }}
}}
"#
    )
}

/// Unsigned (wrapping, masked to the element width) inverse 5/3 lift on the global `buf`, shared by the
/// single-tile [`WAVELET_SHADER`] and the batched [`WAVELET_B_BODY`] (spliced in for `//@WAVE_LIFT@`; both
/// declare `buf`). Mirrors `exv_codec::wavelet::lift_inv`; correct for esz<=2 (`l + r` cannot overflow u32).
const WAVE_LIFT: &str = r#"
fn mask_of(esz: u32) -> u32 { if (esz >= 4u) { return 0xffffffffu; } return (1u << (8u * esz)) - 1u; }
fn lift_inv_line(base: u32, stride: u32, n: u32, mask: u32) {
    if (n < 2u) { return; }
    // undo update (even)
    var j = 0u;
    loop {
        if (j >= n) { break; }
        let jm1 = select(1u, j - 1u, j >= 1u);
        let jp1 = select(j - 1u, j + 1u, j + 1u < n);
        let t = (buf[base + jm1 * stride] + buf[base + jp1 * stride] + 2u) >> 2u;
        buf[base + j * stride] = (buf[base + j * stride] - t) & mask;
        j = j + 2u;
    }
    // undo predict (odd)
    j = 1u;
    loop {
        if (j >= n) { break; }
        let jp1 = select(j - 1u, j + 1u, j + 1u < n);
        let t = (buf[base + (j - 1u) * stride] + buf[base + jp1 * stride]) >> 1u;
        buf[base + j * stride] = (buf[base + j * stride] + t) & mask;
        j = j + 2u;
    }
}
"#;

/// [`WAVELET_B_BODY`] with the shared lift spliced in.
pub(crate) fn wavelet_b_src() -> String {
    WAVELET_B_BODY.replace("//@WAVE_LIFT@", WAVE_LIFT)
}

/// [`WAVELET_SHADER`] with the shared lift spliced in.
pub(crate) fn wavelet_src() -> String {
    WAVELET_SHADER.replace("//@WAVE_LIFT@", WAVE_LIFT)
}

/// The single-stream rANS kernel [`SHADER`] with [`RANS_COMMON`] spliced in.
pub(crate) fn rans_src() -> String {
    SHADER.replace("//@RANS_COMMON@", RANS_COMMON)
}

/// WGSL interleaved rANS decoder. Replicates the CPU steps 1:1 (`rans::byte_coder`):
/// `slot = state & mask; sym = slot[slot]; state = f*(state>>tl) + (state&mask) - cum; renorm`.
/// All operations are `u32`, no division in the hot loop (only shift/mask/multiply).
pub(crate) const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       params:   array<u32>;   // [lanes, table_log, n]
@group(0) @binding(1) var<storage, read>       freq:     array<u32>;   // [256]
@group(0) @binding(2) var<storage, read>       cum:      array<u32>;   // [256]
@group(0) @binding(3) var<storage, read>       slot:     array<u32>;   // [M]
@group(0) @binding(4) var<storage, read>       streams:  array<u32>;   // packed bytes (LE)
@group(0) @binding(5) var<storage, read>       lane_off: array<u32>;   // [2*lanes] [start,end) byte span per lane
@group(0) @binding(6) var<storage, read_write> outp:     array<u32>;   // [n] decoded symbols
//@RANS_COMMON@

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lanes = params[0];
    let table_log = params[1];
    let n = params[2];
    let k = gid.x;
    if (k >= lanes) { return; }

    let mask = (1u << table_log) - 1u;
    let end = lane_end(k);
    var pos = lane_start(k);
    var state = rd_state(pos, end); // init state: 4 bytes LE
    pos = pos + 4u;

    var cnt = 0u;
    if (n > k) { cnt = (n - k + lanes - 1u) / lanes; } // ceil((n-k)/lanes)

    for (var j = 0u; j < cnt; j = j + 1u) {
        let s = slot[state & mask];
        let r = renorm(freq[s] * (state >> table_log) + (state & mask) - cum[s], pos, end);
        state = r.x;
        pos = r.y;
        outp[k + j * lanes] = s;
    }
}
"#;

/// The six read-only rANS inputs of ONE stream, uploaded once (params, tables, concatenated lane streams,
/// lane start offsets). ONE upload codepath for every single-stream rANS user ([`decode_block_gpu`],
/// the wave and fast tile decoders) — it used to be copy-pasted per decoder.
pub(crate) struct RansBufs {
    params: wgpu::Buffer,
    freq: wgpu::Buffer,
    cum: wgpu::Buffer,
    slot: wgpu::Buffer,
    streams: wgpu::Buffer,
    off: wgpu::Buffer,
}

impl RansBufs {
    /// Upload `prep` (a `rans::parse`d/prepared stream). `tag` labels the buffers. The lane streams are
    /// concatenated and zero-padded to a `u32` boundary; `off[k]` = the byte start of lane `k`.
    pub(crate) fn new(ctx: &GpuContext, tag: &str, prep: &Prepared) -> GpuResult<Self> {
        let n_u32 = u32::try_from(prep.n).map_err(|_| GpuError::Limit(format!("{tag}: {} symbols exceed u32", prep.n)))?;
        let mut stream_bytes: Vec<u8> = Vec::new();
        let mut lane_off: Vec<u32> = Vec::with_capacity(prep.lane_streams.len() * 2);
        for st in &prep.lane_streams {
            let start = stream_bytes.len();
            stream_bytes.extend_from_slice(st);
            push_lane(&mut lane_off, start, stream_bytes.len())?;
        }
        stream_bytes.resize(stream_bytes.len().next_multiple_of(4), 0);
        ctx.check_buf(&format!("{tag}.streams"), stream_bytes.len() as u64)?;
        let streams = words_of(&stream_bytes);
        let params = [prep.cfg.lanes as u32, prep.cfg.table_log as u32, n_u32];
        let freq: Vec<u32> = prep.freq.iter().map(|&f| f as u32).collect();
        let cum: Vec<u32> = prep.cum.iter().map(|&c| c as u32).collect();
        let slot: Vec<u32> = prep.slot_to_sym.iter().map(|&s| s as u32).collect();
        let up = |name: &str, data: &[u32]| {
            storage_u32(&ctx.device, &ctx.queue, &format!("{tag}.{name}"), data, wgpu::BufferUsages::empty())
        };
        Ok(Self {
            params: up("params", &params),
            freq: up("freq", &freq),
            cum: up("cum", &cum),
            slot: up("slot", &slot),
            streams: up("streams", &streams),
            off: up("lane_off", &lane_off),
        })
    }

    /// Bind group of the single-stream rANS kernel [`SHADER`] writing byte-symbols into `out`.
    pub(crate) fn bind(&self, ctx: &GpuContext, out: &wgpu::Buffer) -> wgpu::BindGroup {
        let bufs = [&self.params, &self.freq, &self.cum, &self.slot, &self.streams, &self.off, out];
        bind_bufs(&ctx.device, &ctx.pipelines.rans, "rans.bind", &bufs)
    }
}

/// Decode a single rANS block on the GPU and **read the result back** (test-only verification).
///
/// In the format's hot path no readback is done — the result stays in VRAM. Here readback is needed
/// only to compare byte-for-byte against the CPU reference (`rans::decode`). Returns the decoded
/// symbols (length `prep.n`) or an error.
pub fn decode_block_gpu(ctx: &GpuContext, prep: &Prepared) -> GpuResult<Vec<u8>> {
    if prep.n == 0 {
        return Ok(Vec::new());
    }
    let out_size = (prep.n as u64) * 4;
    ctx.check_buf("rans.out", out_size)?;
    ctx.scoped(|| {
        let bufs = RansBufs::new(ctx, "rans", prep)?;
        let b_out = new_buf(&ctx.device, "rans.out", out_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let bind = bufs.bind(ctx, &b_out);
        let mut enc = encoder(&ctx.device, "rans.encoder");
        pass(ctx, &mut enc, "rans.pass", &ctx.pipelines.rans, &bind, [(prep.cfg.lanes as u32).div_ceil(WG), 1, 1])?;
        // --- readback (TEST-ONLY verification; not part of the format) ---
        let rb = staging(&ctx.device, "rans.readback", out_size);
        enc.copy_buffer_to_buffer(&b_out, 0, &rb, 0, out_size);
        ctx.queue.submit([enc.finish()]);
        let words = words_of(&map_read_blocking(&ctx.device, &rb)?);
        Ok(words.iter().map(|&w| (w & 0xff) as u8).collect())
    })
}

// === Batched (descriptor-grid) wave decode: ONE workgroup per unit, ALL units in ONE dispatch per
// stage. Removes the per-unit buffers/submits that the measurement showed dominate. The host
// concatenates every unit's rANS data into shared buffers + a `Desc` array (std430: 16 u32 = 64 B).
// `lane_off` holds ABSOLUTE byte offsets into the shared `streams` buffer (built host-side as
// stream_base[u] + local_lane_off — the single biggest correctness hazard). The final stage writes
// straight into the combined composite at the unit's tile origin (no scatter). ===
// `unit_id`: the batched kernels dispatch ONE workgroup per unit on a 2-D grid (`GpuContext::split_1d`) so more
// units than `max_compute_workgroups_per_dimension` still fit; the linear unit index is rebuilt from the grid.
pub(crate) const DESC_STRUCT: &str = "struct Desc { lanes:u32, table_log:u32, n:u32, lane_off_base:u32, freq_base:u32, cum_base:u32, slot_base:u32, sym_base:u32, esz:u32, count:u32, tw:u32, th:u32, full_w:u32, x0:u32, y0:u32, out_base:u32, };\nfn unit_id(wid: vec3<u32>, nwg: vec3<u32>) -> u32 { return wid.y * nwg.x + wid.x; }
";

pub(crate) const RANS_B_BODY: &str = r#"
@group(0) @binding(0) var<storage, read>       descs:    array<Desc>;
@group(0) @binding(1) var<storage, read>       freq:     array<u32>;
@group(0) @binding(2) var<storage, read>       cum:      array<u32>;
@group(0) @binding(3) var<storage, read>       slot:     array<u32>;
@group(0) @binding(4) var<storage, read>       streams:  array<u32>;
@group(0) @binding(5) var<storage, read>       lane_off: array<u32>;
@group(0) @binding(6) var<storage, read_write> outp:     array<u32>;
//@RANS_COMMON@
// Per-workgroup (per-unit) rANS tables in shared memory: the decode inner loop does 3 table lookups per
// symbol (slot/freq/cum); staging them once converts those from global to shared loads — the #1 perf
// lever for the latency-bound rANS. slot ≤ M = 2^table_log ≤ SLOT_MAX (16 KiB) + freq/cum 256 each → 18 KiB.
// `SLOT_MAX` is substituted by `rans_b_src` from the Rust constant (single source with the host gate).
var<workgroup> sh_slot: array<u32, SLOT_MAX>;
var<workgroup> sh_freq: array<u32, ALPHA>;
var<workgroup> sh_cum:  array<u32, ALPHA>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let u = unit_id(wid, nwg);
    if (u >= arrayLength(&descs)) { return; } // workgroup-uniform (grid overshoot), before the barrier
    let d = descs[u];
    let m = 1u << d.table_log; // M (≤ SLOT_MAX)
    // Cooperative load of this unit's tables — ALL 64 threads participate (the barrier must be uniform,
    // so this runs BEFORE the lane early-return).
    var i = lid.x;
    loop { if (i >= m) { break; } sh_slot[i] = slot[d.slot_base + i]; i = i + WG; }
    i = lid.x;
    loop { if (i >= ALPHA) { break; } sh_freq[i] = freq[d.freq_base + i]; sh_cum[i] = cum[d.cum_base + i]; i = i + WG; }
    workgroupBarrier();
    let mask = m - 1u;
    // Lanes are strided over the 64 threads (default encoder emits 128 lanes, `rans::parse` allows up to 255):
    // thread `lid` decodes lanes lid, lid+64, ... Lane order is irrelevant (lanes are independent streams).
    for (var k = lid.x; k < d.lanes; k = k + WG) {
        let end = lane_end(d.lane_off_base + k);
        var pos = lane_start(d.lane_off_base + k);
        var state = rd_state(pos, end);
        pos = pos + 4u;
        var cnt = 0u;
        if (d.n > k) { cnt = (d.n - k + d.lanes - 1u) / d.lanes; }
        for (var j = 0u; j < cnt; j = j + 1u) {
            let s = sh_slot[state & mask];
            let r = renorm(sh_freq[s] * (state >> d.table_log) + (state & mask) - sh_cum[s], pos, end);
            state = r.x;
            pos = r.y;
            outp[d.sym_base + k + j * d.lanes] = s;
        }
    }
}
"#;

/// The batched rANS kernel source with the table capacity substituted ([`SLOT_MAX`]).
pub(crate) fn rans_b_src() -> String {
    RANS_B_BODY.replace("//@RANS_COMMON@", RANS_COMMON)
}

// EXV-RT-specific rANS pass (B-RT.3 Increment 5) — **1 BLOCK per THREAD**, each block an `LANES`-way
// interleaved substream decoded with `LANES`-way instruction-level parallelism. The batched `RANS_B_BODY`
// above maps 1 block per WORKGROUP (63/64 lanes idle + the shared table re-loaded once per block). EXV-RT
// is the ideal case for a thread-dense kernel: every block of a TILE shares ONE table (freq/cum/slot bases,
// uniform `table_log`). The host groups blocks by table and pads each run to a multiple of `WG` (`rt.rs`), so a
// workgroup loads the table of its first unit into workgroup memory ONCE (all 64 threads cooperate) and hands
// each thread ONE whole block → 64 blocks per workgroup, `ceil(units/64)` workgroups.
//
// **Why N lanes per thread (Increment 5, PATH A):** the serial per-block rANS is a latency-bound dependent
// chain — each renorm reads stream bytes from global memory before the next symbol can be decoded. Splitting
// a block into `LANES` INTERLEAVED rANS substreams (lane k owns symbols k, k+LANES, 2·LANES+k, …; exactly
// `rans::encode_block(.., LANES)`) gives `LANES` INDEPENDENT dependent-chains that one thread advances in
// lockstep → `LANES`-way ILP hides the per-symbol latency. `LANES` is a COMPILE-TIME constant here so the
// `for k in 0..LANES` loops unroll and `state[k]`/`pos[k]` stay in registers (dynamic indexing would spill
// to thread-local memory and defeat the ILP). One specialized pipeline is built per `RT_LANE_SET` value.
// The decoded byte order (`outp[sym_base + k + j·LANES]`) is byte-identical to `RANS_B_BODY`, so the inverse
// stage is unchanged and bit-exactness is preserved for any `LANES` (rANS is exact regardless of lane count).
//
// `LANES==1` reduces to the former single-lane fast path. Used for block<=16 chunks of enough blocks (see
// `rt.rs`); block>16, an unsupported lane count, or a small chunk fall back to the workgroup-per-block `rans_b`.

/// Build the thread-dense fast rANS kernel body specialized for a compile-time lane count `lanes`.
/// `DESC_STRUCT` is prepended by the caller (as for the other batched kernels).
pub(crate) fn rans_b_rt_src(lanes: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>       descs:    array<Desc>;
@group(0) @binding(1) var<storage, read>       freq:     array<u32>;
@group(0) @binding(2) var<storage, read>       cum:      array<u32>;
@group(0) @binding(3) var<storage, read>       slot:     array<u32>;
@group(0) @binding(4) var<storage, read>       streams:  array<u32>;
@group(0) @binding(5) var<storage, read>       lane_off: array<u32>;
@group(0) @binding(6) var<storage, read_write> outp:     array<u32>;
{RANS_COMMON}
const LANES: u32 = {lanes}u;
// 1-lane blocks emit symbols in order, so a thread packs 4 per word: ONE store per 4 symbols instead of 4 scattered
// single-word stores (one cache line per thread = 32 sectors per warp store; measured 2.4x of the stage, OPT.md O).
// The host marks such chunks `packed` and `sym_at` (inverse) unpacks. Multi-lane order is k + j*LANES: unpacked.
const PACK: bool = LANES == {RT_PACK_LANES}u;
var<workgroup> sh_slot: array<u32, SLOT_MAX>;
var<workgroup> sh_freq: array<u32, ALPHA>;
var<workgroup> sh_cum:  array<u32, ALPHA>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    // One table per WORKGROUP: the host orders blocks so the `WG` units of a workgroup share the table of its
    // first unit (equal tables adjacent, each run padded to a multiple of `WG`), `table_log` included. The
    // guard is workgroup-uniform and precedes the barrier. All 64 threads cooperate in the table load BEFORE
    // any per-thread early return so `workgroupBarrier` stays uniform.
    let first = unit_id(wid, nwg) * WG;
    if (first >= arrayLength(&descs)) {{ return; }}
    let d0 = descs[first];
    let table_log = d0.table_log;
    let m = 1u << table_log; // M (<= SLOT_MAX)
    var i = lid.x;
    loop {{ if (i >= m) {{ break; }} sh_slot[i] = slot[d0.slot_base + i]; i = i + WG; }}
    i = lid.x;
    loop {{ if (i >= ALPHA) {{ break; }} sh_freq[i] = freq[d0.freq_base + i]; sh_cum[i] = cum[d0.cum_base + i]; i = i + WG; }}
    workgroupBarrier();
    let bi = first + lid.x;
    if (bi >= arrayLength(&descs)) {{ return; }}
    let d = descs[bi];
    let mask = m - 1u;
    let n = d.n;
    // `LANES` independent rANS states this thread advances together (register-resident; loops unroll).
    var state: array<u32, LANES>;
    var pos:   array<u32, LANES>;
    var cnt:   array<u32, LANES>;
    var lend:  array<u32, LANES>;
    for (var k = 0u; k < LANES; k = k + 1u) {{
        // lane_off holds the ABSOLUTE per-lane [start,end) (host parsed the substream's `LANES-1` lane header).
        let p = lane_start(d.lane_off_base + k);
        lend[k] = lane_end(d.lane_off_base + k);
        state[k] = rd_state(p, lend[k]);
        pos[k] = p + 4u;
        var c = 0u;
        if (n > k) {{ c = (n - k + LANES - 1u) / LANES; }} // ceil((n-k)/LANES) symbols in lane k
        cnt[k] = c;
    }}
    // Lane 0 always has the largest count → its length bounds the shared j loop; ragged lanes skip via `j < cnt[k]`.
    let jmax = cnt[0];
    var acc = 0u; // PACK: the 4 symbols of the current output word
    for (var j = 0u; j < jmax; j = j + 1u) {{
        for (var k = 0u; k < LANES; k = k + 1u) {{
            if (j < cnt[k]) {{
                let s = sh_slot[state[k] & mask];
                let r = renorm(sh_freq[s] * (state[k] >> table_log) + (state[k] & mask) - sh_cum[s], pos[k], lend[k]);
                state[k] = r.x;
                pos[k] = r.y;
                if (PACK) {{
                    acc = acc | (s << ((j & 3u) * 8u));
                    if ((j & 3u) == 3u || j + 1u == jmax) {{ outp[d.sym_base + (j >> 2u)] = acc; acc = 0u; }}
                }} else {{
                    outp[d.sym_base + k + j * LANES] = s;
                }}
            }}
        }}
    }}
}}
"#
    )
}

/// Lane counts for which a specialized thread-dense fast rANS kernel is precompiled (aligned with
/// `Pipelines::rans_b_rt`). A EXV-RT blob whose per-plane `lanes` is in this set takes the fast
/// N-lane-per-thread path; any other lane count falls back to the slow `rans_b` kernel (which maps
/// lanes→threads and handles an arbitrary N, one block per workgroup).
pub(crate) const RT_LANE_SET: [u32; 4] = [1, 2, 4, 8];

/// The lane count whose thread-dense rANS kernel byte-packs its symbols (4 per word, `rans_b_rt_src` `PACK`); the host
/// flags such chunks `packed` in the per-tile record so `rt_prep` sizes the regions and the inverse unpacks (`sym_at`).
pub(crate) const RT_PACK_LANES: u32 = 1;

pub(crate) const REPACK_B_BODY: &str = r#"
@group(0) @binding(0) var<storage, read>       descs: array<Desc>;
@group(0) @binding(1) var<storage, read>       syms:  array<u32>;
@group(0) @binding(2) var<storage, read_write> outp:  array<u32>;
@compute @workgroup_size(WG)
fn repack(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let u = unit_id(wid, nwg);
    if (u >= arrayLength(&descs)) { return; }
    let d = descs[u];
    var e = lid.x;
    loop {
        if (e >= d.count) { break; }
        var v = 0u;
        for (var i = 0u; i < d.esz; i = i + 1u) { v = v | ((syms[d.sym_base + e * d.esz + i] & 0xffu) << (8u * i)); }
        outp[d.out_base + (d.y0 + e / d.tw) * d.full_w + (d.x0 + e % d.tw)] = v;
        e = e + WG;
    }
}
"#;

pub(crate) const WAVELET_B_BODY: &str = r#"
@group(0) @binding(0) var<storage, read>       descs: array<Desc>;
@group(0) @binding(1) var<storage, read_write> buf:   array<u32>;
//@WAVE_LIFT@
@compute @workgroup_size(WG)
fn inv_cols(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let u = unit_id(wid, nwg);
    if (u >= arrayLength(&descs)) { return; }
    let d = descs[u];
    let base = d.out_base + d.y0 * d.full_w + d.x0;
    var c = lid.x;
    loop { if (c >= d.tw) { break; } lift_inv_line(base + c, d.full_w, d.th, mask_of(d.esz)); c = c + WG; }
}
@compute @workgroup_size(WG)
fn inv_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let u = unit_id(wid, nwg);
    if (u >= arrayLength(&descs)) { return; }
    let d = descs[u];
    let base = d.out_base + d.y0 * d.full_w + d.x0;
    // A line shorter than 2 is a no-op (as in `lift_inv_line`); folded into the bound so no thread returns before the barrier.
    let n = select(0u, d.tw, d.tw >= 2u);
    let mask = mask_of(d.esz);
    // ROWS = the same two lifting steps as `lift_inv_line`, but the WG threads cooperate ALONG each row (adjacent
    // lanes = adjacent words, coalesced) instead of one thread walking a whole row: a thread-per-row mapping touches
    // `WG` different rows (pages) at once and fell off a cliff on frames wider than ~6k (OPT.md L). Step order is
    // unchanged: the even update reads only odds and writes only evens, the odd predict reads the updated evens and
    // writes only odds, so each step is race-free across threads and one barrier separates them for ALL rows.
    for (var y = 0u; y < d.th; y = y + 1u) {
        let rb = base + y * d.full_w;
        for (var j = lid.x * 2u; j < n; j = j + 2u * WG) {
            let jm1 = select(1u, j - 1u, j >= 1u);
            let jp1 = select(j - 1u, j + 1u, j + 1u < n);
            let t = (buf[rb + jm1] + buf[rb + jp1] + 2u) >> 2u;
            buf[rb + j] = (buf[rb + j] - t) & mask;
        }
    }
    storageBarrier();
    workgroupBarrier();
    for (var y = 0u; y < d.th; y = y + 1u) {
        let rb = base + y * d.full_w;
        for (var j = lid.x * 2u + 1u; j < n; j = j + 2u * WG) {
            let jp1 = select(j - 1u, j + 1u, j + 1u < n);
            let t = (buf[rb + j - 1u] + buf[rb + jp1]) >> 1u;
            buf[rb + j] = (buf[rb + j] + t) & mask;
        }
    }
}
"#;

/// Batched inverse of reorder+delta (`exv_codec::reorder::join_undelta`) over a descriptor grid — the LosslessFast
/// second stage after the batched rANS pass (`RANS_B_BODY`, same `Desc` rows). The byte-delta is a serial prefix sum
/// WITHIN one plane of one tile (`esz` planes per tile, plane-major in `syms`); the pairs are independent. One
/// serial thread per pair was a 65536-step chain (~13 ms per channel regardless of image size), so ONE WORKGROUP
/// per `(tile, plane)` now computes the same sum as a parallel prefix scan: segments of `WG * UD_K` elements, each
/// thread sums its `UD_K` consecutive elements, a Hillis-Steele scan combines the thread totals, a carry links the
/// segments. Addition mod 2^32 is associative and `& 0xff` commutes with it, so the bytes are identical to the serial
/// chain. Output is the `BytePerWord` composite (one byte per word at pixel `(y0 + row, x0 + col)`, plane `p` of `esz`).
/// The workgroup id is rebuilt from a 2-D grid (`GpuContext::split_1d`): `MAX_ESZ` workgroups per unit (planes `>= esz` exit).
pub(crate) const UNDELTA_B_BODY: &str = r#"
@group(0) @binding(0) var<storage, read>       descs: array<Desc>;
@group(0) @binding(1) var<storage, read>       syms:  array<u32>;
@group(0) @binding(2) var<storage, read_write> outp:  array<u32>;
var<workgroup> sh_sum: array<u32, WG>;
var<workgroup> sh_n: u32;
// Elements one thread owns per segment (consecutive, so its partial sums stay in registers).
const UD_K: u32 = 8u;
@compute @workgroup_size(WG)
fn undelta(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = unit_id(wid, nwg);
    let u = t / MAX_ESZ;
    let p = t % MAX_ESZ;
    if (u >= arrayLength(&descs)) { return; }
    let d = descs[u];
    // The element count is made workgroup-uniform (a plane `p >= esz` has none) so the barriers below are in uniform flow.
    if (lid.x == 0u) { sh_n = select(0u, d.count, p < d.esz); }
    let n = workgroupUniformLoad(&sh_n);
    var carry = 0u; // running sum of every earlier segment (mod 2^32; only the low byte is kept, and `& 0xff` commutes with +)
    for (var seg = 0u; seg < n; seg = seg + WG * UD_K) {
        // 1) each thread: inclusive running sums of its UD_K consecutive elements.
        var v: array<u32, UD_K>;
        var s = 0u;
        let e0 = seg + lid.x * UD_K;
        for (var k = 0u; k < UD_K; k = k + 1u) {
            if (e0 + k < n) { s = s + syms[d.sym_base + p * d.count + e0 + k]; }
            v[k] = s;
        }
        // 2) Hillis-Steele inclusive scan of the WG thread totals.
        sh_sum[lid.x] = s;
        workgroupBarrier();
        for (var off = 1u; off < WG; off = off << 1u) {
            var a = sh_sum[lid.x];
            if (lid.x >= off) { a = a + sh_sum[lid.x - off]; }
            workgroupBarrier();
            sh_sum[lid.x] = a;
            workgroupBarrier();
        }
        let excl = carry + sh_sum[lid.x] - s;
        let total = sh_sum[WG - 1u];
        // 3) scatter this segment (BytePerWord composite: pixel (y0 + row, x0 + col), plane p of esz).
        for (var k = 0u; k < UD_K; k = k + 1u) {
            let e = e0 + k;
            if (e < n) {
                let row = e / d.tw;
                let col = e - row * d.tw;
                outp[d.out_base + ((d.y0 + row) * d.full_w + d.x0 + col) * d.esz + p] = (excl + v[k]) & 0xffu;
            }
        }
        carry = carry + total;
        workgroupBarrier(); // `sh_sum` is rewritten by the next segment
    }
}
"#;

/// WGSL `hdr::decode_value(c, scale = 256)` + `hdr::f32_to_half` + `unzig` + `emit_bits`, shared by every EXV-RT inverse kernel.
/// decode: `k = min(|c|, 64*256) - 32*256`, `a = floor(k/256)` (arithmetic
/// shift), `r = k mod 256` (mask), `value = bitcast<f32>((a+127) << 23) * T[r]` — bit-identical to the CPU.
/// The including module supplies `fn exp2_t(r: u32) -> f32` (the table lookup). Spliced in for the
/// `//@HDR_DECODE@` token by [`exvrt_inv_src`]/[`exvrt_fused_src`].
const HDR_DECODE: &str = r#"
fn decode_value(c: i32) -> f32 {
    if (c == 0) { return 0.0; }
    // abs(i32::MIN) wraps to i32::MIN in WGSL; u32() then min() saturates it like the CPU's unsigned_abs().
    let k = i32(min(u32(abs(c)), u32(HDR_MAX_CODE))) - HDR_BIAS;
    let a = k >> HDR_SHIFT;
    let r = u32(k & HDR_MASK);
    let v = bitcast<f32>(u32(a + 127) << 23u) * exp2_t(r);
    return select(v, -v, c < 0);
}

// Un-zigzag a regrouped coefficient code (sign in the LSB): `(z >> 1) ^ -(z & 1)`.
fn unzig(z: u32) -> i32 {
    let s = i32(z & 1u);
    return (i32(z >> 1u)) ^ (-s);
}

// Final f16 pattern of one inverse-lifted coefficient: q == 0 = clamp to the u16 range; q > 0 = clamp the code
// to [-HDR_MAX_CODE, HDR_MAX_CODE] (== `from_codes`' MAX_CODE clamp), `decode_value`, then `f32_to_half`.
// `decode_value(0) == 0.0` -> 0x0000, matching the CPU exactly. ONE copy for every EXV-RT inverse kernel.
fn emit_bits(coeff: i32, q: u32) -> u32 {
    if (q == 0u) { return u32(clamp(coeff, 0, 65535)); }
    return f32_to_half(decode_value(clamp(coeff, -HDR_MAX_CODE, HDR_MAX_CODE)));
}

// hdr::f32_to_half transliterated EXACTLY (round-to-nearest-even, subnormal, overflow→inf, NaN). Returns
// the u16 pattern in a u32 (only the low 16 bits are read back). `x = bitcast<u32>(v)` == Rust `v.to_bits()`.
fn f32_to_half(v: f32) -> u32 {
    let x = bitcast<u32>(v);
    let sign = (x >> 16u) & 0x8000u;
    let expo = i32((x >> 23u) & 0xffu) - 127 + 15;
    let mant = x & 0x7fffffu;
    if (expo >= 0x1f) {
        if ((x & 0x7fffffffu) > 0x7f800000u) {
            return sign | 0x7e00u; // NaN
        }
        return sign | 0x7c00u; // inf
    }
    if (expo <= 0) {
        if (expo < -10) {
            return sign;
        }
        let m = mant | 0x800000u;
        let shift = u32(14 - expo);
        let half_mant = m >> shift;
        let rem = m & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        let do_round = (rem > halfway) || (rem == halfway && (half_mant & 1u) == 1u);
        return sign | (half_mant + select(0u, 1u, do_round));
    }
    let half = sign | (u32(expo) << 10u) | (mant >> 13u);
    let rem = mant & 0x1fffu;
    let do_round = (rem > 0x1000u) || (rem == 0x1000u && (half & 1u) == 1u);
    return half + select(0u, 1u, do_round);
}
"#;

// === EXV-RT (blockrt_dwt) GPU decoder — B-RT.3 Increment 1 (q=0, BIT-EXACT). The rANS stage REUSES
// `RANS_B_BODY`/`ctx.pipelines.rans_b` VERBATIM (each block = one 1-lane descriptor; its single-lane
// substream is byte-identical to `rans::decode_block(.., 1)`, all blocks sharing ONE per-plane table).
// The NEW stage below regroups the decoded byte-symbols, un-zigzags, runs the signed-i32 inverse 5/3
// lifting (arithmetic `>>`, NOT the unsigned `WAVELET_B_BODY`), and emits the f16 bit pattern with the
// crop fused in. Increment 1 = q=0 (clamp to u16, BIT-EXACT). Increment 2 = q>0: detail dequant BEFORE
// the lifting, then the hdr `decode_value`→`f32_to_half` log path AFTER it — bit-exact vs the CPU (the log
// decode is a table fetch, see [`EXP2_TAB_256`]; no hardware `exp2`). `q` rides in the descriptor. ===
pub(crate) const RT_DESC_STRUCT: &str = "struct RtDesc { width:u32, sym_base:u32, block:u32, x0:u32, y0:u32, full_w:u32, x1:u32, y1:u32, q:u32, out_base:u32, packed:u32, };\n";

/// Per-tile record of the EXV-RT descriptor-prep kernel ([`RT_PREP_BODY`]); [`TILE_DESC_WORDS`] members, field
/// order == `rt::tile_rec`. `unit0`/`units` = the tile's row range in the chunk's descriptor arrays (`units - nblk`
/// trailing rows are the inert padding of a table run); `sym_base` chunk-relative; `w_base`/`w_len` = the tile's span
/// of the frame-wide `widths` bytes (the clamp for untrusted indices).
pub(crate) const TILE_DESC_STRUCT: &str = "struct TileDesc { unit0:u32, nblk:u32, units:u32, lanes:u32, table_log:u32, freq_base:u32, cum_base:u32, slot_base:u32, sym_base:u32, lane_base:u32, w_base:u32, w_len:u32, block:u32, nbx:u32, x0:u32, y0:u32, full_w:u32, x1:u32, y1:u32, q:u32, out_base:u32, packed:u32, };
";

/// EXV-RT descriptor prep: ONE thread per tile loops its rows and writes the SAME `Desc` / `RtDesc` rows the host
/// used to build (so `rans_b_rt` / `exvrt_inv_*` are unchanged). Prepended: `DESC_STRUCT` (+`unit_id`), `RT_DESC_STRUCT`,
/// [`TILE_DESC_STRUCT`]. **Untrusted input:** a width byte is read only inside the tile's `widths` span and a width
/// outside 1..=4 (or past the span) makes the row INERT (`n = 0` / zero `RtDesc`, which both hot kernels skip) and
/// does not advance `sym`, so a forged record can never address outside its buffers: every row write is bounded
/// by `arrayLength`.
pub(crate) const RT_PREP_BODY: &str = r#"
@group(0) @binding(0) var<storage, read>       tiles:   array<TileDesc>;
@group(0) @binding(1) var<storage, read>       widths:  array<u32>;   // per-block byte widths, packed LE
@group(0) @binding(2) var<storage, read_write> rdescs:  array<Desc>;
@group(0) @binding(3) var<storage, read_write> rtdescs: array<RtDesc>;

fn width_at(i: u32) -> u32 { return (widths[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu; }

@compute @workgroup_size(WG)
fn rt_prep(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_index) li: u32) {
    let t = unit_id(wid, nwg) * WG + li;
    if (t >= arrayLength(&tiles)) { return; }
    let td = tiles[t];
    let ncoeff = td.block * td.block;
    let nbx = max(td.nbx, 1u);
    let wcap = arrayLength(&widths) * 4u;
    let rows = min(arrayLength(&rdescs), arrayLength(&rtdescs));
    var sym = td.sym_base;
    for (var i = 0u; i < td.units; i++) {
        let row = td.unit0 + i;
        if (row >= rows) { break; }
        let real = i < td.nblk;
        var wb = 0u;
        if (real && i < td.w_len && td.w_base + i < wcap) { wb = width_at(td.w_base + i); }
        if (wb > 4u) { wb = 0u; }
        let n_sym = ncoeff * wb;
        let lane_off_base = select(td.lane_base, td.lane_base + i * td.lanes, real);
        rdescs[row] = Desc(td.lanes, td.table_log, n_sym, lane_off_base, td.freq_base, td.cum_base, td.slot_base, sym,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        if (wb == 0u) {
            rtdescs[row] = RtDesc();
        } else {
            rtdescs[row] = RtDesc(wb, sym, td.block, td.x0 + (i % nbx) * td.block, td.y0 + (i / nbx) * td.block,
                                  td.full_w, td.x1, td.y1, td.q, td.out_base, td.packed);
        }
        // Byte-packed symbols take `ceil(n_sym / 4)` words (a block's region stays word-aligned: threads own disjoint words).
        sym += select(n_sym, (n_sym + 3u) >> 2u, td.packed != 0u);
    }
}
"#;

pub(crate) const EXVRT_INV_BODY: &str = r#"
@group(0) @binding(0) var<storage, read>       rtdescs: array<RtDesc>;
@group(0) @binding(1) var<storage, read>       syms:    array<u32>;  // decoded byte-symbols (low byte per word)
@group(0) @binding(2) var<storage, read_write> outp:    array<u32>;  // cropped w*h plane, one u16 f16 pattern per word
@group(0) @binding(3) var<storage, read>       exp2t:   array<u32>;  // EXP2_TAB_256 f32 bit patterns (see decode_value)
fn exp2_t(r: u32) -> f32 { return bitcast<f32>(exp2t[r]); }
// Symbol `i` of a block whose symbols start at word `base`: byte-packed LE, 4 per word, when `packed` (the 1-lane
// thread-dense rANS kernel's layout, `rans_b_rt_src`), else one symbol per word (every other rANS kernel).
fn sym_at(base: u32, packed: u32, i: u32) -> u32 {
    if (packed != 0u) { return (syms[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu; }
    return syms[base + i] & 0xffu;
}
// block ≤ 64 → n*n ≤ 4096 i32 of per-block scratch (16 KiB workgroup storage). Only lane 0 touches it.
var<workgroup> sh_coeff: array<i32, MAX_COEFF>;

// Inverse 5/3 lift on `sh_coeff` (block <= 64 -> n*n <= 4096 i32 of per-block scratch).
//@LIFT_INV@

//@HDR_DECODE@


@compute @workgroup_size(WG)
fn exvrt_inv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) { return; } // Increment 1: one thread decodes the whole block (simplest bit-exact path).
    let bi = wid.y * nwg.x + wid.x; // 2-D unit grid (`GpuContext::split_1d`)
    if (bi >= arrayLength(&rtdescs)) { return; }
    let d = rtdescs[bi];
    if (d.block == 0u) { return; }
    let n = d.block;
    let ncoeff = n * n;

    // 1) regroup: `width` LE byte-symbols per coeff → u32 code → un-zigzag → i32 coeff (row-major k = y*n + x).
    var k = 0u;
    loop {
        if (k >= ncoeff) { break; }
        var z = 0u;
        var b = 0u;
        loop {
            if (b >= d.width) { break; }
            z = z | (sym_at(d.sym_base, d.packed, k * d.width + b) << (8u * b));
            b = b + 1u;
        }
        sh_coeff[k] = unzig(z);
        k = k + 1u;
    }

    // 1b) dequantize detail subbands (q>0 only) BEFORE the inverse lifting — mirrors `lossy::dequantize`
    //     (block-local coords, LL where both indices are even is untouched). i32 `*` wraps mod 2^32 just
    //     like the CPU's `wrapping_mul`; the host gate guarantees q<16 so `1<<q` never overflows i32.
    let q = d.q;
    if (q != 0u) {
        let step = i32(1u << q);
        var yq = 0u;
        loop {
            if (yq >= n) { break; }
            var xq = 0u;
            loop {
                if (xq >= n) { break; }
                if (((xq & 1u) | (yq & 1u)) != 0u) {
                    let kk = yq * n + xq;
                    sh_coeff[kk] = sh_coeff[kk] * step;
                }
                xq = xq + 1u;
            }
            yq = yq + 1u;
        }
    }

    // 2) inverse 2D 5/3: COLUMNS first (stride n), then ROWS (stride 1) — the exact order of `lossy::inv_2d`.
    var col = 0u;
    loop { if (col >= n) { break; } lift_inv(col, n, n); col = col + 1u; }
    var row = 0u;
    loop { if (row >= n) { break; } lift_inv(row * n, 1u, n); row = row + 1u; }

    // 3) emit: crop to the TILE rect x1×y1 (a padded pixel outside the tile rect is simply not written — mirrors
    //    `decode_plane`'s trailing crop). q=0 (Increment 1) = clamp to the u16 range; q>0 (Increment 2) =
    //    the hdr log path: clamp the code to [-16384, 16384] (== `from_codes`' MAX_CODE clamp), `decode_value`
    //    then `f32_to_half`. `decode_value(0)==0.0` → `f32_to_half`→0x0000, matching the CPU exactly.
    var yy = 0u;
    loop {
        if (yy >= n) { break; }
        let py = d.y0 + yy;
        if (py < d.y1) {
            var xx = 0u;
            loop {
                if (xx >= n) { break; }
                let px = d.x0 + xx;
                if (px < d.x1) {
                    let coeff = sh_coeff[yy * n + xx];
                    outp[d.out_base + py * d.full_w + px] = emit_bits(coeff, q);
                }
                xx = xx + 1u;
            }
        }
        yy = yy + 1u;
    }
}

// === B-RT.3 Increment 5, STEP 1 — the FAST inverse: **1 block per THREAD**. Identical MATH to `exvrt_inv`
// above (regroup → un-zigzag → dequant → inverse 2D 5/3 → emit), but the per-block coefficients live in a
// `var<function> array<i32,256>` PER-THREAD scratch instead of the workgroup-storage `sh_coeff[4096]`.
// 256 = 16×16 covers block<=16 (the DEFAULT). Dropping the workgroup scratch is what lets thousands of
// blocks stay in flight (high occupancy) — the old kernel is forced to 1 block/workgroup by its 16 KiB
// `sh_coeff`. Routed here for block<=16; block>16 keeps `exvrt_inv`. Bit-exactness is preserved because
// only the storage class of the scratch changed, not a single arithmetic op. ===

// Inverse 5/3 lift on the per-thread scratch (pointer param, `function` address space).
//@LIFT_INV_F@

@compute @workgroup_size(WG)
fn exvrt_inv_fast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bi = gid.x;
    if (bi >= arrayLength(&rtdescs)) { return; } // one block per thread; guard the ragged last workgroup.
    let d = rtdescs[bi];
    let n = d.block;      // block <= 16 on this path → n*n <= 256 fits the per-thread scratch.
    let ncoeff = n * n;
    var coeff: array<i32, FAST_COEFF>;

    // 1) regroup: `width` LE byte-symbols per coeff → u32 code → un-zigzag → i32 coeff (row-major k=y*n+x).
    var k = 0u;
    loop {
        if (k >= ncoeff) { break; }
        var z = 0u;
        var b = 0u;
        loop {
            if (b >= d.width) { break; }
            z = z | (sym_at(d.sym_base, d.packed, k * d.width + b) << (8u * b));
            b = b + 1u;
        }
        coeff[k] = unzig(z);
        k = k + 1u;
    }

    // 1b) dequantize detail subbands (q>0 only) BEFORE the inverse lifting — mirrors `lossy::dequantize`.
    let q = d.q;
    if (q != 0u) {
        let step = i32(1u << q);
        var yq = 0u;
        loop {
            if (yq >= n) { break; }
            var xq = 0u;
            loop {
                if (xq >= n) { break; }
                if (((xq & 1u) | (yq & 1u)) != 0u) {
                    let kk = yq * n + xq;
                    coeff[kk] = coeff[kk] * step;
                }
                xq = xq + 1u;
            }
            yq = yq + 1u;
        }
    }

    // 2) inverse 2D 5/3: COLUMNS first (stride n), then ROWS (stride 1) — the exact order of `lossy::inv_2d`.
    var col = 0u;
    loop { if (col >= n) { break; } lift_inv_f(&coeff, col, n, n); col = col + 1u; }
    var row = 0u;
    loop { if (row >= n) { break; } lift_inv_f(&coeff, row * n, 1u, n); row = row + 1u; }

    // 3) emit: crop to the tile rect x1×y1; q=0 = clamp to u16, q>0 = the hdr log path (`decode_value`→`f32_to_half`).
    var yy = 0u;
    loop {
        if (yy >= n) { break; }
        let py = d.y0 + yy;
        if (py < d.y1) {
            var xx = 0u;
            loop {
                if (xx >= n) { break; }
                let px = d.x0 + xx;
                if (px < d.x1) {
                    let coeffv = coeff[yy * n + xx];
                    outp[d.out_base + py * d.full_w + px] = emit_bits(coeffv, q);
                }
                xx = xx + 1u;
            }
        }
        yy = yy + 1u;
    }
}

// === EXV-RT SHARED inverse — the shipped one: **`INV_BLOCKS` blocks per WORKGROUP** (`INV_WG` = INV_BLOCKS * FAST_BLOCK
// threads). Same MATH as `exvrt_inv` (regroup → un-zigzag → dequant → inverse 2D 5/3 columns-then-rows → emit): only
// the storage + the WHO-does-which-element mapping differ, so it stays bit-exact. Why this shape (OPT.md O): the old
// 1-block-per-64-thread kernel kept 16 of 64 threads busy in the lifts and paid 3-4 barriers per block. Here the lifts
// are one thread per block LINE (thread `t` = block slot `t / FAST_BLOCK`, line `t % FAST_BLOCK`; a block of edge
// `n < FAST_BLOCK` leaves lines `>= n` idle), while the regroup and the emit are COOPERATIVE with iteration `it` = block
// slot `it` for all `INV_WG == FAST_COEFF` threads, so consecutive threads touch consecutive symbols / pixels
// (coalesced) and each descriptor is a workgroup-uniform broadcast, staged once in `sh_d`. The row stride is padded to
// `INV_STRIDE` (odd) so the row lifts (thread stride = row stride) are bank-conflict free. Measured (OPT.md O): giving
// every thread its own row to regroup/emit (2 barriers, uncoalesced) was slower than this.
//
// Uniformity: every `workgroupBarrier` is reached by every thread (no early return); inert rows (`block == 0`, the
// padding of a table run / the ragged tail past `arrayLength`) are zero in `sh_d` and just idle. The grid is
// `ceil(units / INV_BLOCKS)` workgroups on the
// usual 2-D split (`wid.y * nwg.x + wid.x`).

// Inverse 5/3 lift on the workgroup scratch (`sh_blk`): a thread owns a whole line, distinct lines touch disjoint
// indices (columns: stride INV_STRIDE; rows: stride 1).
//@LIFT_INV_SH@

// INV_BLOCKS blocks of FAST_BLOCK rows x INV_STRIDE (padded) i32.
var<workgroup> sh_blk: array<i32, INV_SH>;
var<workgroup> sh_d: array<RtDesc, INV_BLOCKS>; // the workgroup's block descriptors, staged once (zero = inert)

@compute @workgroup_size(INV_WG)
fn exvrt_inv_shared(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(num_workgroups) nwg: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let u0 = (wid.y * nwg.x + wid.x) * INV_BLOCKS;  // first unit (rtdescs row) of this workgroup
    let nu = arrayLength(&rtdescs);
    if (t < INV_BLOCKS) {
        var dd: RtDesc;
        if (u0 + t < nu) { dd = rtdescs[u0 + t]; }
        sh_d[t] = dd;
    }
    workgroupBarrier();

    // 1) COOPERATIVE regroup + un-zigzag (+ detail dequant for q>0, LL = both indices even untouched; i32 `*` wraps
    //    like the CPU's `wrapping_mul`, the host gate keeps q < 16): iteration `it` is block slot `it` for ALL threads
    //    (`t / FAST_COEFF == 0` since INV_WG == FAST_COEFF), so the descriptor is a workgroup-uniform broadcast and
    //    consecutive threads read consecutive symbols (coalesced).
    for (var it = 0u; it < INV_BLOCKS; it = it + 1u) {
        let d = sh_d[it];
        let n = d.block;                             // 0 = inert row
        if (t < n * n) {
            let x = t % n;
            let y = t / n;
            var z = 0u;
            for (var b = 0u; b < d.width; b = b + 1u) { z = z | (sym_at(d.sym_base, d.packed, t * d.width + b) << (8u * b)); }
            var c = unzig(z);
            if (d.q != 0u && ((x | y) & 1u) != 0u) { c = c * i32(1u << d.q); }
            sh_blk[it * (FAST_BLOCK * INV_STRIDE) + y * INV_STRIDE + x] = c;
        }
    }
    workgroupBarrier();

    // 2) inverse 2D 5/3: COLUMNS first — thread `t` = block slot `t / FAST_BLOCK`, column `t % FAST_BLOCK` (stride
    //    INV_STRIDE: consecutive threads, consecutive banks) — the exact order of `lossy::inv_2d`. A thread past the
    //    block edge (n < FAST_BLOCK) or of an inert / out-of-range row idles.
    let slot = t / FAST_BLOCK;
    let line = t % FAST_BLOCK;
    let n = sh_d[slot].block;
    let row0 = slot * (FAST_BLOCK * INV_STRIDE) + line * INV_STRIDE;
    if (line < n) { lift_inv_sh(slot * (FAST_BLOCK * INV_STRIDE) + line, INV_STRIDE, n); }
    workgroupBarrier();

    // 2b) then ROWS (thread `t` owns row `line` of block `slot`; INV_STRIDE is odd, so threads hit distinct banks).
    if (line < n) { lift_inv_sh(row0, 1u, n); }
    workgroupBarrier();

    // 3) emit, cooperatively again so the stores of a warp are consecutive pixels of a block row (coalesced): crop to
    //    the TILE rect (x1/y1). q=0 = clamp to u16, q>0 = the hdr log path (`decode_value`→`f32_to_half`).
    for (var it = 0u; it < INV_BLOCKS; it = it + 1u) {
        let d = sh_d[it];
        let bn = d.block;
        if (t < bn * bn) {
            let y = t / bn;
            let x = t % bn;
            let px = d.x0 + x;
            let py = d.y0 + y;
            if (px < d.x1 && py < d.y1) {
                outp[d.out_base + py * d.full_w + px] = emit_bits(sh_blk[it * (FAST_BLOCK * INV_STRIDE) + y * INV_STRIDE + x], d.q);
            }
        }
    }
}
"#;

// === B-RT.3 Increment 5, PATH B — the FUSED single-pass EXV-RT decoder: rANS expand + cooperative
// shared-memory inverse in ONE kernel, ONE dispatch, with NO global `syms` VRAM round-trip. Selected for
// block<=16 && lanes==1 (the shipping default `RT_LANES==1`). 1 block per WORKGROUP, @workgroup_size(WG),
// dispatched on the SAME 2D (nbx,nby) grid as `exvrt_inv_shared` (`bi = wid.y*nwg.x + wid.x` is
// workgroup-uniform → every `workgroupBarrier` stays in uniform control flow).
//
// WHY: the old path decoded rANS symbols into a global `syms` buffer (pass 1) and read them back in the
// inverse (pass 2). Fusing keeps the decoded coefficients in the SAME `var<workgroup> sh_blk` the
// cooperative inverse already reads — no `syms` alloc, no write, no read-back, one dispatch.
//
// Phase 1 (thread 0, SERIAL single-lane rANS): decode this block's substream reproducing `RANS_B_BODY`'s
// math byte-for-byte, folding the pass-2 regroup+un-zigzag directly into the decode. Single-lane means the
// decoder emits symbols in LINEAR order `j = 0..ncoeff*width`, i.e. COEFF-MAJOR (coeff k owns symbols
// `[k*width .. (k+1)*width)`), so thread 0 assembles each coeff's `width` LE byte-symbols into `z` and
// writes the un-zigzagged i32 straight into `sh_blk[k]` — no intermediate buffer. The rANS tables
// (freq/cum/slot) are read from GLOBAL (small + L2-resident); they are NOT staged into workgroup memory,
// so the ONLY workgroup storage is `sh_blk` (1 KiB) → occupancy stays as high as the A2 shared inverse
// (the very reason A2 beat the per-thread-scratch fast path). The other 63 threads idle in phase 1.
//
// Phase 2 (all 64 threads, COOPERATIVE inverse): the EXACT `exvrt_inv_shared` body from the dequant step
// on (dequant q>0 → inverse COLUMNS → barrier → inverse ROWS → barrier → emit), on the same `sh_blk`.
//
// Bit-exactness is preserved: the rANS math == `RANS_B_BODY`, the regroup/un-zigzag == the pass-2 regroup,
// and the inverse == `exvrt_inv_shared` — only the STORAGE of the decoded coeffs changed (global `syms` →
// workgroup `sh_blk`), not a single arithmetic op. Built with BOTH `Desc` (rANS) and `RtDesc` (geometry)
// structs prepended (see `build_pipelines`): the rANS `Desc[bi]` supplies `table_log`/`lane_off_base`/table
// bases, the `RtDesc[bi]` supplies `width`/`block`/`x0`/`y0`/`full_w`/`x1`/`y1`/`q`.
pub(crate) const EXVRT_FUSED_BODY: &str = r#"
@group(0) @binding(0) var<storage, read>       descs:    array<Desc>;    // rANS Desc (lanes==1): table_log, lane_off_base, freq/cum/slot_base(=0)
@group(0) @binding(1) var<storage, read>       freq:     array<u32>;
@group(0) @binding(2) var<storage, read>       cum:      array<u32>;
@group(0) @binding(3) var<storage, read>       slot:     array<u32>;
@group(0) @binding(4) var<storage, read>       streams:  array<u32>;
@group(0) @binding(5) var<storage, read>       lane_off: array<u32>;
@group(0) @binding(6) var<storage, read>       rtdescs:  array<RtDesc>; // width, block, x0, y0, full_w, x1, y1, q, out_base
@group(0) @binding(7) var<storage, read_write> outp:     array<u32>;    // cropped plane, one u16 f16 pattern per word
//@RANS_COMMON@
// The fused kernel already uses all 8 default storage bindings, so its decode table is a constant array
// (substituted by `exvrt_fused_src`); a dynamic index copies it to a local — acceptable for an experiment.
const EXP2_T: array<u32, EXP2_N> = array<u32, EXP2_N>(EXP2_TAB_LIST);
fn exp2_t(r: u32) -> f32 { var t = EXP2_T; return bitcast<f32>(t[r]); }
// Only workgroup storage is this 256-i32 (1 KiB) coeff scratch — 256 = 16x16 covers block<=16 (the DEFAULT).
var<workgroup> sh_blk: array<i32, FAST_COEFF>;

//@LIFT_INV_SH@

//@HDR_DECODE@

@compute @workgroup_size(WG)
fn exvrt_fused(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wid.y * nwg.x + wid.x;                 // linear block index (host order byi*nbx + bxi)
    if (bi >= arrayLength(&rtdescs)) { return; }    // workgroup-uniform guard for the ragged tail
    let rd = rtdescs[bi];
    let n = rd.block;                               // block <= 16 -> ncoeff <= 256 fits sh_blk
    let ncoeff = n * n;
    let width = rd.width;                           // adaptive LE byte-symbols per coeff
    let q = rd.q;
    let t = lid.x;
    let wsz = WG;

    // --- Phase 1: thread 0 serially decodes the block's single-lane rANS substream, folding regroup +
    //     un-zigzag into the decode (coeff-major symbol order) straight into sh_blk. Reproduces
    //     `RANS_B_BODY` byte-for-byte (state init, per-symbol slot/freq/cum lookup, 4-byte-max renorm). ---
    if (t == 0u) {
        let d = descs[bi];
        let table_log = d.table_log;
        let m = 1u << table_log;                    // M (<= 4096)
        let mask = m - 1u;
        let end = lane_end(d.lane_off_base);        // lanes==1 -> one lane span per block (absolute bytes)
        var pos = lane_start(d.lane_off_base);
        var state = rd_state(pos, end);
        pos = pos + 4u;
        var k = 0u;
        loop {
            if (k >= ncoeff) { break; }
            var z = 0u;
            var b = 0u;
            loop {
                if (b >= width) { break; }
                let s = slot[d.slot_base + (state & mask)];
                let r = renorm(freq[d.freq_base + s] * (state >> table_log) + (state & mask) - cum[d.cum_base + s], pos, end);
                state = r.x;
                pos = r.y;
                z = z | ((s & 0xffu) << (8u * b));
                b = b + 1u;
            }
            sh_blk[k] = unzig(z);
            k = k + 1u;
        }
    }
    workgroupBarrier();

    // --- Phase 2: cooperative inverse — EXACT `exvrt_inv_shared` body (dequant -> cols -> rows -> emit). ---
    // 1b) dequant detail subbands (q>0) — parallel over coeffs; LL (both indices even) untouched. `q` is
    //     block-uniform so this whole branch (and its barrier) is uniform.
    if (q != 0u) {
        let step = i32(1u << q);
        var kk = t;
        loop {
            if (kk >= ncoeff) { break; }
            let xq = kk % n;
            let yq = kk / n;
            if (((xq & 1u) | (yq & 1u)) != 0u) {
                sh_blk[kk] = sh_blk[kk] * step;
            }
            kk = kk + wsz;
        }
        workgroupBarrier();
    }

    // 2) inverse 2D 5/3: COLUMNS first (stride n) — the n columns are independent 1D lifts, one per thread.
    var col = t;
    loop {
        if (col >= n) { break; }
        lift_inv_sh(col, n, n);
        col = col + wsz;
    }
    workgroupBarrier();
    // 2b) then ROWS (stride 1) — the exact order of `lossy::inv_2d`.
    var row = t;
    loop {
        if (row >= n) { break; }
        lift_inv_sh(row * n, 1u, n);
        row = row + wsz;
    }
    workgroupBarrier();

    // 3) emit: parallel over the block's pixels (index p == yy*n+xx), cropped to the TILE rect (x1/y1). q=0 =
    //    clamp to u16, q>0 = the hdr log path (`decode_value`->`f32_to_half`). Disjoint pixels -> no conflict.
    var p = t;
    loop {
        if (p >= ncoeff) { break; }
        let xx = p % n;
        let yy = p / n;
        let px = rd.x0 + xx;
        let py = rd.y0 + yy;
        if (px < rd.x1 && py < rd.y1) {
            let coeff = sh_blk[p];
            outp[rd.out_base + py * rd.full_w + px] = emit_bits(coeff, q);
        }
        p = p + wsz;
    }
}
"#;

/// `EXVRT_INV_BODY` (3 entry points) with the shared hdr decode spliced in.
pub(crate) fn exvrt_inv_src() -> String {
    EXVRT_INV_BODY
        .replace("//@LIFT_INV@", &lift_inv_src("lift_inv", "", "sh_coeff"))
        .replace("//@LIFT_INV_F@", &lift_inv_src("lift_inv_f", "c: ptr<function, array<i32, FAST_COEFF>>, ", "(*c)"))
        .replace("//@LIFT_INV_SH@", &lift_inv_src("lift_inv_sh", "", "sh_blk"))
        .replace("//@HDR_DECODE@", HDR_DECODE)
}

/// `EXVRT_FUSED_BODY` with the shared hdr decode and the constant decode table spliced in.
pub(crate) fn exvrt_fused_src() -> String {
    let list = EXP2_TAB_256.iter().map(|b| format!("{b:#x}u")).collect::<Vec<_>>().join(",");
    EXVRT_FUSED_BODY
        .replace("//@RANS_COMMON@", RANS_COMMON)
        .replace("//@LIFT_INV_SH@", &lift_inv_src("lift_inv_sh", "", "sh_blk"))
        .replace("//@HDR_DECODE@", HDR_DECODE)
        .replace("EXP2_TAB_LIST", &list)
}

/// WGSL inverse 5/3 (exactly mirrors `exv_codec::wavelet::lift_inv`): two passes (columns, rows),
/// one thread per line. u32 arithmetic — correct for esz≤2 (l+r does not overflow u32); esz=4 requires INT64.
pub(crate) const WAVELET_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       params: array<u32>; // [tw, th, esz, full_w, x0, y0]
@group(0) @binding(1) var<storage, read_write> buf:    array<u32>; // target buffer (a per-tile buffer when
                                                                   // full_w==tw,x0=y0=0; else the full-image
                                                                   // composite — lifting runs in place on the
                                                                   // tile's strided sub-region, no scatter).

//@WAVE_LIFT@

@compute @workgroup_size(WG)
fn inv_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = params[0]; let th = params[1]; let full_w = params[3];
    let base = params[5] * full_w + params[4]; // y0*full_w + x0
    if (gid.x >= tw) { return; }
    lift_inv_line(base + gid.x, full_w, th, mask_of(params[2]));
}

@compute @workgroup_size(WG)
fn inv_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = params[0]; let th = params[1]; let full_w = params[3];
    let base = params[5] * full_w + params[4];
    if (gid.x >= th) { return; }
    lift_inv_line(base + gid.x * full_w, 1u, tw, mask_of(params[2]));
}
"#;

/// GPU inverse 5/3 wavelet (for esz≤2). Readback — test-only verification; in the hot path
/// the result stays in VRAM. Bit-for-bit equal to `exv_codec::wavelet::inverse_2d`.
pub fn inverse_wavelet_gpu(ctx: &GpuContext, coeffs: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<Vec<u8>> {
    let esz = dtype.size();
    if esz > 2 {
        return Err(GpuError::Unsupported(format!("wavelet GPU decode needs esz≤2 (INT64 for {esz}-byte); use the CPU")));
    }
    if w == 0 || h == 0 {
        return Ok(coeffs.to_vec());
    }
    let elems = to_elems(coeffs, esz);
    let size = (elems.len() * 4) as u64;
    ctx.check_buf("wav.buf", size)?;
    ctx.scoped(|| {
        let device = &ctx.device;
        let params = [w, h, esz as u32, w, 0, 0]; // 6-field strided params; identity strides (full_w=w, x0=y0=0)
        let b_params = storage_u32(device, &ctx.queue, "wav.params", &params, wgpu::BufferUsages::empty());
        let b_buf = storage_u32(device, &ctx.queue, "wav.buf", &elems, wgpu::BufferUsages::COPY_SRC);
        let bg = |p: &wgpu::ComputePipeline| bind_bufs(device, p, "wav.bind", &[&b_params, &b_buf]);
        let (p_cols, p_rows) = (&ctx.pipelines.wav_cols, &ctx.pipelines.wav_rows);
        let mut enc = encoder(device, "wav.enc");
        // two SEPARATE passes → barrier between columns and rows
        pass(ctx, &mut enc, "cols", p_cols, &bg(p_cols), [w.div_ceil(WG), 1, 1])?;
        pass(ctx, &mut enc, "rows", p_rows, &bg(p_rows), [h.div_ceil(WG), 1, 1])?;
        let rb = staging(device, "wav.readback", size);
        enc.copy_buffer_to_buffer(&b_buf, 0, &rb, 0, size);
        ctx.queue.submit([enc.finish()]);
        Ok(from_elems(&words_of(&map_read_blocking(device, &rb)?), esz))
    })
}

/// Bytes → u32 elements (LE, zero-extended) and back — for the wavelet buffer.
fn to_elems(bytes: &[u8], esz: usize) -> Vec<u32> {
    bytes
        .chunks_exact(esz)
        .map(|c| {
            let mut v = 0u32;
            for (i, &b) in c.iter().enumerate() {
                v |= (b as u32) << (8 * i);
            }
            v
        })
        .collect()
}
pub(crate) fn from_elems(elems: &[u32], esz: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(elems.len() * esz);
    for &v in elems {
        for i in 0..esz {
            out.push((v >> (8 * i)) as u8);
        }
    }
    out
}

/// Repack: from the stream of rANS byte-symbols (u32 per byte) assemble elements (esz bytes → one u32).
pub(crate) const REPACK_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       rp:    array<u32>;  // [esz, count, tw, full_w, x0, y0]
@group(0) @binding(1) var<storage, read>       syms:  array<u32>;  // n_sym byte-symbols
@group(0) @binding(2) var<storage, read_write> elems: array<u32>;  // ElemPerWord target (tile buffer or composite)

// 2-D grid: a tile above 64 * max_compute_workgroups_per_dimension elements (2048^2 on desktop drivers)
// does not fit a 1-D dispatch, so the linear workgroup index is rebuilt from (wid.y, num_workgroups.x).
@compute @workgroup_size(WG)
fn repack(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let esz = rp[0]; let count = rp[1]; let tw = rp[2]; let full_w = rp[3]; let x0 = rp[4]; let y0 = rp[5];
    let e = (wid.y * nwg.x + wid.x) * WG + lid.x; if (e >= count) { return; }
    var v = 0u;
    for (var i = 0u; i < esz; i = i + 1u) {
        v = v | ((syms[e * esz + i] & 0xffu) << (8u * i));
    }
    // write element e at its strided position in the target (identity when full_w==tw,x0=y0=0).
    elems[(y0 + e / tw) * full_w + (x0 + e % tw)] = v;
}
"#;

/// Route one tagged tile blob for the test/export wrappers: rANS → `vram` (GPU decode) + [`read_back`],
/// raw → as-is, every other tag → a typed [`GpuError::Unsupported`] (the caller decodes it on the CPU /
/// jph). Shared by [`decode_tile_gpu_wave`] and [`decode_tile_gpu_fast`] (was duplicated).
fn tile_bytes(
    ctx: &GpuContext,
    blob: &[u8],
    plane_len: usize,
    vram: impl FnOnce(&[u8]) -> GpuResult<GpuTile>,
) -> GpuResult<Vec<u8>> {
    match classify_blob(blob) {
        Some(BlobKind::Rans(p)) => read_back(ctx, &vram(p)?),
        Some(BlobKind::Raw(p)) => {
            if p.len() == plane_len { Ok(p.to_vec()) } else { Err(GpuError::Corrupt("raw tile size mismatch".to_string())) }
        }
        Some(BlobKind::Deflate(_)) => Err(GpuError::Unsupported("deflate-coded tile: decode on CPU".to_string())),
        // HTJ2K tiles are decoded by jph-rs (CPU) / jph-wgpu (GPU), not the rANS kernels.
        Some(BlobKind::Htj2k(_)) => Err(GpuError::Unsupported("htj2k-coded tile: decode via jph-rs/jph-wgpu".to_string())),
        // EXV-RT block tiles have their own GPU path ([`decode_tile_exvrt_vram`]).
        Some(BlobKind::BlockRt(_)) => Err(GpuError::Unsupported("exv-rt tile: use decode_tile_exvrt_vram".to_string())),
        // BC6H block tiles have their own GPU path ([`decode_tile_bc6h_vram`]).
        Some(BlobKind::Bc6h(_)) => Err(GpuError::Unsupported("bc6h tile: use decode_tile_bc6h_vram".to_string())),
        None => Err(GpuError::Corrupt("bad blob tag".to_string())),
    }
}

/// **GPU tile decode for the LosslessWave profile** (esz≤2). Dispatches on the blob tag
/// ([`classify_blob`]): a rANS tile decodes on the GPU into VRAM ([`decode_tile_gpu_wave_vram`]) and
/// is read back here (test/export only); a raw tile is returned as-is; a deflate tile → CPU fallback
/// (`Err`, caught by the caller). Bit-for-bit == CPU `decode_channel`.
pub fn decode_tile_gpu_wave(ctx: &GpuContext, blob: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<Vec<u8>> {
    let esz = dtype.size();
    if esz > 2 {
        return Err(GpuError::Unsupported(format!("wavelet GPU tile decode needs esz≤2 ({esz}-byte → CPU)")));
    }
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    tile_bytes(ctx, blob, w as usize * h as usize * esz, |p| decode_tile_gpu_wave_vram(ctx, p, dtype, w, h))
}

/// **GPU tile decode (LosslessWave, esz≤2) into VRAM — NO readback (the GPU-first hot path).**
///
/// One encoder, four passes: rANS → repack → inverse-wavelet(columns→rows). All buffers stay in VRAM;
/// the decoded plane is returned as a [`GpuTile`] for on-GPU consumption (compositing/viewer). Unpack
/// to packed bytes via [`read_back`] (test/export only) → bit-for-bit == CPU `decode_channel`.
/// `payload` is the rANS container (blob tag already stripped by [`classify_blob`]).
///
/// **Safety contract:** a forged initial rANS lane state is rejected by `rans::parse` on the CPU
/// before the shader runs; the WGSL decoder does not (and need not) replicate the reject — a malformed
/// blob never reaches the GPU, and the WGSL RENORM_GUARD (=4, identical to the CPU) prevents any hang.
/// The rANS kernel strides lanes over its 64 threads, so any `lanes` ≤ 255 decodes (default encoder: 128).
pub fn decode_tile_gpu_wave_vram(ctx: &GpuContext, payload: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<GpuTile> {
    let esz = dtype.size();
    if esz > 2 {
        return Err(GpuError::Unsupported(format!("wavelet GPU decode needs esz≤2 ({esz}-byte → CPU)")));
    }
    let count = w as usize * h as usize; // elements (no u32-overflow)
    let prep = rans::parse(payload).ok_or_else(|| corrupt("bad rANS container"))?;
    if prep.n != count * esz {
        return Err(GpuError::Corrupt(format!("size mismatch: n={} != {count}*{esz}", prep.n)));
    }
    // GPU rANS holds 1×u32 per decoded byte → the sym buffer is 4× the plane (8192² f16 ≈ 537 MiB > the
    // default 128 MiB binding limit). A typed Limit error → the caller reduces the tile or uses the CPU.
    ctx.check_buf("wave.sym", prep.n as u64 * 4)?;
    ctx.check_buf("wave.elem", count as u64 * 4)?;
    ctx.scoped(|| {
        let bufs = RansBufs::new(ctx, "t", &prep)?;
        let device = &ctx.device;
        let no = wgpu::BufferUsages::empty();
        let b_rp = storage_u32(device, &ctx.queue, "t.rp", &[esz as u32, count as u32, w, w, 0, 0], no);
        // outputs/intermediates (stay in VRAM)
        let b_sym = ctx.scratch("t.sym", prep.n as u64 * 4);
        let b_elem = new_buf(device, "t.elem", count as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let b_wavp = storage_u32(device, &ctx.queue, "t.wavp", &[w, h, esz as u32, w, 0, 0], no);

        let pl = &ctx.pipelines;
        let bg_rans = bufs.bind(ctx, &b_sym);
        let bg_rep = bind_bufs(device, &pl.repack, "t.repack", &[&b_rp, &b_sym, &b_elem]);
        let bg_cols = bind_bufs(device, &pl.wav_cols, "t.cols", &[&b_wavp, &b_elem]);
        let bg_rows = bind_bufs(device, &pl.wav_rows, "t.rows", &[&b_wavp, &b_elem]);

        let mut enc = encoder(device, "t.enc");
        pass(ctx, &mut enc, "rans", &pl.rans, &bg_rans, [(prep.cfg.lanes as u32).div_ceil(WG), 1, 1])?;
        // Repack is element-parallel: 2-D grid so a tile > 64*max_dim elements still dispatches (H2).
        pass(ctx, &mut enc, "repack", &pl.repack, &bg_rep, ctx.split_1d("wave.repack", (count as u64).div_ceil(WG as u64))?)?;
        pass(ctx, &mut enc, "cols", &pl.wav_cols, &bg_cols, [w.div_ceil(WG), 1, 1])?;
        pass(ctx, &mut enc, "rows", &pl.wav_rows, &bg_rows, [h.div_ceil(WG), 1, 1])?;
        ctx.queue.submit([enc.finish()]);
        Ok(GpuTile { buf: b_elem, off_words: 0, words: count, w, h, dtype, enc: TileEnc::ElemPerWord(esz) })
    })
}

/// Inverse for reorder+delta (`exv_codec::reorder::join_undelta`): per-plane prefix-sum + scatter.
/// One thread per plane (stride≤4 threads) — correct, but serial within a plane (byte-delta is inverted
/// only by prefix-sum); hence the wavelet profile is GPU-preferred. esz=stride. Single-tile (<= 4 threads) form; the
/// whole-frame path uses the batched [`UNDELTA_B_BODY`] (one thread per tile-plane).
pub(crate) const REORDER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       params: array<u32>; // [stride, n, tw, full_w, x0, y0]
@group(0) @binding(1) var<storage, read>       din:    array<u32>; // decorrelated bytes (u32-per-byte), plane-major
@group(0) @binding(2) var<storage, read_write> dout:   array<u32>; // BytePerWord target (tile buffer or composite)

@compute @workgroup_size(WG)
fn undelta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = params[0]; let n = params[1]; let tw = params[2]; let full_w = params[3]; let x0 = params[4]; let y0 = params[5];
    let p = gid.x; if (p >= stride) { return; }
    var prev = 0u;
    for (var e = 0u; e < n; e = e + 1u) {
        let b = (din[p * n + e] + prev) & 0xffu;
        prev = b;
        dout[((y0 + e / tw) * full_w + (x0 + e % tw)) * stride + p] = b;
    }
}
"#;

/// **GPU tile decode for the LosslessFast profile** (rANS → reorder-undelta). Dispatches on the blob
/// tag ([`classify_blob`]): a rANS tile decodes on the GPU into VRAM ([`decode_tile_gpu_fast_vram`])
/// and is read back here (test/export only); a raw tile is returned as-is; a deflate tile → CPU
/// fallback (`Err`). Bit-for-bit == CPU `decode_channel`.
pub fn decode_tile_gpu_fast(ctx: &GpuContext, blob: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<Vec<u8>> {
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    tile_bytes(ctx, blob, w as usize * h as usize * dtype.size(), |p| decode_tile_gpu_fast_vram(ctx, p, dtype, w, h))
}

/// **GPU tile decode (LosslessFast) into VRAM — NO readback (the GPU-first hot path).**
///
/// Two passes: rANS → reorder-undelta. The plane stays in VRAM as a [`GpuTile`] (one byte per u32
/// word). Unpack via [`read_back`] (test/export only) → bit-for-bit == CPU `decode_channel`. `payload`
/// is the rANS container (blob tag already stripped). The reorder undelta is serial within a plane
/// (byte-delta prefix-sum), so the wavelet profile is the GPU-preferred one.
///
/// **Safety contract:** as in [`decode_tile_gpu_wave_vram`] — a forged rANS state is rejected by
/// `rans::parse` on the CPU; the WGSL RENORM_GUARD prevents any hang.
pub fn decode_tile_gpu_fast_vram(ctx: &GpuContext, payload: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<GpuTile> {
    let esz = dtype.size();
    let count = w as usize * h as usize; // plane length (n in reorder; no u32-overflow)
    let prep = rans::parse(payload).ok_or_else(|| corrupt("bad rANS container"))?;
    if prep.n != count * esz {
        return Err(GpuError::Corrupt(format!("size mismatch: n={} != {count}*{esz}", prep.n)));
    }
    let nbytes = prep.n as u64 * 4;
    ctx.check_buf("fast.out", nbytes)?;
    ctx.scoped(|| {
        let bufs = RansBufs::new(ctx, "f", &prep)?;
        let device = &ctx.device;
        let b_rp = storage_u32(device, &ctx.queue, "f.rp", &[esz as u32, count as u32, w, w, 0, 0], wgpu::BufferUsages::empty());
        let b_sym = ctx.scratch("f.sym", nbytes);
        let b_out = new_buf(device, "f.out", nbytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let p_re = &ctx.pipelines.reorder;
        let bg_rans = bufs.bind(ctx, &b_sym);
        let bg_re = bind_bufs(device, p_re, "f.reorder", &[&b_rp, &b_sym, &b_out]);
        let mut enc = encoder(device, "f.enc");
        pass(ctx, &mut enc, "rans", &ctx.pipelines.rans, &bg_rans, [(prep.cfg.lanes as u32).div_ceil(WG), 1, 1])?;
        pass(ctx, &mut enc, "reorder", p_re, &bg_re, [(esz as u32).div_ceil(WG), 1, 1])?;
        ctx.queue.submit([enc.finish()]);
        Ok(GpuTile { buf: b_out, off_words: 0, words: prep.n, w, h, dtype, enc: TileEnc::BytePerWord })
    })
}

/// WGSL compute decoder for `BC6H_SF16` Mode 11 (`0b00011`) 4x4 hardware blocks + optional 16-bit lossless residual.
/// Replicates `exv_codec::bc6h::decode_bc6h_mono_block` + residual addition bit-for-bit on the GPU.
///
/// Bindings:
/// - `params`: `[tw, th, full_w, x0, y0, has_res, blk_enc]` (`has_res`: `0` = Lossy, `1` = BytePerWord residual from `reorder`,
///   `2` = packed LE `u16` residual from `TAG_RAW`; `blk_enc`: `0` = 4 packed `u32` words/block, `1` = 16 `BytePerWord` words/block).
/// - `blocks`: `BC6H_SF16` Mode 11 block buffer (`nbx * nby * 4` words when `blk_enc == 0`, `nbx * nby * 16` words when `blk_enc == 1`).
/// - `res`: optional residual words (`BytePerWord` when `has_res == 1`, packed `u32` words when `has_res == 2`).
/// - `elems`: output `ElemPerWord(2)` buffer (`(y0 + y) * full_w + (x0 + x)`).
pub(crate) const BC6H_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       params: array<u32>;
@group(0) @binding(1) var<storage, read>       blocks: array<u32>;
@group(0) @binding(2) var<storage, read>       res:    array<u32>;
@group(0) @binding(3) var<storage, read_write> elems:  array<u32>;

fn sign_ext_10(v: u32) -> i32 {
    let m = i32(v & 0x3ffu);
    return select(m, m - 1024, (m & 0x200) != 0);
}

fn unquantize_sf16_10(q: i32) -> i32 {
    if (q == 0) { return 0; }
    if (q < 0) {
        if (q <= -512) { return -0x7fff; }
        return -((((-q) << 15u) + 0x4000) >> 9u);
    }
    if (q >= 511) { return 0x7fff; }
    return ((q << 15u) + 0x4000) >> 9u;
}

fn finish_unquantize_sf16(interp: i32) -> u32 {
    if (interp < 0) {
        let mag = u32(((-interp) * 31) >> 5u);
        return select((mag & 0x7fffu) | 0x8000u, 0u, mag == 0u);
    }
    return u32((interp * 31) >> 5u) & 0x7fffu;
}

fn weight4(idx: u32) -> i32 {
    var w = array<i32, 16>(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 0x40);
    return w[idx & 15u];
}

fn pack4(base: u32) -> u32 {
    return (blocks[base] & 0xffu)
         | ((blocks[base + 1u] & 0xffu) << 8u)
         | ((blocks[base + 2u] & 0xffu) << 16u)
         | ((blocks[base + 3u] & 0xffu) << 24u);
}

@compute @workgroup_size(WG)
fn bc6h_decode(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let tw = params[0]; let th = params[1]; let full_w = params[2];
    let x0 = params[3]; let y0 = params[4]; let has_res = params[5]; let blk_enc = params[6];
    let count = tw * th;
    let px = (wid.y * nwg.x + wid.x) * WG + lid.x;
    if (px >= count) { return; }

    let x = px % tw;
    let y = px / tw;
    let nbx = (tw + 3u) >> 2u;
    let bx = x >> 2u;
    let by = y >> 2u;
    var w0 = 0u; var w1 = 0u; var w2 = 0u; var w3 = 0u;
    if (blk_enc == 0u) {
        let b_idx = (by * nbx + bx) * 4u;
        w0 = blocks[b_idx];
        w1 = blocks[b_idx + 1u];
        w2 = blocks[b_idx + 2u];
        w3 = blocks[b_idx + 3u];
    } else {
        let b_idx = (by * nbx + bx) * 16u;
        w0 = pack4(b_idx);
        w1 = pack4(b_idx + 4u);
        w2 = pack4(b_idx + 8u);
        w3 = pack4(b_idx + 12u);
    }

    let q0 = sign_ext_10(w0 >> 5u);
    let q1 = sign_ext_10(w1 >> 3u);
    let u0 = unquantize_sf16_10(q0);
    let u1 = unquantize_sf16_10(q1);

    let t = (y & 3u) * 4u + (x & 3u);
    var idx = 0u;
    if (t == 0u) {
        idx = (w2 >> 1u) & 0x07u;
    } else if (t < 8u) {
        idx = (w2 >> (4u * t)) & 0x0fu;
    } else {
        idx = (w3 >> (4u * (t - 8u))) & 0x0fu;
    }

    let w_val = weight4(idx);
    let interp = (u0 * (0x40 - w_val) + u1 * w_val + 32) >> 6u;
    var val = finish_unquantize_sf16(interp);

    if (has_res == 1u) {
        let r_lo = res[px * 2u] & 0xffu;
        let r_hi = res[px * 2u + 1u] & 0xffu;
        val = (val + (r_lo | (r_hi << 8u))) & 0xffffu;
    } else if (has_res == 2u) {
        val = (val + (res[px] & 0xffffu)) & 0xffffu;
    }

    elems[(y0 + y) * full_w + (x0 + x)] = val;
}
"#;

/// Decode a `TAG_BC6H` payload (`payload` after stripping `TAG_BC6H`) on the GPU into `dst_elem` at `(x0, y0)` with
/// row stride `full_w` (`ElemPerWord(2)` layout). Used both for single-tile VRAM decode (`decode_tile_bc6h_vram`)
/// and for whole-frame composite decoding (`decode_bc6h_batched`).
#[allow(clippy::too_many_arguments)]
pub fn decode_bc6h_into(
    ctx: &GpuContext,
    payload: &[u8],
    dtype: Dtype,
    tw: u32,
    th: u32,
    full_w: u32,
    x0: u32,
    y0: u32,
    dst_elem: &wgpu::Buffer,
) -> GpuResult<()> {
    if dtype != Dtype::F16 {
        return Err(GpuError::Unsupported(format!("BC6H GPU decode requires F16, got {dtype:?}")));
    }
    let count = (tw as usize).checked_mul(th as usize).ok_or_else(|| corrupt("BC6H tile dimensions overflow"))?;
    if count == 0 {
        return Ok(());
    }
    let nbx = (tw.div_ceil(4)) as usize;
    let nby = (th.div_ceil(4)) as usize;
    let n_blocks = nbx.checked_mul(nby).ok_or_else(|| corrupt("BC6H block count overflow"))?;
    let blocks_bytes = n_blocks.checked_mul(16).ok_or_else(|| corrupt("BC6H block size overflow"))?;
    const BLOCKS_RANS_FLAG: u8 = 0x80; // == exv_codec::bc6h::BLOCKS_RANS_FLAG
    let raw_mode = *payload.first().ok_or_else(|| corrupt("empty BC6H payload"))?;
    let blocks_rans = (raw_mode & BLOCKS_RANS_FLAG) != 0;
    let mode = raw_mode & !BLOCKS_RANS_FLAG;

    let device = &ctx.device;
    let no = wgpu::BufferUsages::empty();
    let mut enc = encoder(device, "bc6h.enc");

    let (blk_enc, b_blocks, tail) = if !blocks_rans {
        let end = 1usize.checked_add(blocks_bytes).ok_or_else(|| corrupt("BC6H blocks overflow"))?;
        let Some(blocks_slice) = payload.get(1..end) else {
            return Err(corrupt("BC6H payload truncated in block array"));
        };
        let block_words = to_elems(blocks_slice, 4);
        let b_blocks = storage_u32(device, &ctx.queue, "bc6h.blocks", &block_words, no);
        (0u32, b_blocks, &payload[end..])
    } else {
        let len_bytes: [u8; 4] = payload
            .get(1..5)
            .ok_or_else(|| corrupt("BC6H rANS block header truncated"))?
            .try_into()
            .map_err(|_| corrupt("BC6H rANS block header"))?;
        let r_len = u32::from_le_bytes(len_bytes) as usize;
        let end = 5usize.checked_add(r_len).ok_or_else(|| corrupt("BC6H rANS block length overflow"))?;
        let Some(r_slice) = payload.get(5..end) else {
            return Err(corrupt("BC6H payload truncated in rANS block stream"));
        };
        let prep = rans::parse(r_slice).ok_or_else(|| corrupt("bad BC6H block rANS container"))?;
        if prep.n != blocks_bytes {
            return Err(GpuError::Corrupt(format!("BC6H blocks size mismatch: n={} != {blocks_bytes}", prep.n)));
        }
        let nbytes = prep.n as u64 * 4;
        ctx.check_buf("bc6h.blocks", nbytes)?;
        let bufs = RansBufs::new(ctx, "bc6h.blk_r", &prep)?;
        let b_rp = storage_u32(device, &ctx.queue, "bc6h.blk_rp", &[16, n_blocks as u32, nbx as u32, nbx as u32, 0, 0], no);
        let b_sym = new_buf(device, "bc6h.blk_sym", nbytes, wgpu::BufferUsages::STORAGE);
        let b_blocks = new_buf(device, "bc6h.blocks", nbytes, wgpu::BufferUsages::STORAGE);
        let bg_rans = bufs.bind(ctx, &b_sym);
        let bg_re = bind_bufs(device, &ctx.pipelines.reorder, "bc6h.blk_reorder", &[&b_rp, &b_sym, &b_blocks]);
        pass(ctx, &mut enc, "bc6h.blk_rans", &ctx.pipelines.rans, &bg_rans, [(prep.cfg.lanes as u32).div_ceil(WG), 1, 1])?;
        pass(ctx, &mut enc, "bc6h.blk_reorder", &ctx.pipelines.reorder, &bg_re, [1, 1, 1])?;
        (1u32, b_blocks, &payload[end..])
    };

    let (has_res, b_res) = match mode {
        0 => {
            if !tail.is_empty() {
                return Err(corrupt("BC6H lossy payload has trailing bytes"));
            }
            (0u32, storage_u32(device, &ctx.queue, "bc6h.res_dummy", &[0], no))
        }
        1 => match classify_blob(tail) {
            Some(BlobKind::Rans(r_payload)) => {
                let prep = rans::parse(r_payload).ok_or_else(|| corrupt("bad BC6H residual rANS container"))?;
                if prep.n != count * 2 {
                    return Err(GpuError::Corrupt(format!("BC6H residual size mismatch: n={} != {}", prep.n, count * 2)));
                }
                let nbytes = prep.n as u64 * 4;
                ctx.check_buf("bc6h.res", nbytes)?;
                let bufs = RansBufs::new(ctx, "bc6h.r", &prep)?;
                let b_rp = storage_u32(device, &ctx.queue, "bc6h.rp", &[2, count as u32, tw, tw, 0, 0], no);
                let b_sym = ctx.scratch("bc6h.sym", nbytes);
                let b_res = new_buf(device, "bc6h.res", nbytes, wgpu::BufferUsages::STORAGE);
                let bg_rans = bufs.bind(ctx, &b_sym);
                let bg_re = bind_bufs(device, &ctx.pipelines.reorder, "bc6h.reorder", &[&b_rp, &b_sym, &b_res]);
                pass(ctx, &mut enc, "bc6h.rans", &ctx.pipelines.rans, &bg_rans, [(prep.cfg.lanes as u32).div_ceil(WG), 1, 1])?;
                pass(ctx, &mut enc, "bc6h.reorder", &ctx.pipelines.reorder, &bg_re, [1, 1, 1])?;
                (1u32, b_res)
            }
            Some(BlobKind::Raw(r_raw)) => {
                if r_raw.len() != count * 2 {
                    return Err(corrupt("BC6H raw residual length mismatch"));
                }
                let words = to_elems(r_raw, 2);
                (2u32, storage_u32(device, &ctx.queue, "bc6h.res_raw", &words, no))
            }
            _ => return Err(corrupt("BC6H residual has invalid tag")),
        },
        _ => return Err(corrupt("BC6H payload has unknown mode byte")),
    };

    let b_params = storage_u32(device, &ctx.queue, "bc6h.params", &[tw, th, full_w, x0, y0, has_res, blk_enc], no);
    let bg_bc6h = bind_bufs(device, &ctx.pipelines.bc6h, "bc6h.bg", &[&b_params, &b_blocks, &b_res, dst_elem]);
    pass(ctx, &mut enc, "bc6h", &ctx.pipelines.bc6h, &bg_bc6h, ctx.split_1d("bc6h", (count as u64).div_ceil(WG as u64))?)?;
    ctx.queue.submit([enc.finish()]);
    Ok(())
}

/// **GPU tile decode for the BC6H profile into VRAM — NO readback.**
///
/// Decodes `BC6H_SF16` Mode 11 blocks (plus optional rANS 16-bit lossless residual) on the GPU into an
/// `ElemPerWord(2)` [`GpuTile`], bit-for-bit identical to CPU `exv_codec::bc6h::decode_plane_into`.
pub fn decode_tile_bc6h_vram(ctx: &GpuContext, payload: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<GpuTile> {
    if dtype != Dtype::F16 {
        return Err(GpuError::Unsupported(format!("BC6H GPU decode requires F16, got {dtype:?}")));
    }
    let count = (w as usize).checked_mul(h as usize).ok_or_else(|| corrupt("BC6H tile dimensions overflow"))?;
    ctx.check_buf("bc6h.elem", count as u64 * 4)?;
    ctx.scoped(|| {
        let b_elem = new_buf(&ctx.device, "bc6h.elem", (count as u64 * 4).max(4), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        decode_bc6h_into(ctx, payload, dtype, w, h, w, 0, 0, &b_elem)?;
        Ok(GpuTile { buf: b_elem, off_words: 0, words: count, w, h, dtype, enc: TileEnc::ElemPerWord(2) })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shader source of the crate as `(name, source WITHOUT the constants prelude)` — exactly what
    /// `build_pipelines` / the lazy pipeline getters hand to `make_pipe` (which prepends the prelude).
    fn sources() -> Vec<(String, String)> {
        let mut v = vec![
            ("rans".to_string(), rans_src()),
            ("rans_b".to_string(), format!("{DESC_STRUCT}{}", rans_b_src())),
            ("repack_b".to_string(), format!("{DESC_STRUCT}{REPACK_B_BODY}")),
            ("undelta_b".to_string(), format!("{DESC_STRUCT}{UNDELTA_B_BODY}")),
            ("wav_b".to_string(), format!("{DESC_STRUCT}{}", wavelet_b_src())),
            ("wav".to_string(), wavelet_src()),
            ("repack".to_string(), REPACK_SHADER.to_string()),
            ("reorder".to_string(), REORDER_SHADER.to_string()),
            ("bc6h".to_string(), BC6H_SHADER.to_string()),
            ("exvrt_inv".to_string(), format!("{RT_DESC_STRUCT}{}", exvrt_inv_src())),
            ("exvrt_fused".to_string(), format!("{DESC_STRUCT}{RT_DESC_STRUCT}{}", exvrt_fused_src())),
            ("rt_prep".to_string(), format!("{DESC_STRUCT}{RT_DESC_STRUCT}{TILE_DESC_STRUCT}{RT_PREP_BODY}")),
        ];
        for lanes in RT_LANE_SET {
            v.push((format!("rans_b_rt{lanes}"), format!("{DESC_STRUCT}{}", rans_b_rt_src(lanes))));
        }
        v.extend(display_srcs().into_iter().map(|(label, src, _)| (label.to_string(), src)));
        v
    }

    fn parse(name: &str, src: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        module
    }

    /// Every spliced kernel (shared rANS reader, lift templates, lane variants, display passes) is valid WGSL
    /// WITH the generated constants prelude — catches a broken splice token or a misspelled constant name
    /// without a GPU.
    #[test]
    fn composed_shaders_validate() {
        parse("prelude", prelude());
        for (name, src) in sources() {
            parse(&name, &wgsl(&src));
        }
    }

    /// The descriptor widths the hosts stride by equal the WGSL struct member counts (struct layout is the
    /// other half of every `descs.extend_from_slice(&[..16 words..])`).
    #[test]
    fn desc_words_match_structs() {
        let words = |src: &str, name: &str| {
            let m = parse(name, &wgsl(src));
            m.types
                .iter()
                .find_map(|(_, t)| match (&t.name, &t.inner) {
                    (Some(n), naga::TypeInner::Struct { members, .. }) if n == name => Some(members.len()),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("struct {name} not found"))
        };
        assert_eq!(words(DESC_STRUCT, "Desc"), DESC_WORDS);
        assert_eq!(words(RT_DESC_STRUCT, "RtDesc"), RT_DESC_WORDS);
        assert_eq!(words(TILE_DESC_STRUCT, "TileDesc"), TILE_DESC_WORDS);
    }

    /// Strip `// ...` line comments (prose may legitimately mention numbers).
    fn code_of(src: &str) -> String {
        src.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("
")
    }

    /// SSOT guard: a literal that duplicates a Rust constant ([`WG`], [`SLOT_MAX`], [`ALPHA`], `RANS_L`, the
    /// scratch sizes, the lossy-log constants) must never reappear in a shader — only the generated NAMES.
    /// Tokens are whole words, so hex table entries such as `0x3f800064u` do not trip the `64u` check.
    #[test]
    fn shaders_use_named_consts() {
        const BANNED: [&str; 12] =
            ["64", "64u", "256", "256u", "4096", "4096u", "8388608", "8388608u", "16384", "16384u", "8192", "8192u"];
        for (name, src) in sources() {
            let code = code_of(&src);
            for tok in code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                assert!(!BANNED.contains(&tok), "{name}: bare literal `{tok}` duplicates a Rust constant — use the prelude name");
            }
            for (i, _) in code.match_indices("@workgroup_size(") {
                let next = code[i + "@workgroup_size(".len()..].chars().next();
                assert!(next.is_some_and(|c| !c.is_ascii_digit()), "{name}: numeric @workgroup_size — use WG/WG2");
            }
        }
    }

    #[test]
    fn push_lane_writes_start_end_pairs() {
        let mut v = Vec::new();
        push_lane(&mut v, 0, 10).unwrap();
        push_lane(&mut v, 10, 17).unwrap();
        assert_eq!(v, [0, 10, 10, 17]);
        assert!(push_lane(&mut v, 0, usize::MAX).is_err());
    }
}
