//! `rt.rs` — EXV-RT (`exv_codec::blockrt_dwt`) GPU decode as a WHOLE-FRAME batch.
//!
//! # Why
//! The perf audit measured 89-99% of a cold EXV-RT frame decode as host overhead: every `(tile, channel)` blob
//! used to get its own parse, ~7 storage buffers, 3 bind groups, one encoder and one submit. The kernels
//! already take descriptor grids (`Desc`/`RtDesc` per block), so a whole frame is just MORE descriptors:
//! [`RtBatch`] parses every blob ([`RtTile::parse`], pure and parallelisable by the caller), concatenates
//! streams / tables / lane spans into ONE set of storage buffers per frame, and records every dispatch into
//! ONE encoder / ONE submit ([`RtResident::dispatch`]).
//!
//! # Layout
//! * Tiles are sorted by `(channel, block, lanes, table)` and cut into *chunks* ([`plan_chunks`]): a chunk never
//!   mixes channels (each channel has its own output buffer / bind group) and its symbol scratch stays below
//!   the chunk cap ([`GpuContext::chunk_words`], bounded by the device buffer limit) — an 8k+ frame is several chunks, not an error. The scratch is
//!   ONE pooled buffer reused by every chunk (passes are ordered by wgpu).
//! * The thread-dense fast rANS kernel loads ONE table per workgroup (`descs[wid * WG]`). Units of equal table
//!   are therefore adjacent and each table run is padded to a multiple of `WG` with inert descriptors
//!   (`n = 0` / `block = 0`, skipped by the kernels), so a workgroup never straddles two tables.
//! * Frame-wide buffers (`streams`, tables, `lane_off`) are shared by all chunks; only the two descriptor
//!   arrays are per chunk.
//!
//! # Where used
//! `exv-gpu::dispatch::decode_blockrt_batched` (whole still), `decode_exvrt_into` / [`decode_tile_exvrt_vram`]
//! (single tile = a one-tile batch, ONE codepath) and [`ExvrtWarmTile`] (resident re-decode, the warm bench).

use crate::prelude::*;
use crate::*;
use rans::{RANS_L, RANS_UPPER};
use std::collections::HashMap;

/// Minimum number of blocks in a chunk before the thread-dense kernel pays off. It packs `WG` blocks per
/// workgroup, so a single 256-block tile (4 workgroups) starves the SMs and is SLOWER than the
/// one-block-per-workgroup kernel; a frame chunk has thousands of blocks and clears the floor. Measured on an
/// RTX 3080 Ti (provisional STEP-1 heuristic of the old per-tile path, kept verbatim).
const FAST_MIN_BLOCKS: usize = 4096;

/// Blocks per parallel task of [`RtTile::parse`]'s validation loop.
const PARSE_MIN_BLOCKS: usize = 1024;

/// One parsed + validated EXV-RT blob (`blockrt_dwt` frozen layout, tag stripped), borrowing the payload.
///
/// Validation mirrors the CPU `blockrt_dwt::decode_plane` / `rans::decode_block` contract (header, offset index,
/// per-block lane layout, initial rANS states) and rejects as [`GpuError::Corrupt`] BEFORE any GPU work. Pure
/// and `Send`: callers parse many tiles in parallel and push them into an [`RtBatch`] sequentially.
pub struct RtTile<'a> {
    block: u32,
    q: u32,
    lanes: u32,
    table_log: u32,
    nbx: u32,
    /// The tile's pixel size: its blocks overhanging it (`tile % block != 0`) must not be written.
    tile_w: u32,
    tile_h: u32,
    /// The 512 B frequency table as stored: the dedupe key of identical models across tiles.
    freq_key: &'a [u8],
    model: rans::Model,
    region: &'a [u8],
    /// Per-block coefficient byte width (1..=4), validated.
    widths: &'a [u8],
    /// `[start, end)` byte span of every lane of every block, RELATIVE to `region` ([`push_lane`] layout).
    lane_off: Vec<u32>,
    /// Total rANS byte-symbols of the tile (`sum block^2 * width`): its share of the symbol scratch.
    syms: usize,
    /// Words of the byte-packed symbol layout (`sum ceil(block^2 * width / 4)`): the scratch share of a `packed` chunk.
    words: usize,
}

impl<'a> RtTile<'a> {
    /// Parse and validate `payload` for a `tile_w` x `tile_h` tile.
    pub fn parse(payload: &'a [u8], tile_w: u32, tile_h: u32) -> GpuResult<Self> {
        if tile_w == 0 || tile_h == 0 {
            return Err(corrupt("exv-rt GPU: empty tile"));
        }
        let hdr = payload.get(..8).ok_or_else(|| corrupt("exv-rt GPU: short header"))?;
        let (block, q, table_log) = (hdr[0] as u32, hdr[1] as u32, hdr[2] as u32);
        // rANS interleave width (per-plane, shared by every block; entropy-only). The fast kernel is specialised
        // for `RT_LANE_SET`; the slow `rans_b` kernel strides any `lanes` <= 255 over its 64 threads.
        let lanes = hdr[3] as u32;
        if block == 0 || block > RT_MAX_BLOCK {
            return Err(GpuError::Corrupt(format!("exv-rt GPU: bad block {block}")));
        }
        if lanes == 0 {
            return Err(corrupt("exv-rt GPU: zero lanes"));
        }
        // `q` is the detail-subband quant shift; the GPU dequant multiplies by `1<<q` in i32 and must stay below
        // the F16 element width (`valid_q` for F16 = q < 16). Our encoder never emits q>=16, so it is a hard
        // `Err` (the CPU is the strict fallback), NOT a silent mimic of the CPU's "skip quant on invalid q".
        if q >= 16 {
            return Err(GpuError::Unsupported(format!("exv-rt GPU: q={q} >= 16 invalid for f16 (needs CPU)")));
        }
        if table_log > MAX_GPU_TABLE_LOG as u32 {
            return Err(GpuError::Unsupported(format!("exv-rt GPU: table_log {table_log} > {MAX_GPU_TABLE_LOG} (CPU fallback)")));
        }
        let nbx = u16::from_le_bytes([hdr[4], hdr[5]]) as u32;
        let nby = u16::from_le_bytes([hdr[6], hdr[7]]) as u32;
        if nbx != tile_w.div_ceil(block) || nby != tile_h.div_ceil(block) {
            return Err(corrupt("exv-rt GPU: tile geometry mismatch"));
        }
        let mut p = 8usize;
        let freq_key = payload.get(p..p + 512).ok_or_else(|| corrupt("exv-rt GPU: short freq table"))?;
        let model = rans::Model::from_freqs(freq_key, table_log).ok_or_else(|| corrupt("exv-rt GPU: bad freq table"))?;
        p += 512;
        let nblk = (nbx as usize) * (nby as usize);
        let idx = payload.get(p..p + (nblk + 1) * 4).ok_or_else(|| corrupt("exv-rt GPU: short offset index"))?;
        p += (nblk + 1) * 4;
        let widths = payload.get(p..p + nblk).ok_or_else(|| corrupt("exv-rt GPU: short width array"))?;
        p += nblk;
        let region = payload.get(p..).ok_or_else(|| corrupt("exv-rt GPU: no substream region"))?;
        let off = |i: usize| -> u32 { u32::from_le_bytes([idx[4 * i], idx[4 * i + 1], idx[4 * i + 2], idx[4 * i + 3]]) };
        // Offset index must be monotonic and end exactly at the region length (no OOB, no trailing bytes).
        if (0..nblk).any(|i| off(i + 1) < off(i)) {
            return Err(corrupt("exv-rt GPU: non-monotonic offset index"));
        }
        if off(nblk) as usize != region.len() {
            return Err(corrupt("exv-rt GPU: offset index end != region len"));
        }

        // Per-block substream = `[lane_len[lanes-1]:u32][lane streams...]` (last lane = the remainder), parsed
        // EXACTLY like `rans::decode_block`: every lane >= 4 bytes with an initial state in `[RANS_L, RANS_UPPER)`.
        let lanes_us = lanes as usize;
        let hdr_bytes = (lanes_us - 1) * 4;
        // Per-block floor of the CPU contract (header + >= 4 bytes per lane): a forged `nblk` that cannot even
        // fit its floor in the region is rejected BEFORE any `nblk`-sized reservation.
        let floor = nblk.checked_mul(8 * lanes_us - 4).ok_or_else(|| corrupt("exv-rt GPU: block count overflow"))?;
        if floor > region.len() {
            return Err(corrupt("exv-rt GPU: region too small for the declared block count"));
        }
        let ncoeff = (block * block) as usize;
        // Blocks are independent, so a big single tile (an 8k plane = 262k blocks) validates across the pool; the
        // minimum split keeps a 256-block tile on one thread (its parse already runs in a parallel per-tile map).
        use rayon::prelude::*;
        let to_u32 = |v: usize| u32::try_from(v).map_err(|_| GpuError::Limit(format!("lane offset {v} exceeds the u32 shader range")));
        let mut lane_off = vec![0u32; nblk * lanes_us * 2];
        let (syms, words) = lane_off
            .par_chunks_mut(lanes_us * 2)
            .zip(widths.par_iter())
            .enumerate()
            .with_min_len(PARSE_MIN_BLOCKS)
            .map(|(bi, (out, &wb))| -> GpuResult<(usize, usize)> {
                let wbytes = wb as usize;
                if !(1..=4).contains(&wbytes) {
                    return Err(GpuError::Corrupt(format!("exv-rt GPU: bad width {wbytes}")));
                }
                let start0 = off(bi) as usize;
                let sub = region.get(start0..off(bi + 1) as usize).ok_or_else(|| corrupt("exv-rt GPU: block substream OOB"))?;
                let streams_part = sub.get(hdr_bytes..).ok_or_else(|| corrupt("exv-rt GPU: short lane header"))?;
                let lane_len = |k: usize| -> usize {
                    u32::from_le_bytes([sub[4 * k], sub[4 * k + 1], sub[4 * k + 2], sub[4 * k + 3]]) as usize
                };
                let mut known = 0usize;
                for k in 0..lanes_us - 1 {
                    // `hdr_bytes <= sub.len()` was checked above, so every header word is in range.
                    known = known.checked_add(lane_len(k)).ok_or_else(|| corrupt("exv-rt GPU: lane length overflow"))?;
                }
                let last = streams_part.len().checked_sub(known).ok_or_else(|| corrupt("exv-rt GPU: lane lengths exceed the block"))?;
                let mut lq = 0usize;
                for (k, span) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let len = if k + 1 < lanes_us { lane_len(k) } else { last };
                    let lane = streams_part.get(lq..lq + len).ok_or_else(|| corrupt("exv-rt GPU: lane OOB"))?;
                    let init = lane.get(..4).ok_or_else(|| corrupt("exv-rt GPU: lane shorter than its 4-byte state"))?;
                    let state = u32::from_le_bytes([init[0], init[1], init[2], init[3]]);
                    if !(RANS_L..RANS_UPPER).contains(&state) {
                        return Err(corrupt("exv-rt GPU: initial rANS state out of range"));
                    }
                    // absolute [start, end) byte span within `region` (the kernel bounds every read by `end`)
                    let start = start0 + hdr_bytes + lq;
                    span[0] = to_u32(start)?;
                    span[1] = to_u32(start + len)?;
                    lq += len;
                }
                Ok((ncoeff * wbytes, (ncoeff * wbytes).div_ceil(4)))
            })
            .try_reduce(|| (0usize, 0usize), |a, b| Ok((a.0 + b.0, a.1 + b.1)))?;
        Ok(Self { block, q, lanes, table_log, nbx, tile_w, tile_h, freq_key, model, region, widths, lane_off, syms, words })
    }
}

/// Where a pushed tile lands: output channel `ch` (index into [`RtBatch::build`]'s `outs`), a `full_w` x `full_h` plane
/// that starts at word `base` of that buffer, and the tile's pixel origin `(x0, y0)` inside the plane. Blocks overhanging
/// the plane edge are simply not written (mirrors `decode_plane`'s crop). `base` lets MANY differently-sized planes
/// (the tiles of a resident-atlas request) share ONE output buffer per channel, hence one dispatch per channel.
#[derive(Clone, Copy, Debug)]
pub struct RtDest {
    pub ch: usize,
    pub base: u32,
    pub x0: u32,
    pub y0: u32,
    pub full_w: u32,
    pub full_h: u32,
}

impl RtDest {
    /// A plane at the start of channel `ch`'s buffer (`base = 0`).
    pub fn plane(ch: usize, x0: u32, y0: u32, full_w: u32, full_h: u32) -> Self {
        Self { ch, base: 0, x0, y0, full_w, full_h }
    }
}

/// A pushed tile with its destination and its frame-wide table / stream / lane bases.
struct Placed<'a> {
    ch: usize,
    base: u32,
    x0: u32,
    y0: u32,
    full_w: u32,
    full_h: u32,
    /// Exclusive crop limits: the tile's own right/bottom edge clamped to the plane (CPU `decode_plane` crops to the
    /// TILE, not the plane — a block overhanging an interior tile must not touch its neighbour).
    x1: u32,
    y1: u32,
    block: u32,
    q: u32,
    lanes: u32,
    nbx: u32,
    region: &'a [u8],
    widths: &'a [u8],
    /// Region-relative lane spans ([`RtTile::lane_off`]); rebased by `stream_base` when uploaded.
    lane_off: Vec<u32>,
    syms: usize,
    words: usize,
    /// Index into [`RtBatch::tables`].
    table: usize,
    /// Byte offset of the tile's region inside the frame-wide `streams` buffer.
    stream_base: u32,
    /// Frame-wide index of the tile's first lane (`lane_off` pair index); block `b` starts at `+ b * lanes`.
    lane_base: u32,
    /// Byte offset of the tile's per-block widths inside the frame-wide `widths` buffer.
    w_base: u32,
}

/// One deduplicated rANS model of the frame (its words land in the frame-wide `freq`/`cum`/`slot` buffers at
/// `*_base` words).
struct Table {
    log: u32,
    model: rans::Model,
    freq: u32,
    cum: u32,
    slot: u32,
}

/// Host-side accumulator of a whole frame's EXV-RT tiles (module docs). `'a` = the borrowed blob payloads.
/// [`push`](Self::push) only does O(1) bookkeeping (dedupe a table, assign stream / lane bases); every byte is
/// laid out by [`build`](Self::build), in parallel, straight into the upload staging memory.
#[derive(Default)]
pub struct RtBatch<'a> {
    tiles: Vec<Placed<'a>>,
    tables: Vec<Table>,
    table_ix: HashMap<(u32, &'a [u8]), usize>,
    /// Running totals of the frame-wide buffers (words / bytes) — the bases of the next pushed tile.
    freq_words: usize,
    slot_words: usize,
    stream_bytes: usize,
    lane_words: usize,
    /// Total per-block width bytes (the frame-wide `widths` buffer the prep kernel reads).
    width_bytes: usize,
}

/// One chunk's recorded work: its two pipelines + bind groups + grids (see [`RtResident::dispatch`]).
struct RtPass {
    rans: wgpu::ComputePipeline,
    bg_rans: wgpu::BindGroup,
    rans_grid: [u32; 3],
    inv: wgpu::ComputePipeline,
    bg_inv: wgpu::BindGroup,
    inv_grid: [u32; 3],
}

/// A built batch: every buffer + bind group resident, ready to [`dispatch`](Self::dispatch) (repeatedly: both
/// stages are idempotent over the same inputs, which is the warm-bench / resident-player primitive).
pub struct RtResident {
    passes: Vec<RtPass>,
    /// The chunks' shared symbol scratch (returned to the pool on drop; bind groups hold the buffer too).
    _sym: Scratch,
    fast: bool,
}

/// `u32` words as bytes for a [`storage_parts`] fill (zero-copy view; little-endian host, see `storage_u32`).
fn put_words(dst: &mut wgpu::WriteOnly<'_, [u8]>, words: &[u32]) {
    dst.copy_from_slice(bytemuck::cast_slice(words));
}

/// Expand per-tile records into the per-block `Desc` (rANS) and `RtDesc` (inverse) arrays of `units` rows ON THE GPU
/// (`rt_prep`, one thread per tile) and submit it: the one owner of the descriptor derivation (the chunked batch build and
/// the forged-record test both call it). `recs` field order = WGSL `TileDesc`; `widths` = the frame-wide packed per-block
/// width bytes the records index (`w_base`/`w_len`, clamped in the kernel). Rows of units no record covers stay
/// zero = inert. Ordered before every later submit of the queue.
pub fn prep_rows(ctx: &GpuContext, recs: &[[u32; TILE_DESC_WORDS]], widths: &wgpu::Buffer, units: usize) -> GpuResult<(wgpu::Buffer, wgpu::Buffer)> {
    let device = &ctx.device;
    let b_recs = storage_u32(device, &ctx.queue, "rt.tiles", bytemuck::cast_slice::<_, u32>(recs), wgpu::BufferUsages::empty());
    // COPY_SRC: costs nothing and lets tests / diagnostics read the derived rows back.
    let ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let rows = units.max(1) as u64;
    let b_rd = new_buf(device, "rt.rans_desc", rows * DESC_WORDS as u64 * 4, ro);
    let b_rt = new_buf(device, "rt.inv_desc", rows * RT_DESC_WORDS as u64 * 4, ro);
    let p = &ctx.pipelines.rt_prep;
    let bg = bind_bufs(device, p, "rt.bg_prep", &[&b_recs, widths, &b_rd, &b_rt]);
    let grid = ctx.split_1d("exv-rt prep", (recs.len() as u64).div_ceil(WG as u64))?;
    let mut enc = encoder(device, "rt.prep");
    pass(ctx, &mut enc, "rt.prep", p, &bg, grid)?;
    ctx.queue.submit([enc.finish()]);
    Ok((b_rd, b_rt))
}

impl<'a> RtBatch<'a> {
    /// Empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of pushed tiles.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether no tile was pushed.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Add `t`, destined for `dest` (see [`RtDest`]).
    pub fn push(&mut self, t: RtTile<'a>, dest: RtDest) -> GpuResult<()> {
        let RtDest { ch, base, x0, y0, full_w, full_h } = dest;
        let lim = |what: &str| GpuError::Limit(format!("exv-rt batch: {what} exceeds the u32 shader index range"));
        let RtTile { block, q, lanes, table_log, nbx, tile_w, tile_h, freq_key, model, region, widths, lane_off, syms, words } = t;
        // Absolute lane spans index the frame-wide `streams`: they must stay u32-addressable.
        let stream_base = u32::try_from(self.stream_bytes).map_err(|_| lim("substream bytes"))?;
        let stream_end = self.stream_bytes + region.len();
        u32::try_from(stream_end).map_err(|_| lim("substream bytes"))?;
        let lane_base = u32::try_from(self.lane_words / 2).map_err(|_| lim("lane count"))?;
        let table = match self.table_ix.get(&(table_log, freq_key)) {
            Some(&i) => i,
            None => {
                let base = |words: usize| u32::try_from(words).map_err(|_| lim("table words"));
                let m = model.slot_to_sym().len();
                let tb = Table { log: table_log, model, freq: base(self.freq_words)?, cum: base(self.freq_words)?, slot: base(self.slot_words)? };
                self.freq_words += 256;
                self.slot_words += m;
                self.tables.push(tb);
                self.table_ix.insert((table_log, freq_key), self.tables.len() - 1);
                self.tables.len() - 1
            }
        };
        let w_base = u32::try_from(self.width_bytes).map_err(|_| lim("width bytes"))?;
        self.width_bytes += widths.len();
        self.stream_bytes = stream_end;
        self.lane_words += lane_off.len();
        self.tiles.push(Placed { ch, base, x0, y0, full_w, full_h, x1: x0.saturating_add(tile_w).min(full_w), y1: y0.saturating_add(tile_h).min(full_h), block, q, lanes, nbx, region, widths, lane_off, syms, words, table, stream_base, lane_base, w_base });
        Ok(())
    }

    /// Lay out the frame-wide buffers (parallel), plan the chunks and build every descriptor array + bind group.
    /// `outs[ch]` is the destination composite of channel `ch` (`STORAGE`, >= `full_w * full_h` words).
    ///
    /// `cancel` is polled ([`cancelled_gpu`]) before the frame-wide uploads and before each chunk's staging + descriptor
    /// submit, so a seek stops a 16k build at the next chunk (the previous 20 ms uncancellable tail) instead of at its end.
    /// Only the batch's own buffers exist at a checkpoint: an `Err` just drops them. A one-shot caller passes `Cancel::new()`.
    pub fn build(self, ctx: &GpuContext, outs: &[&wgpu::Buffer], cancel: &bufpool::Cancel) -> GpuResult<RtResident> {
        if self.tiles.is_empty() {
            return Err(corrupt("exv-rt batch: no tiles"));
        }
        cancelled_gpu(cancel)?;
        for p in &self.tiles {
            let out = outs.get(p.ch).ok_or_else(|| corrupt("exv-rt batch: channel without an output buffer"))?;
            let need = (u64::from(p.base) + u64::from(p.full_w) * u64::from(p.full_h)) * 4;
            if !out.usage().contains(wgpu::BufferUsages::STORAGE) || out.size() < need {
                return Err(GpuError::Corrupt(format!("exv-rt batch: output must be STORAGE and >= {need} B (has {} B)", out.size())));
            }
        }
        for (what, bytes) in [
            ("exv-rt freq", self.freq_words * 4),
            ("exv-rt slot", self.slot_words * 4),
            ("exv-rt streams", self.stream_bytes.next_multiple_of(4)),
            ("exv-rt lane_off", self.lane_words * 4),
            ("exv-rt widths", self.width_bytes.next_multiple_of(4)),
        ] {
            ctx.check_buf(what, bytes as u64)?;
        }
        let device = &ctx.device;
        // Frame-wide buffers, every part filled by its own rayon task.
        let width_parts: Vec<usize> = self.tiles.iter().map(|p| p.widths.len()).collect();
        let b_widths = storage_parts(ctx, "rt.widths", &width_parts, |i, mut d| d.copy_from_slice(self.tiles[i].widths))?;
        let tab_parts = |words: fn(&Table) -> usize| -> Vec<usize> { self.tables.iter().map(|t| words(t) * 4).collect() };
        let b_freq = storage_parts(ctx, "rt.freq", &tab_parts(|_| 256), |i, mut d| {
            let v: Vec<u32> = self.tables[i].model.freq().iter().map(|&f| f as u32).collect();
            put_words(&mut d, &v);
        })?;
        let b_cum = storage_parts(ctx, "rt.cum", &tab_parts(|_| 256), |i, mut d| {
            let v: Vec<u32> = self.tables[i].model.cum().iter().map(|&c| c as u32).collect();
            put_words(&mut d, &v);
        })?;
        let b_slot = storage_parts(ctx, "rt.slot", &tab_parts(|t| t.model.slot_to_sym().len()), |i, mut d| {
            let v: Vec<u32> = self.tables[i].model.slot_to_sym().iter().map(|&s| s as u32).collect();
            put_words(&mut d, &v);
        })?;
        // Streams are raw bytes read through `rd_byte` (u32 words, LE): the tiles' regions, back to back.
        let stream_parts: Vec<usize> = self.tiles.iter().map(|p| p.region.len()).collect();
        let b_streams = storage_parts(ctx, "rt.streams", &stream_parts, |i, mut d| d.copy_from_slice(self.tiles[i].region))?;
        // Absolute lane spans = the tile-relative ones + the tile's stream base.
        let lane_parts: Vec<usize> = self.tiles.iter().map(|p| p.lane_off.len() * 4).collect();
        let b_off = storage_parts(ctx, "rt.off", &lane_parts, |i, mut d| {
            let p = &self.tiles[i];
            let v: Vec<u32> = p.lane_off.iter().map(|&o| o + p.stream_base).collect();
            put_words(&mut d, &v);
        })?;

        // Sort so that chunks are runs of one (channel, block, lanes) and equal tables are adjacent.
        let mut order: Vec<usize> = (0..self.tiles.len()).collect();
        order.sort_by_key(|&i| {
            let p = &self.tiles[i];
            (p.ch, p.block, p.lanes, p.table)
        });
        let group = |p: &Placed| ((p.ch as u64) << 16) | ((p.block as u64) << 8) | p.lanes as u64;
        let biggest = self.tiles.iter().map(|p| p.syms).max().unwrap_or(0);
        let chunks = plan_chunks(order.iter().map(|&i| (group(&self.tiles[i]), self.tiles[i].syms)), ctx.chunk_words(biggest))?;
        // Per chunk: its tiles, whether the thread-dense rANS kernel runs (`use_fast`) and whether that kernel byte-packs
        // its symbols ([`RT_PACK_LANES`]). The scratch is sized by the real words per chunk (a packed chunk needs ~1/4).
        let plan: Vec<(Vec<&Placed>, bool, bool)> = chunks
            .iter()
            .map(|c| {
                let tiles: Vec<&Placed> = order[c.units.clone()].iter().map(|&i| &self.tiles[i]).collect();
                let first = tiles[0];
                let blocks: usize = tiles.iter().map(|p| p.widths.len()).sum();
                let use_fast = first.block <= RT_FAST_BLOCK && blocks >= FAST_MIN_BLOCKS && ctx.pipelines.rans_b_rt_for(first.lanes).is_some();
                (tiles, use_fast, use_fast && first.lanes == RT_PACK_LANES)
            })
            .collect();
        let sym_words = plan.iter().zip(&chunks).map(|((tiles, _, packed), c)| if *packed { tiles.iter().map(|p| p.words).sum() } else { c.words }).max().unwrap_or(0);
        let sym = ctx.scratch("rt.sym", (sym_words.max(1) * 4) as u64);

        let mut passes = Vec::with_capacity(chunks.len());
        let mut fast = false;
        for (tiles, use_fast, packed) in &plan {
            cancelled_gpu(cancel)?;
            let (use_fast, packed) = (*use_fast, *packed);
            let first = tiles[0];
            let rans_fast = ctx.pipelines.rans_b_rt_for(first.lanes);
            let (b_rdesc, b_rtdesc, units) = self.chunk_descs(ctx, tiles, use_fast, packed, &b_widths)?;
            // Fast kernel: WG blocks per workgroup; slow `rans_b`: one block per workgroup. The cooperative inverse takes
            // `RT_INV_BLOCKS` blocks per workgroup (block<=16, below), else the slow 1-thread kernel.
            let rans_groups = if use_fast { (units as u64).div_ceil(WG as u64) } else { units as u64 };
            let p_rans = match (use_fast, rans_fast) {
                (true, Some(p)) => p.clone(),
                _ => ctx.pipelines.rans_b.clone(),
            };
            // Cooperative inverse (block <= 16): `RT_INV_BLOCKS` blocks per workgroup; else the 1-thread-per-block kernel, one block per workgroup.
            let (p_inv, inv_groups) = if first.block <= RT_FAST_BLOCK {
                (&ctx.pipelines.exvrt_inv_shared, (units as u64).div_ceil(u64::from(RT_INV_BLOCKS)))
            } else {
                (&ctx.pipelines.exvrt_inv, units as u64)
            };
            let bg_rans = bind_bufs(device, &p_rans, "rt.bg_rans", &[&b_rdesc, &b_freq, &b_cum, &b_slot, &b_streams, &b_off, &sym]);
            let bg_inv = bind_bufs(device, p_inv, "rt.bg_inv", &[&b_rtdesc, &sym, outs[first.ch], &ctx.pipelines.exp2_t]);
            passes.push(RtPass {
                rans: p_rans,
                bg_rans,
                rans_grid: ctx.split_1d("exv-rt rans", rans_groups)?,
                inv: p_inv.clone(),
                bg_inv,
                inv_grid: ctx.split_1d("exv-rt inverse", inv_groups)?,
            });
            fast |= use_fast;
        }
        Ok(RtResident { passes, _sym: sym, fast })
    }

    /// The rANS `Desc` and inverse `RtDesc` buffers of one chunk (chunk-relative symbol bases) and its unit count
    /// INCLUDING the inert padding. With `fast`, every table run is padded to a multiple of `WG` (module docs):
    /// padding rows are `n = 0` (rANS no-op) / `block = 0` (inverse early-out). The host only emits one
    /// [`TILE_DESC_WORDS`] record per tile; the `rt_prep` kernel ([`prep_rows`]) derives every per-block row on the GPU
    /// (82 of the 91 MB a host-built 8k frame uploaded).
    fn chunk_descs(&self, ctx: &GpuContext, tiles: &[&Placed], fast: bool, packed: bool, widths: &wgpu::Buffer) -> GpuResult<(wgpu::Buffer, wgpu::Buffer, usize)> {
        let mut recs: Vec<[u32; TILE_DESC_WORDS]> = Vec::with_capacity(tiles.len());
        let (mut units, mut sym_base) = (0usize, 0u64);
        let lim = |what: &str, v: u64| u32::try_from(v).map_err(|_| GpuError::Limit(format!("exv-rt chunk {what} exceeds u32")));
        for (k, p) in tiles.iter().enumerate() {
            let tb = &self.tables[p.table];
            let nblk = p.widths.len();
            let unit0 = units;
            units += nblk;
            // The run ends here when the next tile has another table (or the chunk ends).
            if fast && tiles.get(k + 1).is_none_or(|nx| nx.table != p.table) {
                units = units.next_multiple_of(WG as usize);
            }
            // Field order == WGSL `TileDesc`.
            recs.push([
                lim("units", unit0 as u64)?, nblk as u32, (units - unit0) as u32, p.lanes, tb.log, tb.freq, tb.cum, tb.slot,
                lim("symbols", sym_base)?, p.lane_base, p.w_base, nblk as u32, p.block, p.nbx, p.x0, p.y0, p.full_w, p.x1, p.y1, p.q, p.base, u32::from(packed),
            ]);
            sym_base += (if packed { p.words } else { p.syms }) as u64;
        }
        lim("units", units as u64)?;
        // A block of 1-byte widths needs far fewer symbol bytes than its 64 B descriptor: the descriptor buffers can hit the
        // device limit before the symbol scratch does (typed `Limit`, not a wgpu validation error).
        ctx.check_buf("exv-rt rans desc", (units * DESC_WORDS * 4) as u64)?;
        ctx.check_buf("exv-rt inverse desc", (units * RT_DESC_WORDS * 4) as u64)?;
        let (rd, rt) = prep_rows(ctx, &recs, widths, units)?;
        Ok((rd, rt, units))
    }

    /// Build + dispatch under the crate's error-capture scope (the one-shot decode of a whole frame).
    pub fn decode(self, ctx: &GpuContext, outs: &[&wgpu::Buffer]) -> GpuResult<()> {
        ctx.scoped(|| self.build(ctx, outs, &bufpool::Cancel::new())?.dispatch(ctx))
    }
}

impl RtResident {
    /// Record every chunk's rANS + inverse pass into ONE encoder and submit once — NO allocation, NO upload.
    pub fn dispatch(&self, ctx: &GpuContext) -> GpuResult<()> {
        self.dispatch_stages(ctx, true, true)
    }

    /// [`dispatch`](Self::dispatch) with either stage switchable: the per-stage GPU time of a frame is
    /// `time(rans only)` / `time(inverse only)` (the inverse re-reads the previous run's symbols, so run a full
    /// dispatch once first). Bench probe only: no hot path passes `false`.
    pub fn dispatch_stages(&self, ctx: &GpuContext, rans: bool, inv: bool) -> GpuResult<()> {
        let mut enc = encoder(&ctx.device, "rt.enc");
        for p in &self.passes {
            if rans {
                pass(ctx, &mut enc, "rt.rans", &p.rans, &p.bg_rans, p.rans_grid)?;
            }
            if inv {
                pass(ctx, &mut enc, "rt.inv", &p.inv, &p.bg_inv, p.inv_grid)?;
            }
        }
        ctx.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Whether any chunk took the thread-dense fast rANS kernel (bench reporting).
    pub fn is_fast(&self) -> bool {
        self.fast
    }

    /// Number of chunks (dispatch pairs) — more than one means the symbol scratch was split by the device limit.
    pub fn chunks(&self) -> usize {
        self.passes.len()
    }
}

/// **GPU tile decode for EXV-RT (`exv_codec::blockrt_dwt`) into VRAM (NO readback).**
///
/// Bit-for-bit == `exv_codec::blockrt_dwt::decode_plane` at q=0 (pure-integer path: 5/3 inverse lifting + clamp)
/// and bit-exact for q>0 too (the hdr log decode is a table fetch, no hardware `exp2`). Two GPU stages over every
/// block of the f16 plane: **rANS** (one `Desc` per block, the plane's shared table, `lanes`-way interleaved
/// substreams; output = a symbol scratch, one byte-symbol per u32 word, or 4 byte-packed per word for `lanes == 1` on the fast route, see `RT_PACK_LANES`) then **inverse+emit**
/// (`exvrt_inv_shared`: regroup `width` LE byte-symbols -> un-zigzag -> i32 -> (q>0) detail dequant x`2^q` ->
/// inverse 2D 5/3 columns then rows -> q=0 clamp to u16 or the hdr log path -> f16 bits, crop fused in).
///
/// `payload` is the blob WITHOUT the leading tag byte (as `classify_blob` yields for `BlobKind::BlockRt`).
/// Device-limit gates (dispatch grid, storage-binding size, `table_log > 12`, `q >= 16`) return `Err` (the strict
/// CPU fallback). The plane stays in VRAM as a [`GpuTile`]; unpack via [`read_back`] (test/export).
pub fn decode_tile_exvrt_vram(ctx: &GpuContext, payload: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<GpuTile> {
    let esz = dtype.size();
    // The EXV-RT plane is always f16 (2-byte); `decode_plane` emits f16 bits regardless of the container dtype.
    if esz != 2 {
        return Err(GpuError::Unsupported(format!("exv-rt GPU: plane is f16 (esz=2), got {esz}-byte dtype")));
    }
    if w == 0 || h == 0 {
        return Err(corrupt("exv-rt GPU: empty plane"));
    }
    // Single-tile decode = the shared helper writing into a PRIVATE `w*h` buffer, treating the plane as its OWN
    // composite (full = w x h, origin 0,0): ONE codepath with the whole-image batch.
    let out_words = (w as usize) * (h as usize);
    ctx.check_buf("exv-rt out", out_words as u64 * 4)?;
    ctx.scoped(|| {
        let b_out = new_buf(&ctx.device, "rt.out", (out_words.max(1) * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        decode_exvrt_into(ctx, payload, w, h, &b_out, w, h, 0, 0)?;
        Ok(GpuTile { buf: b_out, off_words: 0, words: out_words, w, h, dtype, enc: TileEnc::ElemPerWord(2) })
    })
}

/// Decode ONE EXV-RT tile (tag stripped) straight into `out_buf` — a VRAM composite storing one f16 bit pattern
/// per u32 word — at the composite pixel origin `(dst_x0, dst_y0)`: a one-tile [`RtBatch`]. `tile_w`/`tile_h` =
/// the tile's pixel dims; `comp_full_w`/`comp_full_h` = the destination plane dims. `out_buf` MUST hold >=
/// `comp_full_w * comp_full_h` u32 words and carry `STORAGE` usage. Every gate returns `Err`, never a silent mimic.
// The arity is irreducible: a tile decode needs its payload + the FULL tile geometry (tile w/h, composite full
// w/h, destination offsets); a struct of coordinate scalars would not clarify anything.
#[allow(clippy::too_many_arguments)] // irreducible tile geometry, see the note above
pub fn decode_exvrt_into(
    ctx: &GpuContext,
    payload: &[u8],
    tile_w: u32,
    tile_h: u32,
    out_buf: &wgpu::Buffer,
    comp_full_w: u32,
    comp_full_h: u32,
    dst_x0: u32,
    dst_y0: u32,
) -> GpuResult<()> {
    let mut batch = RtBatch::new();
    batch.push(RtTile::parse(payload, tile_w, tile_h)?, RtDest::plane(0, dst_x0, dst_y0, comp_full_w, comp_full_h))?;
    batch.decode(ctx, &[out_buf])
}

/// A WARM, RESIDENT single-plane EXV-RT decode: every buffer + bind group + the destination are built ONCE and
/// the kernels are re-runnable via [`ExvrtWarmTile::redecode`] with ZERO per-call allocation/upload — the
/// resident-player primitive AND the harness for TRUE kernel-only throughput (time `dispatch + drain` only).
pub struct ExvrtWarmTile {
    res: RtResident,
    out_buf: wgpu::Buffer,
    out_words: usize,
    w: u32,
    h: u32,
    dtype: Dtype,
}

impl ExvrtWarmTile {
    /// Re-run the GPU passes over the RESIDENT buffers (no alloc/upload). Fence with [`GpuContext::drain`].
    pub fn redecode(&self, ctx: &GpuContext) -> GpuResult<()> {
        ctx.scoped(|| self.res.dispatch(ctx))
    }

    /// A [`GpuTile`] view of the resident output for [`read_back`] — shares the same VRAM buffer (Arc-cloned).
    pub fn tile(&self) -> GpuTile {
        GpuTile {
            buf: self.out_buf.clone(),
            off_words: 0,
            words: self.out_words,
            w: self.w,
            h: self.h,
            dtype: self.dtype,
            enc: TileEnc::ElemPerWord(2),
        }
    }

    /// Whether the thread-dense fast kernel was selected (for bench reporting).
    pub fn is_fast(&self) -> bool {
        self.res.is_fast()
    }
}

/// Build an [`ExvrtWarmTile`]: parse the blob and allocate + upload ALL GPU buffers ONCE (the "cold" cost the
/// public [`decode_tile_exvrt_vram`] pays on every call). Same F16-plane contract and gates as that function.
pub fn prepare_tile_exvrt_vram(ctx: &GpuContext, payload: &[u8], dtype: Dtype, w: u32, h: u32) -> GpuResult<ExvrtWarmTile> {
    let esz = dtype.size();
    if esz != 2 {
        return Err(GpuError::Unsupported(format!("exv-rt GPU: plane is f16 (esz=2), got {esz}-byte dtype")));
    }
    if w == 0 || h == 0 {
        return Err(corrupt("exv-rt GPU: empty plane"));
    }
    let out_words = (w as usize) * (h as usize);
    ctx.check_buf("exv-rt out", out_words as u64 * 4)?;
    ctx.scoped(|| {
        let out_buf = new_buf(&ctx.device, "rt.out.warm", (out_words.max(1) * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let mut batch = RtBatch::new();
        batch.push(RtTile::parse(payload, w, h)?, RtDest::plane(0, 0, 0, w, h))?;
        let res = batch.build(ctx, &[&out_buf], &bufpool::Cancel::new())?;
        Ok(ExvrtWarmTile { res, out_buf, out_words, w, h, dtype })
    })
}
