//! High-level interleaved rANS on top of the byte-coder.
//!
//! # Parallelism model (GPU-friendly)
//! N **independent** sub-streams: lane `k` owns symbols `k, k+N, 2N+k, …`.
//! Each sub-stream is a self-contained rANS with its own state and byte slice. On the GPU, thread `k`
//! decodes its slice in lockstep with no interaction with its neighbors (see `docs/archive/PLAN.md` §3.1).
//!
//! # Container (LE)
//! ```text
//! MAGIC:u32 | table_log:u8 | lanes:u8 | n:u32 | freq[256×u16] | lane_len[lanes×u32] | streams…
//! ```
//! Self-contained: the frequency table and lane metadata are inside → bit-exact decode with no external state.
//!
//! [`prepare`] yields the unpacked pieces (model + per-lane streams) for the GPU decoder
//! (`exv-gpu`) so it does not re-parse the container. [`encode`] is `prepare` + serialization.
//!
//! # CPU vs GPU strictness
//! The CPU decoders ([`decode_capped`], [`decode_block`]) enforce the exact-end rule per lane (state back at
//! `RANS_L`, every byte consumed). [`parse`] (the GPU entry) checks the same STRUCTURE the CPU rejects up
//! front — magic, `lanes ≥ 1`, `n` ceiling, freq table sum, lane slices in bounds, initial states in range —
//! but cannot check lane ends without decoding; the WGSL output on a lane that violates the exact-end rule
//! is unspecified (it is never produced by [`encode`]).

use std::cell::{Cell, RefCell};
use std::fmt;

use crate::byte_coder::{Encoder, Lane, LaneOut, RANS_L, RANS_UPPER, enc_cap};
use crate::model::{ALPHABET, Model};

/// Container signature: ASCII "RANS" in LE.
pub const MAGIC: u32 = 0x534E4152;

/// Ceiling on symbols per block, enforced symmetrically: [`encode`] refuses to write a stream the decoder
/// would then reject, and decode treats a larger `n` as a decompression bomb (`n` is untrusted → a huge
/// allocation would kill the process). 1 GiB symbols.
pub const MAX_DECODE_SYMBOLS: usize = 1 << 30;

/// Why an encode was refused (always a caller/geometry problem, never a data-dependent surprise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// More than [`MAX_DECODE_SYMBOLS`] symbols: the decoder would reject the stream.
    TooLarge(usize),
    /// A lane stream does not fit its `u32` length field.
    LaneOverflow,
    /// `encode_block` was given a symbol whose frequency in the shared model is 0 (it was not in the pool
    /// the model was built from), so it cannot be coded.
    SymbolNotInModel(u8),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge(n) => write!(f, "{n} symbols exceed the {MAX_DECODE_SYMBOLS}-symbol rANS block limit"),
            Self::LaneOverflow => write!(f, "rANS lane stream exceeds the u32 length field"),
            Self::SymbolNotInModel(s) => write!(f, "symbol {s} has zero frequency in the shared model"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// CPU↔WGSL contract (locked after Spike A). `symbol_bits` is fixed = 8 (byte alphabet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RansConfig {
    /// Number of independent lanes (= warp/SIMD width on the GPU). ≥ 1.
    pub lanes: u8,
    /// log2 of the frequency sum M. Range 8..=15 (M=256..32768).
    pub table_log: u8,
}

impl RansConfig {
    /// New config clamped into the valid ranges (`lanes≥1`, `table_log∈8..=15`).
    pub fn new(lanes: u8, table_log: u8) -> Self {
        Self { lanes: lanes.max(1), table_log: table_log.clamp(8, 15) }
    }
}

impl Default for RansConfig {
    /// 128 lanes and M=4096. 128 (4 warps' worth of independent rANS states per tile) feeds GPU
    /// occupancy: the decode is a serial dependent recurrence per lane, so more concurrent lanes are
    /// what hides its latency (32 lanes ≈ 1 active warp/tile ≈ ~8% occupancy — measured too slow). Costs
    /// a little ratio on small tiles (per-lane state overhead); the format stores `lanes`, so decode
    /// adapts automatically (no back-compat with 32-lane files — those simply re-encode).
    fn default() -> Self {
        Self { lanes: 128, table_log: 12 }
    }
}

/// Unpacked data of a single rANS block: decode model + per-lane byte streams.
///
/// This is exactly what the GPU decoder needs to load into storage buffers
/// (`freq`/`cum`/`slot_to_sym` + concatenation of `lane_streams` with lane offsets).
///
/// `L` is how a lane stream is held: OWNED (`Vec<u8>`, the default: what [`prepare`] produces and [`parse`]
/// copies out) or BORROWED from the container (`&[u8]`, [`parse_ref`]: no per-lane allocation, which was ~128
/// `Vec`s per tile, ~400k per 8k frame, on the GPU host-planning path).
pub struct Prepared<L = Vec<u8>> {
    pub cfg: RansConfig,
    pub n: usize,
    pub freq: [u16; ALPHABET],
    pub cum: [u16; ALPHABET],
    pub slot_to_sym: Vec<u8>,
    pub lane_streams: Vec<L>,
}

/// Symbol counts. Four interleaved sub-histograms: on skewed data the same bin is hit back to back, and one
/// table would serialize every increment behind a store-to-load forward (~5 cycles each).
fn histogram(symbols: &[u8]) -> [u32; ALPHABET] {
    let mut h = [[0u32; ALPHABET]; 4];
    let (quads, rest) = symbols.as_chunks::<4>();
    for q in quads {
        for (t, &b) in h.iter_mut().zip(q) {
            t[b as usize] += 1;
        }
    }
    for &b in rest {
        h[0][b as usize] += 1;
    }
    let mut counts = [0u32; ALPHABET];
    for (c, ((a, b), (d, e))) in counts.iter_mut().zip(h[0].iter().zip(&h[1]).zip(h[2].iter().zip(&h[3]))) {
        *c = a + b + d + e;
    }
    counts
}

/// Code every lane (lane `k` owns symbols `k, k+lanes, …`) against `model`, in REVERSE index order so the
/// decoder yields forward order. The single encode loop of [`prepare`], [`encode_into`] and [`encode_block_into`];
/// the finished lanes live in the per-thread scratch and are handed to `emit` BORROWED (it copies them where it
/// needs them: no per-lane `Vec`).
///
/// ROUND-MAJOR: all lanes advance together, one symbol each (round `j` of lane `k` is `symbols[j*lanes + k]`),
/// so the `lanes` independent state chains overlap in the CPU pipeline instead of one serial chain per lane.
/// Each lane still sees exactly its own symbols in descending order, so the bytes are identical to coding the
/// lanes one after another. A symbol with zero model frequency is an `Err` (it never reaches the reciprocal).
fn with_lanes<T>(symbols: &[u8], lanes: usize, model: &Model, emit: impl FnOnce(&[LaneOut<'_>]) -> T) -> Result<T, EncodeError> {
    let tab = model.enc_syms();
    if let Some(&s) = symbols.iter().find(|&&s| !tab[s as usize].is_coded()) {
        return Err(EncodeError::SymbolNotInModel(s));
    }
    let n = symbols.len();
    let caps: Vec<usize> = (0..lanes).map(|k| enc_cap(if n > k { (n - k).div_ceil(lanes) } else { 0 })).collect();
    let mut scratch = SCRATCH.with(Cell::take);
    let total = caps.iter().sum();
    if scratch.len() < total {
        scratch.resize(total, 0);
    }
    let mut rest = scratch.as_mut_slice();
    let mut encs: Vec<Encoder> = caps
        .iter()
        .map(|&c| {
            let (lane, tail) = std::mem::take(&mut rest).split_at_mut(c);
            rest = tail;
            Encoder::new(lane)
        })
        .collect();
    let (rounds, last) = symbols.split_at(n - n % lanes);
    // The partial round (lanes 0..n%lanes) holds each of those lanes' final symbol, so it is coded first.
    for (enc, &s) in encs.iter_mut().zip(last) {
        enc.put(&tab[s as usize]);
    }
    for round in rounds.chunks_exact(lanes).rev() {
        for (enc, &s) in encs.iter_mut().zip(round) {
            enc.put(&tab[s as usize]);
        }
    }
    let parts: Vec<LaneOut<'_>> = encs.into_iter().map(Encoder::finish).collect();
    let out = emit(&parts);
    drop(parts); // ends the borrow of `scratch`
    if scratch.len() <= SCRATCH_KEEP {
        SCRATCH.with(|c| c.set(scratch));
    }
    Ok(out)
}

// Per-thread encode scratch (all lanes' emission buffers, back to back), kept between calls: a fresh zeroed
// `2n`-byte allocation per call was ~20% of encode time (memset + page faults after the allocator decommitted).
// Only the live prefix of each lane is ever read, so stale contents are harmless.
thread_local! {
    static SCRATCH: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
}

/// Largest scratch a thread keeps (bigger one-off blocks are freed again).
const SCRATCH_KEEP: usize = 16 << 20;

/// The clamped config and the model of `symbols`, after the symbol-count ceiling check. The shared first step of
/// [`prepare`] and [`encode_into`].
fn plan(symbols: &[u8], cfg: RansConfig) -> Result<(RansConfig, Model), EncodeError> {
    let cfg = RansConfig::new(cfg.lanes, cfg.table_log);
    if symbols.len() > MAX_DECODE_SYMBOLS {
        return Err(EncodeError::TooLarge(symbols.len()));
    }
    Ok((cfg, Model::from_counts(&histogram(symbols), cfg.table_log as u32)))
}

/// Build the model and encode each lane (owned per-lane `Vec`s: the form the GPU upload path consumes).
///
/// # Errors
/// [`EncodeError::TooLarge`] beyond [`MAX_DECODE_SYMBOLS`] symbols (the decoder would refuse the stream).
pub fn prepare(symbols: &[u8], cfg: RansConfig) -> Result<Prepared, EncodeError> {
    let (cfg, model) = plan(symbols, cfg)?;
    let lane_streams = with_lanes(symbols, cfg.lanes as usize, &model, |parts| {
        parts
            .iter()
            .map(|p| {
                let mut v = Vec::with_capacity(p.len());
                p.write(&mut v);
                v
            })
            .collect()
    })?;
    Ok(Prepared {
        cfg,
        n: symbols.len(),
        freq: *model.freq(),
        cum: *model.cum(),
        slot_to_sym: model.slot_to_sym().to_vec(),
        lane_streams,
    })
}

/// Encode bytes into a self-contained container (a new `Vec`; see [`encode_into`]).
pub fn encode(symbols: &[u8], cfg: RansConfig) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::new();
    encode_into(symbols, cfg, &mut out)?;
    Ok(out)
}

/// APPEND the self-contained container of `symbols` to `out` (the lanes are copied from the encoder scratch
/// straight into it: no per-lane `Vec`, no second container copy). Errors as [`prepare`], plus
/// [`EncodeError::LaneOverflow`] (the header stores `n` and every lane length as `u32`); on `Err` `out` is left
/// exactly as it was.
pub fn encode_into(symbols: &[u8], cfg: RansConfig, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    let (cfg, model) = plan(symbols, cfg)?;
    let start = out.len();
    let res = with_lanes(symbols, cfg.lanes as usize, &model, |parts| {
        // Exact size up front: growing a Vec by doubling re-copied the whole container ~2x.
        out.reserve(4 + 1 + 1 + 4 + ALPHABET * 2 + 4 * parts.len() + parts.iter().map(LaneOut::len).sum::<usize>());
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.push(cfg.table_log);
        out.push(cfg.lanes);
        out.extend_from_slice(&(symbols.len() as u32).to_le_bytes()); // n <= MAX_DECODE_SYMBOLS < 2^32 (checked in plan)
        for &f in model.freq() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for p in parts {
            out.extend_from_slice(&u32::try_from(p.len()).map_err(|_| EncodeError::LaneOverflow)?.to_le_bytes());
        }
        parts.iter().for_each(|p| p.write(out));
        Ok(())
    });
    let res = res.and_then(|r| r);
    if res.is_err() {
        out.truncate(start);
    }
    res
}

/// Little-endian `u32` at `*p` (advances). `None` when fewer than 4 bytes remain.
fn rd_u32(d: &[u8], p: &mut usize) -> Option<u32> {
    let s = d.get(*p..p.checked_add(4)?)?;
    *p += 4;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// A validated container header: the single parse of the wire format shared by [`parse`] (GPU path) and
/// [`decode_capped`] (CPU path), so the two can never disagree on what is valid. The lane slices are NOT
/// materialised here: [`Head::slices`] walks them in place, so a decode keeps its cursors on the stack and
/// [`parse_span`] collects the one `Vec` it must own.
struct Head<'a> {
    table_log: u8,
    n: usize,
    lanes: usize,
    /// The `lanes` little-endian `u32` stream lengths.
    lens: &'a [u8],
    /// Everything after the length table: the lane streams back to back (plus any trailing bytes).
    streams: &'a [u8],
}

impl<'a> Head<'a> {
    /// The lane streams in order; `None` for a lane that runs past the end of the data.
    fn slices(&self) -> impl Iterator<Item = Option<&'a [u8]>> {
        let (streams, mut q) = (self.streams, 0usize);
        self.lens.as_chunks::<4>().0.iter().map(move |l| {
            let s = streams.get(q..q.checked_add(u32::from_le_bytes(*l) as usize)?)?;
            q += s.len();
            Some(s)
        })
    }
}

// The per-thread decode model: the ~20 KiB slot tables of a container's frequency table were allocated per
// blob (0.15x of a tile's raw bytes of churn on the FAST/WAVE paths); the tables are rebuilt in place instead.
thread_local! {
    static MODEL: RefCell<Model> = RefCell::new(Model::default());
}

/// Run `f` over the reused per-thread [`Model`] (its contents are whatever the last user left: load it with
/// [`Model::load_freqs`] before use). Not re-entrant by design: a nested call (never made by this crate) gets a
/// throw-away model instead of a borrow panic.
pub fn with_model<T>(f: impl FnOnce(&mut Model) -> T) -> T {
    MODEL.with(|m| match m.try_borrow_mut() {
        Ok(mut m) => f(&mut m),
        Err(_) => f(&mut Model::default()),
    })
}

/// `s` itself when it starts with a legal lane state (the structure check of the borrowed [`parse_ref`] form;
/// [`Lane::new`] makes the same check for the decoders).
fn lane_ok(s: &[u8]) -> Option<&[u8]> {
    let state = u32::from_le_bytes(s.get(0..4)?.try_into().ok()?);
    (RANS_L..RANS_UPPER).contains(&state).then_some(s)
}

/// Walk the container header: magic, `lanes ≥ 1`, `n ≤ min(MAX_DECODE_SYMBOLS, max_n)`, a valid freq table (loaded
/// into `model`) and a length table inside `data`. The caller walks the lanes ([`Head::slices`]) and applies the
/// initial-state check ([`Lane::new`] / [`lane_ok`]: the range [RANS_L, RANS_UPPER) must hold for the GPU path
/// too, else the GPU diverges byte-wise from the CPU). Nothing is allocated from `n`.
fn head<'a>(data: &'a [u8], max_n: usize, model: &mut Model) -> Option<Head<'a>> {
    let mut p = 0usize;
    if rd_u32(data, &mut p)? != MAGIC {
        return None;
    }
    let table_log = *data.get(p)?;
    p += 1;
    // lanes==0 would decode nothing and return a silently wrong `vec![0; n]`; encode always writes ≥ 1.
    let lanes = *data.get(p)? as usize;
    p += 1;
    if lanes == 0 {
        return None;
    }
    let n = rd_u32(data, &mut p)? as usize;
    if n > MAX_DECODE_SYMBOLS || n > max_n {
        return None; // decompression-bomb guard (absolute + caller threshold)
    }
    let freq_bytes = data.get(p..p.checked_add(ALPHABET * 2)?)?;
    model.load_freqs(freq_bytes, table_log as u32)?;
    p += ALPHABET * 2;
    // The length table, then the streams back to back: read the lengths in place (no `Vec` of them).
    let lens = data.get(p..p + lanes * 4)?; // lanes <= 255: no overflow
    Some(Head { table_log, n, lanes, lens, streams: data.get(p + lens.len()..)? })
}

/// Header walk + lane cursors on the STACK + the decode, for [`decode_capped_into`] / [`decode_into`]: `out_for(n)`
/// supplies the `n`-byte output once every lane has validated (so a container with a bad lane never sizes it). The
/// cursor array is `255` (a `u8` lane count) x `size_of::<Lane>()` of stack: a heap `Vec` per container was one
/// allocation of ~4 KiB per channel per tile (docs/OPT.md R).
fn decode_with<'o>(data: &[u8], max_n: usize, out_for: impl FnOnce(usize) -> Option<&'o mut [u8]>) -> Option<()> {
    with_model(|model| {
        let h = head(data, max_n, model)?;
        let mut cur = [Lane::EMPTY; u8::MAX as usize];
        let lanes = cur.get_mut(..h.lanes)?;
        for (c, s) in lanes.iter_mut().zip(h.slices()) {
            *c = Lane::new(s?)?;
        }
        run_lanes(lanes, model, out_for(h.n)?)
    })
}

/// Run every lane's decoder over `n` interleaved symbols (lane `k` owns `k, k+lanes, …`) and require each
/// lane to finish EXACTLY ([`Lane::done`]). `None` on a truncated / padded / forged lane — a forced
/// stream is a hard error instead of plausible-looking garbage. A lane that reads past its last byte is
/// caught after the round it happens in ([`Lane::overrun`]): a forged `n` therefore costs only as many
/// rounds as the stream's bytes can feed (a near-deterministic model legitimately decodes many symbols per
/// byte — the output buffer, bounded by the caller's `max_n`, is the budget for that case). Decodes into the
/// caller's `out` (`n = out.len()` symbols; every byte is overwritten on success). Shared by the container and
/// block paths.
///
/// ROUND-MAJOR (see `byte_coder.rs`): round `j` decodes symbol `j*lanes + k` of every lane `k`, so the lane
/// chains are independent in the pipeline and the output is written sequentially. The decoded bytes are
/// identical to the lane-major order the WGSL kernels use (each lane is still decoded front to back).
fn run_lanes(lanes: &mut [Lane<'_>], model: &Model, out: &mut [u8]) -> Option<()> {
    let tab = model.dec_tab();
    let mut rounds = out.chunks_exact_mut(lanes.len());
    for round in &mut rounds {
        for (lane, o) in lanes.iter_mut().zip(round) {
            *o = lane.next(&tab);
        }
        if lanes.iter().any(Lane::overrun) {
            return None;
        }
    }
    // The partial last round: only the first `n % lanes` lanes own a symbol in it.
    for (lane, o) in lanes.iter_mut().zip(rounds.into_remainder()) {
        *o = lane.next(&tab);
    }
    lanes.iter().all(Lane::done).then_some(())
}

/// Parse a container into [`Prepared`] with BORROWED lane streams (model + slices into `data`) — for GPU decode
/// of a saved blob. Symmetric to [`encode`]; `None` on a corrupt stream. Structure only (see the module docs):
/// whether each lane ends exactly (truncation/padding) is verified by decoding ([`decode_capped`]); GPU output
/// on such a lane is unspecified. Allocates the 4 KiB `slot_to_sym` and one `Vec` of slices, nothing per lane.
pub fn parse_ref(data: &[u8]) -> Option<Prepared<&[u8]>> {
    parse_span(data).map(|(p, _)| p)
}

/// [`parse_ref`] plus the lane streams as ONE contiguous slice (they are stored back to back after the length table,
/// so `lane_streams` are adjacent sub-slices of it). The GPU host copies a unit's streams with a single memcpy.
pub fn parse_span(data: &[u8]) -> Option<(Prepared<&[u8]>, &[u8])> {
    with_model(|model| {
        let h = head(data, MAX_DECODE_SYMBOLS, model)?;
        let lane_streams = h.slices().map(|s| lane_ok(s?)).collect::<Option<Vec<_>>>()?;
        let span = h.streams.get(..lane_streams.iter().map(|s| s.len()).sum())?;
        let prep = Prepared {
            cfg: RansConfig { lanes: h.lanes as u8, table_log: h.table_log },
            n: h.n,
            freq: *model.freq(),
            cum: *model.cum(),
            slot_to_sym: model.slot_to_sym().to_vec(),
            lane_streams,
        };
        Some((prep, span))
    })
}

/// [`parse_ref`] with every lane copied into its own `Vec` (the owned form; prefer [`parse_ref`] on hot paths).
pub fn parse(data: &[u8]) -> Option<Prepared> {
    let p = parse_ref(data)?;
    Some(Prepared {
        cfg: p.cfg,
        n: p.n,
        freq: p.freq,
        cum: p.cum,
        slot_to_sym: p.slot_to_sym,
        lane_streams: p.lane_streams.into_iter().map(<[u8]>::to_vec).collect(),
    })
}

/// Decode a container back into bytes. `max_n` is the caller's REQUIRED ceiling on the symbol count (the
/// expected output length, e.g. `w*h*esz` of a plane; applied together with [`MAX_DECODE_SYMBOLS`]): a
/// bomb (a tiny container with a huge `n`) is rejected BEFORE the output is allocated, and there is
/// deliberately no cap-less variant. Every lane must end exactly (state `RANS_L`, all bytes consumed):
/// truncated or padded containers return `None` (never a panic). The output is allocated fallibly.
pub fn decode_capped(data: &[u8], max_n: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    decode_capped_into(data, max_n, &mut out)?;
    Some(out)
}

/// [`decode_capped`] into the caller's `Vec`, which is REPLACED by the `n` decoded bytes (its capacity is reused:
/// a hot path keeps one scratch `Vec` instead of allocating the stream per blob; `n` is only ever reserved after the
/// header passed the `max_n` ceiling). On `None` the contents of `out` are unspecified.
pub fn decode_capped_into(data: &[u8], max_n: usize, out: &mut Vec<u8>) -> Option<()> {
    decode_with(data, max_n, |n| {
        out.clear();
        out.try_reserve_exact(n).ok()?;
        out.resize(n, 0);
        Some(out.as_mut_slice())
    })
}

/// Decode a container into the caller's buffer, whose length is the EXPECTED symbol count: a container that
/// declares any other `n` is `None`, as is every violation [`decode_capped`] rejects. No allocation besides the
/// per-lane cursors (the tables live in a per-thread [`Model`]). On `None` the contents of `out` are unspecified.
pub fn decode_into(data: &[u8], out: &mut [u8]) -> Option<()> {
    let want = out.len();
    decode_with(data, want, |n| (n == want).then_some(out))
}

// ---------------------------------------------------------------------------
// Shared-table per-block primitives (EXV-RT / Tier-2 block codec).
//
// The interleaved container above embeds its own frequency table per `encode()`. A block codec that
// wants THOUSANDS of independently GPU-decodable blocks cannot pay one table per block — instead ONE
// table is built over the POOLED symbols of all blocks (stored once by the caller) and each block is a
// HEADERLESS substream coded against that shared table. This reuses the exact byte-coder + model (zero
// new entropy math, same CPU↔WGSL contract) — only the framing differs.
//
// Substream layout (per block): `[lane_len[lanes-1]:u32][lane streams…]` — the LAST lane length is
// implied by the block's slice size (from the caller's offset index), so a 1-lane block has NO header.
// `n` (symbol count) is geometry, supplied by the caller (block w×h×bytes), never stored.
// ---------------------------------------------------------------------------

/// Build a shared frequency [`Model`] from the POOL of all blocks' symbols (concatenated). The SAME
/// model must encode every block and decode them; the caller stores it ONCE per tile (`freqs_le()` →
/// 512 bytes) and rebuilds it with [`Model::from_freqs`]. `table_log` clamped to 8..=15.
pub fn build_table(pool: &[u8], table_log: u8) -> Model {
    Model::from_counts(&histogram(pool), table_log as u32)
}

/// Encode ONE block's `symbols` against a shared external `model` as a HEADERLESS substream (no
/// MAGIC/table/n). `lanes` interleaves WITHIN the block (1 = serial; K = K GPU sub-lanes per block for
/// warp utilisation). Deterministic; mirrors [`prepare`]'s per-lane reverse coding.
///
/// # Errors
/// [`EncodeError::SymbolNotInModel`] when a symbol was not in the pool `model` was built from (zero
/// frequency ⇒ un-codable), [`EncodeError::TooLarge`] beyond [`MAX_DECODE_SYMBOLS`],
/// [`EncodeError::LaneOverflow`] for a lane over `u32` bytes.
pub fn encode_block(symbols: &[u8], model: &Model, lanes: u8) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::new();
    encode_block_into(symbols, model, lanes, &mut out)?;
    Ok(out)
}

/// [`encode_block`] APPENDING to `out` (the EXV-RT plane encoder concatenates thousands of blocks: no `Vec` per
/// block). On `Err` `out` is left exactly as it was.
pub fn encode_block_into(symbols: &[u8], model: &Model, lanes: u8, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    let lanes = lanes.max(1) as usize;
    if symbols.len() > MAX_DECODE_SYMBOLS {
        return Err(EncodeError::TooLarge(symbols.len()));
    }
    let start = out.len();
    let res = with_lanes(symbols, lanes, model, |parts| {
        out.reserve(4 * (lanes - 1) + parts.iter().map(LaneOut::len).sum::<usize>());
        // lanes-1 explicit lengths; the last lane fills the remainder of the block's slice.
        for p in parts.iter().take(lanes - 1) {
            out.extend_from_slice(&u32::try_from(p.len()).map_err(|_| EncodeError::LaneOverflow)?.to_le_bytes());
        }
        parts.iter().for_each(|p| p.write(out));
        Ok(())
    });
    let res = res.and_then(|r| r);
    if res.is_err() {
        out.truncate(start);
    }
    res
}

/// Decode ONE block substream (from [`encode_block`]) against the shared `model`. `n` = the block's
/// symbol count (geometry, from the caller); `lanes` MUST match encode. `None` on any corruption
/// (short header, out-of-range state, trailing garbage, bomb `n`) — never panics.
pub fn decode_block(sub: &[u8], model: &Model, n: usize, lanes: u8) -> Option<Vec<u8>> {
    if n > MAX_DECODE_SYMBOLS {
        return None;
    }
    let mut out = Vec::new();
    out.try_reserve_exact(n).ok()?;
    out.resize(n, 0);
    decode_block_into(sub, model, &mut out, lanes).map(|()| out)
}

/// [`decode_block`] into the caller's buffer (`n = out.len()`): the EXV-RT decoder reuses one buffer for every
/// block of a plane. On `None` the contents of `out` are unspecified.
pub fn decode_block_into(sub: &[u8], model: &Model, out: &mut [u8], lanes: u8) -> Option<()> {
    let lanes = lanes.max(1) as usize;
    if out.len() > MAX_DECODE_SYMBOLS {
        return None;
    }
    if lanes == 1 {
        // The EXV-RT shape (thousands of 1-lane blocks per plane): no header, the whole slice is the lane, and no
        // `Vec` of cursors per block.
        return run_lanes(&mut [Lane::new(sub)?], model, out);
    }
    let lens = sub.get(..4 * (lanes - 1))?;
    let streams = sub.get(lens.len()..)?;
    let lens = lens.as_chunks::<4>().0;
    let known = lens.iter().try_fold(0usize, |a, l| a.checked_add(u32::from_le_bytes(*l) as usize))?;
    let last = streams.len().checked_sub(known)?; // last lane = remainder of the slice
    let mut cur = [Lane::EMPTY; u8::MAX as usize]; // `lanes` is a `u8`: the cursors stay on the stack
    let mut q = 0usize;
    for (c, len) in cur.iter_mut().zip(lens.iter().map(|l| u32::from_le_bytes(*l) as usize).chain([last])) {
        let slice = streams.get(q..q.checked_add(len)?)?;
        *c = Lane::new(slice)?;
        q += len;
    }
    if q != streams.len() {
        return None; // trailing garbage after the last lane
    }
    run_lanes(cur.get_mut(..lanes)?, model, out)
}
