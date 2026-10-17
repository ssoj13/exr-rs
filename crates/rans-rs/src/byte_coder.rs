//! Canonical 32-bit rANS byte-coder (port of `rans_byte.h`, Fabian Giesen, public domain).
//!
//! # Invariants
//! - State `state ∈ [RANS_L, 256·RANS_L)`; renormalization is byte-wise.
//! - **All operations are `u32`, no 64-bit** → ports to WGSL 1:1 (see `docs/archive/PLAN.md` §3.1). The only
//!   64-bit arithmetic is the CPU-encoder's exact division-by-reciprocal ([`EncSym`]), which computes the SAME
//!   `x / freq` the shader never needs (the decoder has no division) — the produced bytes are identical.
//! - `scale_bits = table_log` (≤ 15) → `freq ≤ M = 2^scale_bits`, which prevents `u32` overflow
//!   either in `x_max` (≤ 2^31) or in the decode step `freq*(x>>sb)` (< 2^32). See the derivation in `docs/archive/PLAN.md`.
//!
//! The encoder emits bytes **backward** (in memory order they are reversed once at [`Encoder::finish`]), the
//! decoder reads **forward** → symbols encoded in reverse order are decoded in forward order.
//!
//! # Why the lane types are split this way (CPU speed)
//! One rANS lane is a serial dependent chain (`state → slot → table → state`), ~10-30 cycles per symbol. The
//! lane count is large (128), so the callers in `codec.rs` run all lanes ROUND-MAJOR (symbol `j` of every lane,
//! then `j+1`): the chains are independent, so the out-of-order core overlaps them. [`Lane`] / [`Encoder`] are
//! therefore small, inlinable, branch-light per-symbol steps, and the per-symbol bounds/overrun branch of the
//! old lane-major loop became one check per round.

/// Lower bound of the normalized state (`1 << 23`).
pub const RANS_L: u32 = 1 << 23;

/// Upper bound of the normalized state (`256·RANS_L = 1 << 31`). The state of a valid
/// stream is always `< RANS_UPPER`; a forced larger `state` makes `freq*(state>>sb)` exceed u32 in
/// [`Lane::next`] → overflow/panic (debug) or garbage (release). The decoder must
/// reject such a `state` on input (see [`Lane::new`]).
pub const RANS_UPPER: u32 = RANS_L << 8;

/// Maximum renormalization bytes per [`Lane::next`]. Valid data needs ≤3
/// (23 state bits / 8 bits per byte). This is the **single CPU↔WGSL contract**: both the CPU and the GPU shader
/// must use exactly this value — otherwise, on forced/truncated input, their
/// renorm loops would read a different number of bytes → states diverge (violating the byte-exact
/// CPU==GPU invariant). The CPU used to be at 8 and WGSL at 4 — a mismatch on a corrupt stream.
pub const RENORM_GUARD: u32 = 4;

/// Per-symbol encode constants, built once per model ([`crate::Model`]) so the hot loop has no division.
///
/// `x / freq` is replaced by the Granlund–Montgomery reciprocal: with `l = ceil(log2 freq)` and
/// `m = floor(2^(31+l) / freq) + 1`, `floor(x / freq) == (x·m) >> (31+l)` for EVERY `x < 2^31`. The encoder
/// state is `< x_max ≤ 2^31` before the step, so the identity is exact (proved above, and swept in
/// `tests::reciprocal_division_is_exact`); `m·x < 2^63` fits `u64`.
#[derive(Clone, Copy)]
pub struct EncSym {
    /// Renorm threshold `((RANS_L >> sb) << 8) * freq`; `u32::MAX` for an absent symbol (never renormalizes).
    x_max: u32,
    /// `m` of the reciprocal (see the type docs).
    rcp: u64,
    /// `31 + ceil(log2 freq)`.
    shift: u32,
    /// `M - freq`: the state becomes `x + start + q·(M - freq)` (≡ `(q << sb) + x % freq + start`).
    m_sub_f: u32,
    start: u32,
    freq: u32,
}

impl EncSym {
    /// Constants of a symbol with `start = cum[s]`, `freq = freq[s]` under `scale_bits`. `freq == 0` (a symbol
    /// not in the model) yields an inert entry whose [`EncSym::is_coded`] is `false`; the caller rejects it.
    pub fn new(start: u32, freq: u32, scale_bits: u32) -> Self {
        if freq == 0 {
            return Self { x_max: u32::MAX, rcp: 0, shift: 0, m_sub_f: 0, start: 0, freq: 0 };
        }
        let l = u32::BITS - (freq - 1).leading_zeros(); // ceil(log2 freq); freq==1 → 0
        let shift = 31 + l;
        Self {
            x_max: ((RANS_L >> scale_bits) << 8) * freq,
            rcp: (1u64 << shift) / freq as u64 + 1,
            shift,
            m_sub_f: (1u32 << scale_bits) - freq,
            start,
            freq,
        }
    }

    /// `false` for a symbol whose model frequency is 0 (it cannot be coded).
    pub fn is_coded(&self) -> bool {
        self.freq != 0
    }
}

/// Encoder for a single rANS stream (one lane) writing into caller-provided scratch.
pub struct Encoder<'a> {
    /// Bytes in EMISSION order (the stream is the reverse of this, see [`Encoder::finish`]); only `..pos` is live.
    buf: &'a mut [u8],
    pos: usize,
    state: u32,
}

/// Scratch bytes one lane needs for `symbols` symbols: the proven worst case `symbols·2 + 8`. A symbol emits
/// ≤ 2 renorm bytes (`x < 2^31` shrinks below `x_max ≥ 2^16` in two byte-shifts), plus the 4-byte flush.
pub fn enc_cap(symbols: usize) -> usize {
    symbols * 2 + 8
}

impl<'a> Encoder<'a> {
    /// New encoder over `buf` (`buf.len() >= enc_cap(symbols)`). The scratch is reused across calls by the
    /// caller, so nothing here allocates or zero-fills; stale bytes past `pos` are never read.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0, state: RANS_L }
    }

    /// Encode a symbol. `e` must satisfy [`EncSym::is_coded`] — the callers (`prepare`, `encode_block`)
    /// establish that BEFORE coding, so a zero-frequency symbol never reaches the reciprocal below.
    ///
    /// The first renorm byte is emitted BRANCH-FREE (written unconditionally, `pos` advances only when the
    /// state really overflowed): on high-entropy data that branch is taken ~50-75% of the time at random, and
    /// a mispredict (~15 cycles) cost more than the whole symbol step. The rarely needed second byte stays a loop.
    #[inline(always)]
    pub fn put(&mut self, e: &EncSym) {
        debug_assert!(e.freq > 0, "rANS put: freq must be > 0 (zero-frequency symbol cannot be coded)");
        let mut x = self.state;
        let c = x >= e.x_max;
        // `pos ≤ 2·symbols so far < cap` (see `new`), so the slot always exists: `get_mut` is a never-taken branch
        // instead of a panic path (no `unsafe` unchecked store).
        if let Some(b) = self.buf.get_mut(self.pos) {
            *b = x as u8;
        }
        self.pos += usize::from(c);
        x = if c { x >> 8 } else { x };
        while x >= e.x_max {
            if let Some(b) = self.buf.get_mut(self.pos) {
                *b = x as u8;
            }
            self.pos += 1;
            x >>= 8;
        }
        let q = ((x as u64 * e.rcp) >> e.shift) as u32;
        self.state = x + e.start + q * e.m_sub_f;
    }

    /// Flush the lane in place: the forward-readable stream is `state.to_le_bytes()` followed by the returned
    /// body. Nothing is copied or allocated here — the caller appends both parts straight into its output, so a
    /// 128-lane block no longer makes 128 small `Vec`s plus a second copy into the container.
    pub fn finish(self) -> LaneOut<'a> {
        // Emission order is [renorm bytes…] and the state is flushed last: the stream is
        // [state LE][renorm bytes, last emitted first] — exactly what a backward-filled buffer held.
        let Self { buf, pos, state } = self;
        let live = buf.get_mut(..pos).unwrap_or_default();
        live.reverse();
        LaneOut { state, body: live }
    }
}

/// A finished encoder lane: the stream is `state` (4 bytes LE) then `body`.
pub struct LaneOut<'a> {
    state: u32,
    body: &'a [u8],
}

impl LaneOut<'_> {
    /// Stream length in bytes (the container's `lane_len`).
    pub fn len(&self) -> usize {
        4 + self.body.len()
    }

    /// Append the stream to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.state.to_le_bytes());
        out.extend_from_slice(self.body);
    }
}

/// Decode tables of a model, flattened for one load per symbol (built by [`crate::Model`]).
#[derive(Clone, Copy)]
pub struct DecTab<'a> {
    /// Per slot: `freq << 16 | (slot - cum[sym])`. `freq ≤ 2^15` and the in-symbol offset `< 2^15` each fit 16 bits.
    pub slot: &'a [u32],
    /// Per slot: the owning symbol.
    pub sym: &'a [u8],
    pub scale_bits: u32,
}

/// Decoder for a single rANS stream (one lane). `data` is the slice of exactly this lane.
#[derive(Clone, Copy)]
pub struct Lane<'a> {
    data: &'a [u8],
    pos: usize,
    state: u32,
}

impl<'a> Lane<'a> {
    /// A placeholder cursor (no data, invalid state) that fills a stack array of lanes before the real ones are
    /// written over it; decoding with it never succeeds (`state < RANS_L`, [`Lane::done`] is false).
    pub const EMPTY: Lane<'static> = Lane { data: &[], pos: 0, state: 0 };

    /// Initialize the state from the first 4 bytes (LE). `None` if the stream is shorter than 4 bytes.
    pub fn new(data: &'a [u8]) -> Option<Self> {
        let state = u32::from_le_bytes(data.get(..4)?.try_into().ok()?);
        // A valid stream always finishes with state ∈ [RANS_L, RANS_UPPER).
        //  * state < RANS_L (e.g. 0): x==0 in `next` → renorm reads zeros, the state never grows.
        //  * state ≥ RANS_UPPER: `freq*(state>>sb)` overflows u32 in `next` (panic debug / garbage).
        // Reject both edges up front → inside `next` the u32 arithmetic is guaranteed not to overflow,
        // and renorm terminates. The GPU decoder must check the same range (single contract).
        if !(RANS_L..RANS_UPPER).contains(&state) {
            return None;
        }
        Some(Self { data, pos: 4, state })
    }

    /// `true` iff the stream was consumed EXACTLY: state back at the encoder's initial [`RANS_L`] and the
    /// cursor on the last byte. Holds for every valid stream after its last symbol; truncation (reads
    /// past the end, `pos > len`), trailing junk (`pos < len`) and forged bytes all break it, so this is
    /// a free integrity check. (The GPU kernels do not check it: their output on such a stream is unspecified.)
    pub fn done(&self) -> bool {
        self.state == RANS_L && self.pos == self.data.len()
    }

    /// `true` once the decoder has tried to read past the lane's last byte. A valid stream never does, so
    /// callers abort instead of decoding `n` symbols of zero-fed garbage (bounds the CPU work a forged
    /// stream with a huge `n` can cause by the bytes it actually carries; checked once per round of lanes).
    pub fn overrun(&self) -> bool {
        self.pos > self.data.len()
    }

    /// Decode one symbol and advance the state (`state = freq·(state >> sb) + slot - cum`; no division).
    /// A slot outside the table cannot happen (`slot = state & (M-1)`, `M` = table length); should the tables
    /// ever disagree the lookup yields 0 and the exact-end check ([`Lane::done`]) rejects the stream.
    #[inline(always)]
    pub fn next(&mut self, t: &DecTab<'_>) -> u8 {
        let slot = (self.state & ((1u32 << t.scale_bits) - 1)) as usize;
        let e = t.slot.get(slot).copied().unwrap_or(0);
        let sym = t.sym.get(slot).copied().unwrap_or(0);
        let mut x = (e >> 16) * (self.state >> t.scale_bits) + (e & 0xffff);
        // Valid data renorms ≤3 bytes (invariant state∈[RANS_L,RANS_UPPER) from Lane::new).
        // `guard` bounds the loop by the number shared with WGSL ([`RENORM_GUARD`]) → on any invalid
        // input the CPU and GPU read the same number of bytes and do not diverge (and no hang).
        // The first byte is renormalized BRANCH-FREE (≈ every symbol needs one on high-entropy data, at random
        // → a mispredict per symbol otherwise); a second/third byte is rare and stays a loop. Counting the
        // first step in `guard` keeps the total at ≤ RENORM_GUARD, bit-identical to the plain loop.
        let c = x < RANS_L;
        let b = self.data.get(self.pos).copied().unwrap_or(0) as u32;
        self.pos += usize::from(c);
        x = if c { (x << 8) | b } else { x };
        let mut guard = u32::from(c);
        while x < RANS_L && guard < RENORM_GUARD {
            // Past the end we read 0 so the loop stays panic-free; `pos` still advances, so
            // [`Self::overrun`]/[`Self::done`] report it and the caller rejects the stream.
            let b = self.data.get(self.pos).copied().unwrap_or(0) as u32;
            self.pos += 1;
            x = (x << 8) | b;
            guard += 1;
        }
        self.state = x;
        sym
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reciprocal replaces `x / freq` and `x % freq` EXACTLY for every legal `(freq, x)`: all 32768
    /// frequencies at `table_log` 15 against the multiples of `freq` (the only places an inexact reciprocal
    /// can slip by one), the `x_max` edge and a deterministic pseudo-random spread.
    #[test]
    fn reciprocal_division_is_exact() {
        let mut seed = 0x9e37_79b9u32;
        for freq in 1..=(1u32 << 15) {
            let e = EncSym::new(0, freq, 15);
            let q = |x: u32| ((x as u64 * e.rcp) >> e.shift) as u32;
            let x_max = e.x_max; // x < x_max ≤ 2^31 is the only domain `put` uses
            let mut xs = vec![0, 1, freq.saturating_sub(1), freq, x_max - 1];
            for k in [1u32, 2, 3, 255, 4096, 65535, 1 << 20] {
                if let Some(m) = k.checked_mul(freq).filter(|&m| m < x_max) {
                    xs.extend([m.saturating_sub(1), m, m + 1].into_iter().filter(|&v| v < x_max));
                }
            }
            for _ in 0..24 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                xs.push(seed % x_max);
            }
            for x in xs {
                assert_eq!(q(x), x / freq, "freq {freq} x {x}");
                assert_eq!(x + q(x) * e.m_sub_f, ((x / freq) << 15) + x % freq, "state update freq {freq} x {x}");
            }
        }
    }
}
