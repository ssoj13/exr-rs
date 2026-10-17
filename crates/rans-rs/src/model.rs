//! Frequency model for static rANS.
//!
//! # What and why
//! Counts symbol frequencies, normalizes to a sum of `M = 2^table_log` (a power of two —
//! a format invariant: `% M`/`/ M` degenerate into a mask/shift, which is critical for the GPU),
//! cumulative sums, and the inverse `slot → symbol` table for decode.
//!
//! # Determinism
//! Normalization is fully determined by the input counts and `table_log` → encoder and decoder
//! build identical tables. The alphabet is byte-sized (≤256), `table_log ∈ 8..=15`
//! (`M ≤ 32768` fits in `u16`; `M ≥ 256` ≥ alphabet → sum correction is always possible).

use std::sync::OnceLock;

use crate::byte_coder::{DecTab, EncSym};

/// Size of the byte alphabet.
pub const ALPHABET: usize = 256;

/// Normalized frequency model (`sum(freq) == 1 << table_log`).
///
/// Fields are private: the only constructors ([`Model::from_counts`], [`Model::from_freqs`]) establish the
/// invariants every decoder relies on (`freq` sums to `M`, `slot_to_sym.len() == M` is a power of two in
/// `256..=32768`, `cum` is the prefix sum), so a hand-built inconsistent model cannot exist.
#[derive(Clone)]
pub struct Model {
    freq: [u16; ALPHABET],
    cum: [u16; ALPHABET], // cum[s] = sum of freq[0..s]
    slot_to_sym: Vec<u8>, // length M; owner of the slot
    /// Per slot `freq << 16 | (slot - cum[sym])`: the decoder's single-load step table (CPU speed only; derived
    /// from `freq`/`cum`, never serialized — the wire format and the GPU tables are unchanged).
    slot_dec: Vec<u32>,
    /// Encoder constants per symbol (reciprocals), built on first encode: decode-only models never pay for it.
    enc: OnceLock<Box<[EncSym; ALPHABET]>>,
}

impl Model {
    /// Build a model from symbol counts. `table_log` is clamped into `8..=15` (the same range
    /// [`crate::RansConfig::new`] enforces), so the normalization below is always well-defined.
    pub fn from_counts(counts: &[u32; ALPHABET], table_log: u32) -> Self {
        let m = 1u32 << table_log.clamp(8, 15);
        let total: u64 = counts.iter().map(|&c| c as u64).sum();
        let mut freq = [0u16; ALPHABET];

        if total == 0 {
            // Degenerate input (no symbols): the model is not used for encoding,
            // but slot_to_sym must cover M slots — assign everything to symbol 0.
            freq[0] = m as u16;
            return Self::from_freq(freq, m);
        }
        // Initial scaling: proportional, a present symbol gets ≥ 1.
        for (f, &c) in freq.iter_mut().zip(counts) {
            if c != 0 {
                *f = ((c as u64 * m as u64) / total).max(1) as u16;
            }
        }
        // Correct to the exact sum M in ±1 steps on the current maximum (ties → the highest symbol).
        // Shrinking never zeroes a symbol: while `sum > M ≥ 256` the maximum over 256 slots is ≥ 2.
        // |diff| ≤ ALPHABET, cheap.
        let mut sum: i64 = freq.iter().map(|&f| f as i64).sum();
        let target = m as i64;
        while sum != target {
            let a = (0..ALPHABET).max_by_key(|&s| freq[s]).unwrap_or(0);
            if sum < target {
                freq[a] += 1;
                sum += 1;
            } else {
                freq[a] -= 1;
                sum -= 1;
            }
        }
        Self::from_freq(freq, m)
    }

    /// Cumulative sums + the inverse slot→symbol table from a frequency array that sums to `m`.
    fn from_freq(freq: [u16; ALPHABET], m: u32) -> Self {
        let mut model = Self { freq, cum: [0; ALPHABET], slot_to_sym: Vec::new(), slot_dec: Vec::new(), enc: OnceLock::new() };
        model.fill(m);
        model
    }

    /// (Re)build `cum` and both slot tables from `self.freq` (which sums to `m`), REUSING the table allocations:
    /// a decoder that loads one model per blob keeps a single model per thread ([`Model::load_freqs`]) instead of
    /// allocating ~20 KiB of tables per blob. Every slot is overwritten (the runs cover exactly `m`).
    fn fill(&mut self, m: u32) {
        let Self { freq, cum, slot_to_sym, slot_dec, enc } = self;
        slot_to_sym.resize(m as usize, 0);
        slot_dec.resize(m as usize, 0);
        enc.take(); // the encoder constants belong to the old frequencies
        // Each symbol owns the next `freq[s]` slots: carve them off the front of both tables (`split_at_mut`) and
        // fill them as whole runs, so there is no per-slot bounds check (`freq` sums to `m` = the table length).
        let (mut acc, mut syms, mut decs) = (0u32, slot_to_sym.as_mut_slice(), slot_dec.as_mut_slice());
        for ((s, &f), c) in (0u8..=u8::MAX).zip(freq.iter()).zip(cum.iter_mut()) {
            *c = acc as u16;
            let fu = f as usize;
            let (Some((sy, sr)), Some((de, dr))) =
                (std::mem::take(&mut syms).split_at_mut_checked(fu), std::mem::take(&mut decs).split_at_mut_checked(fu))
            else {
                break; // `freq` sums past `m`: excluded by both constructors
            };
            sy.fill(s);
            for (d, k) in de.iter_mut().zip(0u32..) {
                *d = (u32::from(f) << 16) | k;
            }
            (syms, decs) = (sr, dr);
            acc += u32::from(f);
        }
        debug_assert_eq!(acc, m);
    }

    /// Normalized frequency of every symbol (sums to `1 << table_log`).
    pub fn freq(&self) -> &[u16; ALPHABET] {
        &self.freq
    }

    /// Cumulative frequency (`cum[s] = sum(freq[..s])`) of every symbol.
    pub fn cum(&self) -> &[u16; ALPHABET] {
        &self.cum
    }

    /// Inverse table: the symbol owning each of the `M` slots.
    pub fn slot_to_sym(&self) -> &[u8] {
        &self.slot_to_sym
    }

    /// The flattened decode tables for the CPU lane loop ([`crate::byte_coder::Lane::next`]).
    pub(crate) fn dec_tab(&self) -> DecTab<'_> {
        DecTab { slot: &self.slot_dec, sym: &self.slot_to_sym, scale_bits: self.table_log() }
    }

    /// Encoder constants of every symbol (built once per model, thread-safe).
    pub(crate) fn enc_syms(&self) -> &[EncSym; ALPHABET] {
        self.enc.get_or_init(|| {
            let tl = self.table_log();
            let mut t = Box::new([EncSym::new(0, 0, tl); ALPHABET]);
            for (e, (&f, &c)) in t.iter_mut().zip(self.freq.iter().zip(&self.cum)) {
                *e = EncSym::new(c as u32, f as u32, tl);
            }
            t
        })
    }

    /// `log2(M)`: `M` is `slot_to_sym.len()`, a power of two by construction.
    pub fn table_log(&self) -> u32 {
        self.slot_to_sym.len().trailing_zeros()
    }

    /// Serialize the frequency table as `ALPHABET×u16` LE — the 512-byte shared-table blob stored once
    /// per EXV-RT tile (inverse of [`Model::from_freqs`]).
    pub fn freqs_le(&self) -> [u8; ALPHABET * 2] {
        let mut out = [0u8; ALPHABET * 2];
        for (o, &f) in out.as_chunks_mut::<2>().0.iter_mut().zip(&self.freq) {
            *o = f.to_le_bytes();
        }
        out
    }

    /// Reconstruct the model from a frequency table (inverse of [`Model::freqs_le`]).
    /// Returns `None` if the table is incomplete or the sum ≠ M (corrupt stream).
    pub fn from_freqs(freq_bytes: &[u8], table_log: u32) -> Option<Self> {
        let mut model = Self::default();
        model.load_freqs(freq_bytes, table_log)?;
        Some(model)
    }

    /// [`Model::from_freqs`] into `self`, reusing its table allocations (the per-thread decode model of
    /// [`crate::with_model`]). `None` leaves `self` untouched (validation happens before any write).
    pub fn load_freqs(&mut self, freq_bytes: &[u8], table_log: u32) -> Option<()> {
        // table_log from an untrusted container: outside 8..=15 → `1<<table_log` is a shift overflow
        // (panic in debug). Validate before any shifts/allocations.
        if !(8..=15).contains(&table_log) {
            return None;
        }
        let pairs = freq_bytes.as_chunks::<2>().0;
        if pairs.len() < ALPHABET {
            return None;
        }
        let mut freq = [0u16; ALPHABET];
        let mut sum: u32 = 0;
        for (f, b) in freq.iter_mut().zip(pairs) {
            *f = u16::from_le_bytes(*b);
            sum += u32::from(*f);
        }
        let m = 1u32 << table_log;
        if sum != m {
            return None;
        }
        self.freq = freq;
        self.fill(m);
        Some(())
    }
}

impl Default for Model {
    /// The degenerate valid model (`M = 256`, every slot symbol 0): a starting point for [`Model::load_freqs`].
    fn default() -> Self {
        let mut freq = [0u16; ALPHABET];
        freq[0] = 256;
        Self::from_freq(freq, 256)
    }
}
