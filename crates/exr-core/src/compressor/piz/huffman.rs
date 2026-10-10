//! 16-bit Huffman compression/decompression for the PIZ codec — 1:1 port of
//! OpenEXR's `ImfHuf.cpp` (`hufCompress` / `hufUncompress`, the canonical-code
//! build `hufCanonicalCodeTable`, `hufBuildEncTable` / `hufBuildDecTable`, the
//! 6-bit code-length packing and 8-bit run-length packing).
//!
//! The encoder is fully deterministic and **byte-identical** to C++: the same
//! frequencies produce the same canonical Huffman codes, the same packed table,
//! and the same bit stream.  The reference test `tests::golden_vector` pins
//! the output against a vector captured from genuine C++ OpenEXR.
//!
//! # Output layout (`hufCompress`)
//!
//! A 20-byte little-endian header followed by the packed code table and the
//! entropy-coded data:
//!
//! ```text
//!   [0..4)   im        (min non-zero symbol index)        u32 LE
//!   [4..8)   iM        (max non-zero symbol index)        u32 LE
//!   [8..12)  tableLen  (packed-table length in bytes)     u32 LE
//!   [12..16) nBits     (entropy stream length in bits)    u32 LE
//!   [16..20) 0         (reserved)                         u32 LE
//!   [20..)   packed code table, then the bit stream
//! ```
//!
//! # Panic-free decoder
//!
//! Every read is bounds-checked and every structural invariant validated; a
//! corrupt or truncated stream yields an [`ExrError`], never a panic, `unwrap`,
//! or out-of-bounds access.  This mirrors the C++ decoder's `throw InputExc`
//! sites (`invalidCode`, `tooMuchData`, `notEnoughData`, `tableTooLong`, …).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfHuf.cpp>
//!
//! The behavioural target (C++ 3.4.15) decodes
//! through OpenEXRCore's `internal_huf.c`, not this legacy `Imf` file —
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_huf.c>.
//! [`decompress`] is the real `hufUncompress` dispatch: `nBits > 128` (real
//! chunks essentially always are) takes the accelerated [`super::fast_huf`]
//! decoder, `nBits <= 128` takes the generic scalar path ported here
//! (`decode_generic`). See [`compress_with_limit`]. A former
//! FastHuf/slow-path accept-reject gap is closed now that both paths are
//! reached through one real dispatch rather than compared against each other
//! as an approximation of it.

use super::fast_huf;
use crate::error::{ExrError, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Constants — mirror `ImfHuf.cpp`
// ---------------------------------------------------------------------------

/// Literal (value) bit length — `HUF_ENCBITS`.
const ENCODE_BITS: u32 = 16;
/// Decoding hash bit size — `HUF_DECBITS` (>= 8).
const DECODE_BITS: u32 = 14;

/// Encoding table size — `HUF_ENCSIZE = (1 << ENCBITS) + 1`.
const ENCODING_TABLE_SIZE: usize = (1 << ENCODE_BITS) + 1;
/// Decoding table size — `HUF_DECSIZE = 1 << DECBITS`.
const DECODING_TABLE_SIZE: usize = 1 << DECODE_BITS;
/// Decoding table mask — `HUF_DECMASK`.
const DECODE_MASK: u64 = DECODING_TABLE_SIZE as u64 - 1;

/// Packed-table run-length sentinels — mirror `ImfHuf.cpp`.
pub(super) const SHORT_ZEROCODE_RUN: u64 = 59;
pub(super) const LONG_ZEROCODE_RUN: u64 = 63;
pub(super) const SHORTEST_LONG_RUN: u64 = 2 + LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN;
const LONGEST_LONG_RUN: u64 = 255 + SHORTEST_LONG_RUN;

// ---------------------------------------------------------------------------
// (code, length) packing helpers — `hufCode` / `hufLength`
// ---------------------------------------------------------------------------

/// Code-length field (low 6 bits) — `hufLength`.
#[inline]
fn length(code: u64) -> u64 {
    code & 63
}

/// Code-value field (bits 6..) — `hufCode`.
#[inline]
fn code(code: u64) -> u64 {
    code >> 6
}

// ---------------------------------------------------------------------------
// Decoding table entry
// ---------------------------------------------------------------------------

/// A decoding-table slot — the safe-Rust equivalent of C++ `struct HufDec`.
///
/// * `Empty` — unused slot.
/// * `Short` — a code of length `<= DECODE_BITS`, resolved by a single table
///   lookup (`len` + symbol `value`).
/// * `Long` — one or more codes longer than `DECODE_BITS` sharing this prefix;
///   the symbols are searched linearly (rare, exactly as in C++).
#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug))]
enum Dec {
    #[default]
    Empty,
    Short { value: u32, len: u8 },
    Long(Vec<u32>),
}

/// Persistent scratch of the GENERIC scalar decoder ([`decode_generic`], the
/// `nBits <= 128` path): the unpacked code table (~512 KiB) and the decoding
/// hash table (~512 KiB). They were fresh `vec!`s on every such chunk — a
/// constant or near-constant chunk, the common case in flat render regions,
/// paid two half-megabyte allocations + memsets for a handful of symbols.
/// Lives inside [`fast_huf::FastHufDecoder`] so the ONE persistent decoder
/// object every caller already owns (PIZ, DWA) carries it; no signature change.
///
/// `enc` needs no per-call reset ([`unpack_encoding_table`] writes every index
/// of the `[im, iM]` range it hands on), `dec` is re-filled with
/// [`Dec::Empty`] by [`build_decoding_table`]. `Dec::Long` buckets (codes
/// longer than [`DECODE_BITS`], never seen with `nBits <= 128`) still
/// allocate their own small `Vec`, as in C++'s `HufDec::p`.
#[derive(Default)]
#[cfg_attr(test, derive(Debug))]
pub(super) struct GenScratch {
    enc: Vec<u64>,
    dec: Vec<Dec>,
}

#[cfg(test)]
impl GenScratch {
    /// Test-only: fill both tables with implausible values (and one
    /// `Dec::Long` bucket) so a missed reset shows up as a decode difference.
    pub(super) fn poison_for_test(&mut self) {
        self.enc = vec![0xAAAA_AAAA_AAAA_AAAA; ENCODING_TABLE_SIZE];
        self.dec = vec![Dec::Short { value: 0xAAAA_AAAA, len: 0xAA }; DECODING_TABLE_SIZE];
        self.dec[0] = Dec::Long(vec![0xAAAA_AAAA; 4]);
    }
}

// ---------------------------------------------------------------------------
// Bit reader / writer
// ---------------------------------------------------------------------------

/// LSB-first-into-MSB-out bit writer matching C++ `outputBits`.
struct BitWriter {
    out: Vec<u8>,
    bits: u64,
    count: u32,
}

impl BitWriter {
    /// Build a writer over a caller-owned, reusable buffer
    /// instead of always allocating a fresh one — `out.clear()` first since a
    /// reused buffer may carry a PRIOR call's bytes; `push`-based writes from
    /// then on don't care about the buffer's prior capacity/contents beyond
    /// that.
    fn with_buf(mut out: Vec<u8>) -> Self {
        out.clear();
        Self { out, bits: 0, count: 0 }
    }

    /// Append `n_bits` low bits of `value` — port of `outputBits`.
    #[inline]
    fn write(&mut self, n_bits: u32, value: u64) {
        self.bits = (self.bits << n_bits) | (value & ((1u64 << n_bits) - 1));
        self.count += n_bits;
        while self.count >= 8 {
            self.count -= 8;
            self.out.push((self.bits >> self.count) as u8);
        }
    }

    /// Write a full code (length + value) — port of `outputCode`.
    #[inline]
    fn write_code(&mut self, packed: u64) {
        self.write(length(packed) as u32, code(packed));
    }

    /// Flush any remaining bits as one final byte (left-justified) and return
    /// the entropy-stream length in **bits** (`(out - outStart) * 8 + lc`).
    fn finish(mut self) -> (Vec<u8>, u64) {
        let data_bits = self.out.len() as u64 * 8 + self.count as u64;
        if self.count > 0 {
            // C++: *out = (c << (8 - lc)) & 0xff;  (does NOT advance the count)
            let last = ((self.bits << (8 - self.count)) & 0xff) as u8;
            self.out.push(last);
        }
        (self.out, data_bits)
    }
}

/// MSB-first bit reader matching C++ `getBits` / `getChar`.
///
/// `pub(super)`: also reused by [`super::fast_huf`] to parse the packed
/// code-length table (`fasthuf_initialize` reads that table with its own
/// `fasthuf_read_bits`, but that is the exact same MSB-first wire format as
/// `unpack_encoding_table`'s use of this reader, so one reader serves both).
pub(super) struct BitReader<'a> {
    input: &'a [u8],
    pub(super) pos: usize,
    bits: u64,
    count: u32,
}

impl<'a> BitReader<'a> {
    pub(super) fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            bits: 0,
            count: 0,
        }
    }

    /// Pull one byte into the accumulator — port of `getChar`.  Errors on EOF.
    #[inline]
    fn read_byte(&mut self) -> Result<()> {
        let b = *self
            .input
            .get(self.pos)
            .ok_or_else(|| ExrError::Input("PIZ huffman: unexpected end of input".into()))?;
        self.pos += 1;
        self.bits = (self.bits << 8) | b as u64;
        self.count += 8;
        Ok(())
    }

    /// Read `n_bits` bits — port of `getBits`.  Errors on EOF.
    #[inline]
    pub(super) fn read_bits(&mut self, n_bits: u32) -> Result<u64> {
        while self.count < n_bits {
            self.read_byte()?;
        }
        self.count -= n_bits;
        Ok((self.bits >> self.count) & ((1u64 << n_bits) - 1))
    }
}

// ---------------------------------------------------------------------------
// Frequency counting — `countFrequencies`
// ---------------------------------------------------------------------------

/// Count occurrences of every symbol into `freq` — port of
/// `countFrequencies`.
///
/// `freq` is a caller-owned, reusable buffer (part of
/// [`HufEncScratch`]): grown to [`ENCODING_TABLE_SIZE`] once (via
/// [`crate::compressor::ensure_len`]) and then fully re-zeroed EVERY call --
/// this is an accumulator (`freq[v] += 1`), so a stale count left over from a
/// PRIOR chunk's histogram at an index this chunk doesn't touch would
/// silently corrupt this chunk's table. The win over the old
/// `vec![0u64; ENCODING_TABLE_SIZE]` per call is the allocation (~512 KiB),
/// not the zero-fill, which stays unavoidable here.
fn count_frequencies(data: &[u16], freq: &mut Vec<u64>) {
    crate::compressor::ensure_len(freq, ENCODING_TABLE_SIZE);
    freq.fill(0);
    for &v in data {
        freq[v as usize] += 1;
    }
}

// ---------------------------------------------------------------------------
// Canonical code table — `hufCanonicalCodeTable`
// ---------------------------------------------------------------------------

/// Replace code lengths in `table` with canonical `(code, length)` pairs —
/// 1:1 port of `hufCanonicalCodeTable`.
///
/// Shorter codes get numerically higher values; equal-length codes increase
/// with symbol index.  Determinism here is what makes the encoder byte-exact.
fn build_canonical_table(table: &mut [u64]) -> Result<()> {
    // n[i] counts codes of length i (0..=58).
    let mut n = [0u64; 59];
    for &c in table.iter() {
        let l = c as usize;
        // Code lengths are produced by build_encoding_table (<= 58); guard so a
        // corrupt unpacked table cannot index out of bounds on the decode path.
        if l >= n.len() {
            return Err(ExrError::Input(
                "PIZ huffman: code length exceeds 58 in canonical table".into(),
            ));
        }
        n[l] += 1;
    }

    // For i from 58 down to 1: lowest code of length i.
    let mut c: u64 = 0;
    for i in (1..=58usize).rev() {
        let nc = (c + n[i]) >> 1;
        n[i] = c;
        c = nc;
    }

    // Assign each symbol the next code of its length.
    for entry in table.iter_mut() {
        let l = *entry as usize;
        if l > 0 {
            *entry |= n[l] << 6;
            n[l] += 1;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding table — `hufBuildEncTable`
// ---------------------------------------------------------------------------

/// Heap element keyed by frequency then symbol index, reproducing C++'s
/// `FHeapCompare` tie-break (`(*a > *b) || (*a == *b && a > b)`) so the
/// resulting Huffman tree — and therefore the code lengths — are identical
/// across platforms.
///
/// C++ uses a *max-heap* with `operator()` returning `a` *after* `b` when `a`
/// has the larger frequency, so `make_heap` keeps the **smallest** frequency on
/// top.  `BinaryHeap` is a max-heap, so `Ord` is inverted accordingly.
#[derive(Eq, PartialEq, Copy, Clone)]
struct HeapItem {
    frequency: u64,
    position: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Smallest frequency (then smallest position) must be "greatest" so it
        // pops first from the max-heap.
        other
            .frequency
            .cmp(&self.frequency)
            .then_with(|| other.position.cmp(&self.position))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build the encoding table from `frequencies` (destroyed in place) — 1:1 port
/// of `hufBuildEncTable`.  Returns `(im, iM)` = the min / max non-zero symbol
/// indices (with the run-length pseudo-symbol added at `iM`).
///
/// `hlink`/`scode`/`heap_indices` are caller-owned, reusable buffers (part of
/// [`HufEncScratch`]) instead of fresh per-call allocations:
/// * `hlink[i] = i` is written unconditionally for every `i` in
///   `[min_index, ENCODING_TABLE_SIZE)` by the loop just below, regardless of
///   what a prior call left there, and indices below `min_index` are never
///   read this call (every merge-list walk starts from a `heap_indices` entry
///   or a value already written this call, both `>= min_index`) — so
///   `ensure_len` (grow only, no reset) is sound.
/// * `scode` is the opposite case: `build_canonical_table` below reads EVERY
///   one of its `ENCODING_TABLE_SIZE` entries (`pack_encoding_table` later
///   also walks the full `[im, im_max]` range expecting an untouched symbol
///   to read as length 0), but the merge loop only writes entries reachable
///   from a heap symbol — so a stale nonzero length surviving from a PRIOR
///   chunk at an index THIS chunk never touches would corrupt the packed
///   table. Must be fully re-zeroed every call, same reasoning as `freq`.
/// * `heap_indices` is `clear()`ed and re-`push`ed (no zero-fill needed —
///   only pushed entries are ever read).
/// * `heap` (up to 64K x 16 B, was a fresh `BinaryHeap::with_capacity` per
///   chunk) is `clear()`ed and refilled; its pop order depends only on the
///   total `(frequency, position)` order, never on the buffer's history.
fn build_encoding_table(
    frequencies: &mut [u64],
    hlink: &mut Vec<usize>,
    scode: &mut Vec<u64>,
    heap_indices: &mut Vec<usize>,
    heap: &mut BinaryHeap<HeapItem>,
) -> Result<(usize, usize)> {
    crate::compressor::ensure_len(hlink, ENCODING_TABLE_SIZE);
    crate::compressor::ensure_len(scode, ENCODING_TABLE_SIZE);
    scode.fill(0);
    heap_indices.clear();

    // im = first non-zero entry (0 if the data were all-zero — frequencies[0]
    // is then forced below by the pseudo-symbol, matching C++ which assumes the
    // caller passes non-empty data).
    let min_index = frequencies.iter().position(|&f| f != 0).unwrap_or(0);

    let mut max_index = 0usize;
    for i in min_index..ENCODING_TABLE_SIZE {
        hlink[i] = i;
        if frequencies[i] != 0 {
            heap_indices.push(i);
            max_index = i;
        }
    }

    // Add the run-length pseudo-symbol (frequency 1) just past iM.
    max_index += 1;
    frequencies[max_index] = 1;
    heap_indices.push(max_index);

    heap.clear();
    if heap.capacity() >= heap_indices.len() {
        crate::probe::hit(crate::probe::Probe::PizHufHeapReuse);
    }
    heap.reserve(heap_indices.len());
    for &i in heap_indices.iter() {
        heap.push(HeapItem {
            frequency: frequencies[i],
            position: i,
        });
    }

    // Repeatedly merge the two least-frequent symbols.
    while heap.len() > 1 {
        let smallest = heap
            .pop()
            .ok_or_else(|| ExrError::Logic("PIZ huffman: heap underflow".into()))?;
        let second = heap
            .pop()
            .ok_or_else(|| ExrError::Logic("PIZ huffman: heap underflow".into()))?;

        let mm = smallest.position;
        let m = second.position;

        // C++: frq[m] += frq[mm]; push_heap with the new combined node at m.
        frequencies[m] += frequencies[mm];
        heap.push(HeapItem {
            frequency: frequencies[m],
            position: m,
        });

        // Add a bit to every code in the first list (starting at m), then merge
        // the second list (starting at mm) onto its tail.
        let mut j = m;
        loop {
            scode[j] += 1;
            if scode[j] > 58 {
                return Err(ExrError::Input(
                    "PIZ huffman: code length overflow (>58 bits)".into(),
                ));
            }
            if hlink[j] == j {
                hlink[j] = mm; // merge
                break;
            }
            j = hlink[j];
        }

        // Add a bit to every code in the second list.
        let mut j = mm;
        loop {
            scode[j] += 1;
            if scode[j] > 58 {
                return Err(ExrError::Input(
                    "PIZ huffman: code length overflow (>58 bits)".into(),
                ));
            }
            if hlink[j] == j {
                break;
            }
            j = hlink[j];
        }
    }

    // Turn the lengths in scode into canonical codes, then copy back into frq.
    build_canonical_table(scode)?;
    frequencies.copy_from_slice(scode);

    Ok((min_index, max_index))
}

// ---------------------------------------------------------------------------
// Pack / unpack the encoding table — `hufPackEncTable` / `hufUnpackEncTable`
// ---------------------------------------------------------------------------

/// Pack the code lengths (with zero-run compression) into `writer` — 1:1 port
/// of `hufPackEncTable`.
fn pack_encoding_table(frequencies: &[u64], im: usize, im_max: usize, writer: &mut BitWriter) {
    let mut i = im;
    while i <= im_max {
        let l = length(frequencies[i]);
        if l == 0 {
            // Count the zero run (bounded by LONGEST_LONG_RUN).
            let mut zerun = 1u64;
            while i < im_max && zerun < LONGEST_LONG_RUN {
                if length(frequencies[i + 1]) > 0 {
                    break;
                }
                i += 1;
                zerun += 1;
            }
            if zerun >= 2 {
                if zerun >= SHORTEST_LONG_RUN {
                    writer.write(6, LONG_ZEROCODE_RUN);
                    writer.write(8, zerun - SHORTEST_LONG_RUN);
                } else {
                    writer.write(6, SHORT_ZEROCODE_RUN + zerun - 2);
                }
                i += 1;
                continue;
            }
        }
        writer.write(6, l);
        i += 1;
    }
}

/// Unpack a code-length table written by [`pack_encoding_table`] into the
/// caller-owned `table` — 1:1 port of `hufUnpackEncTable`.  Validates every run
/// against `iM` (`tableTooLong`).
///
/// Only `table[im..=im_max]` is meaningful afterwards, and every index of it
/// is written here (a length, or the explicit zero of a run), so a reused
/// `table` needs no re-zeroing. C++ zeroes the whole `HUF_ENCSIZE` array, but
/// nothing outside `[im, iM]` is ever read (`build_decoding_table` and
/// `decode_data` only index symbols in that range), and
/// [`build_canonical_table`] is run over the range alone: zero entries are
/// skipped there, so the codes are identical.
fn unpack_encoding_table(
    reader: &mut BitReader,
    im: usize,
    im_max: usize,
    table: &mut Vec<u64>,
) -> Result<()> {
    crate::compressor::ensure_len(table, ENCODING_TABLE_SIZE);

    let mut i = im;
    while i <= im_max {
        let l = reader.read_bits(6)?;
        table[i] = l;

        if l == LONG_ZEROCODE_RUN {
            let zerun = (reader.read_bits(8)? + SHORTEST_LONG_RUN) as usize;
            if i + zerun > im_max + 1 {
                return Err(ExrError::Input(
                    "PIZ huffman: packed table run too long (LONG_ZEROCODE_RUN)".into(),
                ));
            }
            for slot in table.iter_mut().skip(i).take(zerun) {
                *slot = 0;
            }
            i += zerun;
        } else if l >= SHORT_ZEROCODE_RUN {
            let zerun = (l - SHORT_ZEROCODE_RUN + 2) as usize;
            if i + zerun > im_max + 1 {
                return Err(ExrError::Input(
                    "PIZ huffman: packed table run too long (SHORT_ZEROCODE_RUN)".into(),
                ));
            }
            for slot in table.iter_mut().skip(i).take(zerun) {
                *slot = 0;
            }
            i += zerun;
        } else {
            i += 1;
        }
    }

    build_canonical_table(&mut table[im..=im_max])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Decoding table — `hufBuildDecTable`
// ---------------------------------------------------------------------------

/// Build the decoding hash table from the encoding table into the caller-owned
/// `table` — 1:1 port of `hufBuildDecTable`.  Validates code/length consistency
/// (`invalidTableEntry`). Every slot is reset to [`Dec::Empty`] first: slots are
/// only ever written when still empty, so stale entries would be rejected as
/// "overlapping codes".
fn build_decoding_table(
    encoding_table: &[u64],
    im: usize,
    im_max: usize,
    table: &mut Vec<Dec>,
) -> Result<()> {
    crate::compressor::ensure_len(table, DECODING_TABLE_SIZE);
    table.fill(Dec::Empty);

    for (symbol, &enc) in encoding_table
        .iter()
        .enumerate()
        .take(im_max + 1)
        .skip(im)
    {
        let c = code(enc);
        let l = length(enc);

        // c must fit in l bits.
        if l > 0 && (c >> l) != 0 {
            return Err(ExrError::Input(
                "PIZ huffman: invalid code table entry (code exceeds its length)".into(),
            ));
        }

        if l > DECODE_BITS as u64 {
            // Long code: append the symbol to the shared-prefix bucket.
            let idx = (c >> (l - DECODE_BITS as u64)) as usize;
            let slot = table
                .get_mut(idx)
                .ok_or_else(|| ExrError::Input("PIZ huffman: long-code index out of range".into()))?;
            match slot {
                Dec::Empty => *slot = Dec::Long(vec![symbol as u32]),
                Dec::Long(v) => v.push(symbol as u32),
                Dec::Short { .. } => {
                    return Err(ExrError::Input(
                        "PIZ huffman: invalid table entry (short code where long expected)".into(),
                    ));
                }
            }
        } else if l > 0 {
            // Short code: fill every primary entry sharing this prefix.
            let start = (c << (DECODE_BITS as u64 - l)) as usize;
            let count = 1usize << (DECODE_BITS as u64 - l);
            let end = start
                .checked_add(count)
                .filter(|&e| e <= DECODING_TABLE_SIZE)
                .ok_or_else(|| {
                    ExrError::Input("PIZ huffman: short-code range out of bounds".into())
                })?;
            let len_u8 = u8::try_from(l)
                .map_err(|_| ExrError::Input("PIZ huffman: code length too large".into()))?;
            for slot in &mut table[start..end] {
                if !matches!(slot, Dec::Empty) {
                    return Err(ExrError::Input(
                        "PIZ huffman: invalid table entry (overlapping codes)".into(),
                    ));
                }
                *slot = Dec::Short {
                    value: symbol as u32,
                    len: len_u8,
                };
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding — `hufEncode` / `sendCode`
// ---------------------------------------------------------------------------

/// Emit a run of `run_count + 1` instances of `s_code`, or the run-length
/// encoded form when shorter — 1:1 port of `sendCode`.
#[inline]
fn send_code(writer: &mut BitWriter, s_code: u64, run_count: u64, run_code: u64) {
    if length(s_code) + length(run_code) + 8 < length(s_code) * run_count {
        writer.write_code(s_code);
        writer.write_code(run_code);
        writer.write(8, run_count);
    } else {
        // C++: while (runCount-- >= 0) outputCode(...); → runCount+1 outputs.
        for _ in 0..=run_count {
            writer.write_code(s_code);
        }
    }
}

/// Entropy-encode `data` with the codes in `encoding_table` — 1:1 port of
/// `hufEncode`.  Returns the produced bytes and the stream length in bits.
///
/// `writer_buf` is a caller-owned, reusable buffer for the
/// entropy stream's bytes instead of always allocating a fresh one.
fn encode_data(
    encoding_table: &[u64],
    data: &[u16],
    run_length_code: usize,
    writer_buf: Vec<u8>,
) -> (Vec<u8>, u64) {
    let mut writer = BitWriter::with_buf(writer_buf);

    // data is non-empty (checked by the caller).
    let mut s = data[0];
    let mut cs: u64 = 0;

    for &cur in &data[1..] {
        if s == cur && cs < 255 {
            cs += 1;
        } else {
            send_code(
                &mut writer,
                encoding_table[s as usize],
                cs,
                encoding_table[run_length_code],
            );
            cs = 0;
        }
        s = cur;
    }

    send_code(
        &mut writer,
        encoding_table[s as usize],
        cs,
        encoding_table[run_length_code],
    );

    writer.finish()
}

// ---------------------------------------------------------------------------
// Decoding — `hufDecode` / `getCode`
// ---------------------------------------------------------------------------

/// Emit one decoded symbol (`po`) into `out`, expanding a run when `po` is the
/// run-length code — 1:1 port of the `getCode` macro.  Fully bounds-checked.
#[inline]
fn read_code_into(
    symbol: u32,
    run_length_code: u32,
    reader: &mut BitReader,
    out: &mut Vec<u16>,
    expected: usize,
) -> Result<()> {
    if symbol == run_length_code {
        // Read the 8-bit repetition count and replicate the previous symbol.
        if reader.count < 8 {
            reader.read_byte()?;
        }
        reader.count -= 8;
        let cs = ((reader.bits >> reader.count) & 0xff) as usize;

        if out.len() + cs > expected {
            return Err(ExrError::Input(
                "PIZ huffman: decoded data longer than expected (run)".into(),
            ));
        }
        let last = *out
            .last()
            .ok_or_else(|| ExrError::Input("PIZ huffman: run code before any symbol".into()))?;
        out.extend(std::iter::repeat_n(last, cs));
    } else if out.len() < expected {
        let v = u16::try_from(symbol)
            .map_err(|_| ExrError::Input("PIZ huffman: symbol exceeds u16 range".into()))?;
        out.push(v);
    } else {
        return Err(ExrError::Input(
            "PIZ huffman: decoded data longer than expected".into(),
        ));
    }
    Ok(())
}

/// Entropy-decode `input` (`n_bits` bits) — 1:1 port of `hufDecode`.  Validates
/// all input; any corruption returns an [`ExrError`] (never a panic).
///
/// `out` is NOT cleared here — the one public entry point ([`decompress`])
/// clears it once on entry; every internal function assumes
/// `out` already starts empty.
fn decode_data(
    encoding_table: &[u64],
    decoding_table: &[Dec],
    input: &[u8],
    n_bits: i64,
    run_length_code: u32,
    expected: usize,
    out: &mut Vec<u16>,
) -> Result<()> {
    let mut reader = BitReader::new(input);

    // Number of input bytes covered by n_bits (C++ `ie = in + (ni+7)/8`).
    let in_byte_len = (n_bits as u64).div_ceil(8) as usize;
    if in_byte_len > input.len() {
        return Err(ExrError::Input(
            "PIZ huffman: bit count exceeds available input".into(),
        ));
    }

    // Main loop: pull bytes while data remains.
    while reader.pos < in_byte_len {
        reader.read_byte()?;

        while reader.count >= DECODE_BITS {
            let idx = ((reader.bits >> (reader.count - DECODE_BITS)) & DECODE_MASK) as usize;
            match &decoding_table[idx] {
                Dec::Short { value, len } => {
                    let len = *len as u32;
                    if len > reader.count {
                        return Err(ExrError::Input(
                            "PIZ huffman: invalid code (length too long)".into(),
                        ));
                    }
                    reader.count -= len;
                    let value = *value;
                    read_code_into(value, run_length_code, &mut reader, out, expected)?;
                }
                Dec::Long(symbols) => {
                    // Linear-search the bucket for the matching long code.
                    let mut matched = None;
                    for &sym in symbols {
                        let enc = encoding_table[sym as usize];
                        let l = length(enc);
                        // Pull more bits if needed (bounded by available input).
                        while (reader.count as u64) < l && reader.pos < in_byte_len {
                            reader.read_byte()?;
                        }
                        if reader.count as u64 >= l {
                            let want = (reader.bits >> (reader.count as u64 - l))
                                & ((1u64 << l) - 1);
                            if code(enc) == want {
                                reader.count -= l as u32;
                                matched = Some(sym);
                                break;
                            }
                        }
                    }
                    let sym = matched.ok_or_else(|| {
                        ExrError::Input("PIZ huffman: invalid code (long code not found)".into())
                    })?;
                    read_code_into(sym, run_length_code, &mut reader, out, expected)?;
                }
                Dec::Empty => {
                    return Err(ExrError::Input("PIZ huffman: invalid code (empty slot)".into()));
                }
            }
        }
    }

    // Drain the trailing partial code(s) (C++ tail loop).
    let shift = ((8 - n_bits) & 7) as u32;
    reader.bits >>= shift;
    reader.count = reader
        .count
        .checked_sub(shift)
        .ok_or_else(|| ExrError::Input("PIZ huffman: corrupt trailing bits".into()))?;

    while reader.count > 0 {
        let idx = ((reader.bits << (DECODE_BITS - reader.count)) & DECODE_MASK) as usize;
        match &decoding_table[idx] {
            Dec::Short { value, len } => {
                let len = *len as u32;
                if len > reader.count {
                    return Err(ExrError::Input(
                        "PIZ huffman: invalid trailing code (length too long)".into(),
                    ));
                }
                reader.count -= len;
                let value = *value;
                read_code_into(value, run_length_code, &mut reader, out, expected)?;
            }
            _ => {
                return Err(ExrError::Input(
                    "PIZ huffman: invalid trailing code".into(),
                ));
            }
        }
    }

    if out.len() != expected {
        return Err(ExrError::Input(format!(
            "PIZ huffman: decoded {} values, expected {expected}",
            out.len()
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public interface — `hufCompress` / `hufUncompress`
// ---------------------------------------------------------------------------

/// Read a little-endian `u32` at `buf[off..off+4]`, error on short buffer.
#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> Result<u32> {
    let b = buf
        .get(off..off + 4)
        .ok_or_else(|| ExrError::Input("PIZ huffman: header truncated".into()))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Persistent encode-side scratch for [`compress`]/[`compress_with_limit`]
/// `freq`/`hlink`/`scode` are each [`ENCODING_TABLE_SIZE`]
/// (`(1 << 16) + 1`) elements — about 2 MiB combined — and were previously
/// fresh `vec![...]` allocations on EVERY chunk; `heap_indices` and the two
/// `BitWriter` buffers (`table_buf`/`data_buf`) are smaller but were also
/// reallocated per call. Lives in `PizCompressor`/`DwaCompressor` (DWA's
/// static-Huffman AC path uses the same encoder).
///
/// See `count_frequencies`/`build_encoding_table`'s docs for exactly which
/// fields need a full re-zero every call (`freq`, `scode` — both accumulators
/// read in full downstream) versus which only need `ensure_len` (`hlink`,
/// whose write loop covers every reachable index unconditionally; the two
/// `BitWriter` buffers, which are `push`-only).
#[derive(Default)]
pub(crate) struct HufEncScratch {
    freq: Vec<u64>,
    hlink: Vec<usize>,
    scode: Vec<u64>,
    heap_indices: Vec<usize>,
    heap: BinaryHeap<HeapItem>,
    table_buf: Vec<u8>,
    data_buf: Vec<u8>,
}

/// Huffman-compress `raw` into `out` — 1:1 port of `hufCompress` — with an
/// injectable bit-count ceiling (real callers always pass `u32::MAX as u64`;
/// tests inject a tiny one to exercise the out-of-range path below
/// without a real >512 MiB fixture).
///
/// There is no separate zero-argument `compress` entry point (unlike the
/// earlier shape): `PizCompressor`/`DwaCompressor` call this directly with the
/// real ceiling, and a thin wrapper with no other caller would just be dead
/// code under this module's `pub(crate)` boundary.
///
/// Returns `Ok(true)` with `out` holding the full 20-byte-header-prefixed
/// compressed buffer, `Ok(false)` (with `out` left untouched) when the
/// entropy-coded bit count does not fit a `u32`, or `Err` on an internal
/// table-building failure. An empty input yields `Ok(true)` with `out`
/// cleared (matching `nRaw == 0 → return 0`).
///
/// `hufEncode` (`internal_huf.c:846-848`,
/// <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_huf.c#L846-L848>)
/// returns `EXR_ERR_ARGUMENT_OUT_OF_RANGE` when the entropy-coded bit count
/// `c` exceeds `UINT32_MAX`, and `internal_piz.c:596-603`
/// (<https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/internal_piz.c#L596-L603>)
/// catches that by storing the chunk raw instead. The old Rust code
/// unconditionally wrote `n_bits as u32` into the header, silently wrapping a
/// huge bit count into a corrupt, unreadable chunk.
pub(crate) fn compress_with_limit(
    raw: &[u16],
    scratch: &mut HufEncScratch,
    max_n_bits: u64,
    out: &mut Vec<u8>,
) -> Result<bool> {
    if raw.is_empty() {
        out.clear();
        return Ok(true);
    }

    count_frequencies(raw, &mut scratch.freq);
    let (im, im_max) = build_encoding_table(
        &mut scratch.freq,
        &mut scratch.hlink,
        &mut scratch.scode,
        &mut scratch.heap_indices,
        &mut scratch.heap,
    )?;

    // Pack the code table, into the reused `table_buf`.
    let mut table_writer = BitWriter::with_buf(std::mem::take(&mut scratch.table_buf));
    pack_encoding_table(&scratch.freq, im, im_max, &mut table_writer);
    let (table_bytes, _) = table_writer.finish();
    let table_length = table_bytes.len();

    // Entropy-encode the data (the run-length code is symbol iM), into the
    // reused `data_buf`.
    let (data_bytes, n_bits) = encode_data(&scratch.freq, raw, im_max, std::mem::take(&mut scratch.data_buf));

    // `hufEncode`'s own out-of-range check, on the bit count only (the
    // table length is bounded by the caller's allocation and cannot
    // realistically overflow a u32).
    if n_bits > max_n_bits {
        // Give the buffers back to `scratch` for the NEXT call even on this
        // early return -- they're still perfectly good, just abandoned bytes.
        scratch.table_buf = table_bytes;
        scratch.data_buf = data_bytes;
        return Ok(false);
    }

    // Assemble: 20-byte LE header + packed table + data, directly into the
    // caller's `out` (no intermediate `Vec` this function returns and
    // the caller immediately copies elsewhere).
    out.clear();
    out.reserve(20 + table_bytes.len() + data_bytes.len());
    out.extend_from_slice(&(im as u32).to_le_bytes());
    out.extend_from_slice(&(im_max as u32).to_le_bytes());
    out.extend_from_slice(&(table_length as u32).to_le_bytes());
    out.extend_from_slice(&(n_bits as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    out.extend_from_slice(&table_bytes);
    out.extend_from_slice(&data_bytes);

    // Give the buffers back to `scratch` for the next call to reuse.
    scratch.table_buf = table_bytes;
    scratch.data_buf = data_bytes;

    Ok(true)
}

/// The fixed 20-byte header's fields, parsed ONCE (the old
/// `decompress`/`decompress_dispatch` pair each parsed and validated this
/// same header independently, `internal_huf.c`'s own `hufUncompress` does it
/// once). `body` is `compressed[20..]`, borrowed from the caller's slice.
struct ParsedHeader<'a> {
    im: usize,
    im_max: usize,
    n_bits: u64,
    body: &'a [u8],
}

/// Parse and validate the 20-byte header — port of `hufUncompress`'s own
/// header read (`internal_huf.c:1934-1949`). `Ok(None)` means "compressed is
/// the C++ `< 20 bytes, nRaw == 0` empty-input case" — the caller returns
/// `Ok(())` with an already-cleared `out`.
fn parse_header(compressed: &[u8], expected: usize) -> Result<Option<ParsedHeader<'_>>> {
    // C++: < 20 bytes is only valid when nRaw == 0.
    if compressed.len() < 20 {
        if expected != 0 {
            return Err(ExrError::Input(
                "PIZ huffman: compressed data shorter than the 20-byte header".into(),
            ));
        }
        return Ok(None);
    }

    let im = read_u32_le(compressed, 0)? as usize;
    let im_max = read_u32_le(compressed, 4)? as usize;
    // table_length at [8..12) is not needed (the table is self-delimiting).
    let n_bits = read_u32_le(compressed, 12)? as u64;

    if im >= ENCODING_TABLE_SIZE || im_max >= ENCODING_TABLE_SIZE {
        return Err(ExrError::Input(
            "PIZ huffman: invalid code table size (im/iM out of range)".into(),
        ));
    }
    if im > im_max {
        return Err(ExrError::Input(
            "PIZ huffman: invalid code table size (im > iM)".into(),
        ));
    }

    Ok(Some(ParsedHeader { im, im_max, n_bits, body: &compressed[20..] }))
}

/// The generic scalar decoder — 1:1 port of `hufDecode`'s calling context in
/// `hufUncompress` (`internal_huf.c:1976-1990`'s `else` branch: unpack the
/// packed code-length table, build the decode table, then `hufDecode`). Takes
/// an ALREADY-PARSED header (this used to re-parse the same
/// 20-byte header [`decompress`] itself, now the public entry point,
/// had already parsed).
fn decode_generic(
    header: &ParsedHeader<'_>,
    expected: usize,
    out: &mut Vec<u16>,
    scratch: &mut GenScratch,
) -> Result<()> {
    let ParsedHeader { im, im_max, n_bits, body } = *header;
    let n_bits = n_bits as i64;

    let n_bytes = (n_bits as u64).div_ceil(8) as usize;
    if n_bytes > body.len() {
        return Err(ExrError::Input(
            "PIZ huffman: not enough data for the declared bit count".into(),
        ));
    }

    // Unpack the code table, then build the decode table (both into `scratch`).
    if scratch.enc.capacity() >= ENCODING_TABLE_SIZE && scratch.dec.capacity() >= DECODING_TABLE_SIZE {
        crate::probe::hit(crate::probe::Probe::PizHufGenericReuse);
    }
    let mut table_reader = BitReader::new(body);
    unpack_encoding_table(&mut table_reader, im, im_max, &mut scratch.enc)?;

    // The entropy stream starts right after the consumed table bytes.
    let consumed = table_reader.pos;
    let data = body
        .get(consumed..)
        .ok_or_else(|| ExrError::Input("PIZ huffman: table overran the buffer".into()))?;

    if n_bits < 0 || n_bits as u64 > 8 * data.len() as u64 {
        return Err(ExrError::Input(
            "PIZ huffman: invalid number of bits".into(),
        ));
    }

    build_decoding_table(&scratch.enc, im, im_max, &mut scratch.dec)?;

    decode_data(
        &scratch.enc,
        &scratch.dec,
        data,
        n_bits,
        im_max as u32,
        expected,
        out,
    )
}

/// Huffman-decompress `compressed` into exactly `expected` `u16` values — the
/// ONE public entry point, 1:1 port of `hufUncompress` itself
/// (`internal_huf.c:1953-1989`; this used to be split across a
/// public `decompress` — actually just the generic scalar path — and a
/// separate public `decompress_dispatch` that duplicated this same header
/// parse before delegating). `nBits > 128` always takes the accelerated
/// [`super::fast_huf`] decoder on every platform `fasthuf_decode_enabled()`
/// recognises (we target that — "as C on x64 or better" everywhere), and any
/// error it returns is final — there is **no** fallback to the generic
/// scalar path ([`decode_generic`]) in that case. `nBits <= 128` uses
/// [`decode_generic`]. This dispatch — not the generic path alone — is what
/// real PIZ/DWA chunks decode through in practice (their entropy streams are
/// essentially always `nBits > 128`), so it dominates measured
/// PIZ/DWA decode time.
///
/// `out` is cleared exactly once, here, on entry — every function this calls
/// into ([`decode_generic`]/[`decode_data`]/`fast_huf::decode`) assumes `out`
/// already starts empty rather than clearing it again itself.
///
/// # Errors
/// Never panics; returns [`ExrError::Input`] for any truncation, oversized
/// table indices, invalid codes, or a length mismatch — matching FastHuf's
/// own error conditions when `nBits > 128` (this closes a previously
/// documented FastHuf/slow-path accept-reject gap: both paths now share one
/// real dispatch, not an approximation of one).
///
/// `fhd` is the caller's persistent [`fast_huf::FastHufDecoder`]
/// — reused across calls instead of a fresh decoder (with its ~20 KiB
/// acceleration table plus a variable-size `id_to_symbol`) being built and
/// returned every time.
pub(crate) fn decompress(
    compressed: &[u8],
    expected: usize,
    out: &mut Vec<u16>,
    fhd: &mut fast_huf::FastHufDecoder,
) -> Result<()> {
    out.clear();
    let header = match parse_header(compressed, expected)? {
        Some(h) => h,
        None => return Ok(()),
    };

    if header.n_bits <= 128 {
        return decode_generic(&header, expected, out, &mut fhd.generic);
    }

    let n_bytes = header.n_bits.div_ceil(8) as usize;
    if n_bytes > header.body.len() {
        return Err(ExrError::Input(
            "PIZ huffman: not enough data for the declared bit count".into(),
        ));
    }

    let consumed = fast_huf::initialize(header.body, header.im as u32, header.im_max as u32, fhd)?;
    let data = header
        .body
        .get(consumed..)
        .ok_or_else(|| ExrError::Input("PIZ fast-huf: table overran the buffer".into()))?;
    if n_bytes > data.len() {
        return Err(ExrError::Input(
            "PIZ fast-huf: not enough data for the declared bit count".into(),
        ));
    }

    fast_huf::decode(fhd, data, header.n_bits, expected, out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only wrapper restoring the earlier `Result<Option<Vec<u8>>>`
    /// shape for the many existing call sites in this module -- NOT a second
    /// production API, exactly like `dwa`'s own `compress_dwa`/
    /// `decompress_dwa` test wrappers. Real callers (`PizCompressor`/
    /// `DwaCompressor`) always use the persistent `HufEncScratch`/`out`
    /// fields directly via `super::compress_with_limit`.
    fn compress(raw: &[u16]) -> Result<Option<Vec<u8>>> {
        compress_with_limit(raw, u32::MAX as u64)
    }

    fn compress_with_limit(raw: &[u16], max_n_bits: u64) -> Result<Option<Vec<u8>>> {
        let mut scratch = HufEncScratch::default();
        let mut out = Vec::new();
        let ok = super::compress_with_limit(raw, &mut scratch, max_n_bits, &mut out)?;
        Ok(if ok { Some(out) } else { None })
    }

    /// Test-only wrapper exercising the GENERIC scalar decoder directly
    /// (`decode_generic`, now private since the header
    /// parse was folded into the one public `decompress` dispatcher) — for tests that
    /// want to verify the generic path's own behaviour regardless of which
    /// path a real `decompress()` call would take for the same bytes (most
    /// of this file's fixtures have `n_bits > 128` and would otherwise always
    /// hit FastHuf). Named for what it actually calls (`decode_generic`), NOT
    /// `decompress` — the earlier name invited
    /// exactly the confusion this rename fixes: the wrapper that called
    /// `decode_generic` directly was named `decompress`, while the wrapper
    /// for the real, dispatching `decompress` was named `decompress_dispatch`.
    /// NOT a second production API — production code only ever calls the
    /// real `decompress`.
    fn decode_generic_t(compressed: &[u8], expected: usize) -> Result<Vec<u16>> {
        let mut out = Vec::new();
        let header = match parse_header(compressed, expected)? {
            Some(h) => h,
            None => return Ok(out),
        };
        decode_generic(&header, expected, &mut out, &mut GenScratch::default())?;
        Ok(out)
    }
    /// Test-only convenience wrapper around the grow-only, caller-owns-the-
    /// buffer real `decompress` dispatcher (matching `PizCompressor`'s
    /// persistent `tmp` field / C's reused `decode->scratch_buffer_1`) — NOT
    /// a second production API, just saving every one-off test call site
    /// from spelling out its own throwaway `Vec::new()`. Named to match what
    /// it actually calls (`super::decompress`, the real dispatcher) — see
    /// `decode_generic_t`'s doc for why the old names (`decompress` for this
    /// generic-only wrapper, `decompress_dispatch` for this one) were
    /// swapped from what they should have been.
    fn decompress(compressed: &[u8], expected: usize) -> Result<Vec<u16>> {
        let mut out = Vec::new();
        let mut fhd = fast_huf::FastHufDecoder::default();
        super::decompress(compressed, expected, &mut out, &mut fhd)?;
        Ok(out)
    }

    /// Reference vector captured from genuine C++ OpenEXR `hufCompress` of
    /// `UNCOMPRESSED_ARRAY` — proves our encoder is BYTE-IDENTICAL to C++.
    const UNCOMPRESSED_ARRAY: [u16; 100] = [
        3852, 2432, 33635, 49381, 10100, 15095, 62693, 63738, 62359, 5013, 7715, 59875, 28182,
        34449, 19983, 20399, 63407, 29486, 4877, 26738, 44815, 14042, 46091, 48228, 25682, 35412,
        7582, 65069, 6632, 54124, 13798, 27503, 52154, 61961, 30474, 46880, 39097, 15754, 52897,
        42371, 54053, 14178, 48276, 34591, 42602, 32126, 42062, 31474, 16274, 55991, 2882, 17039,
        56389, 20835, 57057, 54081, 3414, 33957, 52584, 10222, 25139, 40002, 44980, 1602, 48021,
        19703, 6562, 61777, 41582, 201, 31253, 51790, 15888, 40921, 3627, 12184, 16036, 26349,
        3159, 29002, 14535, 50632, 18118, 33583, 18878, 59470, 32835, 9347, 16991, 21303, 26263,
        8312, 14017, 41777, 43240, 3500, 60250, 52437, 45715, 61520,
    ];

    const COMPRESSED_ARRAY: [u8; 703] = [
        0xc9, 0x0, 0x0, 0x0, 0x2e, 0xfe, 0x0, 0x0, 0x56, 0x2, 0x0, 0x0, 0xa2, 0x2, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xd6, 0x47,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x28, 0x1f, 0xff, 0xff, 0xed, 0x87, 0xff, 0xff, 0xf0,
        0x91, 0xff, 0xf8, 0x1f, 0xf4, 0xf1, 0xff, 0x78, 0x1f, 0xfd, 0xa1, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xfa, 0xc7, 0xfe, 0x4, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xed, 0x1f, 0xf3, 0xf1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe8, 0x7, 0xfd, 0xf8,
        0x7f, 0xff, 0xff, 0xff, 0xfd, 0x10, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x51, 0xff,
        0xff, 0xff, 0xff, 0xfe, 0x1, 0xff, 0x73, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x0, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xfc, 0xa4, 0x7f, 0xf5, 0x7, 0xfc, 0x48, 0x7f, 0xe0, 0x47, 0xff, 0xff,
        0xf5, 0x91, 0xff, 0xff, 0xff, 0xff, 0xf1, 0xf1, 0xff, 0xff, 0xff, 0xff, 0xf8, 0x21, 0xff,
        0x7f, 0x1f, 0xf8, 0xd1, 0xff, 0xe7, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xbc, 0x1f, 0xf2, 0x91,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x1c, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xe7,
        0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfc, 0x8c, 0x7f, 0xff, 0xff, 0xc, 0x1f, 0xff, 0xff,
        0xe5, 0x7, 0xff, 0xff, 0xfa, 0x81, 0xff, 0xff, 0xff, 0x20, 0x7f, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xbc, 0x7f, 0xff, 0xff, 0xff, 0xfc, 0x38, 0x7f, 0xff,
        0xff, 0xff, 0xfc, 0xd0, 0x7f, 0xd3, 0xc7, 0xff, 0xff, 0xf7, 0x91, 0xff, 0xff, 0xff, 0xff,
        0xfe, 0xc1, 0xff, 0xff, 0xff, 0xff, 0xf9, 0x61, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc7,
        0x87, 0xff, 0xff, 0xfd, 0x81, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf1, 0x87, 0xff, 0xff,
        0xff, 0xff, 0xfe, 0x87, 0xff, 0x58, 0x7f, 0xff, 0xff, 0xff, 0xfd, 0xec, 0x7f, 0xff, 0xff,
        0xff, 0xfe, 0xd0, 0x7f, 0xff, 0xff, 0xff, 0xff, 0x6c, 0x7f, 0xcb, 0x47, 0xff, 0xff, 0xf3,
        0x61, 0xff, 0xff, 0xff, 0x80, 0x7f, 0xe1, 0xc7, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x1f,
        0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x18, 0x1f, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xfd, 0xcc, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf8, 0x11, 0xff, 0xff,
        0xff, 0xff, 0xf8, 0x41, 0xff, 0xbc, 0x1f, 0xff, 0xff, 0xc4, 0x47, 0xff, 0xff, 0xf2, 0x91,
        0xff, 0xe0, 0x1f, 0xff, 0xff, 0xff, 0xff, 0x6d, 0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0x2, 0x1f, 0xf9, 0xe1, 0xff, 0xff, 0xff, 0xff, 0xfc, 0xe1,
        0xff, 0xff, 0xfd, 0xb0, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe1, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0x5a, 0x1f, 0xfc, 0x81, 0xbf, 0x29, 0x1b, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xf3, 0x61, 0xbf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc8, 0x1b,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf6, 0xb1, 0xbf, 0xff, 0xfd, 0x80, 0x6f, 0xff,
        0xff, 0xf, 0x1b, 0xf8, 0xc1, 0xbf, 0xff, 0xfc, 0xb4, 0x6f, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xda, 0x46, 0xfc, 0x54, 0x6f, 0xc9, 0x6, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x21, 0x1b, 0xff, 0xff, 0xe0, 0x86, 0xff, 0xff,
        0xff, 0xff, 0xe2, 0xc6, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xf3, 0x91, 0xbf, 0xff, 0xfe, 0x24, 0x6f, 0xff, 0xff, 0x6b,
        0x1b, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd, 0xb1, 0xbf, 0xfa, 0x1b, 0xfb, 0x11,
        0xbf, 0xff, 0xfe, 0x8, 0x6f, 0xff, 0xff, 0x42, 0x1b, 0xff, 0xff, 0xff, 0xff, 0xb9, 0x1b,
        0xff, 0xff, 0xcf, 0xc6, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xf1, 0x31,
        0x86, 0x10, 0x9, 0xb4, 0xe4, 0x4c, 0xf7, 0xef, 0x42, 0x87, 0x6a, 0xb5, 0xc2, 0x34, 0x9e,
        0x2f, 0x12, 0xae, 0x21, 0x68, 0xf2, 0xa8, 0x74, 0x37, 0xe1, 0x98, 0x14, 0x59, 0x57, 0x2c,
        0x24, 0x3b, 0x35, 0x6c, 0x1b, 0x8b, 0xcc, 0xe6, 0x13, 0x38, 0xc, 0x8e, 0xe2, 0xc, 0xfe,
        0x49, 0x73, 0xbc, 0x2b, 0x7b, 0x9, 0x27, 0x79, 0x14, 0xc, 0x94, 0x42, 0xf8, 0x7c, 0x1,
        0x8d, 0x26, 0xde, 0x87, 0x26, 0x71, 0x50, 0x45, 0xc6, 0x28, 0x40, 0xd5, 0xe, 0x8d, 0x8,
        0x1e, 0x4c, 0xa4, 0x79, 0x57, 0xf0, 0xc3, 0x6d, 0x5c, 0x6d, 0xc0,
    ];

    /// Byte-exact parity with C++ OpenEXR `hufCompress`.
    #[test]
    fn golden_vector() {
        let out = compress(&UNCOMPRESSED_ARRAY).unwrap().unwrap();
        assert_eq!(out, COMPRESSED_ARRAY.to_vec(), "huffman bytes diverge from C++");
    }

    #[test]
    fn round_trip_known() {
        let out = compress(&UNCOMPRESSED_ARRAY).unwrap().unwrap();
        let back = decode_generic_t(&out, UNCOMPRESSED_ARRAY.len()).unwrap();
        assert_eq!(back, UNCOMPRESSED_ARRAY.to_vec());
    }

    #[test]
    fn round_trip_noise_and_runs() {
        let mut state: u32 = 0xC0FF_EE11;
        // Mixed: high-entropy region + long runs of 0 and 0xffff.
        let mut data: Vec<u16> = (0..50_000)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u16
            })
            .collect();
        data.extend(std::iter::repeat_n(0u16, 5000));
        data.extend(std::iter::repeat_n(0xffffu16, 5000));
        data.extend(std::iter::repeat_n(28182u16, 300));

        let out = compress(&data).unwrap().unwrap();
        let back = decode_generic_t(&out, data.len()).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn round_trip_all_zero() {
        let data = vec![0u16; 1234];
        let out = compress(&data).unwrap().unwrap();
        let back = decode_generic_t(&out, data.len()).unwrap();
        assert_eq!(back, data);
    }

    /// `HufEncScratch`'s ~2 MiB `freq`/`hlink`/`scode` tables
    /// (plus the two `BitWriter` buffers) are now reused across calls instead
    /// of being fresh `vec![...]`s every time. Compress a SEQUENCE of chunks
    /// with genuinely DIFFERENT histograms -- a near-max-code-length
    /// Fibonacci distribution, an RLE-heavy run-length-friendly stream, an
    /// all-zero stream, and the real golden-vector fixture -- on ONE
    /// persistent scratch, and diff each result byte-for-byte against a
    /// FRESH scratch compressing the exact same input in isolation. Any
    /// stale byte surviving from a PRIOR chunk's tables (`freq`/`scode` in
    /// particular -- see their own docs for exactly why a stale entry would
    /// corrupt the NEXT chunk's table) would show up here as a divergence.
    #[test]
    fn reused_scratch_across_different_histograms_matches_fresh_scratch() {
        // Near-max code length: true Fibonacci frequencies (same underlying
        // skew as `build_encoding_table_reaches_near_maximum_code_length`,
        // fewer terms so the actual DATA vector stays a reasonable size --
        // the skew, not the absolute counts, is what drives long codes).
        let fib_data: Vec<u16> = {
            let mut freq = [0u64; 20];
            let (mut a, mut b) = (1u64, 1u64);
            for slot in freq.iter_mut() {
                *slot = a;
                let next = a + b;
                a = b;
                b = next;
            }
            let mut data = Vec::new();
            for (sym, &count) in freq.iter().enumerate() {
                data.extend(std::iter::repeat_n(sym as u16, count as usize));
            }
            data
        };

        // RLE-heavy: long runs interleaved with a little high-entropy noise.
        let rle_data: Vec<u16> = {
            let mut state: u32 = 0x1234_5678;
            let mut data = Vec::new();
            data.extend(std::iter::repeat_n(0u16, 2000));
            data.extend((0..500).map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u16
            }));
            data.extend(std::iter::repeat_n(0xffffu16, 2000));
            data
        };

        let all_zero: Vec<u16> = vec![0u16; 500];

        let sequence: [&[u16]; 4] =
            [&fib_data, &rle_data, &all_zero, &UNCOMPRESSED_ARRAY];

        let mut shared_scratch = HufEncScratch::default();
        for &data in &sequence {
            let mut shared_out = Vec::new();
            let shared_ok =
                super::compress_with_limit(data, &mut shared_scratch, u32::MAX as u64, &mut shared_out)
                    .unwrap();

            let mut fresh_scratch = HufEncScratch::default();
            let mut fresh_out = Vec::new();
            let fresh_ok =
                super::compress_with_limit(data, &mut fresh_scratch, u32::MAX as u64, &mut fresh_out)
                    .unwrap();

            assert_eq!(shared_ok, fresh_ok);
            assert_eq!(
                shared_out, fresh_out,
                "a reused HufEncScratch produced different compressed bytes than a fresh \
                 one for the same input -- some field is not fully reset between calls"
            );

            // Round trip through the shared-scratch result too, for good measure.
            let back = decode_generic_t(&shared_out, data.len()).unwrap();
            assert_eq!(back, data);
        }
    }

    /// `fast_huf::FastHufDecoder` is now reused across calls
    /// via `initialize`/`decompress` (its ~20 KiB acceleration table + a
    /// variable-size `id_to_symbol`) instead of being fresh every time.
    /// POISONS the persistent decoder (`poison_for_test`) BEFORE the first
    /// real decode, then decompresses the SAME sequence of different
    /// histograms as `reused_scratch_across_different_histograms_matches_fresh_scratch`
    /// through the real `super::decompress` dispatch (so this exercises
    /// whichever path -- `decode_generic` or `fast_huf` -- each chunk's real
    /// `n_bits` actually takes), diffing against a completely fresh decoder
    /// each time. `fib_data`/`UNCOMPRESSED_ARRAY` are large enough to clear
    /// the `n_bits > 128` FastHuf threshold, so this genuinely exercises
    /// `FastHufDecoder` reuse, not just `decode_generic`.
    #[test]
    fn reused_fast_huf_decoder_across_different_histograms_matches_fresh_decoder() {
        let fib_data: Vec<u16> = {
            let mut freq = [0u64; 20];
            let (mut a, mut b) = (1u64, 1u64);
            for slot in freq.iter_mut() {
                *slot = a;
                let next = a + b;
                a = b;
                b = next;
            }
            let mut data = Vec::new();
            for (sym, &count) in freq.iter().enumerate() {
                data.extend(std::iter::repeat_n(sym as u16, count as usize));
            }
            data
        };
        let rle_data: Vec<u16> = {
            let mut state: u32 = 0x1234_5678;
            let mut data = Vec::new();
            data.extend(std::iter::repeat_n(0u16, 2000));
            data.extend((0..500).map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u16
            }));
            data.extend(std::iter::repeat_n(0xffffu16, 2000));
            data
        };
        let all_zero: Vec<u16> = vec![0u16; 500];
        let sequence: [&[u16]; 4] = [&fib_data, &rle_data, &all_zero, &UNCOMPRESSED_ARRAY];

        let mut shared_scratch = HufEncScratch::default();
        let mut shared_fhd = fast_huf::FastHufDecoder::default();
        shared_fhd.poison_for_test();

        let mut hit_fast_huf = false;
        for &data in &sequence {
            let mut compressed = Vec::new();
            super::compress_with_limit(data, &mut shared_scratch, u32::MAX as u64, &mut compressed).unwrap();

            // Real n_bits, straight from the compressed header (bytes
            // [12..16), LE u32) -- used only to track whether this sample
            // actually exercised FastHuf, not to change behavior.
            if compressed.len() >= 16 {
                let n_bits = u32::from_le_bytes([compressed[12], compressed[13], compressed[14], compressed[15]]);
                if n_bits > 128 {
                    hit_fast_huf = true;
                }
            }

            let mut shared_out = Vec::new();
            super::decompress(&compressed, data.len(), &mut shared_out, &mut shared_fhd).unwrap();

            let mut fresh_fhd = fast_huf::FastHufDecoder::default();
            let mut fresh_out = Vec::new();
            super::decompress(&compressed, data.len(), &mut fresh_out, &mut fresh_fhd).unwrap();

            assert_eq!(
                shared_out, fresh_out,
                "a poisoned, reused FastHufDecoder produced different decoded values than a \
                 fresh one for the same input -- some field is not fully overwritten on reuse"
            );
            assert_eq!(shared_out, data);
        }
        assert!(hit_fast_huf, "test setup bug: no sample in this sequence exercised the FastHuf path (n_bits > 128)");

        // `table_symbol`/`table_code_len`'s sentinel reset is only ever
        // OBSERVABLE for an INCOMPLETE canonical code (a complete code's
        // `table_min` gating means the fast path never reads the sentinel --
        // see `build_tables`'s own doc). None of the real, complete-code
        // chunks above exercise that path, so finish by reusing THIS SAME
        // `shared_fhd` (already loaded with a REAL table from the loop
        // above, not just `poison_for_test`'s synthetic garbage) for the
        // identical incomplete-code construction `build_tables_overrun`
        // uses, and confirm it still correctly reports "Overrun" -- proving
        // `reset()` re-arms the sentinel state after genuine prior use, not
        // only after synthetic poisoning.
        // `pack_bits(&[(6, 1)])` (see `fast_huf_error_sites`'s helper, not
        // reachable from this outer `mod tests`) -- one 6-bit code-length
        // field of value 1, left-justified into a single byte: 1 << 2 = 4.
        let incomplete_body = vec![0x04u8];
        let shared_err = fast_huf::initialize(&incomplete_body, 10, 10, &mut shared_fhd).unwrap_err();
        assert!(shared_err.to_string().contains("Overrun"), "got: {shared_err}");
        let mut fresh_fhd2 = fast_huf::FastHufDecoder::default();
        let fresh_err = fast_huf::initialize(&incomplete_body, 10, 10, &mut fresh_fhd2).unwrap_err();
        assert!(fresh_err.to_string().contains("Overrun"), "got: {fresh_err}");
    }

    /// Corrupt input must error, never panic.
    #[test]
    fn corrupt_input_errors_not_panics() {
        // Valid stream, then mangle it various ways.
        let good = compress(&UNCOMPRESSED_ARRAY).unwrap().unwrap();

        // Truncated header.
        let e1 = decode_generic_t(&good[..10], UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(e1.to_string().contains("shorter than the 20-byte header"), "got: {e1}");

        // Garbage body of plausible length -- deterministic for this
        // fixture: the corrupted table hits end-of-input before completing.
        let mut garbage = good.clone();
        for b in garbage.iter_mut().skip(20) {
            *b = 0xAB;
        }
        let e2 = decode_generic_t(&garbage, UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(e2.to_string().contains("unexpected end of input"), "got: {e2}");

        // Bogus im/iM.
        let mut bad_hdr = good.clone();
        bad_hdr[0..4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let e3 = decode_generic_t(&bad_hdr, UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(e3.to_string().contains("im/iM out of range"), "got: {e3}");

        // Wrong expected length (too small) → error.
        let e4 = decode_generic_t(&good, 1).unwrap_err();
        assert!(e4.to_string().contains("decoded data longer than expected"), "got: {e4}");

        // All-bytes-0xFF of arbitrary length -- the all-0xFF im/iM header
        // fields alone are already out of range.
        let e5 = decode_generic_t(&[0xFFu8; 64], 100).unwrap_err();
        assert!(e5.to_string().contains("im/iM out of range"), "got: {e5}");
    }

    #[test]
    fn empty_input_empty_output() {
        assert!(compress(&[]).unwrap().unwrap().is_empty());
        assert!(decode_generic_t(&[], 0).unwrap().is_empty());
    }

    /// When the entropy-coded bit count exceeds the injected ceiling,
    /// `compress_with_limit` must return `Ok(None)` instead of writing a
    /// wrapped `n_bits` into the header (`hufEncode`'s
    /// `c > UINT32_MAX` check, see `compress_with_limit`'s doc comment).
    /// A real 512 MiB fixture is impractical, so this uses a tiny injected
    /// limit that `UNCOMPRESSED_ARRAY`'s real bit count exceeds.
    #[test]
    fn compress_with_limit_returns_none_over_ceiling() {
        // A real compress of this fixture needs far more than 8 bits.
        let out = compress_with_limit(&UNCOMPRESSED_ARRAY, 8).unwrap();
        assert!(out.is_none(), "over-ceiling bit count must yield None, not a wrapped header");
    }

    /// Below the ceiling, `compress_with_limit` behaves exactly like `compress`.
    #[test]
    fn compress_with_limit_below_ceiling_matches_compress() {
        let via_limit = compress_with_limit(&UNCOMPRESSED_ARRAY, u32::MAX as u64)
            .unwrap()
            .unwrap();
        let via_compress = compress(&UNCOMPRESSED_ARRAY).unwrap().unwrap();
        assert_eq!(via_limit, via_compress);
    }

    /// Mutation-testing gap fix: the boundary itself, `n_bits > max_n_bits`
    /// (C++ `hufEncode`'s `c > UINT32_MAX` check cited in this function's own
    /// doc comment) - AT the ceiling exactly must still succeed (`Ok(Some)`),
    /// only ABOVE it must fall back to `Ok(None)`. `n_bits` is extracted from
    /// a real `compress()` output's own header (bytes 12..16, the `n_bits`
    /// field per this function's header-assembly code just above), not
    /// recomputed by calling `compress_with_limit` itself, so this is an
    /// independently-obtained expected value for the boundary test.
    #[test]
    fn compress_with_limit_at_exact_ceiling_still_succeeds() {
        let good = compress(&UNCOMPRESSED_ARRAY).unwrap().unwrap();
        let n_bits = u32::from_le_bytes([good[12], good[13], good[14], good[15]]) as u64;
        assert!(n_bits > 0, "test setup: fixture must produce a real nonzero bit count");
        let at_ceiling = compress_with_limit(&UNCOMPRESSED_ARRAY, n_bits).unwrap();
        assert!(at_ceiling.is_some(), "max_n_bits == n_bits exactly must still succeed, not fall back to None");
    }

    // -----------------------------------------------------------------------
    // FastHuf differential tests — slow (`decode_generic_t`) vs accelerated
    // (`decompress`, the real dispatcher — `hufUncompress`'s real
    // `nBits > 128` path) must agree on every valid stream, and
    // `decompress`'s rejection of a corrupt stream must match FastHuf's own
    // explicit error conditions in `internal_huf.c`, not merely "some Err".
    // -----------------------------------------------------------------------

    /// Extract the header's `n_bits` field the same way `decompress` does.
    fn header_n_bits(compressed: &[u8]) -> u64 {
        u32::from_le_bytes([compressed[12], compressed[13], compressed[14], compressed[15]]) as u64
    }

    /// The longest code length our own encoder actually assigned, read back
    /// from the compressed stream's own packed code-length table (not
    /// re-derived independently) — used to ASSERT a test fixture really
    /// reaches the code-length range it claims to target, instead of just
    /// asserting round-trip correctness.
    fn max_code_length(compressed: &[u8]) -> u64 {
        let header = parse_header(compressed, usize::MAX)
            .unwrap()
            .expect("test fixture must produce a real (non-empty) compressed stream");
        let mut reader = BitReader::new(header.body);
        let mut table = Vec::new();
        unpack_encoding_table(&mut reader, header.im, header.im_max, &mut table)
            .expect("test fixture's own compressed table must unpack");
        table[header.im..=header.im_max]
            .iter()
            .map(|&entry| entry & 0x3f)
            .max()
            .unwrap_or(0)
    }

    /// Assert `decode_generic_t` (slow) and `decompress` (real dispatcher,
    /// the C++ path) produce byte-identical output for `data`, and record
    /// whether the fixture actually exercised the FastHuf branch
    /// (`n_bits > 128`) so tests can assert they reached the code path they
    /// intend to.
    fn assert_fast_matches_slow(data: &[u16]) -> bool {
        let compressed = compress(data).unwrap().unwrap();
        let via_slow = decode_generic_t(&compressed, data.len()).unwrap();
        let via_dispatch = decompress(&compressed, data.len()).unwrap();
        assert_eq!(via_slow, data, "reference slow decoder diverged from input");
        assert_eq!(via_dispatch, data, "the real dispatcher diverged from input");
        header_n_bits(&compressed) > 128
    }

    #[test]
    fn fast_huf_matches_slow_on_existing_fixtures() {
        // Reuse every existing round-trip fixture; all comfortably exceed the
        // 128-bit dispatch threshold given their size, so this alone already
        // covers the "real chunk" case end to end.
        assert!(assert_fast_matches_slow(&UNCOMPRESSED_ARRAY));

        let mut state: u32 = 0xC0FF_EE11;
        let mut noisy: Vec<u16> = (0..50_000)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u16
            })
            .collect();
        noisy.extend(std::iter::repeat_n(0u16, 5000));
        noisy.extend(std::iter::repeat_n(0xffffu16, 5000));
        noisy.extend(std::iter::repeat_n(28182u16, 300));
        assert!(assert_fast_matches_slow(&noisy));

        // All-zero data collapses to a single RLE code and stays under the
        // 128-bit dispatch threshold (correctly takes the slow path, not
        // FastHuf) — checked for correctness only, not for which path fired.
        assert_fast_matches_slow(&vec![0u16; 1234]);
    }

    /// Short-codes-only: a small, heavily-skewed alphabet produces every code
    /// length `<= TABLE_LOOKUP_BITS` (12) and stays inside FastHuf's direct
    /// lookup table exclusively (no brute-force fallback).
    #[test]
    fn fast_huf_short_codes_only() {
        // A handful of symbols with a steep frequency skew: canonical Huffman
        // gives the rarest symbol the longest code, but with so few symbols
        // no code can exceed a few bits — well under TABLE_LOOKUP_BITS.
        let mut data = Vec::new();
        for (sym, count) in [(1u16, 4000), (2, 2000), (3, 1000), (4, 500), (5, 1)] {
            data.extend(std::iter::repeat_n(sym, count));
        }
        assert!(assert_fast_matches_slow(&data));
    }

    /// Forces codes longer than `TABLE_LOOKUP_BITS` (12 bits), exercising
    /// FastHuf's brute-force `_ljBase`/`_ljOffset` fallback search.
    #[test]
    fn fast_huf_brute_force_long_codes() {
        // A Zipf-ish distribution over many distinct symbols: canonical
        // Huffman on ~2000 symbols with power-law frequencies reliably
        // produces some codes well past 12 bits for the rarest symbols.
        let mut data = Vec::new();
        for sym in 0u16..2000 {
            let count = (2_000_000 / (sym as u64 + 2)).max(1) as usize;
            data.extend(std::iter::repeat_n(sym, count.min(5000)));
        }
        assert!(assert_fast_matches_slow(&data));
        // The whole point of this fixture is to force FastHuf's brute-force
        // `_ljBase`/`_ljOffset` fallback (codes past TABLE_LOOKUP_BITS = 12) —
        // assert that actually happened instead of just checking round-trip
        // correctness, which a short-codes-only stream would also pass.
        let max_len = max_code_length(&compress(&data).unwrap().unwrap());
        assert!(
            max_len > 12,
            "fixture must actually exercise FastHuf's brute-force fallback \
             (a code longer than TABLE_LOOKUP_BITS = 12), got max length {max_len}"
        );
    }

    /// Run-length-heavy data, including a run whose 8-bit count read crosses
    /// a `refill` boundary (the RLE count itself is read via the same 64-bit
    /// buffer as ordinary codes, so a run right at a buffer-exhaustion point
    /// exercises `refill`'s "in the middle of the stream" path).
    #[test]
    fn fast_huf_rle_heavy_runs() {
        let mut data = Vec::new();
        for i in 0..300u16 {
            // Alternate long runs (>= SHORTEST_LONG_RUN) with short bursts so
            // both the encoder's short-run and long-run `sendCode` branches,
            // and correspondingly both of FastHuf's RLE-count reads, fire
            // repeatedly across the whole stream.
            data.extend(std::iter::repeat_n(i % 37, 200));
            data.push(i);
        }
        assert!(assert_fast_matches_slow(&data));
    }

    /// A single symbol repeated enough times to certainly produce `MAX_CODE_LEN`
    /// (58) codes is not achievable with real Huffman (more symbols always
    /// shorten the worst code), so instead this drives a long, adversarial
    /// Fibonacci-like frequency distribution, the classic way to force
    /// near-maximum-length canonical Huffman codes.
    ///
    /// The old version of this test capped each symbol's count at 50 "so the
    /// test stays fast" — but that cap kicks in once the real Fibonacci
    /// growth passes 50 (around the 9th term), flattening every later
    /// symbol's frequency to the SAME value and breaking the exact
    /// sum-of-the-two-smallest cascade the worst case depends on. The result
    /// was a shallow ~12-bit tree that never actually left FastHuf's direct
    /// lookup table, silently defeating the test's own stated purpose. Real,
    /// uncapped counts are needed to reach a genuinely long code — feasible
    /// here because 28 terms keeps the total sample count under a million.
    #[test]
    fn fast_huf_near_maximum_code_length() {
        // Fibonacci-weighted symbol counts, UNCAPPED: canonical Huffman on
        // Fibonacci frequencies is the textbook worst case for code length.
        let mut counts = vec![1u64, 1u64];
        while counts.len() < 28 {
            let n = counts.len();
            counts.push(counts[n - 1] + counts[n - 2]);
        }
        let mut data = Vec::new();
        for (sym, &count) in counts.iter().enumerate() {
            data.extend(std::iter::repeat_n(sym as u16, count as usize));
        }
        assert!(assert_fast_matches_slow(&data));
        let max_len = max_code_length(&compress(&data).unwrap().unwrap());
        assert!(
            max_len >= 20,
            "uncapped Fibonacci frequencies must drive real codes well past the \
             12-bit direct-lookup table, got {max_len}"
        );
    }

    /// `build_encoding_table` is the SSOT for code-length assignment, so it can
    /// be driven with true (uncapped) Fibonacci frequencies directly — no need
    /// to materialize the astronomically large sample count [`fast_huf_near_maximum_code_length`]'s
    /// real round trip would need to reach the SAME frequencies for real
    /// (`fib(58)` is in the hundreds of billions). A frequency array costs
    /// nothing to represent regardless of how large its counts are, so this
    /// reaches genuinely near-`MAX_CODE_LEN` (58) lengths and asserts it,
    /// rather than merely asserting round-trip correctness.
    #[test]
    fn build_encoding_table_reaches_near_maximum_code_length() {
        let mut freq = vec![0u64; ENCODING_TABLE_SIZE];
        let (mut a, mut b) = (1u64, 1u64);
        for slot in freq.iter_mut().take(45) {
            *slot = a;
            let next = a + b;
            a = b;
            b = next;
        }
        let mut hlink = Vec::new();
        let mut scode = Vec::new();
        let mut heap_indices = Vec::new();
        let (im, im_max) =
            build_encoding_table(&mut freq, &mut hlink, &mut scode, &mut heap_indices, &mut BinaryHeap::new())
                .unwrap();
        let max_len = freq[im..=im_max].iter().map(|&e| e & 0x3f).max().unwrap_or(0);
        assert!(
            max_len >= 40,
            "true Fibonacci frequencies must drive canonical Huffman close to \
             MAX_CODE_LEN (58), got {max_len}"
        );
    }

    /// The `nBits > 128` dispatch boundary itself
    /// (`internal_huf.c:1963`/`decompress`, the real dispatcher): just below
    /// must match `decode_generic_t`'s (slow) output, just above must
    /// actually reach FastHuf and still agree.
    #[test]
    fn fast_huf_dispatch_boundary() {
        // A tiny, low-entropy fixture whose compressed n_bits sits somewhere
        // in the two-digit range; grow it one symbol at a time until we have
        // observed both sides of the 128-bit boundary.
        let mut saw_below = false;
        let mut saw_above = false;
        for n in 1..64usize {
            let data: Vec<u16> = (0..n as u16).collect();
            let compressed = compress(&data).unwrap().unwrap();
            let n_bits = header_n_bits(&compressed);
            let via_dispatch = decompress(&compressed, data.len()).unwrap();
            assert_eq!(via_dispatch, data);
            if n_bits <= 128 {
                saw_below = true;
            } else {
                saw_above = true;
            }
            if saw_below && saw_above {
                break;
            }
        }
        assert!(saw_below, "test setup: never observed n_bits <= 128");
        assert!(saw_above, "test setup: never observed n_bits > 128");
    }

    /// Corrupted `nBits > 128` streams must be rejected by `decompress` (the
    /// real dispatcher) exactly as they would be by `decode_generic_t` (the
    /// slow path) — never panic; a real accept/mismatch here would be a
    /// byte-identity gap since FastHuf is what C++ actually runs on such
    /// data — mirrors `corrupt_input_errors_not_panics` but forces the
    /// dispatch onto the FastHuf branch first.
    #[test]
    fn fast_huf_corrupt_input_errors_not_panics() {
        let good = compress(&UNCOMPRESSED_ARRAY).unwrap().unwrap();
        assert!(header_n_bits(&good) > 128, "test setup: fixture must dispatch to FastHuf");

        // Truncated header.
        let e1 = decompress(&good[..10], UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(e1.to_string().contains("shorter than the 20-byte header"), "got: {e1}");

        // Truncated entropy stream (drop the last several bytes).
        let truncated = &good[..good.len() - 8];
        let e2 = decompress(truncated, UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(e2.to_string().contains("not enough data for the declared bit count"), "got: {e2}");

        // Garbage body (table + entropy overwritten with 0xAB) — deterministic
        // for this fixture: the corrupted table never completes before the
        // declared entropy region runs out of bytes.
        let mut garbage = good.clone();
        for b in garbage.iter_mut().skip(20) {
            *b = 0xAB;
        }
        let e3 = decompress(&garbage, UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(e3.to_string().contains("truncated code-length table"), "got: {e3}");

        // Bogus im/iM.
        let mut bad_hdr = good.clone();
        bad_hdr[0..4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let e4 = decompress(&bad_hdr, UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(e4.to_string().contains("im/iM out of range"), "got: {e4}");

        // Wrong expected length (too small) → error: the entropy stream has
        // real bits left over once `expected` symbols are already decoded.
        let e5 = decompress(&good, 1).unwrap_err();
        assert!(e5.to_string().contains("bits of compressed data remains"), "got: {e5}");

        // All-bytes-0xFF of arbitrary length, with a fabricated header
        // claiming n_bits > 128 to force the FastHuf branch specifically —
        // the all-0xFF im/iM header fields alone are already out of range.
        let mut bogus = vec![0xFFu8; 64];
        bogus[12..16].copy_from_slice(&200u32.to_le_bytes());
        let e6 = decompress(&bogus, 100).unwrap_err();
        assert!(e6.to_string().contains("im/iM out of range"), "got: {e6}");
    }

    /// A stream flipped to claim a huge `n_bits` that does not fit the actual
    /// data must be rejected before any FastHuf table work — matches
    /// `hufUncompress`'s own `nBits > 8*nLeft`-style guard (here, the
    /// `n_bytes > body.len()` check in `decompress`, the real dispatcher).
    #[test]
    fn fast_huf_rejects_n_bits_exceeding_available_data() {
        let good = compress(&UNCOMPRESSED_ARRAY).unwrap().unwrap();
        let mut bad = good;
        bad[12..16].copy_from_slice(&(u32::MAX).to_le_bytes());
        let e = decompress(&bad, UNCOMPRESSED_ARRAY.len()).unwrap_err();
        assert!(
            e.to_string().contains("not enough data for the declared bit count"),
            "got: {e}"
        );
    }

    /// `FastHufDecoder`'s `_tableSymbol`
    /// sentinel (`0xffff`) is the SAME bit pattern as the legitimate pixel
    /// value 65535 (`HUF_ENCSIZE = 65537` makes every symbol `0..=65536`
    /// real). Force symbol 65535 to be the most frequent value so canonical
    /// Huffman gives it a short (`<= TABLE_LOOKUP_BITS`), table-accelerated
    /// code, then confirm `decompress` (the real dispatcher) decodes it correctly — proving
    /// the shared bit pattern never actually gets confused with the "no
    /// match yet" sentinel at runtime (the
    /// `table_min` guard guarantees a real match was already written before
    /// this slot is ever read).
    #[test]
    fn fast_huf_symbol_65535_gets_a_short_code_not_confused_with_sentinel() {
        let mut data = Vec::new();
        data.extend(std::iter::repeat_n(0xffffu16, 5000));
        data.extend(std::iter::repeat_n(1u16, 2000));
        data.extend(std::iter::repeat_n(2u16, 1000));
        data.extend(std::iter::repeat_n(3u16, 1));
        assert!(
            assert_fast_matches_slow(&data),
            "test setup: fixture must dispatch to FastHuf"
        );
    }

    /// The last per-chunk allocations of the PIZ/DWA Huffman
    /// paths are gone — `build_encoding_table`'s `BinaryHeap` now lives in
    /// `HufEncScratch::heap`, and the generic (`nBits <= 128`) decoder's two
    /// ~512 KiB tables live in `GenScratch` inside the persistent
    /// `FastHufDecoder`. Every reused buffer is POISONED before the first call
    /// (a missed reset would change the bytes), each result is diffed against a
    /// fresh-instance run, and the engagement probes must have fired: a byte
    /// match alone cannot tell reuse from a silent fresh allocation.
    ///
    /// The tiny inputs really take the generic path (asserted from each
    /// stream's own `nBits`); the Fibonacci input (20 symbols, codes up to ~19
    /// bits, so `Dec::Long` buckets) is forced through `decode_generic` to
    /// cover the poisoned `Dec::Long` slot too.
    #[test]
    fn generic_and_heap_scratch_is_reused_and_poison_proof() {
        use crate::probe::{self, Probe};

        let fib_data: Vec<u16> = {
            let (mut a, mut b) = (1u64, 1u64);
            let mut data = Vec::new();
            for sym in 0..20u16 {
                data.extend(std::iter::repeat_n(sym, a as usize));
                (a, b) = (b, a + b);
            }
            data
        };
        let tiny: [Vec<u16>; 3] = [
            vec![0u16; 500],
            vec![1234u16; 1500],
            vec![5, 5, 5, 5, 9, 9, 9, 9, 5, 5, 5, 5],
        ];

        let mut enc = HufEncScratch {
            freq: vec![0xAAAA_AAAA_AAAA_AAAA; ENCODING_TABLE_SIZE + 7],
            hlink: vec![0xAAAA_AAAA; ENCODING_TABLE_SIZE + 7],
            scode: vec![0xAAAA_AAAA_AAAA_AAAA; ENCODING_TABLE_SIZE + 7],
            heap_indices: vec![0xAAAA; 300],
            ..Default::default()
        };
        for i in 0..300 {
            enc.heap.push(HeapItem { frequency: 1 + i as u64 % 3, position: 40_000 + i });
        }
        let mut fhd = fast_huf::FastHufDecoder::default();
        fhd.poison_for_test();

        probe::reset(Probe::PizHufHeapReuse);
        probe::reset(Probe::PizHufGenericReuse);
        let (mut compressions, mut generic_decodes) = (0u64, 0u64);

        for data in tiny.iter().chain(std::iter::once(&fib_data)) {
            let mut shared = Vec::new();
            super::compress_with_limit(data, &mut enc, u32::MAX as u64, &mut shared).unwrap();
            compressions += 1;
            let mut fresh = Vec::new();
            super::compress_with_limit(data, &mut HufEncScratch::default(), u32::MAX as u64, &mut fresh).unwrap();
            assert_eq!(shared, fresh, "poisoned HufEncScratch changed the compressed bytes");

            let header = parse_header(&shared, data.len()).unwrap().unwrap();
            let is_fib = std::ptr::eq(data, &fib_data);
            if !is_fib {
                assert!(header.n_bits <= 128, "test setup: tiny input must take the generic path (n_bits {})", header.n_bits);
            }
            let mut out = Vec::new();
            if is_fib {
                decode_generic(&header, data.len(), &mut out, &mut fhd.generic).unwrap();
            } else {
                super::decompress(&shared, data.len(), &mut out, &mut fhd).unwrap();
            }
            generic_decodes += 1;
            assert_eq!(&out, data, "poisoned, reused generic decoder mis-decoded");

            let mut fresh_out = Vec::new();
            decode_generic(&header, data.len(), &mut fresh_out, &mut GenScratch::default()).unwrap();
            assert_eq!(out, fresh_out);
        }

        // The heap was warmed (poisoned with 300 items) before call 1, so every call reuses it.
        assert_eq!(probe::get(Probe::PizHufHeapReuse), compressions, "heap reuse probe");
        assert_eq!(probe::get(Probe::PizHufGenericReuse), generic_decodes, "generic scratch reuse probe");
    }
}

// -----------------------------------------------------------------------
// FastHuf: one targeted test per C error site, each
// crafting a MINIMAL stream that hits exactly that branch and checking the
// error is that specific branch (by message), not merely "some Err" — the
// existing `fast_huf_corrupt_input_errors_not_panics` fuzz-style test only
// proves "never panics", not "rejects for the right reason".
// -----------------------------------------------------------------------
#[cfg(test)]
mod fast_huf_error_sites {
    use super::fast_huf;
    use super::*;

    /// MSB-first bit packer matching `BitReader::read_bits`'s convention
    /// exactly (inverse of it): each `(n_bits, value)` field is appended to
    /// a running MSB-first bitstream, byte-aligned with zero padding at the
    /// end. Test-only — production code never needs to hand-assemble a
    /// packed table.
    fn pack_bits(fields: &[(u32, u64)]) -> Vec<u8> {
        let mut acc: u64 = 0;
        let mut nbits: u32 = 0;
        let mut out = Vec::new();
        for &(n, v) in fields {
            acc = (acc << n) | (v & ((1u64 << n) - 1));
            nbits += n;
            while nbits >= 8 {
                nbits -= 8;
                out.push((acc >> nbits) as u8);
            }
        }
        if nbits > 0 {
            out.push(((acc << (8 - nbits)) & 0xff) as u8);
        }
        out
    }

    /// Assert `err`'s message contains `needle` — our `ExrError::Input` is a
    /// single string variant, so "which branch" is checked by message text
    /// (there is no separate variant per C error site to match against).
    fn assert_err_is(err: &ExrError, needle: &str) {
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "expected an error mentioning {needle:?}, got: {msg}"
        );
    }

    /// `internal_huf.c:1385`/`:1407` — "Truncated table data": the very first
    /// 6-bit code-length read runs off the end of an empty table.
    #[test]
    fn truncated_table_data() {
        let mut fhd = fast_huf::FastHufDecoder::default();
        let err = fast_huf::initialize(&[], 0, 5, &mut fhd).unwrap_err();
        assert_err_is(&err, "truncated code-length table");
    }

    /// `internal_huf.c:1436` — "Run beyond end of table": a `LONG_ZEROCODE_RUN`
    /// (6-bit value 63) followed by an 8-bit count that overshoots `max_symbol`.
    /// `SHORTEST_LONG_RUN = 2 + 63 - 59 = 6`, so count=0 -> run=5,
    /// `symbol(0) + 5 = 5 > max_symbol(3)`.
    #[test]
    fn run_beyond_end_of_table() {
        let body = pack_bits(&[(6, 63), (8, 0)]);
        let mut fhd = fast_huf::FastHufDecoder::default();
        let err = fast_huf::initialize(&body, 0, 3, &mut fhd).unwrap_err();
        assert_err_is(&err, "Run beyond end of table");
    }

    /// `internal_huf.c:1541` — "Invalid symbol in header": exploits the same
    /// verbatim-ported quirk as `num_symbols`'s own computation
    /// (`for i in 0..MAX_CODE_LEN` excludes `codeCount[MAX_CODE_LEN]` itself,
    /// `fast_huf.rs`'s `initialize`) — a single symbol whose ONLY code has
    /// length exactly `MAX_CODE_LEN` (58) is counted into `offset`/`mapping`
    /// (built from `min..=max_code_length`, INCLUSIVE) but excluded from
    /// `num_symbols` (built from `0..MAX_CODE_LEN`, EXCLUSIVE), so pass 2's
    /// `mapping[58] (== offset[58] == 0) >= num_symbols (== 0)` fires on the
    /// very first (and only) real symbol.
    #[test]
    fn invalid_symbol_in_header() {
        let body = pack_bits(&[(6, 58)]);
        let mut fhd = fast_huf::FastHufDecoder::default();
        let err = fast_huf::initialize(&body, 7, 7, &mut fhd).unwrap_err();
        assert_err_is(&err, "Invalid symbol in header");
    }

    /// `internal_huf.c:1201` (`FastHufDecoder_buildTables`) — "Overrun": the
    /// acceleration-table build loop computes an `id` from `lj_base`/
    /// `lj_offset` alone (not from an actually-recorded symbol), independent
    /// of how many symbols were actually recorded — reachable with an
    /// ordinary INCOMPLETE canonical code, no exotic construction needed. A
    /// single symbol with a 1-bit code only "claims" half of the 1-bit code
    /// space (Kraft sum 1/2, not 1); the OTHER half of that space still gets
    /// scanned by the table-build loop (`i` ranges over every possible
    /// 12-bit prefix, not just the ones a real code produces), landing on
    /// `id = lj_offset[1] + 1 = 1`, which is `>= num_symbols (1)`.
    #[test]
    fn build_tables_overrun() {
        let body = pack_bits(&[(6, 1)]); // one symbol, code length 1 (incomplete tree)
        let mut fhd = fast_huf::FastHufDecoder::default();
        let err = fast_huf::initialize(&body, 10, 10, &mut fhd).unwrap_err();
        assert_err_is(&err, "Overrun");
    }

    /// `internal_huf.c:1745` — "RLE code with no previous symbol": the
    /// entropy stream's very FIRST decoded symbol is the run-length code
    /// itself (`dstIdx < 1`). Built by hand: a small real alphabet gives the
    /// RLE symbol (`im_max`) a real packed code via the same
    /// `build_encoding_table`/`pack_encoding_table` this module's own
    /// `compress` uses, then the entropy body is that RLE code's bits
    /// followed by an arbitrary 8-bit count, padded past the FastHuf
    /// dispatch threshold — `decode` errors on the very first symbol, long
    /// before the padding would ever matter.
    #[test]
    fn rle_with_no_previous_symbol() {
        let data: [u16; 3] = [0, 1, 2];
        let mut freq = Vec::new();
        count_frequencies(&data, &mut freq);
        let (mut hlink, mut scode, mut heap_indices) = (Vec::new(), Vec::new(), Vec::new());
        let (im, im_max) =
            build_encoding_table(&mut freq, &mut hlink, &mut scode, &mut heap_indices, &mut BinaryHeap::new())
                .unwrap();
        let rle_code = freq[im_max];

        let mut table_writer = BitWriter::with_buf(Vec::with_capacity(64));
        pack_encoding_table(&freq, im, im_max, &mut table_writer);
        let (table_bytes, _) = table_writer.finish();

        let mut entropy = pack_bits(&[(length(rle_code) as u32, code(rle_code)), (8, 5)]);
        entropy.resize(32, 0); // pad well past the 128-bit FastHuf threshold
        let n_bits: u32 = 200;

        let mut compressed = Vec::with_capacity(20 + table_bytes.len() + entropy.len());
        compressed.extend_from_slice(&(im as u32).to_le_bytes());
        compressed.extend_from_slice(&(im_max as u32).to_le_bytes());
        compressed.extend_from_slice(&(table_bytes.len() as u32).to_le_bytes());
        compressed.extend_from_slice(&n_bits.to_le_bytes());
        compressed.extend_from_slice(&0u32.to_le_bytes());
        compressed.extend_from_slice(&table_bytes);
        compressed.extend_from_slice(&entropy);

        let err = decompress(&compressed, data.len(), &mut Vec::new(), &mut fast_huf::FastHufDecoder::default()).unwrap_err();
        assert_err_is(&err, "RLE code with no previous symbol");
    }

    /// `internal_huf.c:1770` — "Invalid RLE length": an RLE code appearing
    /// AFTER a real symbol (so the "no previous symbol" check passes) with
    /// an 8-bit count of exactly 0.
    #[test]
    fn invalid_rle_length_zero() {
        let data: [u16; 3] = [0, 1, 2];
        let mut freq = Vec::new();
        count_frequencies(&data, &mut freq);
        let (mut hlink, mut scode, mut heap_indices) = (Vec::new(), Vec::new(), Vec::new());
        let (im, im_max) =
            build_encoding_table(&mut freq, &mut hlink, &mut scode, &mut heap_indices, &mut BinaryHeap::new())
                .unwrap();
        let rle_code = freq[im_max];
        let sym0_code = freq[0];

        let mut table_writer = BitWriter::with_buf(Vec::with_capacity(64));
        pack_encoding_table(&freq, im, im_max, &mut table_writer);
        let (table_bytes, _) = table_writer.finish();

        // One real symbol (so dstIdx >= 1 when the RLE code is hit), then the
        // RLE code with an explicit 0 count.
        let mut entropy = pack_bits(&[
            (length(sym0_code) as u32, code(sym0_code)),
            (length(rle_code) as u32, code(rle_code)),
            (8, 0),
        ]);
        entropy.resize(32, 0);
        let n_bits: u32 = 200;

        let mut compressed = Vec::with_capacity(20 + table_bytes.len() + entropy.len());
        compressed.extend_from_slice(&(im as u32).to_le_bytes());
        compressed.extend_from_slice(&(im_max as u32).to_le_bytes());
        compressed.extend_from_slice(&(table_bytes.len() as u32).to_le_bytes());
        compressed.extend_from_slice(&n_bits.to_le_bytes());
        compressed.extend_from_slice(&0u32.to_le_bytes());
        compressed.extend_from_slice(&table_bytes);
        compressed.extend_from_slice(&entropy);

        let err = decompress(&compressed, data.len(), &mut Vec::new(), &mut fast_huf::FastHufDecoder::default()).unwrap_err();
        assert_err_is(&err, "Invalid RLE length");
    }

    /// `internal_huf.c:1963`/dispatch boundary sanity: `fast_huf::decode`
    /// itself refuses to run below the 128-bit threshold — not a C error
    /// site by number, but the precondition every other test here relies on
    /// (a wrong threshold would silently make every crafted stream above
    /// take the WRONG decoder and invalidate every other test in this
    /// module). Uses a COMPLETE two-symbol, length-1-each table (Kraft sum
    /// exactly 1, both halves of the 1-bit code space claimed) so
    /// `initialize` itself succeeds and this test isolates `decode`'s own
    /// threshold check, not an unrelated table-build error.
    #[test]
    fn decode_refuses_at_or_below_threshold() {
        let mut fhd = fast_huf::FastHufDecoder::default();
        fast_huf::initialize(&pack_bits(&[(6, 1), (6, 1)]), 0, 1, &mut fhd).unwrap();
        let err = fast_huf::decode(&fhd, &[0u8; 32], 128, 1, &mut Vec::new()).unwrap_err();
        assert_err_is(&err, "below the FastHuf bit-count threshold");
    }
}
