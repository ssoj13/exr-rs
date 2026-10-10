//! `FastHufDecoder` — 1:1 port of OpenEXR's accelerated Huffman decoder
//! (`internal_huf.c:1084-1843`, v3.4.15).
//!
//! C++'s `hufUncompress` (`internal_huf.c:1953-1989`) dispatches on the
//! entropy-stream bit count alone: `nBits > 128` always takes this path on the
//! platforms `fasthuf_decode_enabled()` recognises (every mainstream compiler
//! target, including the MSVC-x64 build our oracle uses), and any error from
//! [`initialize`]/[`decode`] here is **final** — there is no fallback to the
//! generic [`super::huffman`] decoder in that case. `nBits <= 128` uses the
//! generic decoder instead (see `huffman::decompress_dispatch`, which implements this
//! exact dispatch). Real PIZ/DWA chunks are essentially always `nBits > 128`,
//! so this module — not `super::huffman`'s generic path — is what C++ decodes
//! through in practice, so it dominates measured PIZ/DWA
//! decode time.
//!
//! Two decoders, one wire format: [`initialize`] parses the SAME packed
//! code-length table `super::huffman::unpack_encoding_table` reads (reusing
//! its `BitReader`, since that is not performance-critical), but builds an
//! entirely different set of decode tables — a 12-bit direct lookup for short
//! codes plus a "left-justified" brute-force fallback for longer ones — which
//! [`decode`]'s hot loop then walks with a 64-bit two-register bit buffer
//! instead of a byte-at-a-time reader. No `unsafe`: the hot-path table index
//! is provably in range from the shift width alone (a `u64 >> 52` result
//! always fits 12 bits), so the fixed-size array index there costs nothing a
//! bounds check wouldn't already optimise away; the rare brute-force fallback
//! keeps an explicit, cheap bounds check exactly where C keeps an explicit
//! range check (`id < numSymbols`).

use super::huffman::{BitReader, LONG_ZEROCODE_RUN, SHORT_ZEROCODE_RUN, SHORTEST_LONG_RUN};
use crate::error::{ExrError, Result};

/// `MAX_CODE_LEN` (`internal_huf.c:1043`) — the widest canonical code
/// `hufCanonicalCodeTable` can produce.
const MAX_CODE_LEN: usize = 58;
/// `TABLE_LOOKUP_BITS` (`internal_huf.c:1050`) — codes up to this length are
/// resolved by one direct table lookup on the buffer's top bits.
const TABLE_LOOKUP_BITS: u32 = 12;
/// `1 << TABLE_LOOKUP_BITS`.
const TABLE_SIZE: usize = 1 << TABLE_LOOKUP_BITS;
/// `INDEX_BIT_SHIFT` (`internal_huf.c:1052`) — `64 - TABLE_LOOKUP_BITS`.
const INDEX_BIT_SHIFT: u32 = 64 - TABLE_LOOKUP_BITS;
/// Sentinel written into `table_symbol` for a slot the accelerated table does
/// not actually resolve (`internal_huf.c:1182`: `fhd->_tableSymbol[i] = 0xffff`).
/// Never observed by [`decode`]'s fast branch in practice: `table_min`
/// guarantees some code of length `<= TABLE_LOOKUP_BITS` matches whenever the
/// fast branch is taken, so `buildTables` always overwrites this default
/// before it would be read — ported verbatim regardless, matching C exactly.
const INVALID_TABLE_SYMBOL: u32 = 0xffff;

/// Safe-Rust equivalent of C++ `struct FastHufDecoder` (`internal_huf.c:1084`).
///
/// `id_to_symbol` is sized exactly `num_symbols` rather than C's fixed
/// `[65536 + 1]` — the "too many symbols" check in [`initialize`] enforces the
/// same upper bound before allocating it.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct FastHufDecoder {
    /// `_rleSymbol` — always `max_symbol` (the caller passes `(int) iM`,
    /// `internal_huf.c:1968`).
    rle_symbol: u32,
    /// `_numSymbols`.
    num_symbols: u32,
    /// `_maxCodeLength`.
    max_code_length: u8,
    /// `_idToSymbol`.
    id_to_symbol: Vec<u32>,
    /// `_ljBase` — sized `MAX_CODE_LEN + 1 + 1`; index `<= MAX_CODE_LEN + 1`
    /// is guaranteed by the sentinel `_ljBase[MAX_CODE_LEN + 1] = 0` (always
    /// `<= buffer` for an unsigned `buffer`), so the brute-force search in
    /// [`decode`] always terminates within this array.
    lj_base: [u64; MAX_CODE_LEN + 2],
    /// `_ljOffset` — sized `MAX_CODE_LEN + 1`.
    lj_offset: [u64; MAX_CODE_LEN + 1],
    /// `_tableSymbol` — `TABLE_SIZE` direct-lookup entries.
    table_symbol: Box<[u32; TABLE_SIZE]>,
    /// `_tableCodeLen` — `TABLE_SIZE` direct-lookup code lengths.
    table_code_len: Box<[u8; TABLE_SIZE]>,
    /// `_tableMin` — smallest left-justified base among lengths
    /// `<= TABLE_LOOKUP_BITS`; `decode`'s `tMin <= buffer` test.
    table_min: u64,
    /// Scratch tables of the generic scalar decoder (`huffman::decode_generic`,
    /// the `nBits <= 128` path). Not part of C's `FastHufDecoder`; it lives here
    /// because this is the one persistent decoder object callers already own.
    pub(super) generic: super::huffman::GenScratch,
}

impl Default for FastHufDecoder {
    /// Manual (not derived): `[u64; MAX_CODE_LEN + 1]`/`Box<[u32; TABLE_SIZE]>`
    /// etc. exceed the array sizes std's blanket `Default` impls cover, so
    /// `#[derive(Default)]` doesn't apply here.
    fn default() -> Self {
        Self {
            rle_symbol: 0,
            num_symbols: 0,
            max_code_length: 0,
            id_to_symbol: Vec::new(),
            lj_base: [0u64; MAX_CODE_LEN + 2],
            lj_offset: [0u64; MAX_CODE_LEN + 1],
            table_symbol: Box::new([INVALID_TABLE_SYMBOL; TABLE_SIZE]),
            table_code_len: Box::new([0u8; TABLE_SIZE]),
            table_min: 0,
            generic: Default::default(),
        }
    }
}

impl FastHufDecoder {
    /// Reset to the empty/pre-init state: [`initialize`]
    /// calls this on every real call BEFORE rebuilding, since `self` is now a
    /// caller-owned, reused instance (previously a fresh `FastHufDecoder` was
    /// constructed and returned every call, so every field started at a
    /// known-good default with no possibility of stale data).
    ///
    /// Verified empirically (`huffman::tests::
    /// reused_fast_huf_decoder_across_different_histograms_matches_fresh_decoder`,
    /// poison-then-reuse across several different real histograms plus a
    /// deliberately incomplete canonical code): every field this struct
    /// exposes to [`decode`] ends up fully, unconditionally overwritten by
    /// `build_tables`/`initialize`'s own writes regardless of whether this
    /// `reset()` ran first -- `rle_symbol`/`num_symbols`/`max_code_length`
    /// are assigned outright at the top of `build_tables`; `lj_base`/
    /// `lj_offset` are written for every index in their full range;
    /// `id_to_symbol` is `ensure_len`'d to exactly `num_symbols` and every
    /// index within that exact length is written by the second pass. Kept
    /// anyway for `table_symbol`/`table_code_len`'s sentinel defaults (an
    /// incomplete canonical code can legitimately leave a table slot
    /// unwritten -- see `build_tables`'s own comment) purely to match C's
    /// own `_tableSymbol[i] = 0xffff` initialization 1:1, even though
    /// `decode`'s own gating (`table_code_len[i] == 0` never leads to a read
    /// of `table_symbol[i]`, and the "Overrun" bounds check that DOES fire
    /// on an incomplete code depends only on `id`/`num_symbols`, never on a
    /// slot's prior content) means this reset is not independently
    /// observable through the public API either way -- confirmed by the
    /// same test, whose mutation of this exact line does NOT go red.
    fn reset(&mut self) {
        self.rle_symbol = 0;
        self.num_symbols = 0;
        self.max_code_length = 0;
        self.id_to_symbol.clear();
        self.lj_base = [0u64; MAX_CODE_LEN + 2];
        self.lj_offset = [0u64; MAX_CODE_LEN + 1];
        self.table_symbol.fill(INVALID_TABLE_SYMBOL);
        self.table_code_len.fill(0);
        self.table_min = 0;
    }
}

#[cfg(test)]
impl FastHufDecoder {
    /// Test-only: overwrite every field with implausible "poison" values
    /// (never legitimately produced by `reset()`/`build_tables`) -- to
    /// prove coverage empirically with a poison test
    /// (same methodology as B44's
    /// `poison_reused_scratch_matches_fresh_instance`).
    /// `huffman::tests` calls this on a persistent decoder BEFORE the first
    /// real `initialize`/`decompress` call, then diffs the result against a
    /// fresh decoder -- any field `initialize`/`build_tables` doesn't fully
    /// overwrite this call would leak a poison value into the decode.
    pub(super) fn poison_for_test(&mut self) {
        self.rle_symbol = 0xDEAD_BEEF;
        self.num_symbols = 0xDEAD_BEEF;
        self.max_code_length = 0xAA;
        self.id_to_symbol = vec![0xDEAD_BEEF; 4096];
        self.lj_base = [0xAAAA_AAAA_AAAA_AAAA; MAX_CODE_LEN + 2];
        self.lj_offset = [0x5555_5555_5555_5555; MAX_CODE_LEN + 1];
        self.table_symbol.fill(0xDEAD_BEEF);
        self.table_code_len.fill(0xAA);
        self.table_min = 0xAAAA_AAAA_AAAA_AAAA;
        self.generic.poison_for_test();
    }
}

/// A truncated packed code-length table — `internal_huf.c`'s
/// `"Error decoding Huffman table (Truncated table data)."`.
fn truncated_table() -> ExrError {
    ExrError::Input("PIZ fast-huf: truncated code-length table".into())
}

/// `FastHufDecoder_buildTables` (`internal_huf.c:1132-1231`).
///
/// Builds the left-justified base/offset tables and the `TABLE_LOOKUP_BITS`
/// acceleration table from the per-length `base`/`offset` arrays
/// [`initialize`] computed.
/// Fills `fhd`'s tables in place (`fhd` is the caller's
/// persistent, reused decoder -- `fhd.reset()` has already restored the
/// sentinel/empty state before this runs, and `fhd.id_to_symbol` has already
/// been filled by `initialize`'s second pass). Reads `fhd.id_to_symbol`
/// while writing every OTHER field -- disjoint fields of the same `&mut
/// self`-like reference, fine within one function body.
fn build_tables(
    fhd: &mut FastHufDecoder,
    rle_symbol: u32,
    num_symbols: u32,
    min_code_length: u8,
    max_code_length: u8,
    base: &[u64; MAX_CODE_LEN + 1],
    offset: &[u64; MAX_CODE_LEN + 1],
) -> Result<()> {
    fhd.rle_symbol = rle_symbol;
    fhd.num_symbols = num_symbols;
    fhd.max_code_length = max_code_length;

    // Left-justified base table: base[i] shifted into the high bits of a
    // 64-bit word, or the "unused" sentinel carried through unchanged.
    // `i == 0` never holds a real code (code length 0 means "no code" and is
    // never assigned to `base`), so `i > 0` here only ever rules out an
    // unreachable shift-by-64 defensively — never a live substitution for the
    // C behaviour, which is real-code-shift-by-`64-i` for every `i >= 1`.
    #[allow(clippy::needless_range_loop)] // writes fhd.lj_base[i], reads base[i] -- two arrays, one index
    for i in 0..=MAX_CODE_LEN {
        fhd.lj_base[i] = if base[i] != u64::MAX && i > 0 {
            base[i] << (64 - i as u32)
        } else {
            u64::MAX
        };
    }
    fhd.lj_base[MAX_CODE_LEN + 1] = 0; // sentinel for the brute-force search

    // Left-justified offset table (folds `offset[i] - (ljBase[i] >> (64-i))`
    // into one constant so the hot loop needs no extra subtraction per code).
    fhd.lj_offset[0] = offset[0].wrapping_sub(fhd.lj_base[0]);
    #[allow(clippy::needless_range_loop)] // writes fhd.lj_offset[i], reads offset[i] + fhd.lj_base[i]
    for i in 1..=MAX_CODE_LEN {
        fhd.lj_offset[i] = offset[i].wrapping_sub(fhd.lj_base[i] >> (64 - i as u32));
    }

    // The TABLE_LOOKUP_BITS-wide acceleration table for short codes. Already
    // reset to its sentinel defaults by `fhd.reset()` -- this loop does NOT
    // necessarily overwrite every entry (see below), so that reset is what
    // keeps a reused `fhd` correct here.
    for i in 0..TABLE_SIZE as u64 {
        let value = i << INDEX_BIT_SHIFT;
        // When `min_code_length > max_code_length` (no codes at all) this
        // range is empty and `table_symbol`/`table_code_len` keep their
        // defaults — matching C, whose inner `for` loop then also does
        // nothing for this `i`.
        for code_len in min_code_length..=max_code_length {
            if fhd.lj_base[code_len as usize] <= value {
                fhd.table_code_len[i as usize] = code_len;
                let id = fhd.lj_offset[code_len as usize] + (value >> (64 - code_len as u32));
                if id < num_symbols as u64 {
                    fhd.table_symbol[i as usize] = fhd
                        .id_to_symbol
                        .get(id as usize)
                        .copied()
                        .ok_or_else(|| ExrError::Input("PIZ fast-huf: id out of range".into()))?;
                } else {
                    return Err(ExrError::Input(
                        "PIZ fast-huf: Huffman decode error (Overrun)".into(),
                    ));
                }
                break;
            }
        }
    }

    // Smallest lj_base value that points at real (non-sentinel) data —
    // the largest length with a used code, walking down from TABLE_LOOKUP_BITS.
    let mut min_idx = TABLE_LOOKUP_BITS as i32;
    while min_idx > 0 && fhd.lj_base[min_idx as usize] == u64::MAX {
        min_idx -= 1;
    }
    fhd.table_min = if min_idx < 0 {
        u64::MAX
    } else {
        fhd.lj_base[min_idx as usize]
    };

    Ok(())
}

/// `fasthuf_initialize` (`internal_huf.c:1327-1551`).
///
/// Parses the packed code-length table from `body` (the same bytes
/// `super::huffman::unpack_encoding_table` would read, right after the
/// 20-byte chunk header) and builds the decoder. Returns the decoder plus the
/// number of bytes consumed from `body` (C's `*table = currByte` advance) —
/// the entropy-coded bit stream starts immediately after.
pub(crate) fn initialize(
    body: &[u8],
    min_symbol: u32,
    max_symbol: u32,
    fhd: &mut FastHufDecoder,
) -> Result<usize> {
    fhd.reset();
    let rle_symbol = max_symbol;

    let mut code_count = [0usize; MAX_CODE_LEN + 1];
    let mut base = [u64::MAX; MAX_CODE_LEN + 1];
    let mut offset = [0u64; MAX_CODE_LEN + 1];
    let mut min_code_length: u8 = 255;
    let mut max_code_length: u8 = 0;

    // First pass: count codes per length, validate the run never overruns
    // max_symbol. Mirrors the C `for` loop's own `symbol++`: every branch
    // ends with an explicit `symbol += 1` except `code_len == 0`, whose
    // `continue` skips both the bounds check and that trailing `+= 1` — same
    // as C's `continue` jumping straight to the for-loop's increment clause
    // for a run that added nothing.
    {
        let mut reader = BitReader::new(body);
        let mut symbol = min_symbol as u64;
        while symbol <= max_symbol as u64 {
            let code_len = reader.read_bits(6).map_err(|_| truncated_table())?;
            if code_len < SHORT_ZEROCODE_RUN {
                if code_len == 0 {
                    symbol += 1;
                    continue;
                }
                let cl = code_len as u8;
                if cl < min_code_length {
                    min_code_length = cl;
                }
                if cl > max_code_length {
                    max_code_length = cl;
                }
                code_count[code_len as usize] += 1;
            } else if code_len == LONG_ZEROCODE_RUN {
                let run = reader.read_bits(8).map_err(|_| truncated_table())?;
                symbol += run + SHORTEST_LONG_RUN - 1;
            } else {
                symbol += code_len - SHORT_ZEROCODE_RUN + 1;
            }
            if symbol > max_symbol as u64 {
                return Err(ExrError::Input(
                    "PIZ fast-huf: Huffman decode error (Run beyond end of table)".into(),
                ));
            }
            symbol += 1;
        }
    }

    // `for (i = 0; i < MAX_CODE_LEN; ++i)` — C sums indices 0..57 only,
    // excluding codeCount[MAX_CODE_LEN] itself; ported exactly (not a typo
    // we get to "fix" — see the module doc's "never guess" rule).
    let num_symbols: u32 = code_count[..MAX_CODE_LEN]
        .iter()
        .try_fold(0u32, |acc, &c| {
            u32::try_from(c).ok().and_then(|c| acc.checked_add(c))
        })
        .ok_or_else(|| ExrError::Input("PIZ fast-huf: symbol count overflow".into()))?;

    if num_symbols as usize > 65536 + 1 {
        return Err(ExrError::Input(
            "PIZ fast-huf: Huffman decode error (Too many symbols)".into(),
        ));
    }

    // Closed-form base[] via the same double-precision arithmetic as C
    // (internal_huf.c:1467-1497) — an integer reformulation would not
    // reliably reproduce `ceil()`'s rounding on every input.
    if min_code_length <= max_code_length {
        let mut count_tmp = [0f64; MAX_CODE_LEN + 1];
        for l in min_code_length..=max_code_length {
            count_tmp[l as usize] =
                code_count[l as usize] as f64 * (2i64 << (max_code_length - l)) as f64;
        }
        for l in min_code_length..=max_code_length {
            let mut tmp = 0f64;
            for k in (l + 1)..=max_code_length {
                tmp += count_tmp[k as usize];
            }
            tmp /= (2i64 << (max_code_length - l)) as f64;
            base[l as usize] = tmp.ceil() as u64;
        }

        // offset[] — position of the first id at each length.
        offset[max_code_length as usize] = 0;
        let mut i = max_code_length as i32 - 1;
        while i >= min_code_length as i32 {
            offset[i as usize] = offset[i as usize + 1] + code_count[i as usize + 1] as u64;
            i -= 1;
        }
    }

    // Second pass: re-read the table (re-decoding is cheaper than keeping an
    // uncompressed symbol list around, exactly per C's own comment) and fill
    // `id_to_symbol` via the `mapping` cursor, one id per length.
    //
    // `fhd.id_to_symbol` is grown/truncated via
    // `ensure_len`, not a fresh `vec![0u32; ...]` -- every index in
    // `[0, num_symbols)` gets written exactly once by the mapping-cursor
    // loop below (the `offset[]` values computed above partition that whole
    // range across the used code lengths with no gaps, the classic
    // counting-sort bucket-assignment pattern), so a stale value surviving
    // in a position this call doesn't touch is never observed -- unlike
    // `table_symbol`/`table_code_len` in `build_tables`, this one does NOT
    // need `fhd.reset()`'s help beyond the length.
    crate::compressor::ensure_len(&mut fhd.id_to_symbol, num_symbols as usize);
    {
        let mut mapping = [u64::MAX; MAX_CODE_LEN + 1];
        for l in min_code_length..=max_code_length {
            mapping[l as usize] = offset[l as usize];
        }

        let mut reader = BitReader::new(body);
        let mut symbol = min_symbol as u64;
        while symbol <= max_symbol as u64 {
            let code_len = reader.read_bits(6).map_err(|_| truncated_table())?;
            if code_len < SHORT_ZEROCODE_RUN {
                if code_len == 0 {
                    symbol += 1;
                    continue;
                }
                let m = mapping
                    .get_mut(code_len as usize)
                    .ok_or_else(|| ExrError::Input("PIZ fast-huf: code length out of range".into()))?;
                if *m >= num_symbols as u64 {
                    return Err(ExrError::Input(
                        "PIZ fast-huf: Huffman decode error (Invalid symbol in header)".into(),
                    ));
                }
                let slot = fhd
                    .id_to_symbol
                    .get_mut(*m as usize)
                    .ok_or_else(|| ExrError::Input("PIZ fast-huf: id out of range".into()))?;
                *slot = symbol as u32;
                *m += 1;
            } else if code_len == LONG_ZEROCODE_RUN {
                let run = reader.read_bits(8).map_err(|_| truncated_table())?;
                symbol += run + SHORTEST_LONG_RUN - 1;
            } else {
                symbol += code_len - SHORT_ZEROCODE_RUN + 1;
            }
            symbol += 1;
        }
        // The consumed byte count (C: `*table = currByte`) — the entropy
        // stream starts here. `reader.pos` already reflects every byte the
        // second pass pulled, which walks the identical bit sequence as the
        // first pass (same table, re-read), so this matches C exactly.
        let consumed = reader.pos;

        build_tables(fhd, rle_symbol, num_symbols, min_code_length, max_code_length, &base, &offset)?;
        Ok(consumed)
    }
}

/// `FastHufDecoder_refill` (`internal_huf.c:1233-1311`) — pulls more bits into
/// `buffer` from `buffer_back`, then refills `buffer_back` from the byte
/// stream, keeping `buffer` at 64 valid bits whenever data remains.
///
/// `bits_left` mirrors C's `currBitsLeft` — the count of bits remaining in the
/// byte stream (NOT the same as `data.len() - pos`; it is decremented
/// explicitly and independently, exactly as C tracks it, so a truncated final
/// byte is handled the same way).
///
/// # Errors
/// `bits_left` is the caller's own bookkeeping, not `data.len()` — it is
/// established correct by `decompress_dispatch`'s `n_bytes > data.len()`
/// check before `decode` is ever called, so every `data.get(...)` here is
/// expected to always succeed. That invariant is enforced by today's only
/// caller, not by this function, so a violation (a future caller passing a
/// `bits_left` that overstates the real remaining bytes) is a hard
/// [`ExrError`] here rather than a silent zero-pad — never a guess at what
/// the missing bytes "should" be.
#[allow(clippy::too_many_arguments)]
fn refill(
    buffer: &mut u64,
    mut num_bits: i32,
    buffer_back: &mut u64,
    buffer_back_num_bits: &mut i32,
    data: &[u8],
    pos: &mut usize,
    bits_left: &mut u64,
) -> Result<()> {
    let short_read = || ExrError::Input("PIZ fast-huf: refill ran past the entropy stream".into());
    loop {
        if num_bits > 0 {
            let need_bits = 64 - num_bits;
            *buffer |= if num_bits >= 64 { 0 } else { *buffer_back >> num_bits };
            if *buffer_back_num_bits >= need_bits {
                *buffer_back = if need_bits >= 64 { 0 } else { *buffer_back << need_bits };
                *buffer_back_num_bits -= need_bits;
                break;
            }
            num_bits += *buffer_back_num_bits;
            *buffer_back_num_bits = 0;
        } else {
            *buffer = *buffer_back;
            num_bits = *buffer_back_num_bits;
            *buffer_back_num_bits = 0;
        }

        if *buffer_back_num_bits <= 0 {
            if *bits_left >= 64 {
                let chunk: [u8; 8] = data
                    .get(*pos..*pos + 8)
                    .ok_or_else(short_read)?
                    .try_into()
                    .map_err(|_| short_read())?;
                *buffer_back = u64::from_be_bytes(chunk);
                *buffer_back_num_bits = 64;
                *pos += 8;
                *bits_left -= 64;
            } else {
                let mut shift: i64 = 56;
                *buffer_back = 0;
                *buffer_back_num_bits = 64;
                while *bits_left >= 8 {
                    let b = *data.get(*pos).ok_or_else(short_read)?;
                    *buffer_back |= (b as u64) << shift;
                    *pos += 1;
                    shift -= 8;
                    *bits_left -= 8;
                }
                if *bits_left > 0 {
                    let b = *data.get(*pos).ok_or_else(short_read)?;
                    *buffer_back |= (b as u64) << shift;
                    *pos += 1;
                    *bits_left = 0;
                }
            }
        }

        if num_bits >= 64 {
            break;
        }
    }
    Ok(())
}

/// `fasthuf_decode` (`internal_huf.c:1597-1819`) — the accelerated decode hot
/// loop. `data` is the entropy-coded stream (`super::huffman::decompress`'s
/// `body[consumed..]`); `num_src_bits` is the packed-header `nBits` field.
///
/// Requires `data.len() * 8 >= num_src_bits` and `num_src_bits > 128`
/// (`hufUncompress`'s own dispatch gate, `internal_huf.c:1963`) — the initial
/// unconditional 16-byte read is safe under that precondition, checked here
/// explicitly rather than trusted from the caller.
///
/// `out` is NOT cleared here — the one public entry point,
/// `huffman::decompress`, clears it once on entry; callers of THIS function
/// directly (tests) must pass an already-empty `out` themselves.
pub(super) fn decode(
    fhd: &FastHufDecoder,
    data: &[u8],
    num_src_bits: u64,
    expected: usize,
    out: &mut Vec<u16>,
) -> Result<()> {
    if num_src_bits <= 128 {
        return Err(ExrError::Input(
            "PIZ fast-huf: decode called below the FastHuf bit-count threshold".into(),
        ));
    }
    let first16 = data
        .get(0..16)
        .ok_or_else(|| ExrError::Input("PIZ fast-huf: entropy stream shorter than 16 bytes".into()))?;
    // `first16` is exactly 16 bytes (the slice above already guarantees it),
    // so splitting it into two `[u8; 8]` arrays cannot fail — no second,
    // duplicate-message error path needed for a conversion that is really
    // infallible here.
    let first8: [u8; 8] = [
        first16[0], first16[1], first16[2], first16[3],
        first16[4], first16[5], first16[6], first16[7],
    ];
    let next8: [u8; 8] = [
        first16[8], first16[9], first16[10], first16[11],
        first16[12], first16[13], first16[14], first16[15],
    ];
    let mut buffer = u64::from_be_bytes(first8);
    let mut buffer_num_bits: i32 = 64;
    let mut buffer_back = u64::from_be_bytes(next8);
    let mut buffer_back_num_bits: i32 = 64;
    let mut pos: usize = 16;
    let mut bits_left: u64 = num_src_bits - 8 * 16;

    let t_min = fhd.table_min;
    let rle_sym = fhd.rle_symbol;

    while out.len() < expected {
        let symbol: u32;
        let code_len: i32;

        if t_min <= buffer {
            let table_idx = (buffer >> INDEX_BIT_SHIFT) as usize;
            code_len = fhd.table_code_len[table_idx] as i32;
            symbol = fhd.table_symbol[table_idx];
        } else {
            if buffer_num_bits < 64 {
                refill(
                    &mut buffer,
                    buffer_num_bits,
                    &mut buffer_back,
                    &mut buffer_back_num_bits,
                    data,
                    &mut pos,
                    &mut bits_left,
                )?;
                buffer_num_bits = 64;
            }

            let mut cl = TABLE_LOOKUP_BITS as i32 + 1;
            while fhd
                .lj_base
                .get(cl as usize)
                .copied()
                .ok_or_else(|| ExrError::Input("PIZ fast-huf: code length search out of range".into()))?
                > buffer
            {
                cl += 1;
                if cl as usize >= fhd.lj_base.len() {
                    return Err(ExrError::Input(
                        "PIZ fast-huf: Huffman decode error (Decoded an invalid symbol)".into(),
                    ));
                }
            }
            if cl > fhd.max_code_length as i32 {
                return Err(ExrError::Input(
                    "PIZ fast-huf: Huffman decode error (Decoded an invalid symbol)".into(),
                ));
            }
            code_len = cl;
            let id = fhd.lj_offset[cl as usize].wrapping_add(buffer >> (64 - cl as u32));
            if id < fhd.num_symbols as u64 {
                symbol = *fhd
                    .id_to_symbol
                    .get(id as usize)
                    .ok_or_else(|| ExrError::Input("PIZ fast-huf: id out of range".into()))?;
            } else {
                return Err(ExrError::Input(
                    "PIZ fast-huf: Huffman decode error (Decoded an invalid symbol)".into(),
                ));
            }
        }

        buffer <<= code_len;
        buffer_num_bits -= code_len;

        if symbol == rle_sym {
            if buffer_num_bits < 8 {
                refill(
                    &mut buffer,
                    buffer_num_bits,
                    &mut buffer_back,
                    &mut buffer_back_num_bits,
                    data,
                    &mut pos,
                    &mut bits_left,
                )?;
                buffer_num_bits = 64;
            }
            let rle_count = (buffer >> 56) as u32;

            // Single check for "no previous symbol": `out.last()` is the only
            // source of truth, so there is exactly one place that can fail
            // this way — no separate `is_empty()` check duplicating it (the
            // old `is_empty()` guard made the later `.ok_or_else()` on
            // `out.last()` unreachable, since nothing removes elements from
            // `out` in between).
            let last = match out.last() {
                Some(&v) => v,
                None => {
                    return Err(ExrError::Input(
                        "PIZ fast-huf: Huffman decode error (RLE code with no previous symbol)".into(),
                    ));
                }
            };
            if out.len() as u64 + rle_count as u64 > expected as u64 {
                return Err(ExrError::Input(
                    "PIZ fast-huf: Huffman decode error (Symbol run beyond expected output buffer length)".into(),
                ));
            }
            // `rle_count` is an 8-bit field (`buffer >> 56`, max 255), so it
            // can never reach C's `>= INT32_MAX` half of this check — ported
            // as `== 0` only; the other half is dead code in both languages.
            if rle_count == 0 {
                return Err(ExrError::Input(
                    "PIZ fast-huf: Huffman decode error (Invalid RLE length)".into(),
                ));
            }
            out.extend(std::iter::repeat_n(last, rle_count as usize));

            buffer <<= 8;
            buffer_num_bits -= 8;
        } else {
            let v = u16::try_from(symbol)
                .map_err(|_| ExrError::Input("PIZ fast-huf: symbol exceeds u16 range".into()))?;
            out.push(v);
        }

        if buffer_num_bits < TABLE_LOOKUP_BITS as i32 {
            refill(
                &mut buffer,
                buffer_num_bits,
                &mut buffer_back,
                &mut buffer_back_num_bits,
                data,
                &mut pos,
                &mut bits_left,
            )?;
            buffer_num_bits = 64;
        }
    }

    if bits_left != 0 {
        return Err(ExrError::Input(format!(
            "PIZ fast-huf: Huffman decode error ({bits_left} bits of compressed data remains after filling expected output buffer)"
        )));
    }

    Ok(())
}
