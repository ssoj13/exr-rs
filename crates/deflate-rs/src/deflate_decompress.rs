//! Port of libdeflate `lib/deflate_decompress.c` (+ `lib/decompress_template.h`).
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/deflate_decompress.c
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/decompress_template.h
//!
//! Implements the full DEFLATE inflater: Huffman decode-table builder
//! (`build_decode_table`), all three block types (stored / static / dynamic),
//! the litlen+offset decode loop (fast path + generic path), and the
//! libdeflate-specific subtable layout.
//!
//! The C source uses a macro-template (`decompress_template.h`) compiled
//! multiple times with different `#define`s for BMI2 / plain paths.  In Rust
//! we port it as one function (`deflate_decompress_impl`) because we target
//! safe Rust with no SIMD specialisation at this layer.
//!
//! **No `unsafe` code.** All pointer arithmetic is replaced with index-based
//! slice access; the C bit-buffer tricks are reproduced with plain integer ops.

use crate::common::{bsr32, get_unaligned_le16, get_unaligned_le64, put_unaligned_le64};
use crate::deflate_constants::{
    DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, DEFLATE_BLOCKTYPE_STATIC_HUFFMAN,
    DEFLATE_BLOCKTYPE_UNCOMPRESSED, DEFLATE_MAX_CODEWORD_LEN, DEFLATE_MAX_EXTRA_LENGTH_BITS,
    DEFLATE_MAX_EXTRA_OFFSET_BITS, DEFLATE_MAX_LENS_OVERRUN, DEFLATE_MAX_LITLEN_CODEWORD_LEN,
    DEFLATE_MAX_NUM_SYMS, DEFLATE_MAX_OFFSET_CODEWORD_LEN, DEFLATE_MAX_PRE_CODEWORD_LEN,
    DEFLATE_NUM_LITLEN_SYMS, DEFLATE_NUM_OFFSET_SYMS, DEFLATE_NUM_PRECODE_SYMS,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the DEFLATE decompressor.
///
/// Mirrors `libdeflate_result` (the two error variants we can produce).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompressError {
    /// The compressed data is invalid / corrupt. Mirrors `LIBDEFLATE_BAD_DATA`.
    BadData,
    /// The output buffer is too small for the decompressed data.
    /// Mirrors `LIBDEFLATE_INSUFFICIENT_SPACE`.
    InsufficientSpace,
    /// The stream decompressed to FEWER bytes than the caller said to expect.
    ///
    /// Mirrors `LIBDEFLATE_SHORT_OUTPUT`, which libdeflate returns when a NULL
    /// `actual_out_nbytes_ret` was passed (`libdeflate.h:201-203`), i.e. when
    /// the caller claimed to know the exact uncompressed size. Only the
    /// `*_decompress_exact` entry points can produce it; the counted entry
    /// points return the short byte count instead, exactly like C.
    ShortOutput,
}

impl core::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecompressError::BadData => write!(f, "deflate: bad/corrupt compressed data"),
            DecompressError::InsufficientSpace => {
                write!(f, "deflate: output buffer too small")
            }
            DecompressError::ShortOutput => {
                write!(f, "deflate: stream decompressed to fewer bytes than expected")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decode-table constants  (mirrors deflate_decompress.c)
// ---------------------------------------------------------------------------

/// PRECODE_TABLEBITS == 7, matching `DEFLATE_MAX_PRE_CODEWORD_LEN`.
/// No subtables ever needed for the precode.
const PRECODE_TABLEBITS: u32 = 7;
const PRECODE_ENOUGH: usize = 128; // enough 19 7 7

/// LITLEN_TABLEBITS == 11.  We allow dynamic shrink (stored in `Decompressor`).
const LITLEN_TABLEBITS: u32 = 11;
const LITLEN_ENOUGH: usize = 2342; // enough 288 11 15

/// OFFSET_TABLEBITS == 8.
const OFFSET_TABLEBITS: u32 = 8;
const OFFSET_ENOUGH: usize = 402; // enough 32 8 15

// Litlen / offset decode table entry flags  (mirrors deflate_decompress.c)
const HUFFDEC_LITERAL: u32 = 0x8000_0000;
const HUFFDEC_EXCEPTIONAL: u32 = 0x0000_8000;
const HUFFDEC_SUBTABLE_POINTER: u32 = 0x0000_4000;
const HUFFDEC_END_OF_BLOCK: u32 = 0x0000_2000;

/// Maximum bits that can be consumed decoding a match length.
/// `DEFLATE_MAX_LITLEN_CODEWORD_LEN + DEFLATE_MAX_EXTRA_LENGTH_BITS`
const LENGTH_MAXBITS: u32 = DEFLATE_MAX_LITLEN_CODEWORD_LEN + DEFLATE_MAX_EXTRA_LENGTH_BITS;
/// Fast-path variant (no subtable needed).
const LENGTH_MAXFASTBITS: u32 = LITLEN_TABLEBITS + DEFLATE_MAX_EXTRA_LENGTH_BITS;

/// C ports this as `STATIC_ASSERT(CAN_CONSUME_AND_THEN_PRELOAD(
/// LENGTH_MAXFASTBITS, OFFSET_TABLEBITS))` (`decompress_template.h:503`) right
/// before the fastloop's offset decode — a compile-time check that our fixed
/// 64-bit bitbuf / table-bit constants can always consume a fast-path length
/// entry and then preload an `OFFSET_TABLEBITS`-wide offset table lookup
/// without a mid-sequence refill. `CAN_CONSUME_AND_THEN_PRELOAD(consume,
/// preload)` is `CONSUMABLE_NBITS(56) >= consume && FASTLOOP_PRELOADABLE_NBITS
/// (64, branchless refill) >= consume + preload`. This was previously an
/// unused constant (`#[allow(dead_code)]`) with no assertion backing it —
/// the comment inline at its use site cited the property without the crate
/// actually checking it holds.
const _: () = assert!(
    56 >= LENGTH_MAXFASTBITS && 64 >= LENGTH_MAXFASTBITS + OFFSET_TABLEBITS,
    "CAN_CONSUME_AND_THEN_PRELOAD(LENGTH_MAXFASTBITS, OFFSET_TABLEBITS) does not hold \
     for these constants — the fastloop's offset decode may need an extra mid-sequence \
     refill this port does not perform"
);

/// Maximum bits that can be consumed decoding a match offset.
const OFFSET_MAXBITS: u32 = DEFLATE_MAX_OFFSET_CODEWORD_LEN + DEFLATE_MAX_EXTRA_OFFSET_BITS;
/// Fast-path variant (no subtable needed for offset).
const OFFSET_MAXFASTBITS: u32 = OFFSET_TABLEBITS + DEFLATE_MAX_EXTRA_OFFSET_BITS;

/// Default fastloop setting for the public entry points.
///
/// The fastloop is an *optimisation* over the generic loop: in C they produce
/// byte-identical output, the fastloop just decodes several items per refill
/// (and per-byte instead of per-bit bitbuffer refills). The generic loop
/// remains the byte-exact reference implementation. See
/// [`deflate_decompress_impl`]'s `use_fastloop` parameter — kept as a runtime
/// flag rather than folded into this `const` so the differential test in this
/// module's test suite can exercise the fastloop-off (generic-loop-only) path
/// against the exact same inputs and assert identical output.
const FASTLOOP_ENABLED: bool = true;

/// `PRELOAD_SLACK` (`deflate_decompress.c`): bits guaranteed preloadable but
/// NOT consumable right after `REFILL_BITS_IN_FASTLOOP()`.  On our target
/// (branchless/unaligned refill, 64-bit `bitbuf`): `BITBUF_NBITS=64`,
/// `MAX_BITSLEFT=63` (branchless refill reserves 1 bit),
/// `FASTLOOP_PRELOADABLE_NBITS=BITBUF_NBITS=64` (branchless path), so
/// `PRELOAD_SLACK = max(0, 64-63) = 1`. Used to reproduce C's exact refill
/// thresholds in the fastloop's offset-decode section.
const PRELOAD_SLACK: u32 = 1;

// ---------------------------------------------------------------------------
// Static decode results tables  (mirrors the C arrays of the same name)
// ---------------------------------------------------------------------------

/// `make_decode_table_entry(decode_results, sym, len)` — mirrors the C inline.
///
/// Combines the static part `decode_results[sym]` with the dynamic `len`
/// (remaining codeword length) to form a final decode-table entry.
/// Adding `(len << 8) + len` encodes the length into both byte 0 (consumed
/// bits) and bits 11-8 (remaining codeword length field).
#[inline(always)]
fn make_entry(decode_results: &[u32], sym: usize, len: u32) -> u32 {
    decode_results[sym].wrapping_add((len << 8).wrapping_add(len))
}

/// Static part of precode decode table entries.
/// `precode_decode_results[]` — presym is stored in bits 20-16.
static PRECODE_DECODE_RESULTS: [u32; DEFLATE_NUM_PRECODE_SYMS] = {
    let mut a = [0u32; DEFLATE_NUM_PRECODE_SYMS];
    let mut i = 0usize;
    while i < DEFLATE_NUM_PRECODE_SYMS {
        a[i] = (i as u32) << 16;
        i += 1;
    }
    a
};

/// Static part of litlen decode table entries.
///
/// Mirrors `litlen_decode_results[]`.  Literals set `HUFFDEC_LITERAL | (value << 16)`.
/// EOB sets `HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK`.
/// Lengths encode `(length_base << 16) | num_extra_bits`.
static LITLEN_DECODE_RESULTS: [u32; DEFLATE_NUM_LITLEN_SYMS] = {
    // Literals 0-255
    let mut a = [0u32; DEFLATE_NUM_LITLEN_SYMS];
    let mut i = 0usize;
    while i < 256 {
        a[i] = HUFFDEC_LITERAL | ((i as u32) << 16);
        i += 1;
    }
    // EOB (sym 256)
    a[256] = HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK;
    // Lengths (syms 257-287): (length_base << 16) | num_extra_bits
    // Mirrors the ENTRY(length_base, num_extra_bits) macro.
    #[rustfmt::skip]
    let lengths: [(u32, u32); 29] = [
        (3,   0), (4,   0), (5,   0), (6,   0),
        (7,   0), (8,   0), (9,   0), (10,  0),
        (11,  1), (13,  1), (15,  1), (17,  1),
        (19,  2), (23,  2), (27,  2), (31,  2),
        (35,  3), (43,  3), (51,  3), (59,  3),
        (67,  4), (83,  4), (99,  4), (115, 4),
        (131, 5), (163, 5), (195, 5), (227, 5),
        (258, 0),
    ];
    let mut j = 0usize;
    while j < 29 {
        a[257 + j] = (lengths[j].0 << 16) | lengths[j].1;
        j += 1;
    }
    // Syms 286-287 (extra two entries) — same as last length entry per C source
    // The C array has 31 length entries (257..287 inclusive = 31 entries).
    // Entries 286 and 287 (indices 29 and 30) are ENTRY(258, 0) per the C source.
    // The C macro is `(length_base << 16) | num_extra_bits`; with num_extra_bits == 0
    // the `| 0` is a no-op, so we drop it (byte-identical to ENTRY(258, 0)).
    a[286] = 258u32 << 16;
    a[287] = 258u32 << 16;
    a
};

/// Static part of offset decode table entries.
///
/// Mirrors `offset_decode_results[]`.
/// Format: `(offset_base << 16) | num_extra_bits`.
static OFFSET_DECODE_RESULTS: [u32; DEFLATE_NUM_OFFSET_SYMS] = {
    #[rustfmt::skip]
    let entries: [(u32, u32); 32] = [
        (1,      0),  (2,      0),  (3,      0),  (4,      0),
        (5,      1),  (7,      1),  (9,      2),  (13,     2),
        (17,     3),  (25,     3),  (33,     4),  (49,     4),
        (65,     5),  (97,     5),  (129,    6),  (193,    6),
        (257,    7),  (385,    7),  (513,    8),  (769,    8),
        (1025,   9),  (1537,   9),  (2049,  10),  (3073,  10),
        (4097,  11),  (6145,  11),  (8193,  12),  (12289, 12),
        (16385, 13),  (24577, 13),  (24577, 13),  (24577, 13),
    ];
    let mut a = [0u32; DEFLATE_NUM_OFFSET_SYMS];
    let mut i = 0usize;
    while i < 32 {
        a[i] = (entries[i].0 << 16) | entries[i].1;
        i += 1;
    }
    a
};

// ---------------------------------------------------------------------------
// Decompressor struct  (mirrors `struct libdeflate_decompressor`)
// ---------------------------------------------------------------------------

/// Holds the Huffman decode tables between blocks.
///
/// In libdeflate the struct contains a union to alias precode / litlen storage.
/// In Rust we use separate arrays (no union needed; the lifetimes are clear).
///
/// Fields mirror the C struct layout exactly.
pub struct Decompressor {
    /// Precode codeword lengths (19 bytes).  Mirrors `u.precode_lens[]`.
    precode_lens: [u8; DEFLATE_NUM_PRECODE_SYMS],

    /// Combined litlen + offset + overrun codeword length array.
    /// Mirrors `u.l.lens[]`. C sizes this exactly
    /// `DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS +
    /// DEFLATE_MAX_LENS_OVERRUN` (`deflate_decompress.c:657-659`) -- the
    /// port previously had a `+ 1` here with no cited reason; removed (dead
    /// padding, never written or read; confirmed by the full test suite and
    /// `cargo xtask parity-deflate` staying green, which would immediately
    /// surface an out-of-bounds write in a debug build if that byte were
    /// actually load-bearing).
    lens: [u8; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS + DEFLATE_MAX_LENS_OVERRUN as usize],

    /// Precode decode table.  Mirrors `u.l.precode_decode_table[]`.
    precode_decode_table: [u32; PRECODE_ENOUGH],

    /// Litlen decode table.  Mirrors `u.litlen_decode_table[]`.
    litlen_decode_table: [u32; LITLEN_ENOUGH],

    /// Offset decode table.  Mirrors `offset_decode_table[]`.
    offset_decode_table: [u32; OFFSET_ENOUGH],

    /// Temp sort buffer, reused each call.  Mirrors `sorted_syms[]`.
    sorted_syms: [u16; DEFLATE_MAX_NUM_SYMS],

    /// True when static Huffman tables are already built.
    /// Mirrors `static_codes_loaded`.
    static_codes_loaded: bool,

    /// Actual table_bits used for the litlen table (≤ LITLEN_TABLEBITS).
    /// Mirrors `litlen_tablebits`.
    litlen_tablebits: u32,
}

impl Decompressor {
    /// Create a new, zeroed decompressor.  Mirrors `libdeflate_alloc_decompressor`.
    pub fn new() -> Self {
        Self {
            precode_lens: [0; DEFLATE_NUM_PRECODE_SYMS],
            lens: [0; DEFLATE_NUM_LITLEN_SYMS
                + DEFLATE_NUM_OFFSET_SYMS
                + DEFLATE_MAX_LENS_OVERRUN as usize],
            precode_decode_table: [0; PRECODE_ENOUGH],
            litlen_decode_table: [0; LITLEN_ENOUGH],
            offset_decode_table: [0; OFFSET_ENOUGH],
            sorted_syms: [0; DEFLATE_MAX_NUM_SYMS],
            static_codes_loaded: false,
            litlen_tablebits: 0,
        }
    }
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// build_decode_table  (mirrors the C function of the same name)
// ---------------------------------------------------------------------------

/// Build a Huffman decode table from codeword lengths.
///
/// Returns `true` on success, `false` if the lengths do not form a valid code.
///
/// Mirrors `build_decode_table()` in `deflate_decompress.c` 1:1, including
/// the incremental-doubling fill strategy and the subtable allocation logic.
///
/// # Parameters
/// - `decode_table`: output table (slice of at least `1 << table_bits` entries,
///   plus room for subtables up to `ENOUGH`).
/// - `lens`: codeword lengths (one per symbol, 0 = unused).
/// - `num_syms`: number of symbols.
/// - `decode_results`: static decode-result halves (see `LITLEN_DECODE_RESULTS` etc.).
/// - `table_bits`: log2 of main table size (may be reduced if `table_bits_ret` is Some).
/// - `max_codeword_len`: maximum allowed codeword length.
/// - `sorted_syms`: scratch buffer (at least `num_syms` entries).
/// - `table_bits_ret`: if `Some`, dynamic table_bits is enabled; the actual
///   value used is stored here.
//
// The 8-parameter signature mirrors the C `build_decode_table()` 1:1
// (decode_table, lens, num_syms, decode_results, table_bits, max_codeword_len,
// sorted_syms, table_bits_ret). Keeping the exact param list preserves the
// 1:1 port mapping, so the lint is intentionally allowed here.
#[allow(clippy::too_many_arguments)]
fn build_decode_table(
    decode_table: &mut [u32],
    lens: &[u8],
    num_syms: usize,
    decode_results: &[u32],
    mut table_bits: u32,
    mut max_codeword_len: u32,
    sorted_syms: &mut [u16],
    table_bits_ret: Option<&mut u32>,
) -> bool {
    // Count codewords at each length (0..=max_codeword_len).
    let mut len_counts = [0u32; DEFLATE_MAX_CODEWORD_LEN as usize + 1];
    for &l in &lens[..num_syms] {
        len_counts[l as usize] += 1;
    }

    // Shrink max_codeword_len to the actual maximum used length.
    while max_codeword_len > 1 && len_counts[max_codeword_len as usize] == 0 {
        max_codeword_len -= 1;
    }
    if let Some(ret) = table_bits_ret {
        table_bits = table_bits.min(max_codeword_len);
        *ret = table_bits;
    }

    // Sort symbols by (length, symbol value) via counting sort.
    // Also compute codespace_used = sum_len (len_counts[len] * (1 << (max_codeword_len - len))).
    let mut offsets = [0u32; DEFLATE_MAX_CODEWORD_LEN as usize + 1];
    offsets[0] = 0;
    offsets[1] = len_counts[0];
    let mut codespace_used: u32 = 0;
    for len in 1..max_codeword_len as usize {
        offsets[len + 1] = offsets[len] + len_counts[len];
        // Shift codespace_used left (doubling) and add new codewords at this length.
        codespace_used = codespace_used.wrapping_shl(1).wrapping_add(len_counts[len]);
    }
    codespace_used = codespace_used
        .wrapping_shl(1)
        .wrapping_add(len_counts[max_codeword_len as usize]);

    // Place each symbol into its sorted position.
    {
        let mut off_copy = offsets;
        for (sym, &len) in lens[..num_syms].iter().enumerate() {
            let l = len as usize;
            sorted_syms[off_copy[l] as usize] = sym as u16;
            off_copy[l] += 1;
        }
    }

    // Skip unused symbols (len == 0).
    // In C, sorted_syms is advanced by offsets[0] AFTER the counting-sort pass
    // has mutated offsets[]. In Rust we use a separate off_copy, so offsets[0]
    // is still 0. Use len_counts[0] directly — the number of zero-length symbols.
    let sorted_start = len_counts[0] as usize;

    // Check for overfull code.
    if codespace_used > (1u32 << max_codeword_len) {
        return false;
    }

    // Incomplete code: only valid for 0 or 1 codewords.
    if codespace_used < (1u32 << max_codeword_len) {
        let sym: usize = if codespace_used == 0 {
            0 // arbitrary
        } else {
            // Exactly one codeword of length 1 is the only valid case.
            if codespace_used != (1u32 << (max_codeword_len - 1)) || len_counts[1] != 1 {
                return false;
            }
            sorted_syms[sorted_start] as usize
        };
        let entry = make_entry(decode_results, sym, 1);
        let main_size = 1usize << table_bits;
        for slot in &mut decode_table[..main_size] {
            *slot = entry;
        }
        return true;
    }

    // Complete code: enumerate codewords in lexicographic order and fill the table.
    //
    // The incremental-doubling trick: process lengths 1..=table_bits.
    // At length `len`, the table is treated as having only `2^len` entries,
    // so each codeword uses exactly one slot.  Each time `len` increments,
    // the first half is copied to the second half to double the table size.
    let mut codeword: usize = 0;
    let mut len = 1u32;
    // Find the first non-empty length.
    while len_counts[len as usize] == 0 {
        len += 1;
        if len > max_codeword_len {
            // All symbols have length 0 — shouldn't happen for a complete code,
            // but handle defensively.
            return true;
        }
    }
    let mut count = len_counts[len as usize] as usize;
    let mut cur_table_end: usize = 1 << len;
    let mut sorted_idx = sorted_start;

    while len <= table_bits {
        loop {
            // Fill entry for current codeword.
            decode_table[codeword] =
                make_entry(decode_results, sorted_syms[sorted_idx] as usize, len);
            sorted_idx += 1;

            if codeword == cur_table_end - 1 {
                // Last codeword (all 1s): fill remaining doublings, then done.
                while len < table_bits {
                    decode_table.copy_within(0..cur_table_end, cur_table_end);
                    cur_table_end <<= 1;
                    len += 1;
                }
                return true;
            }

            // Advance to next codeword in bit-reversed order.
            // Equivalent to: flip all bits up to MSB, find the last 0 bit,
            // set it, and clear higher bits. Uses bsr32 on the flipped value.
            let bit = 1usize << bsr32((codeword ^ (cur_table_end - 1)) as u32) as usize;
            codeword &= bit - 1;
            codeword |= bit;

            count -= 1;
            if count == 0 {
                break;
            }
        }

        // Move to next length, doubling the table.
        loop {
            len += 1;
            if len <= table_bits {
                decode_table.copy_within(0..cur_table_end, cur_table_end);
                cur_table_end <<= 1;
            }
            count = len_counts[len as usize] as usize;
            if count != 0 {
                break;
            }
        }
    }

    // Process codewords with len > table_bits — these need subtables.
    cur_table_end = 1 << table_bits;
    let mut subtable_prefix: usize = usize::MAX; // sentinel "none"
    let mut subtable_start: usize = 0;
    // subtable_bits is always assigned before use in the loop below (when prefix changes).
    let mut subtable_bits: u32;

    loop {
        // Start a new subtable when the main-table prefix changes.
        let prefix = codeword & ((1 << table_bits) - 1);
        if prefix != subtable_prefix {
            subtable_prefix = prefix;
            subtable_start = cur_table_end;

            // Determine how many bits this subtable needs.
            subtable_bits = len - table_bits;
            let mut cs = count as u32;
            while cs < (1u32 << subtable_bits) {
                subtable_bits += 1;
                cs = cs
                    .wrapping_shl(1)
                    .wrapping_add(len_counts[(table_bits + subtable_bits) as usize]);
            }
            cur_table_end = subtable_start + (1 << subtable_bits as usize);

            // Write the main-table pointer entry.
            // Format: bits 30-16 = subtable_start, EXCEPTIONAL|SUBTABLE_POINTER,
            //         bits 11-8 = subtable_bits, bits 3-0 = table_bits.
            decode_table[subtable_prefix] = ((subtable_start as u32) << 16)
                | HUFFDEC_EXCEPTIONAL
                | HUFFDEC_SUBTABLE_POINTER
                | (subtable_bits << 8)
                | table_bits;
        }

        // Fill subtable entries for this codeword.
        let entry = make_entry(
            decode_results,
            sorted_syms[sorted_idx] as usize,
            len - table_bits,
        );
        sorted_idx += 1;
        let mut i = subtable_start + (codeword >> table_bits);
        let stride = 1usize << (len - table_bits);
        while i < cur_table_end {
            decode_table[i] = entry;
            i += stride;
        }

        // Advance to next codeword.
        if codeword == (1usize << len) - 1 {
            return true; // last codeword (all 1s)
        }
        let bit = 1usize << bsr32((codeword ^ ((1 << len) - 1)) as u32) as usize;
        codeword &= bit - 1;
        codeword |= bit;
        count -= 1;
        while count == 0 {
            len += 1;
            count = len_counts[len as usize] as usize;
        }
    }
}

// ---------------------------------------------------------------------------
// Precode / litlen / offset table builders  (thin wrappers like in C)
// ---------------------------------------------------------------------------

/// Build the precode decode table.  Mirrors `build_precode_decode_table()`.
fn build_precode_decode_table(d: &mut Decompressor) -> bool {
    // Temporarily copy precode_lens into a local to avoid borrow conflict.
    let lens_copy: [u8; DEFLATE_NUM_PRECODE_SYMS] = d.precode_lens;
    build_decode_table(
        &mut d.precode_decode_table,
        &lens_copy,
        DEFLATE_NUM_PRECODE_SYMS,
        &PRECODE_DECODE_RESULTS,
        PRECODE_TABLEBITS,
        DEFLATE_MAX_PRE_CODEWORD_LEN,
        &mut d.sorted_syms,
        None,
    )
}

/// Build the litlen decode table.  Mirrors `build_litlen_decode_table()`.
fn build_litlen_decode_table(
    d: &mut Decompressor,
    num_litlen_syms: usize,
) -> bool {
    // Disjoint field borrows (lens / litlen_decode_table / sorted_syms are
    // separate `Decompressor` fields) — no per-block heap allocation needed;
    // the borrow checker accepts direct field projections off the same `&mut
    // d` as independent borrows. Previously cloned `d.lens[..]` into a `Vec`
    // here solely to dodge what turned out to be an unnecessary borrow
    // conflict (this call runs once per DEFLATE block, so the clone was a
    // real per-block heap allocation in a hot path).
    let mut tbr = 0u32;
    let ok = build_decode_table(
        &mut d.litlen_decode_table,
        &d.lens[..num_litlen_syms],
        num_litlen_syms,
        &LITLEN_DECODE_RESULTS,
        LITLEN_TABLEBITS,
        DEFLATE_MAX_LITLEN_CODEWORD_LEN,
        &mut d.sorted_syms,
        Some(&mut tbr),
    );
    d.litlen_tablebits = tbr;
    ok
}

/// Build the offset decode table.  Mirrors `build_offset_decode_table()`.
fn build_offset_decode_table(
    d: &mut Decompressor,
    num_litlen_syms: usize,
    num_offset_syms: usize,
) -> bool {
    // See `build_litlen_decode_table`'s comment: disjoint field borrows,
    // no per-block `Vec` clone needed.
    build_decode_table(
        &mut d.offset_decode_table,
        &d.lens[num_litlen_syms..num_litlen_syms + num_offset_syms],
        num_offset_syms,
        &OFFSET_DECODE_RESULTS,
        OFFSET_TABLEBITS,
        DEFLATE_MAX_OFFSET_CODEWORD_LEN,
        &mut d.sorted_syms,
        None,
    )
}

// ---------------------------------------------------------------------------
// Bitstream helpers (inline, mirror the C macros)
// ---------------------------------------------------------------------------

/// Refill the 64-bit bitbuffer from `input[in_pos..]`.
///
/// Mirrors `REFILL_BITS()`.  In the C code this macro uses branchless word
/// reads on fast-unaligned platforms; here we always use the byte-at-a-time
/// path (correct and sufficient for safety — the compiler inlines it well).
/// Near the end of input we insert zero bytes and track `overread_count`.
///
/// `CONSUMABLE_NBITS` for a 64-bit buffer = 63 - 7 = 56.
#[inline(always)]
fn refill_bits(
    bitbuf: &mut u64,
    bitsleft: &mut u32,
    in_next: &mut usize,
    in_end: usize,
    overread_count: &mut usize,
    input: &[u8],
) -> bool {
    // Need at least 56 consumable bits (= CONSUMABLE_NBITS for 64-bit buffer).
    const CONSUMABLE_NBITS: u32 = 56; // 64 - 1 - 7
    while (*bitsleft & 0xFF) < CONSUMABLE_NBITS {
        if *in_next < in_end {
            *bitbuf |= (input[*in_next] as u64) << (*bitsleft & 0xFF);
            *in_next += 1;
        } else {
            *overread_count += 1;
            if *overread_count > 8 {
                // More than sizeof(bitbuf) overread bytes — real overread.
                return false;
            }
        }
        *bitsleft += 8;
    }
    true
}

/// Refill without checking for end of input (fastloop variant).
/// Mirrors `REFILL_BITS_IN_FASTLOOP()`.
#[inline(always)]
fn refill_bits_fast(
    bitbuf: &mut u64,
    bitsleft: &mut u32,
    in_next: &mut usize,
    input: &[u8],
) {
    // Branchless word-read path: load 8 bytes, shift into bitbuf.
    // Mirrors REFILL_BITS_BRANCHLESS().
    // We are guaranteed in_next + 8 <= in_fastloop_end in the fastloop.
    const MAX_BITSLEFT: u32 = 63; // BITBUF_NBITS - 1
    let word = get_unaligned_le64(&input[*in_next..]);
    *bitbuf |= word << (*bitsleft as u8); // low 6 bits of bitsleft
    *in_next += 7 - ((*bitsleft >> 3) & 7) as usize; // advance by bytes loaded
    *bitsleft |= MAX_BITSLEFT & !7; // set bits 3..5, keep bits 0..2
}

/// Extract `count` bits from `word`, masking with `BITMASK(count)`.
/// Mirrors `EXTRACT_VARBITS(word, count)`.
#[inline(always)]
fn extract_varbits(word: u64, count: u32) -> u64 {
    word & bitmask(count)
}

/// Extract `(u8)count` bits from `word`.
/// Mirrors `EXTRACT_VARBITS8(word, count)`.
#[inline(always)]
fn extract_varbits8(word: u64, count: u32) -> u64 {
    word & bitmask(count & 0xFF)
}

/// Build a bitmask of `n` bits: `(1 << n) - 1`.
/// Mirrors the C `BITMASK(n)` macro.
#[inline(always)]
fn bitmask(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// Index into `litlen_decode_table` from the low bits of `bitbuf` masked by
/// `litlen_tablemask`.
///
/// `litlen_tablemask = bitmask(d.litlen_tablebits)` is computed once per block
/// from `d.litlen_tablebits`, a mutable struct field only bounded at runtime
/// (9..=11, see `build_litlen_decode_table`), so LLVM cannot see at compile
/// time that `bitbuf & litlen_tablemask` stays `< LITLEN_ENOUGH` (2342) even
/// though the invariant holds for the whole fastloop/generic loop. The
/// `.min(LITLEN_ENOUGH - 1)` clamp is a true no-op (the real value is always
/// `<= bitmask(11) = 2047 < 2342` by construction) but gives LLVM an explicit,
/// provable upper bound it can use to elide the bounds check on the array
/// access — found while auditing the decompress fastloop's bounds checks;
/// verify empirically via disassembly before trusting
/// this actually helps.
/// `.min(LITLEN_ENOUGH - 1)` was tried against a `& 0x7FF` alternative
/// (the idea: `litlen_tablemask` is always `<= bitmask(11) =
/// 0x7FF` by construction, so ANDing with the compile-time constant `0x7FF`
/// is a true no-op on the value and should be cheaper codegen than a
/// compare-and-conditional-move). Measured via a disasm diff
/// (A/B on the identical
/// `deflate_decompress_impl_bmi2` binary, symbol at the same line both
/// times): `& 0x7FF` was WORSE, not better -- 30 potential-panic call sites
/// (13 panic_bounds_check + 17 slice_index_fail) versus `.min()`'s 27, body
/// 21 lines longer. Reproduced twice (deterministic rebuild). Kept `.min()`
/// as the version proven better; do not "simplify" this back to `& 0x7FF`
/// without re-measuring -- LLVM's bounds-check elimination pass apparently
/// handles the `llvm.umin` intrinsic's range fact better than an AND with a
/// non-constant left-hand mask here, contrary to the intuition that a
/// plain `and` is always cheaper/more provable than `min`.
#[inline(always)]
fn litlen_idx(bitbuf: u64, litlen_tablemask: u64) -> usize {
    ((bitbuf & litlen_tablemask) as usize).min(LITLEN_ENOUGH - 1)
}

// ---------------------------------------------------------------------------
// Public entry point  (mirrors `libdeflate_deflate_decompress`)
// ---------------------------------------------------------------------------

/// Decompress raw DEFLATE data (no zlib header/trailer).
///
/// Returns the number of decompressed bytes written into `out` on success.
///
/// Mirrors `libdeflate_deflate_decompress` / `deflate_decompress_default`.
pub fn deflate_decompress(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<usize, DecompressError> {
    // The non-`_ex` variant discards how many input bytes were consumed
    // (mirrors `libdeflate_deflate_decompress` delegating to `_ex` with a NULL
    // `actual_in_nbytes_ret`).
    deflate_decompress_dispatch(d, input, out, FASTLOOP_ENABLED).map(|(actual_out, _actual_in)| actual_out)
}

/// Decompress raw DEFLATE data into a buffer whose length is the EXACT expected
/// uncompressed size.
///
/// Mirrors `libdeflate_deflate_decompress` called with a NULL
/// `actual_out_nbytes_ret` (`libdeflate.h:222-229`): the stream must fill `out`
/// completely, otherwise [`DecompressError::ShortOutput`] is returned.
pub fn deflate_decompress_exact(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<(), DecompressError> {
    let (actual_out, _actual_in) = deflate_decompress_dispatch(d, input, out, FASTLOOP_ENABLED)?;
    if actual_out != out.len() {
        return Err(DecompressError::ShortOutput);
    }
    Ok(())
}

/// Decompress raw DEFLATE data, additionally reporting how many input bytes
/// were consumed.
///
/// Returns `Ok((actual_out, actual_in))` on success:
/// - `actual_out` — number of decompressed bytes written into `out`.
/// - `actual_in`  — byte offset one past the last consumed DEFLATE byte, i.e.
///   where a following wrapper trailer / next stream member begins. Bytes that
///   were read into the bit buffer but not actually consumed are backed out, so
///   this is the true consumption (not `input.len()`).
///
/// Mirrors `libdeflate_deflate_decompress_ex`.
pub fn deflate_decompress_ex(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
) -> Result<(usize, usize), DecompressError> {
    deflate_decompress_dispatch(d, input, out, FASTLOOP_ENABLED)
}

/// Test-only hook: decompress with the fastloop forced on or off, so the
/// differential test can assert both paths produce byte-identical output
/// (and identical error behaviour) for the same input.
#[cfg(test)]
pub(crate) fn deflate_decompress_ex_with_fastloop(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
    use_fastloop: bool,
) -> Result<(usize, usize), DecompressError> {
    deflate_decompress_dispatch(d, input, out, use_fastloop)
}

/// Route every entry point through the BMI2 runtime dispatch,
/// mirroring `arch_select_decompress_func` (`x86/decompress_impl.h`).
///
/// `deflate_decompress_impl` is the single source of truth either way; the
/// BMI2 path (`deflate_decompress_bmi2` module) is the SAME function, just
/// compiled under `#[target_feature(enable = "bmi2")]` so LLVM can select the
/// `bzhi` instruction for `extract_varbits`/`extract_varbits8`'s `word &
/// BITMASK(count)` pattern -- see that module's doc comment for the full
/// safety/equivalence argument.
#[inline]
fn deflate_decompress_dispatch(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
    use_fastloop: bool,
) -> Result<(usize, usize), DecompressError> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::cpu::bmi2_decompress_available() {
            // SAFETY: `bmi2_decompress_available()` only returns true after
            // `is_x86_feature_detected!("bmi2")` succeeded, so the
            // `#[target_feature(enable = "bmi2")]` precondition holds.
            return unsafe {
                crate::deflate_decompress_bmi2::deflate_decompress_impl_bmi2(
                    d,
                    input,
                    out,
                    use_fastloop,
                )
            };
        }
    }
    deflate_decompress_impl(d, input, out, use_fastloop)
}

// ---------------------------------------------------------------------------
// Core decompression loop  (mirrors decompress_template.h FUNCNAME body)
// ---------------------------------------------------------------------------

/// The actual DEFLATE decompression routine.
///
/// Ported 1:1 from `decompress_template.h`.  The C source is a macro-template
/// compiled into `deflate_decompress_default` (and optionally a BMI2 variant).
/// We implement the single portable path here.
///
/// `use_fastloop` gates the fastloop optimisation (see `FASTLOOP_ENABLED`);
/// the generic loop is always the byte-exact reference path.
///
/// `pub(crate)` + `#[inline(always)]` so `deflate_decompress_bmi2`'s
/// `#[target_feature(enable = "bmi2")]` wrapper can call AND INLINE this body
/// -- required for LLVM to actually recompile it under BMI2 codegen (a plain,
/// non-inlined call from a `#[target_feature]` function does not propagate
/// the target feature to the callee).
#[inline(always)]
pub(crate) fn deflate_decompress_impl(
    d: &mut Decompressor,
    input: &[u8],
    out: &mut [u8],
    use_fastloop: bool,
) -> Result<(usize, usize), DecompressError> {
    // Shorthand for the bad-data error path (mirrors SAFETY_CHECK).
    macro_rules! safety_check {
        ($cond:expr) => {
            if !($cond) {
                return Err(DecompressError::BadData);
            }
        };
    }
    macro_rules! bad_data {
        () => {
            return Err(DecompressError::BadData)
        };
    }

    let in_end = input.len();
    // Fastloop safe zone: enough bytes remain for a full branchless word refill
    // + the maximum bytes read per iteration.
    // FASTLOOP_MAX_BYTES_READ = DIV_ROUND_UP(63 + 2*11 + 20 + 21, 8) + 8 = 18 + 8 = 26
    let fastloop_max_bytes_read: usize = {
        let bits = 63u32 + 2 * LITLEN_TABLEBITS + LENGTH_MAXBITS + OFFSET_MAXBITS;
        (bits as usize).div_ceil(8) + 8
    };
    let in_fastloop_end = in_end.saturating_sub(fastloop_max_bytes_read);

    // FASTLOOP_MAX_BYTES_WRITTEN = 2 + 258 + (5*8) - 1 = 299
    let fastloop_max_bytes_written: usize = 2 + 258 + (5 * 8) - 1;
    let out_len = out.len();
    let out_fastloop_end = out_len.saturating_sub(fastloop_max_bytes_written);

    // Bitstream state (mirrors: in_next, bitbuf, bitsleft, overread_count).
    let mut in_pos: usize = 0;
    let mut bitbuf: u64 = 0;
    let mut bitsleft: u32 = 0;
    let mut overread_count: usize = 0;

    let mut out_pos: usize = 0;
    let mut is_final_block: bool;

    // -----------------------------------------------------------------------
    // Block loop  (mirrors `next_block:` label)
    // -----------------------------------------------------------------------
    'next_block: loop {
        // STATIC_ASSERT(CAN_CONSUME(1 + 2 + 5 + 5 + 4 + 3)) — 20 bits, fits in 56.
        if !refill_bits(
            &mut bitbuf,
            &mut bitsleft,
            &mut in_pos,
            in_end,
            &mut overread_count,
            input,
        ) {
            bad_data!();
        }

        // BFINAL: bit 0
        is_final_block = (bitbuf & 1) != 0;
        // BTYPE: bits 2-1
        let block_type = (bitbuf >> 1) & 3;

        // Engagement probe: one hit per DEFLATE block header actually
        // reached here -- proves a multi-block stream really decoded more
        // than one block (a `fastloop_hit_eob` regression could silently
        // stop after the first block; see `probe::Probe::BlocksDecoded`'s
        // own doc).
        #[cfg(any(test, feature = "probe"))]
        crate::probe::hit(crate::probe::Probe::BlocksDecoded);

        // -------------------------------------------------------------------
        // Decode block header and (for Huffman blocks) build decode tables.
        // -------------------------------------------------------------------
        let num_litlen_syms: usize;
        let num_offset_syms: usize;

        if block_type == DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN as u64 {
            // ------------------------------------------------------------------
            // Dynamic Huffman block header
            // ------------------------------------------------------------------
            // Precode lengths permutation — order they appear in the stream.
            const PRECODE_PERM: [usize; DEFLATE_NUM_PRECODE_SYMS] = [
                16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
            ];

            // HLIT (5 bits, offset 3), HDIST (5 bits, offset 8), HCLEN (4 bits, offset 13).
            // Mirrors: num_litlen_syms = 257 + ((bitbuf >> 3) & BITMASK(5))
            //          num_offset_syms = 1   + ((bitbuf >> 8) & BITMASK(5))
            //          num_explicit   = 4   + ((bitbuf >> 13) & BITMASK(4))
            num_litlen_syms = 257 + ((bitbuf >> 3) & 0x1F) as usize;
            num_offset_syms = 1 + ((bitbuf >> 8) & 0x1F) as usize;
            let num_explicit_precode_lens = 4 + ((bitbuf >> 13) & 0xF) as usize;

            d.static_codes_loaded = false;

            // Consume the 3-field header (BFINAL + BTYPE + HLIT + HDIST + HCLEN = 17 bits).
            // Then read the first precode length merged with the 17-bit consume.
            // The C code reads precode_lens[perm[0]] from bits 17-19.
            let first_prelen = ((bitbuf >> 17) & 7) as u8;
            bitbuf >>= 20;
            bitsleft -= 20;

            if !refill_bits(
                &mut bitbuf,
                &mut bitsleft,
                &mut in_pos,
                in_end,
                &mut overread_count,
                input,
            ) {
                bad_data!();
            }

            d.precode_lens[PRECODE_PERM[0]] = first_prelen;
            // `perm` is the destination index into precode_lens (the permutation
            // slot), read in stream order — mirrors the C `precode_lens[perm[i]]`.
            for &perm in &PRECODE_PERM[1..num_explicit_precode_lens] {
                if (bitsleft & 0xFF) < 3
                    && !refill_bits(
                        &mut bitbuf,
                        &mut bitsleft,
                        &mut in_pos,
                        in_end,
                        &mut overread_count,
                        input,
                    )
                {
                    bad_data!();
                }
                d.precode_lens[perm] = (bitbuf & 7) as u8;
                bitbuf >>= 3;
                bitsleft -= 3;
            }
            // Zero out the remaining precode lengths not explicitly given.
            for &perm in &PRECODE_PERM[num_explicit_precode_lens..DEFLATE_NUM_PRECODE_SYMS] {
                d.precode_lens[perm] = 0;
            }

            // Build the precode decode table.
            safety_check!(build_precode_decode_table(d));

            // Decode litlen + offset codeword lengths using the precode.
            let total_lens = num_litlen_syms + num_offset_syms;
            let mut i = 0usize;
            while i < total_lens {
                if (bitsleft & 0xFF) < DEFLATE_MAX_PRE_CODEWORD_LEN + 7
                    && !refill_bits(
                        &mut bitbuf,
                        &mut bitsleft,
                        &mut in_pos,
                        in_end,
                        &mut overread_count,
                        input,
                    )
                {
                    bad_data!();
                }

                // Decode one precode symbol.
                let entry = d.precode_decode_table
                    [(bitbuf & bitmask(DEFLATE_MAX_PRE_CODEWORD_LEN)) as usize];
                bitbuf >>= entry & 0xFF;
                bitsleft -= entry & 0xFF; // consume only the actual codeword bits (low byte of entry)
                let presym = (entry >> 16) as usize;

                if presym < 16 {
                    // Direct length.
                    d.lens[i] = presym as u8;
                    i += 1;
                    continue;
                }

                // Run-length encoded lengths.
                // For presym 16 and 17, the C code always writes the maximum
                // number of entries (6 and 10 respectively), then adjusts `i`.
                if presym == 16 {
                    // Repeat previous length 3-6 times.
                    safety_check!(i != 0);
                    let rep_val = d.lens[i - 1];
                    let rep_count = 3 + (bitbuf & 3) as usize; // BITMASK(2)
                    bitbuf >>= 2;
                    bitsleft -= 2;
                    // Always write 6, advance by actual count (3..=6).
                    d.lens[i] = rep_val;
                    d.lens[i + 1] = rep_val;
                    d.lens[i + 2] = rep_val;
                    d.lens[i + 3] = rep_val;
                    d.lens[i + 4] = rep_val;
                    d.lens[i + 5] = rep_val;
                    i += rep_count;
                } else if presym == 17 {
                    // Repeat zero 3-10 times.
                    let rep_count = 3 + (bitbuf & 7) as usize; // BITMASK(3)
                    bitbuf >>= 3;
                    bitsleft -= 3;
                    // Always write 10 zeros, advance by actual count (3..=10).
                    d.lens[i] = 0;
                    d.lens[i + 1] = 0;
                    d.lens[i + 2] = 0;
                    d.lens[i + 3] = 0;
                    d.lens[i + 4] = 0;
                    d.lens[i + 5] = 0;
                    d.lens[i + 6] = 0;
                    d.lens[i + 7] = 0;
                    d.lens[i + 8] = 0;
                    d.lens[i + 9] = 0;
                    i += rep_count;
                } else {
                    // presym == 18: repeat zero 11-138 times.
                    let rep_count = 11 + (bitbuf & 127) as usize; // BITMASK(7)
                    bitbuf >>= 7;
                    bitsleft -= 7;
                    let end = i + rep_count;
                    for slot in &mut d.lens[i..end] {
                        *slot = 0;
                    }
                    i += rep_count;
                }
            }
            // Consistency check: exactly `total_lens` symbols decoded.
            safety_check!(i == total_lens);
        } else if block_type == DEFLATE_BLOCKTYPE_UNCOMPRESSED as u64 {
            // ------------------------------------------------------------------
            // Stored (uncompressed) block
            // ------------------------------------------------------------------
            // Consume BFINAL + BTYPE (3 bits already in bitbuf).
            bitsleft -= 3;
            bitsleft &= 0xFF; // mask to low byte

            // Rewind `in_pos` by bytes refilled but not consumed.
            safety_check!(overread_count <= (bitsleft >> 3) as usize);
            in_pos -= (bitsleft >> 3) as usize - overread_count;
            overread_count = 0;
            bitbuf = 0;
            bitsleft = 0;

            // Read LEN + NLEN (4 bytes).
            safety_check!(in_end - in_pos >= 4);
            let len = get_unaligned_le16(&input[in_pos..]) as usize;
            let nlen = get_unaligned_le16(&input[in_pos + 2..]) as usize;
            in_pos += 4;

            // Verify LEN ^ NLEN complement.
            safety_check!(len == (!nlen & 0xFFFF));

            // Bounds checks.
            if len > out_len - out_pos {
                return Err(DecompressError::InsufficientSpace);
            }
            safety_check!(len <= in_end - in_pos);

            // Copy raw bytes.
            out[out_pos..out_pos + len].copy_from_slice(&input[in_pos..in_pos + len]);
            in_pos += len;
            out_pos += len;

            if is_final_block {
                // Stored blocks reset the bit buffer and advance `in_pos` byte
                // by byte, so `in_pos` already points exactly one past the last
                // consumed byte — no readahead to back out.
                return Ok((out_pos, in_pos));
            }
            continue 'next_block;
        } else {
            // ------------------------------------------------------------------
            // Static Huffman block
            // ------------------------------------------------------------------
            safety_check!(block_type == DEFLATE_BLOCKTYPE_STATIC_HUFFMAN as u64);

            bitbuf >>= 3; // consume BFINAL + BTYPE
            bitsleft -= 3;

            if !d.static_codes_loaded {
                d.static_codes_loaded = true;

                // Static litlen codeword lengths (DEFLATE spec, section 3.2.6).
                for l in &mut d.lens[..144] {
                    *l = 8;
                }
                for l in &mut d.lens[144..256] {
                    *l = 9;
                }
                for l in &mut d.lens[256..280] {
                    *l = 7;
                }
                for l in &mut d.lens[280..288] {
                    *l = 8;
                }
                // Static offset lengths: all 5 bits.
                for l in &mut d.lens[288..288 + 32] {
                    *l = 5;
                }
            }

            num_litlen_syms = 288;
            num_offset_syms = 32;

            // Build decode tables (shared path with dynamic block below).
            safety_check!(build_offset_decode_table(d, num_litlen_syms, num_offset_syms));
            safety_check!(build_litlen_decode_table(d, num_litlen_syms));
        }

        // Only static/dynamic Huffman blocks reach here.
        if block_type != DEFLATE_BLOCKTYPE_STATIC_HUFFMAN as u64 {
            // Dynamic block: build tables now (static built them above already).
            safety_check!(build_offset_decode_table(d, num_litlen_syms, num_offset_syms));
            safety_check!(build_litlen_decode_table(d, num_litlen_syms));
        }

        // have_decode_tables:
        let litlen_tablemask: u64 = bitmask(d.litlen_tablebits);

        // ------------------------------------------------------------------
        // Fastloop  (mirrors the `do { ... } while (...)` in the template)
        //
        // NOTE: The fastloop is an optimisation path.  The generic loop below
        // is the reference implementation; both produce byte-identical output
        // (verified by `fastloop_matches_generic_loop_on_varied_inputs` in the
        // test module). Gated by `use_fastloop` (see `FASTLOOP_ENABLED`'s
        // doc-comment). The C entry condition is `in_next < in_fastloop_end &&
        // out_next < out_fastloop_end` (template.h: jump to generic_loop
        // otherwise) — reproduced here, prefixed by the runtime gate.
        // ------------------------------------------------------------------
        // Set when the fastloop consumes a genuine end-of-block entry (mirrors
        // C's `goto block_done`, which jumps straight past the generic loop).
        // A `break 'fastloop` for any other reason (bounds exhausted) must
        // still fall through into the generic loop, exactly like C falling
        // off the bottom of the `do { ... } while (...)` into `generic_loop:`.
        let mut fastloop_hit_eob = false;

        if use_fastloop && in_pos < in_fastloop_end && out_pos < out_fastloop_end {
            // The fastloop's unchecked `out[..]` writes
            // (literals, match copies) are sound only because entry here
            // guarantees at least `fastloop_max_bytes_written` bytes remain
            // past `out_pos` — this is the exact margin `out_fastloop_end`
            // was computed from. Restated as a live debug-build check (not
            // just a comment) so a future refactor that breaks the margin
            // arithmetic fails loudly here instead of silently corrupting
            // memory beyond `out` on hostile input.
            debug_assert!(
                out_pos + fastloop_max_bytes_written <= out_len,
                "fastloop entered without its required output margin: out_pos={out_pos} \
                 + fastloop_max_bytes_written={fastloop_max_bytes_written} > out_len={out_len}"
            );

            refill_bits_fast(&mut bitbuf, &mut bitsleft, &mut in_pos, input);
            let mut entry = d.litlen_decode_table[litlen_idx(bitbuf, litlen_tablemask)];

            'fastloop: loop {
                // Engagement probe: one hit per fastloop ITERATION --
                // proves the fastloop's body actually ran at least once on
                // this input, not just that `FASTLOOP_ENABLED` is `true`
                // (see `probe::Probe::FastloopIters`'s own doc).
                #[cfg(any(test, feature = "probe"))]
                crate::probe::hit(crate::probe::Probe::FastloopIters);

                // saved_bitbuf always holds bitbuf from BEFORE the most recent
                // entry consume — mirrors C's saved_bitbuf pattern exactly.
                let mut saved_bitbuf: u64;

                // Consume litlen entry (primary item).
                saved_bitbuf = bitbuf;
                bitbuf >>= entry & 0xFF;
                bitsleft -= entry & 0xFF;

                if entry & HUFFDEC_LITERAL != 0 {
                    // 1st fast literal.
                    let lit = (entry >> 16) as u8;
                    entry = d.litlen_decode_table[litlen_idx(bitbuf, litlen_tablemask)];
                    saved_bitbuf = bitbuf; // update before 2nd consume
                    bitbuf >>= entry & 0xFF;
                    bitsleft -= entry & 0xFF;
                    out[out_pos] = lit;
                    out_pos += 1;

                    if entry & HUFFDEC_LITERAL != 0 {
                        // 2nd fast literal.
                        let lit2 = (entry >> 16) as u8;
                        entry = d.litlen_decode_table[litlen_idx(bitbuf, litlen_tablemask)];
                        saved_bitbuf = bitbuf; // update before 3rd consume
                        bitbuf >>= entry & 0xFF;
                        bitsleft -= entry & 0xFF;
                        out[out_pos] = lit2;
                        out_pos += 1;

                        if entry & HUFFDEC_LITERAL != 0 {
                            // 3rd fast literal (primary item — restart loop).
                            let lit3 = (entry >> 16) as u8;
                            entry = d.litlen_decode_table[litlen_idx(bitbuf, litlen_tablemask)];
                            refill_bits_fast(&mut bitbuf, &mut bitsleft, &mut in_pos, input);
                            out[out_pos] = lit3;
                            out_pos += 1;

                            if in_pos >= in_fastloop_end || out_pos >= out_fastloop_end {
                                break 'fastloop;
                            }
                            continue 'fastloop;
                        }
                        // 3rd entry is not a literal — fall through to length/eob/subtable
                        // saved_bitbuf was updated to hold bitbuf before 3rd consume ✓
                    } else {
                        // 2nd entry is not a literal — fall through to length/eob/subtable
                        // saved_bitbuf was updated to hold bitbuf before 2nd consume ✓
                    }
                }

                // Exceptional: end-of-block or subtable pointer.
                if entry & HUFFDEC_EXCEPTIONAL != 0 {
                    if entry & HUFFDEC_END_OF_BLOCK != 0 {
                        fastloop_hit_eob = true;
                        break 'fastloop;
                    }
                    // Subtable pointer — follow into subtable.
                    let sub_idx = (entry >> 16) as usize
                        + extract_varbits(bitbuf, (entry >> 8) & 0x3F) as usize;
                    entry = d.litlen_decode_table[sub_idx];
                    saved_bitbuf = bitbuf; // update before subtable entry consume
                    bitbuf >>= entry & 0xFF;
                    bitsleft -= entry & 0xFF;

                    // NOTE: C only refills here when
                    // `!CAN_CONSUME_AND_THEN_PRELOAD(DEFLATE_MAX_LITLEN_CODEWORD_LEN,
                    // LITLEN_TABLEBITS) || !CAN_CONSUME_AND_THEN_PRELOAD(LENGTH_MAXBITS,
                    // OFFSET_TABLEBITS)` — both are compile-time true for our fixed
                    // 64-bit bitbuf / table-bit constants, so C's refill here is dead
                    // code on this target and must NOT be ported unconditionally (doing
                    // so over-consumes `in_next`, corrupting the stream).

                    if entry & HUFFDEC_LITERAL != 0 {
                        let lit = (entry >> 16) as u8;
                        entry = d.litlen_decode_table[litlen_idx(bitbuf, litlen_tablemask)];
                        refill_bits_fast(&mut bitbuf, &mut bitsleft, &mut in_pos, input);
                        out[out_pos] = lit;
                        out_pos += 1;

                        if in_pos >= in_fastloop_end || out_pos >= out_fastloop_end {
                            break 'fastloop;
                        }
                        continue 'fastloop;
                    }
                    if entry & HUFFDEC_END_OF_BLOCK != 0 {
                        fastloop_hit_eob = true;
                        break 'fastloop;
                    }
                    // Length via subtable — fall through to match decode below.
                    // saved_bitbuf was updated to bitbuf before subtable entry consume ✓
                }

                // Decode match length.
                // entry & 0xFF = codeword_len + extra_len_bits (total consumed).
                // (entry >> 8) & 0xFF = codeword_len (shift to expose extra bits).
                // saved_bitbuf holds bitbuf from before the length entry was consumed,
                // so its low bits are [codeword_bits | extra_bits | ...].
                let length_base = entry >> 16;
                let extra = (extract_varbits8(saved_bitbuf, entry) >> ((entry >> 8) & 0xFF)) as u32;
                let length = (length_base + extra) as usize;

                // Decode match offset — preload from OFFSET_TABLEBITS bits.
                // STATIC_ASSERT(CAN_CONSUME_AND_THEN_PRELOAD(LENGTH_MAXFASTBITS, OFFSET_TABLEBITS)) — true on 64-bit.
                entry = d.offset_decode_table[(bitbuf & bitmask(OFFSET_TABLEBITS)) as usize];

                // On 64-bit we can always decode offset + preload litlen without overflow.
                // Possibly one refill needed.
                if entry & HUFFDEC_EXCEPTIONAL != 0 {
                    // Offset codeword needs a subtable.
                    if (bitsleft & 0xFF) < OFFSET_MAXBITS + LITLEN_TABLEBITS - PRELOAD_SLACK {
                        refill_bits_fast(&mut bitbuf, &mut bitsleft, &mut in_pos, input);
                    }
                    bitbuf >>= OFFSET_TABLEBITS;
                    bitsleft -= OFFSET_TABLEBITS;
                    let sub_idx = (entry >> 16) as usize
                        + extract_varbits(bitbuf, (entry >> 8) & 0x3F) as usize;
                    entry = d.offset_decode_table[sub_idx];
                } else if (bitsleft & 0xFF) < OFFSET_MAXFASTBITS + LITLEN_TABLEBITS - PRELOAD_SLACK {
                    refill_bits_fast(&mut bitbuf, &mut bitsleft, &mut in_pos, input);
                }

                let saved_off = bitbuf;
                bitbuf >>= entry & 0xFF;
                bitsleft -= entry & 0xFF;
                let offset_base = entry >> 16;
                let off_extra =
                    (extract_varbits8(saved_off, entry) >> ((entry >> 8) & 0xFF)) as u32;
                let offset = (offset_base + off_extra) as usize;

                // Validate offset.
                safety_check!(offset <= out_pos);

                // Preload next litlen entry + refill bitbuf.
                entry = d.litlen_decode_table[litlen_idx(bitbuf, litlen_tablemask)];
                refill_bits_fast(&mut bitbuf, &mut bitsleft, &mut in_pos, input);

                // Copy match: word-at-a-time (safe unaligned LE u64 load/store),
                // mirroring libdeflate's fastloop copy (decompress_template.h,
                // the `UNALIGNED_ACCESS_IS_FAST` branches). The output buffer's
                // fastloop safe-zone margin (`fastloop_max_bytes_written`)
                // guarantees writing up to 5 extra words past `dst_end` never
                // runs past `out.len()` — overrunning the exact match length is
                // intentional here, not a bug. `src < dst` always (offset >= 1),
                // so whenever the write range is in bounds the read range
                // (`src..src+8`) is automatically in bounds too.
                let dst_end = out_pos + length;
                let mut src = out_pos - offset;
                let mut dst = out_pos;
                out_pos = dst_end;
                if offset >= 8 {
                    // Ordinary case: non-overlapping-in-word-size copy.
                    for _ in 0..5 {
                        put_unaligned_le64(get_unaligned_le64(&out[src..]), &mut out[dst..]);
                        src += 8;
                        dst += 8;
                    }
                    // Tail loop: 5 words per iteration, matching C's `while
                    // (dst < out_next) { store x5; }` shape exactly (not a
                    // 1-word-per-iteration loop with a check after each —
                    // same bytes either way, but this keeps the unroll
                    // factor a 1:1 port instead of an equivalent-but-
                    // differently-shaped rewrite).
                    while dst < dst_end {
                        for _ in 0..5 {
                            put_unaligned_le64(get_unaligned_le64(&out[src..]), &mut out[dst..]);
                            src += 8;
                            dst += 8;
                        }
                    }
                } else if offset == 1 {
                    // RLE of a single repeated byte — broadcast it across a word.
                    let v = 0x0101_0101_0101_0101u64.wrapping_mul(out[src] as u64);
                    for _ in 0..4 {
                        put_unaligned_le64(v, &mut out[dst..]);
                        dst += 8;
                    }
                    // Tail loop: 4 words per iteration, matching C's shape
                    // (see the offset>=8 branch's comment above).
                    while dst < dst_end {
                        for _ in 0..4 {
                            put_unaligned_le64(v, &mut out[dst..]);
                            dst += 8;
                        }
                    }
                } else {
                    // offset in 2..=7: overlapping word copy — each store only
                    // advances by `offset`, so later stores' 8-byte window
                    // legitimately reads bytes this same loop already wrote.
                    put_unaligned_le64(get_unaligned_le64(&out[src..]), &mut out[dst..]);
                    src += offset;
                    dst += offset;
                    put_unaligned_le64(get_unaligned_le64(&out[src..]), &mut out[dst..]);
                    src += offset;
                    dst += offset;
                    loop {
                        put_unaligned_le64(get_unaligned_le64(&out[src..]), &mut out[dst..]);
                        src += offset;
                        dst += offset;
                        put_unaligned_le64(get_unaligned_le64(&out[src..]), &mut out[dst..]);
                        src += offset;
                        dst += offset;
                        if dst >= dst_end {
                            break;
                        }
                    }
                }

                if in_pos >= in_fastloop_end || out_pos >= out_fastloop_end {
                    break 'fastloop;
                }
            } // end 'fastloop
        }

        // ------------------------------------------------------------------
        // Generic loop  (mirrors `generic_loop:` label in the template)
        //
        // Skipped entirely when the fastloop already consumed a genuine
        // end-of-block entry: C's `goto block_done` jumps past this loop, it
        // does not fall into it. Falling through here as an ordinary `if`
        // miss would re-decode the NEXT block's header bits as litlen
        // symbols using the just-finished block's tables — silent corruption.
        // ------------------------------------------------------------------
        if !fastloop_hit_eob {
        loop {
            if !refill_bits(
                &mut bitbuf,
                &mut bitsleft,
                &mut in_pos,
                in_end,
                &mut overread_count,
                input,
            ) {
                bad_data!();
            }

            let mut entry = d.litlen_decode_table[litlen_idx(bitbuf, litlen_tablemask)];
            let mut saved_bitbuf = bitbuf;
            bitbuf >>= entry & 0xFF;
            bitsleft -= entry & 0xFF;

            // Subtable pointer check.
            if entry & HUFFDEC_SUBTABLE_POINTER != 0 {
                let sub_idx = (entry >> 16) as usize
                    + extract_varbits(bitbuf, (entry >> 8) & 0x3F) as usize;
                entry = d.litlen_decode_table[sub_idx];
                saved_bitbuf = bitbuf;
                bitbuf >>= entry & 0xFF;
                bitsleft -= entry & 0xFF;
            }

            let top = (entry >> 16) as usize;

            if entry & HUFFDEC_LITERAL != 0 {
                // Literal.
                if out_pos >= out_len {
                    return Err(DecompressError::InsufficientSpace);
                }
                out[out_pos] = top as u8;
                out_pos += 1;
                continue;
            }
            if entry & HUFFDEC_END_OF_BLOCK != 0 {
                break; // block_done
            }

            // Length.
            let extra_len =
                (extract_varbits8(saved_bitbuf, entry) >> ((entry >> 8) & 0xFF)) as usize;
            let length = top + extra_len;
            if length > out_len - out_pos {
                return Err(DecompressError::InsufficientSpace);
            }

            // Offset decode — may need a refill.
            // CAN_CONSUME(LENGTH_MAXBITS + OFFSET_MAXBITS) = (56 >= 15+5+15+13=48) = true.
            // No extra refill needed on 64-bit.
            let mut entry =
                d.offset_decode_table[(bitbuf & bitmask(OFFSET_TABLEBITS)) as usize];
            if entry & HUFFDEC_EXCEPTIONAL != 0 {
                bitbuf >>= OFFSET_TABLEBITS;
                bitsleft -= OFFSET_TABLEBITS;
                let sub_idx = (entry >> 16) as usize
                    + extract_varbits(bitbuf, (entry >> 8) & 0x3F) as usize;
                entry = d.offset_decode_table[sub_idx];
            }
            let offset_base = entry >> 16;
            let off_extra =
                (extract_varbits8(bitbuf, entry) >> ((entry >> 8) & 0xFF)) as u32;
            let offset = (offset_base + off_extra) as usize;
            bitbuf >>= entry & 0xFF;
            bitsleft -= entry & 0xFF;

            safety_check!(offset <= out_pos);

            // Copy match — always byte-by-byte (handles overlap for offset < length).
            // Mirrors the C generic_loop: *dst++ = *src++ two times, then loop.
            let src_start = out_pos - offset;
            let dst_end = out_pos + length;
            let mut src = src_start;
            let mut dst = out_pos;
            out_pos = dst_end;
            while dst < out_pos {
                let b = out[src];
                out[dst] = b;
                src += 1;
                dst += 1;
            }
        }
        } // end `if !fastloop_hit_eob` (generic loop)

        // block_done:
        if !is_final_block {
            continue 'next_block;
        }
        break;
    }

    // That was the last block.
    bitsleft &= 0xFF;
    // If any overread bytes were consumed (not just buffered), data is bad.
    safety_check!(overread_count <= (bitsleft >> 3) as usize);

    // True input consumption: back out whole bytes pulled into the bit buffer but
    // not actually consumed (mirrors the C
    // `in_next -= (bitsleft >> 3) - overread_count`). The safety_check above
    // guarantees this subtraction does not underflow.
    let actual_in = in_pos - ((bitsleft >> 3) as usize - overread_count);
    Ok((out_pos, actual_in))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: decompress into a fresh Vec.
    fn decomp(compressed: &[u8], out_capacity: usize) -> Result<Vec<u8>, DecompressError> {
        let mut d = Decompressor::new();
        let mut out = vec![0u8; out_capacity];
        let n = deflate_decompress(&mut d, compressed, &mut out)?;
        out.truncate(n);
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Fixtures generated by:
    //   python -c "import zlib; c=zlib.compressobj(9,zlib.DEFLATED,-15); ..."
    // All compressed with Python's stdlib zlib (wbits=-15 = raw DEFLATE, no header).
    // -----------------------------------------------------------------------

    /// Empty input → empty output.
    #[test]
    fn empty() {
        // python: compress_raw(b'')
        let comp: &[u8] = &[3, 0];
        let got = decomp(comp, 0).unwrap();
        assert_eq!(got, b"");
    }

    /// Short string: "Hello, World!"
    #[test]
    fn hello_world() {
        // python: compress_raw(b'Hello, World!')
        let comp: &[u8] = &[
            243, 72, 205, 201, 201, 215, 81, 8, 207, 47, 202, 73, 81, 4, 0,
        ];
        let plain: &[u8] = b"Hello, World!";
        let got = decomp(comp, plain.len()).unwrap();
        assert_eq!(got, plain);
    }

    /// Highly repetitive: b'abcabc' * 200 = 1200 bytes, exercises back-references.
    /// Fixture files generated by `src/testdata/gen_fixtures.py` (see that script).
    #[test]
    fn repetitive_backrefs() {
        let comp: &[u8] = include_bytes!("testdata/repetitive_backrefs.deflate");
        let plain: &[u8] = include_bytes!("testdata/repetitive_backrefs.bin");
        let got = decomp(comp, plain.len()).unwrap();
        assert_eq!(got, plain);
    }

    /// All-same byte run: b'\x42' * 4096 — RLE-like, exercises long matches.
    /// Fixture files generated by `src/testdata/gen_fixtures.py`.
    #[test]
    fn all_same_byte() {
        let comp: &[u8] = include_bytes!("testdata/all_same_byte.deflate");
        let plain: &[u8] = include_bytes!("testdata/all_same_byte.bin");
        let got = decomp(comp, plain.len()).unwrap();
        assert_eq!(got, plain);
    }

    /// Stored (level-0) block: literal copy, no compression.
    #[test]
    fn stored_block() {
        // python: compress_stored(b'stored block data here')  (level=0)
        let comp: &[u8] = &[
            1, 22, 0, 233, 255, 115, 116, 111, 114, 101, 100, 32, 98, 108, 111, 99, 107, 32,
            100, 97, 116, 97, 32, 104, 101, 114, 101,
        ];
        let plain: &[u8] = b"stored block data here";
        let got = decomp(comp, plain.len()).unwrap();
        assert_eq!(got, plain);
    }

    /// ~64 KB of pseudo-random, deterministic data ((i*7+3) mod 256) — exercises
    /// dynamic Huffman + large distances. This was the brittle `large_mixed`
    /// case: the inline byte array had been hand-corrupted (3 extra bytes). The
    /// fixture now lives in committed binary files regenerated by
    /// `src/testdata/gen_fixtures.py`, eliminating the hand-edit hazard.
    #[test]
    fn large_mixed() {
        let comp: &[u8] = include_bytes!("testdata/large_mixed.deflate");
        let plain: &[u8] = include_bytes!("testdata/large_mixed.bin");
        let got = decomp(comp, plain.len()).unwrap();
        assert_eq!(got, plain);
    }

    /// Bad data returns DecompressError::BadData.
    #[test]
    fn bad_data_error() {
        let comp: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF];
        let result = decomp(comp, 64);
        assert!(result.is_err());
    }

    /// `deflate_decompress_ex` reports the exact input consumed even when
    /// trailing bytes follow the raw DEFLATE stream.
    #[test]
    fn ex_reports_consumed_with_trailing_bytes() {
        // Raw DEFLATE of "Hello, World!" (no zlib wrapper) + 5 trailing bytes.
        let mut comp: Vec<u8> = vec![
            243, 72, 205, 201, 201, 215, 81, 8, 207, 47, 202, 73, 81, 4, 0,
        ];
        let stream_len = comp.len();
        comp.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);

        let mut d = Decompressor::new();
        let mut out = vec![0u8; 13];
        let (actual_out, actual_in) = deflate_decompress_ex(&mut d, &comp, &mut out).unwrap();
        assert_eq!(&out[..actual_out], b"Hello, World!");
        // A byte-aligned final block consumes exactly the stream bytes.
        assert_eq!(actual_in, stream_len);
    }

    /// Output buffer too small returns InsufficientSpace.
    #[test]
    fn insufficient_space() {
        let comp: &[u8] = &[
            243, 72, 205, 201, 201, 215, 81, 8, 207, 47, 202, 73, 81, 4, 0,
        ];
        let result = decomp(comp, 4); // "Hello, World!" is 13 bytes, 4 < 13
        assert_eq!(result, Err(DecompressError::InsufficientSpace));
    }

    // -----------------------------------------------------------------------
    // Fastloop vs generic-loop differential test.
    //
    // The fastloop (`use_fastloop = true`) is an optimisation over the
    // generic loop (`use_fastloop = false`): both must produce byte-identical
    // output and identical error behaviour for every input. This guards
    // against exactly the two bugs found while enabling the fastloop:
    //   1. an extra unconditional `REFILL_BITS_IN_FASTLOOP()` after a litlen
    //      subtable consume that C only does on 32-bit / narrow-bitbuf
    //      targets (dead code on our 64-bit config) — over-consumed input.
    //   2. falling through into the generic loop after the fastloop consumed
    //      a genuine end-of-block entry, instead of jumping straight to
    //      `block_done` like C's `goto` — silently re-decoded the next
    //      block's header bits as litlen symbols.
    // -----------------------------------------------------------------------
    use crate::deflate_compress::{Compressor, deflate_compress_bound};

    /// Run the same compressed stream through both decode paths and assert
    /// they agree (`Ok` with identical bytes, or the identical `Err`).
    /// `expected`: when `Some(data)`, additionally assert the decoded bytes
    /// equal the ORIGINAL plaintext, not just that fastloop and generic agree
    /// with each other. Without this, a bug shared by both paths (e.g. a
    /// compressor/decompressor pair that agree on the wrong answer) would
    /// pass silently forever — comparing two implementations to each other is
    /// not the same as comparing either one to ground truth. `None` for
    /// truncated/undersized-output cases, where there is no full plaintext to
    /// compare against (both paths are expected to fail there, or to produce
    /// only a partial prefix).
    fn assert_fastloop_matches_generic(compressed: &[u8], out_cap: usize, expected: Option<&[u8]>) {
        let mut d_fast = Decompressor::new();
        let mut out_fast = vec![0u8; out_cap];
        let r_fast =
            deflate_decompress_ex_with_fastloop(&mut d_fast, compressed, &mut out_fast, true);

        let mut d_generic = Decompressor::new();
        let mut out_generic = vec![0u8; out_cap];
        let r_generic =
            deflate_decompress_ex_with_fastloop(&mut d_generic, compressed, &mut out_generic, false);

        match (r_fast, r_generic) {
            (Ok((n_fast, in_fast)), Ok((n_generic, in_generic))) => {
                assert_eq!(n_fast, n_generic, "output length mismatch");
                assert_eq!(in_fast, in_generic, "consumed-input mismatch");
                assert_eq!(
                    &out_fast[..n_fast],
                    &out_generic[..n_generic],
                    "fastloop and generic loop disagree on decoded bytes"
                );
                if let Some(data) = expected {
                    assert_eq!(
                        &out_fast[..n_fast],
                        data,
                        "both paths agree with each other but NOT with the original plaintext \
                         (ground-truth mismatch)"
                    );
                }
            }
            (Err(e_fast), Err(e_generic)) => {
                assert_eq!(e_fast, e_generic, "error mismatch");
            }
            (fast, generic) => panic!(
                "fastloop and generic loop disagree on success: fast={fast:?} generic={generic:?}"
            ),
        }
    }

    /// Deterministic LCG byte stream (avoids pulling in a `rand` dependency).
    fn lcg_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut rng = seed;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            out.push((rng >> 16) as u8);
        }
        out
    }

    #[test]
    fn fastloop_matches_generic_loop_on_varied_inputs() {
        let mut inputs: Vec<(&'static str, Vec<u8>)> = vec![
            ("empty", vec![]),
            ("one_byte", vec![0x42]),
            ("short", b"hello, world!".to_vec()),
            // Long single-run: forces max-length (258) matches back-to-back,
            // and repeatedly hits the subtable-pointer path (bug #1's site).
            ("all_same_70k", vec![0x5A; 70_000]),
            ("repetitive_12k", b"abcabcabcabc".repeat(1000)),
            // Long enough and varied enough to span multiple DEFLATE blocks
            // at low compression levels, exercising the EOB-inside-fastloop
            // transition to the next block's header (bug #2's exact site).
            ("text_90k", b"The quick brown fox jumps over the lazy dog. ".repeat(2000)),
            ("prng_64k", lcg_bytes(0x1234_5678, 65_536)),
            ("prng_1k", lcg_bytes(0xDEAD_BEEF, 1_024)),
        ];
        // Near-fastloop-boundary sizes: the fastloop's safe-zone margins
        // (`in_fastloop_end` / `out_fastloop_end`) make the last ~300 bytes of
        // output and ~26 bytes of input behave differently (generic loop
        // only); cover lengths just above/below typical margin sizes.
        for len in [0usize, 1, 26, 27, 298, 299, 300, 301, 500] {
            inputs.push((
                "prng_boundary",
                lcg_bytes(0x7777_7777_u32.wrapping_add(len as u32), len),
            ));
        }

        // A fast path can silently stop running while
        // every result-equality assertion below still passes (that is
        // EXACTLY the `fastloop_hit_eob` bug class this test was written to
        // catch, per its own module doc) -- so also assert the relevant
        // PROBE actually fired for these two representative inputs, not just
        // that their output matches the generic loop's.
        //
        // FINDING (verified, not assumed -- the original plan was to
        // "assert iters>0 on text_90k/prng_64k", but `prng_64k` genuinely
        // never engages the fastloop at ANY of the 6 levels tested):
        // `prng_64k`'s bytes (`(rng >> 16) as u8` from a GLIBC-style LCG) are
        // close enough to uniformly distributed that DEFLATE's own cost
        // model correctly picks a STORED block (`block_type == 0`, verified
        // via a temporary debug print of the compressed stream's first
        // block-type bits at every level: always 0) over Huffman coding --
        // stored blocks bypass the Huffman-decode 'fastloop' entirely by
        // construction (there is nothing to decode-loop over, just a raw
        // copy), so `FastloopIters == 0` for this input is CORRECT DEFLATE
        // behaviour, not a fastloop engagement bug. Asserting `iters > 0`
        // here would be a false requirement this crate could never satisfy
        // without breaking parity with real DEFLATE/libdeflate. Checking
        // `BlocksDecoded > 0` instead for `prng_64k` still proves the
        // block-loop probe (and thus the whole decode path) genuinely ran on
        // this input, which is the actual engagement property worth proving
        // for a stored-block input. `text_90k` (highly compressible text)
        // keeps the literal `FastloopIters > 0` check, since it genuinely
        // does engage the Huffman fastloop (confirmed below).
        let mut text_90k_engaged = false;
        let mut prng_64k_engaged = false;

        for level in [0i32, 1, 4, 6, 9, 12] {
            for (name, data) in &inputs {
                let mut c = Compressor::new(level).unwrap();
                let bound = deflate_compress_bound(data.len());
                let mut out = vec![0u8; bound];
                let n = c.deflate_compress(data, &mut out);
                let compressed = &out[..n];

                if *name == "text_90k" || *name == "prng_64k" {
                    crate::probe::reset(crate::probe::Probe::FastloopIters);
                    crate::probe::reset(crate::probe::Probe::BlocksDecoded);
                }

                // Exact-size output buffer: also checked against the
                // original plaintext, not just fast-vs-generic agreement.
                assert_fastloop_matches_generic(compressed, data.len().max(1), Some(data));

                if *name == "text_90k" && crate::probe::get(crate::probe::Probe::FastloopIters) > 0 {
                    text_90k_engaged = true;
                }
                if *name == "prng_64k" && crate::probe::get(crate::probe::Probe::BlocksDecoded) > 0 {
                    prng_64k_engaged = true;
                }

                // Truncated input: must fail identically on both paths (not
                // panic, not silently diverge) for every truncation point,
                // including truncation mid-fastloop-refill and right at the
                // final block's end.
                for cut in [1usize, compressed.len() / 2, compressed.len().saturating_sub(1)] {
                    if cut == 0 || cut >= compressed.len() {
                        continue;
                    }
                    assert_fastloop_matches_generic(&compressed[..cut], data.len().max(1), None);
                }

                let _ = name; // used only for readability while debugging failures
            }
        }

        // Mutation check: temporarily commented out the `crate::probe::hit
        // (FastloopIters)` call at the top of `'fastloop: loop` -- this
        // test's `text_90k` assertion below failed immediately (confirming
        // it is load-bearing, not a tautology), then reverted.
        assert!(
            text_90k_engaged,
            "text_90k never engaged the fastloop at any tested level -- the fastloop's own \
             result-vs-generic-loop equality checks above would still pass even if the \
             fastloop silently stopped running entirely (exactly the fastloop_hit_eob bug \
             class), so this probe assertion is the actual engagement proof"
        );
        // Mutation check: temporarily commented out the `crate::probe::hit
        // (BlocksDecoded)` call at the top of `'next_block: loop` -- this
        // test's `prng_64k` assertion AND
        // `fastloop_boundary_fuzz_exhaustive_truncation_and_output_window`'s
        // `text_multiblock` assertion below both failed immediately,
        // confirming both are load-bearing, then reverted.
        assert!(
            prng_64k_engaged,
            "prng_64k's block-loop probe never fired at any tested level -- this input is a \
             STORED-block case (see the FINDING comment above: DEFLATE correctly stores \
             near-uniform bytes raw rather than Huffman-coding them, so FastloopIters==0 is \
             expected here), but BlocksDecoded should still fire since the decode path DID \
             run; a zero here would mean the block loop itself never executed, a real bug"
        );
    }

    /// Exhaustive truncation + output-window fuzz, targeting exact
    /// `bitsleft` boundaries at fastloop entry/refill.
    ///
    /// `fastloop_matches_generic_loop_on_varied_inputs` above tries only a
    /// few truncation points per input. Rather than hand-craft raw DEFLATE
    /// bit patterns to hit a specific `bitsleft` value (fragile and not
    /// reusable if the encoder's output ever changes), this test gets the
    /// same effective coverage a different way: it tries **every** byte-level
    /// truncation point of several representative compressed streams (so
    /// every possible mid-refill byte boundary is exercised) across several
    /// compression levels (so the DEFLATE block header's bit-alignment, and
    /// therefore the exact `bitsleft` phase at every subsequent refill,
    /// varies too), plus several output-buffer sizes bracketing the exact
    /// expected length (under/over-sized `out`, i.e. "end-of-output window").
    /// This is the same technique libdeflate's own fuzzer uses (exhaustive
    /// small-input / truncation fuzzing) rather than bit-exact synthetic
    /// construction.
    #[test]
    fn fastloop_boundary_fuzz_exhaustive_truncation_and_output_window() {
        let inputs: Vec<(&'static str, Vec<u8>)> = vec![
            // Multi-block at low levels (crosses the EOB-inside-fastloop
            // transition many times); long single-run (repeated max-length
            // matches / subtable path); a PRNG stream (dense litlen/offset
            // subtable usage, no long runs).
            //
            // repeat(2000) (not 400): mutation-tested -- reverting the
            // `fastloop_hit_eob` fix (deflate_decompress.rs, both
            // `fastloop_hit_eob = true;` sites) and rerunning this test with
            // `repeat(400)` did NOT go red (that input never actually
            // crossed a second DEFLATE block at the levels tested, so the
            // EOB-inside-fastloop bug class had no case to catch here even
            // though `fastloop_matches_generic_loop_on_varied_inputs`'s
            // larger `text_90k` (repeat(2000)) input did catch it). Bumped
            // to `repeat(2000)` and reran the same mutation: this test THEN
            // failed red (`output length mismatch: left: <partial> right:
            // <full>`), confirming it's not a tautology. Reverted the
            // mutation immediately after both checks.
            ("text_multiblock", b"The quick brown fox jumps over the lazy dog. ".repeat(2000)),
            ("all_same_run", vec![0x5A; 20_000]),
            ("prng", lcg_bytes(0x9E37_79B9, 20_000)),
        ];

        // A multi-block bug (e.g. reverting the
        // `fastloop_hit_eob` fix) could silently decode only the FIRST block
        // of `text_multiblock` and still pass every result-equality check
        // below on a truncated/short read -- so also assert the block-loop
        // PROBE actually saw >= 2 blocks on the one cut guaranteed to fully
        // decode (`cut == compressed.len()`), accumulated across levels the
        // same way `fastloop_matches_generic_loop_on_varied_inputs` does.
        let mut text_multiblock_saw_multiple_blocks = false;

        for level in [1i32, 6, 9] {
            for (_name, data) in &inputs {
                let mut c = Compressor::new(level).unwrap();
                let bound = deflate_compress_bound(data.len());
                let mut out = vec![0u8; bound];
                let n = c.deflate_compress(data, &mut out);
                let compressed = out[..n].to_vec();

                // Every truncation point: exercises every possible byte
                // boundary where a fastloop refill can run out of input,
                // across a range of block bit-alignments (varies by level).
                // Only the untruncated (cut == full length) case has a full
                // plaintext to compare against; every shorter cut is expected
                // to fail (or is not guaranteed to fully decode), so `None`.
                for cut in 0..=compressed.len() {
                    let expected = if cut == compressed.len() { Some(data.as_slice()) } else { None };
                    if *_name == "text_multiblock" && cut == compressed.len() {
                        // Probe the FASTLOOP path specifically (not the summed
                        // fast+generic total `assert_fastloop_matches_generic`
                        // would give below) -- a regression that stops the
                        // fastloop early after block 1 while the generic loop
                        // still decodes both blocks correctly would otherwise
                        // hide behind a summed count still reading >= 2.
                        crate::probe::reset(crate::probe::Probe::BlocksDecoded);
                        let mut d_probe = Decompressor::new();
                        let mut out_probe = vec![0u8; data.len().max(1)];
                        deflate_decompress_ex_with_fastloop(&mut d_probe, &compressed, &mut out_probe, true)
                            .expect("full-length decode must succeed");
                        if crate::probe::get(crate::probe::Probe::BlocksDecoded) >= 2 {
                            text_multiblock_saw_multiple_blocks = true;
                        }
                    }
                    assert_fastloop_matches_generic(&compressed[..cut], data.len().max(1), expected);
                }

                // Output-window fuzz: undersized (InsufficientSpace should
                // trigger identically on both paths) and a few sizes at/just
                // past the exact expected length ("end-of-output window").
                // Only `out_cap == exact` (a real, full-length buffer) can be
                // checked against the plaintext; smaller/larger caps change
                // `InsufficientSpace`/short-buffer semantics, not correctness
                // of a full decode.
                let exact = data.len().max(1);
                for out_cap in [0usize, 1, exact / 2, exact.saturating_sub(1), exact, exact + 1, exact + 300] {
                    let expected = if out_cap == exact { Some(data.as_slice()) } else { None };
                    assert_fastloop_matches_generic(&compressed, out_cap, expected);
                }
            }
        }

        assert!(
            text_multiblock_saw_multiple_blocks,
            "text_multiblock's fastloop-path decode never saw >= 2 blocks at any tested \
             level -- the byte-identity checks above would still pass even if a regression \
             (e.g. reverting the fastloop_hit_eob fix) made the fastloop silently stop after \
             the first block, since generic-loop fallback still decodes correctly; this probe \
             assertion is what the historical mutation-and-manually-revert check in this \
             test's own module doc is now permanently automated as"
        );
    }
}
