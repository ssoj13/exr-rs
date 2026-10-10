//! Port of libdeflate lib/deflate_compress.c.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/deflate_compress.c
//!
//! Pure-Rust, dependency-free, safe (no `unsafe`, no FFI) 1:1 port of the
//! libdeflate DEFLATE compressor.  This module ports the entire
//! compressor: all 12 compression levels and their strategy mapping, the
//! `none`/`greedy`/`lazy`/`lazy2`/`near-optimal` parsers, the cost model and
//! optimal parser, the block-splitting heuristic, length-limited canonical
//! Huffman code construction, the precode RLE, the output bitstream writer, and
//! `deflate_compress_bound`.
//!
//! # Architecture notes (how the port maps C → Rust)
//! - libdeflate's C output writer uses a `bitbuf_t` machine-word bit buffer plus
//!   raw output pointers (`out_next`/`out_fast_end`).  We port this with a safe
//!   `&mut [u8]` cursor (`OutputBitstream`) and a `u64` bit buffer, flushing whole
//!   words via `to_le_bytes` and a byte-at-a-time tail.  Output is byte-identical.
//! - C uses `const u8 *in_base` + pointer arithmetic for the matchfinders.  The
//!   matchfinders already expose an index-based API (`in_buf: &[u8]`,
//!   `in_base_pos: usize`); we drive them exactly as the C compressor does, just
//!   with indices instead of pointers.
//! - `bitbuf_t` is fixed at `u64` here (libdeflate uses `machine_word_t`, which is
//!   64-bit on all targets exr-rs cares about).  `BITBUF_NBITS` is therefore 63
//!   and `CAN_BUFFER(n)` is `7 + n <= 63`.  The static asserts in C that gate the
//!   "merge precode lens" / "4 literals at once" fast paths all hold for 64-bit.
//!
//! # Things deliberately NOT changed from C semantics
//! - Tie-breaking in `deflate_flush_block` (uncompressed > static > dynamic when
//!   equal cost), the block-split thresholds, the cost model and the optimal
//!   parser are all reproduced exactly so the byte output matches libdeflate.
//!
//! # Clippy: faithful-port allowances
//! This module ports C `for`/`while` loops that walk integer indices over fixed tables
//! (`deflate_length_slot[len]`, `deflate_extra_offset_bits[sym]`, the precode permutation,
//! etc.) verbatim, so the Rust mirrors the C control flow 1:1 and stays auditable against
//! the source.  Rewriting these as `.iter().enumerate()` would obscure that mapping and the
//! interleaved multi-array indexing it relies on.  We therefore allow the index-style lints
//! crate-locally here; the genuinely-improvable lints (abs_diff, default-init, etc.) are
//! fixed in the code rather than allowed.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
// `i + 1 <= last_idx` in build_tree is verbatim C (`i + 1 <= last_idx`).
#![allow(clippy::int_plus_one)]
// `1 << (9 - 9)` / `1 << (5 - 5)` in deflate_init_static_codes are kept in the C form
// `1 << (9 - bits)` to show how each fixed-code frequency is derived from its bit width.
#![allow(clippy::eq_op)]

use crate::common::{bsr32, put_unaligned_le16, put_unaligned_le32, WORDBYTES};
use crate::deflate_constants::*;
use crate::bt_matchfinder::{
    bt_matchfinder_get_matches, bt_matchfinder_skip_byte, BtMatchfinder, LzMatch,
    BT_MATCHFINDER_REQUIRED_NBYTES,
};
use crate::hc_matchfinder::{hc_matchfinder_longest_match, hc_matchfinder_skip_bytes, HcMatchfinder};
use crate::ht_matchfinder::{
    ht_matchfinder_longest_match, ht_matchfinder_skip_bytes, HtMatchfinder,
    HT_MATCHFINDER_REQUIRED_NBYTES,
};
use crate::matchfinder_common::MATCHFINDER_WINDOW_SIZE;

// ============================================================================
// Build-time parameters (mirrors the `#define`s at the top of deflate_compress.c)
// ============================================================================

/// Near-optimal parsing is always compiled in (mirrors `SUPPORT_NEAR_OPTIMAL_PARSING 1`).
/// Minimum block length, in uncompressed bytes.  `MIN_BLOCK_LENGTH`.
const MIN_BLOCK_LENGTH: usize = 5000;
/// Soft maximum block length for greedy/lazy/near-optimal.  `SOFT_MAX_BLOCK_LENGTH`.
const SOFT_MAX_BLOCK_LENGTH: usize = 300000;
/// Sequence-store length for greedy/lazy.  `SEQ_STORE_LENGTH`.
const SEQ_STORE_LENGTH: usize = 50000;
/// Soft maximum block length for the fastest compressor.  `FAST_SOFT_MAX_BLOCK_LENGTH`.
const FAST_SOFT_MAX_BLOCK_LENGTH: usize = 65535;
/// Sequence-store length for the fastest compressor.  `FAST_SEQ_STORE_LENGTH`.
const FAST_SEQ_STORE_LENGTH: usize = 8192;

/// Maximum litlen codeword length.  `MAX_LITLEN_CODEWORD_LEN` (14, intentionally
/// below the DEFLATE max of 15 — see C comment about writing 4 literals at once).
const MAX_LITLEN_CODEWORD_LEN: u32 = 14;
/// Maximum offset codeword length.  `MAX_OFFSET_CODEWORD_LEN`.
const MAX_OFFSET_CODEWORD_LEN: u32 = DEFLATE_MAX_OFFSET_CODEWORD_LEN;
/// Maximum precode codeword length.  `MAX_PRE_CODEWORD_LEN`.
const MAX_PRE_CODEWORD_LEN: u32 = DEFLATE_MAX_PRE_CODEWORD_LEN;

// --- Near-optimal parsing parameters ---

/// Cost scaling factor for the near-optimal parser.  `BIT_COST`.
const BIT_COST: u32 = 16;
/// Default cost (bits) for an unused literal in the previous pass.  `LITERAL_NOSTAT_BITS`.
const LITERAL_NOSTAT_BITS: u32 = 13;
/// Default cost (bits) for an unused length.  `LENGTH_NOSTAT_BITS`.
const LENGTH_NOSTAT_BITS: u32 = 13;
/// Default cost (bits) for an unused offset.  `OFFSET_NOSTAT_BITS`.
const OFFSET_NOSTAT_BITS: u32 = 10;
/// (Slightly less than) the maximum number of cached matches per block.  `MATCH_CACHE_LENGTH`.
const MATCH_CACHE_LENGTH: usize = SOFT_MAX_BLOCK_LENGTH * 5;

/// Maximum number of matches the BT matchfinder can find at one position.
/// `MAX_MATCHES_PER_POS`.
const MAX_MATCHES_PER_POS: usize =
    (DEFLATE_MAX_MATCH_LEN - DEFLATE_MIN_MATCH_LEN + 1) as usize;

/// Largest block length we will ever use.  `MAX_BLOCK_LENGTH`.
const MAX_BLOCK_LENGTH: usize = max_usize(
    SOFT_MAX_BLOCK_LENGTH + MIN_BLOCK_LENGTH - 1,
    SOFT_MAX_BLOCK_LENGTH + 1 + DEFLATE_MAX_MATCH_LEN as usize,
);

/// `const fn` max for use in const contexts.
const fn max_usize(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

// ============================================================================
// Static tables (ported verbatim from deflate_compress.c)
// ============================================================================

/// Table: length slot => length slot base value.  `deflate_length_slot_base`.
const DEFLATE_LENGTH_SLOT_BASE: [u32; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Table: length slot => number of extra length bits.  `deflate_extra_length_bits`.
const DEFLATE_EXTRA_LENGTH_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Table: offset slot => offset slot base value.  `deflate_offset_slot_base`.
const DEFLATE_OFFSET_SLOT_BASE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Table: offset slot => number of extra offset bits.  `deflate_extra_offset_bits`.
const DEFLATE_EXTRA_OFFSET_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Table: length => length slot.  `deflate_length_slot[DEFLATE_MAX_MATCH_LEN + 1]`.
const DEFLATE_LENGTH_SLOT: [u8; 259] = [
    0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12,
    12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16,
    16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18,
    18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
    22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25,
    25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
    25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26,
    26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
    26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 28,
];

/// Table: `offset - 1 => offset_slot` for offset <= 256.  `deflate_offset_slot[256]`.
const DEFLATE_OFFSET_SLOT: [u8; 256] = [
    0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
];

/// The order in which precode codeword lengths are stored.  `deflate_precode_lens_permutation`.
const DEFLATE_PRECODE_LENS_PERMUTATION: [u8; DEFLATE_NUM_PRECODE_SYMS] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Table: precode symbol => number of extra bits.  `deflate_extra_precode_bits`.
const DEFLATE_EXTRA_PRECODE_BITS: [u8; DEFLATE_NUM_PRECODE_SYMS] =
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];

/// Bit-reverse table (one byte).  `bitreverse_tab` (the non-`rbit32` path).
const BITREVERSE_TAB: [u8; 256] = [
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
    0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
    0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
    0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
    0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
    0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
    0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
    0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
    0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
    0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
    0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
];

/// Default litlen costs lookup table.  `default_litlen_costs` (3 subtables).
/// Indexed by estimated match probability (0=0.25, 1=0.50, 2=0.75).  Each entry maps
/// num_used_literals (0..=256) → literal cost, plus a single `len_sym_cost`.
struct DefaultLitlenCosts {
    used_lits_to_lit_cost: [u8; 257],
    len_sym_cost: u8,
}

const DEFAULT_LITLEN_COSTS: [DefaultLitlenCosts; 3] = [
    DefaultLitlenCosts {
        // match_prob = 0.25
        used_lits_to_lit_cost: [
            6, 6, 22, 32, 38, 43, 48, 51, 54, 57, 59, 61, 64, 65, 67, 69, 70, 72, 73, 74, 75, 76,
            77, 79, 80, 80, 81, 82, 83, 84, 85, 85, 86, 87, 88, 88, 89, 89, 90, 91, 91, 92, 92, 93,
            93, 94, 95, 95, 96, 96, 96, 97, 97, 98, 98, 99, 99, 99, 100, 100, 101, 101, 101, 102,
            102, 102, 103, 103, 104, 104, 104, 105, 105, 105, 105, 106, 106, 106, 107, 107, 107,
            108, 108, 108, 108, 109, 109, 109, 109, 110, 110, 110, 111, 111, 111, 111, 112, 112,
            112, 112, 112, 113, 113, 113, 113, 114, 114, 114, 114, 114, 115, 115, 115, 115, 115,
            116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 118, 119,
            119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121,
            122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124,
            124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126,
            126, 127, 127, 127, 127, 127, 127, 127, 127, 128, 128, 128, 128, 128, 128, 128, 128,
            128, 129, 129, 129, 129, 129, 129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 130,
            130, 130, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 132, 132, 132, 132, 132,
            132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 134, 134,
            134, 134, 134, 134, 134, 134,
        ],
        len_sym_cost: 109,
    },
    DefaultLitlenCosts {
        // match_prob = 0.5
        used_lits_to_lit_cost: [
            16, 16, 32, 41, 48, 53, 57, 60, 64, 66, 69, 71, 73, 75, 76, 78, 80, 81, 82, 83, 85, 86,
            87, 88, 89, 90, 91, 92, 92, 93, 94, 95, 96, 96, 97, 98, 98, 99, 99, 100, 101, 101, 102,
            102, 103, 103, 104, 104, 105, 105, 106, 106, 107, 107, 108, 108, 108, 109, 109, 110,
            110, 110, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 115,
            116, 116, 116, 117, 117, 117, 118, 118, 118, 118, 119, 119, 119, 119, 120, 120, 120,
            120, 121, 121, 121, 121, 122, 122, 122, 122, 122, 123, 123, 123, 123, 124, 124, 124,
            124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127,
            128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129, 129, 130, 130, 130, 130, 130,
            130, 131, 131, 131, 131, 131, 131, 131, 132, 132, 132, 132, 132, 132, 133, 133, 133,
            133, 133, 133, 133, 134, 134, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135,
            135, 135, 135, 136, 136, 136, 136, 136, 136, 136, 136, 137, 137, 137, 137, 137, 137,
            137, 137, 138, 138, 138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139, 139, 139,
            139, 139, 139, 140, 140, 140, 140, 140, 140, 140, 140, 140, 141, 141, 141, 141, 141,
            141, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 143,
            143, 143, 143, 143, 143, 143, 143, 143, 143, 144,
        ],
        len_sym_cost: 93,
    },
    DefaultLitlenCosts {
        // match_prob = 0.75
        used_lits_to_lit_cost: [
            32, 32, 48, 57, 64, 69, 73, 76, 80, 82, 85, 87, 89, 91, 92, 94, 96, 97, 98, 99, 101,
            102, 103, 104, 105, 106, 107, 108, 108, 109, 110, 111, 112, 112, 113, 114, 114, 115,
            115, 116, 117, 117, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124,
            124, 124, 125, 125, 126, 126, 126, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130,
            130, 131, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 134, 135, 135,
            135, 135, 136, 136, 136, 136, 137, 137, 137, 137, 138, 138, 138, 138, 138, 139, 139,
            139, 139, 140, 140, 140, 140, 140, 141, 141, 141, 141, 141, 142, 142, 142, 142, 142,
            143, 143, 143, 143, 143, 144, 144, 144, 144, 144, 144, 145, 145, 145, 145, 145, 145,
            146, 146, 146, 146, 146, 146, 147, 147, 147, 147, 147, 147, 147, 148, 148, 148, 148,
            148, 148, 149, 149, 149, 149, 149, 149, 149, 150, 150, 150, 150, 150, 150, 150, 150,
            151, 151, 151, 151, 151, 151, 151, 151, 152, 152, 152, 152, 152, 152, 152, 152, 153,
            153, 153, 153, 153, 153, 153, 153, 154, 154, 154, 154, 154, 154, 154, 154, 154, 155,
            155, 155, 155, 155, 155, 155, 155, 155, 156, 156, 156, 156, 156, 156, 156, 156, 156,
            157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 158, 158, 158, 158, 158, 158, 158,
            158, 158, 158, 158, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 160,
        ],
        len_sym_cost: 84,
    },
];

// ============================================================================
// Huffman code data structures (mirrors deflate_codewords / deflate_lens / deflate_codes)
// ============================================================================

/// Codewords for the DEFLATE Huffman codes.  `struct deflate_codewords`.
#[derive(Clone)]
struct DeflateCodewords {
    litlen: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

/// Codeword lengths (bits) for the DEFLATE Huffman codes.  `struct deflate_lens`.
///
/// Note: in C this is a single struct with `litlen` and `offset` arrays that are
/// contiguous in memory, which `deflate_precompute_huffman_header` exploits via
/// `memmove` to make the lengths contiguous.  Here we store them as one flat
/// `[u8; 288 + 32]` array (`all`) and expose `litlen`/`offset` as index ranges,
/// reproducing the exact contiguous layout the C code relies on.
#[derive(Clone)]
struct DeflateLens {
    /// `litlen[0..288]` then `offset[0..32]`, contiguous (mirrors C struct layout).
    all: [u8; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS],
}

impl DeflateLens {
    const OFFSET_BASE: usize = DEFLATE_NUM_LITLEN_SYMS;

    #[inline(always)]
    fn litlen(&self, i: usize) -> u8 {
        self.all[i]
    }
    #[inline(always)]
    fn offset(&self, i: usize) -> u8 {
        self.all[Self::OFFSET_BASE + i]
    }
}

/// Codewords and lengths for the DEFLATE Huffman codes.  `struct deflate_codes`.
#[derive(Clone)]
struct DeflateCodes {
    codewords: DeflateCodewords,
    lens: DeflateLens,
}

impl DeflateCodes {
    fn new() -> Self {
        DeflateCodes {
            codewords: DeflateCodewords {
                litlen: [0; DEFLATE_NUM_LITLEN_SYMS],
                offset: [0; DEFLATE_NUM_OFFSET_SYMS],
            },
            lens: DeflateLens {
                all: [0; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS],
            },
        }
    }
}

/// Symbol frequency counters for the DEFLATE Huffman codes.  `struct deflate_freqs`.
#[derive(Clone)]
struct DeflateFreqs {
    litlen: [u32; DEFLATE_NUM_LITLEN_SYMS],
    offset: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

impl DeflateFreqs {
    fn new() -> Self {
        DeflateFreqs {
            litlen: [0; DEFLATE_NUM_LITLEN_SYMS],
            offset: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }
    fn reset(&mut self) {
        self.litlen = [0; DEFLATE_NUM_LITLEN_SYMS];
        self.offset = [0; DEFLATE_NUM_OFFSET_SYMS];
    }
}

/// A run of literals followed by a match or end-of-block.  `struct deflate_sequence`.
///
/// `litrunlen_and_length`: bits 0..22 = number of literals in this run; bits 23..31 =
/// length of the following match, or 0 for the final run.  Mirrors C exactly.
#[derive(Clone, Copy, Default)]
struct DeflateSequence {
    litrunlen_and_length: u32,
    offset: u16,
    offset_slot: u16,
}

/// `SEQ_LENGTH_SHIFT`.
const SEQ_LENGTH_SHIFT: u32 = 23;
/// `SEQ_LITRUNLEN_MASK`.
const SEQ_LITRUNLEN_MASK: u32 = (1u32 << SEQ_LENGTH_SHIFT) - 1;

// --- Near-optimal parser structures ---

/// Costs for the near-optimal parsing algorithm.  `struct deflate_costs`.
#[derive(Clone)]
struct DeflateCosts {
    literal: [u32; DEFLATE_NUM_LITERALS as usize],
    length: [u32; DEFLATE_MAX_MATCH_LEN as usize + 1],
    offset_slot: [u32; DEFLATE_NUM_OFFSET_SYMS],
}

impl DeflateCosts {
    fn new() -> Self {
        DeflateCosts {
            literal: [0; DEFLATE_NUM_LITERALS as usize],
            length: [0; DEFLATE_MAX_MATCH_LEN as usize + 1],
            offset_slot: [0; DEFLATE_NUM_OFFSET_SYMS],
        }
    }
}

/// A byte position node in the optimal-parse graph.  `struct deflate_optimum_node`.
///
/// `item`: low `OPTIMUM_OFFSET_SHIFT` bits = length (1 for a literal), high bits =
/// literal byte (if literal) or match offset (if match).
#[derive(Clone, Copy, Default)]
struct DeflateOptimumNode {
    cost_to_end: u32,
    item: u32,
}

/// `OPTIMUM_OFFSET_SHIFT`.
const OPTIMUM_OFFSET_SHIFT: u32 = 9;
/// `OPTIMUM_LEN_MASK`.
const OPTIMUM_LEN_MASK: u32 = (1u32 << OPTIMUM_OFFSET_SHIFT) - 1;

// --- Block split statistics (mirrors block_split_stats) ---

/// `NUM_LITERAL_OBSERVATION_TYPES`.
const NUM_LITERAL_OBSERVATION_TYPES: usize = 8;
/// `NUM_MATCH_OBSERVATION_TYPES`.
const NUM_MATCH_OBSERVATION_TYPES: usize = 2;
/// `NUM_OBSERVATION_TYPES`.
const NUM_OBSERVATION_TYPES: usize = NUM_LITERAL_OBSERVATION_TYPES + NUM_MATCH_OBSERVATION_TYPES;
/// `NUM_OBSERVATIONS_PER_BLOCK_CHECK`.
const NUM_OBSERVATIONS_PER_BLOCK_CHECK: u32 = 512;

/// Block split statistics.  `struct block_split_stats`.
#[derive(Clone)]
struct BlockSplitStats {
    new_observations: [u32; NUM_OBSERVATION_TYPES],
    observations: [u32; NUM_OBSERVATION_TYPES],
    num_new_observations: u32,
    num_observations: u32,
}

impl BlockSplitStats {
    fn new() -> Self {
        BlockSplitStats {
            new_observations: [0; NUM_OBSERVATION_TYPES],
            observations: [0; NUM_OBSERVATION_TYPES],
            num_new_observations: 0,
            num_observations: 0,
        }
    }
}

// ============================================================================
// Output bitstream (safe port of `struct deflate_output_bitstream` + macros)
// ============================================================================

/// `bitbuf_t` — the bit buffer type.  libdeflate uses `machine_word_t`; we fix it
/// at `u64` (64-bit on all exr-rs targets), matching libdeflate's behavior there.
type BitBuf = u64;

/// `BITBUF_NBITS` — capacity of the bit buffer in bits (1 less than the real size,
/// to keep `bitbuf >>= bitcount & ~7` well-defined).  `8*sizeof(bitbuf_t) - 1`.
const BITBUF_NBITS: u32 = 8 * (core::mem::size_of::<BitBuf>() as u32) - 1;

/// `CAN_BUFFER(n)` — can `n` more bits always be added after a flush (which leaves
/// up to 7 bits)?  `7 + n <= BITBUF_NBITS`.
const fn can_buffer(n: u32) -> bool {
    7 + n <= BITBUF_NBITS
}

/// Safe output bitstream.  Mirrors `struct deflate_output_bitstream`, but instead of
/// raw `next`/`end` pointers it uses an output slice plus a write cursor.  The bit
/// buffer/count and the `out_fast_end` whole-word flush fast-path are reproduced
/// so the byte output is identical to libdeflate.
struct OutputBitstream<'a> {
    /// Bits not yet written to the output buffer.  `bitbuf`.
    bitbuf: BitBuf,
    /// Number of bits currently held in `bitbuf` (0..=BITBUF_NBITS).  `bitcount`.
    bitcount: u32,
    /// Output buffer.
    out: &'a mut [u8],
    /// Index of the next byte to write (`next` - start of buffer).
    next: usize,
    /// `true` if the output buffer ran out of space.  `overflow`.
    overflow: bool,
}

impl<'a> OutputBitstream<'a> {
    fn new(out: &'a mut [u8]) -> Self {
        OutputBitstream {
            bitbuf: 0,
            bitcount: 0,
            out,
            next: 0,
            overflow: false,
        }
    }

    /// Total output buffer length (`end - <start>`).
    #[inline(always)]
    fn end(&self) -> usize {
        self.out.len()
    }
}

/// `ADD_BITS(bits, n)` — append `n` bits to the local bit buffer.  Implemented as a
/// macro over the local `bitbuf`/`bitcount` variables so the flush fast-path can use
/// them directly (mirroring the C macro pattern in `deflate_flush_block`).
macro_rules! add_bits {
    ($bitbuf:ident, $bitcount:ident, $bits:expr, $n:expr) => {{
        $bitbuf |= ($bits as BitBuf) << $bitcount;
        $bitcount += $n;
        debug_assert!($bitcount <= BITBUF_NBITS);
    }};
}

/// `FLUSH_BITS()` — flush whole bytes from the local bit buffer to the output.
/// Flushes a whole word when there's room (the fast path), else a byte at a time.
///
/// `out_fast_end` (`end - min(WORDBYTES-1, end - out_next)`) is a PARAMETER, not
/// recomputed here, mirroring C's `deflate_flush_block`: there it's a `u8 * const`
/// local computed ONCE from the block's starting `out_next`, and every `FLUSH_BITS()`
/// call in that block (there can be dozens per literal/match) reads the same value.
/// This macro used to recompute `end`/`out_fast_end` from `$out.len()`
/// on every single call — redundant work in the hottest loop in the compressor,
/// executed once per literal, once or more per match, and once per RLE/precode item.
/// Byte-identical output either way (both branches produce the exact same bytes,
/// this only changes how often the whole-word fast path is taken vs. estimated),
/// verified by `cargo xtask parity-deflate` at all 12 levels.
macro_rules! flush_bits {
    ($bitbuf:ident, $bitcount:ident, $out:expr, $out_next:ident, $out_fast_end:expr) => {{
        if $out_next < $out_fast_end {
            // Flush a whole word.
            let word = $bitbuf.to_le_bytes();
            $out[$out_next..$out_next + WORDBYTES].copy_from_slice(&word);
            $bitbuf >>= $bitcount & !7;
            $out_next += ($bitcount >> 3) as usize;
            $bitcount &= 7;
        } else {
            // Flush a byte at a time.
            while $bitcount >= 8 {
                debug_assert!($out_next < $out.len());
                $out[$out_next] = $bitbuf as u8;
                $out_next += 1;
                $bitcount -= 8;
                $bitbuf >>= 8;
            }
        }
    }};
}

// ============================================================================
// Huffman code construction (sort_symbols / build_tree / compute_length_counts /
// gen_codewords / deflate_make_huffman_code) — 1:1 with C
// ============================================================================

const NUM_SYMBOL_BITS: u32 = 10;
const SYMBOL_MASK: u32 = (1 << NUM_SYMBOL_BITS) - 1;
const FREQ_MASK: u32 = !SYMBOL_MASK;

/// `heapify_subtree` — sift-down for `heap_sort` (1-based indices into `a`).
fn heapify_subtree(a: &mut [u32], length: usize, subtree_idx: usize) {
    let v = a[subtree_idx];
    let mut parent_idx = subtree_idx;
    loop {
        let mut child_idx = parent_idx * 2;
        if child_idx > length {
            break;
        }
        if child_idx < length && a[child_idx + 1] > a[child_idx] {
            child_idx += 1;
        }
        if v >= a[child_idx] {
            break;
        }
        a[parent_idx] = a[child_idx];
        parent_idx = child_idx;
    }
    a[parent_idx] = v;
}

/// `heapify_array`.
fn heapify_array(a: &mut [u32], length: usize) {
    let mut subtree_idx = length / 2;
    while subtree_idx >= 1 {
        heapify_subtree(a, length, subtree_idx);
        subtree_idx -= 1;
    }
}

/// `heap_sort` — sort `a[0..length]`.  C uses `A--` for 1-based indexing; we pass a
/// 1-based view by offsetting indices, so `a` here is the original 0-based slice and
/// we operate on `a[0..length]` using helper functions that treat index 1 as a[0].
///
/// To match the C "A--" trick exactly without unsafe pointer arithmetic, we use a
/// scratch slice of length `length + 1` where index 0 is unused and indices 1..=length
/// alias `a[0..length]`.  This is faithful: `heapify_*` use 1-based indices.
fn heap_sort(a: &mut [u32], length: usize) {
    if length < 2 {
        return;
    }
    // C declares `A--` and reuses the caller's array in place (no
    // allocation at all -- see the module doc comment on `sort_symbols`'s
    // scratch array for why we can't safely do that same pointer trick).
    // Fixed-size stack array instead of a per-call `Vec` -- `length` is
    // always `<= DEFLATE_MAX_NUM_SYMS` (the only caller, `sort_symbols`,
    // sorts at most `num_syms <= DEFLATE_MAX_NUM_SYMS` symbols), mirroring
    // C's own `counters[GET_NUM_COUNTERS(DEFLATE_MAX_NUM_SYMS)]` fixed
    // stack array. This function runs 3+ times per block (once per code
    // built), 20+ times per block at levels 10-12 -- a `Vec` here was a
    // real per-call heap allocation in a hot loop.
    debug_assert!(length <= DEFLATE_MAX_NUM_SYMS);
    let mut work = [0u32; DEFLATE_MAX_NUM_SYMS + 1];
    work[1..=length].copy_from_slice(&a[..length]);

    heapify_array(&mut work[..=length], length);

    let mut len = length;
    while len >= 2 {
        work.swap(len, 1);
        len -= 1;
        heapify_subtree(&mut work[..=len], len, 1);
    }

    a[..length].copy_from_slice(&work[1..=length]);
}

/// `GET_NUM_COUNTERS(num_syms)` — number of count-sort counters.
#[inline(always)]
fn get_num_counters(num_syms: usize) -> usize {
    num_syms
}

/// `sort_symbols` — sort symbols by (frequency, value), discard zero-frequency.
/// Packs `sym | (freq << NUM_SYMBOL_BITS)` into `symout`.  Returns number used.
fn sort_symbols(num_syms: usize, freqs: &[u32], lens: &mut [u8], symout: &mut [u32]) -> usize {
    let num_counters = get_num_counters(num_syms);
    // Fixed-size stack array, mirroring C's `unsigned
    // counters[GET_NUM_COUNTERS(DEFLATE_MAX_NUM_SYMS)]` -- `num_counters ==
    // num_syms <= DEFLATE_MAX_NUM_SYMS` for every caller (litlen: 288,
    // offset: 32, precode: 19). This runs 3+ times per block (once per code
    // built), 20+ times per block at levels 10-12 -- a `Vec` here was a real
    // per-call heap allocation in a hot loop.
    debug_assert!(num_counters <= DEFLATE_MAX_NUM_SYMS);
    let mut counters = [0u32; DEFLATE_MAX_NUM_SYMS];
    let counters = &mut counters[..num_counters];

    // Count the frequencies.
    for sym in 0..num_syms {
        let idx = core::cmp::min(freqs[sym], (num_counters - 1) as u32) as usize;
        counters[idx] += 1;
    }

    // Make the counters cumulative (ignoring counter 0).
    let mut num_used_syms: u32 = 0;
    for i in 1..num_counters {
        let count = counters[i];
        counters[i] = num_used_syms;
        num_used_syms += count;
    }

    // Sort nonzero-frequency symbols; zero-frequency ones get length 0.
    for sym in 0..num_syms {
        let freq = freqs[sym];
        if freq != 0 {
            let idx = core::cmp::min(freq, (num_counters - 1) as u32) as usize;
            symout[counters[idx] as usize] = (sym as u32) | (freq << NUM_SYMBOL_BITS);
            counters[idx] += 1;
        } else {
            lens[sym] = 0;
        }
    }

    // Sort the symbols counted in the last counter.
    let start = counters[num_counters - 2] as usize;
    let count = (counters[num_counters - 1] - counters[num_counters - 2]) as usize;
    heap_sort(&mut symout[start..], count);

    num_used_syms as usize
}

/// `build_tree` — build the stripped-down (non-leaf-only) Huffman tree in `a`.
fn build_tree(a: &mut [u32], sym_count: usize) {
    let last_idx = sym_count - 1;
    let mut i = 0usize; // next lowest-freq leaf needing a parent
    let mut b = 0usize; // next lowest-freq non-leaf needing a parent
    let mut e = 0usize; // next spot for a non-leaf

    loop {
        let new_freq: u32;
        if i + 1 <= last_idx && (b == e || (a[i + 1] & FREQ_MASK) <= (a[b] & FREQ_MASK)) {
            // Two leaves.
            new_freq = (a[i] & FREQ_MASK) + (a[i + 1] & FREQ_MASK);
            i += 2;
        } else if b + 2 <= e && (i > last_idx || (a[b + 1] & FREQ_MASK) < (a[i] & FREQ_MASK)) {
            // Two non-leaves.
            new_freq = (a[b] & FREQ_MASK) + (a[b + 1] & FREQ_MASK);
            a[b] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b] & SYMBOL_MASK);
            a[b + 1] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b + 1] & SYMBOL_MASK);
            b += 2;
        } else {
            // One leaf and one non-leaf.
            new_freq = (a[i] & FREQ_MASK) + (a[b] & FREQ_MASK);
            a[b] = ((e as u32) << NUM_SYMBOL_BITS) | (a[b] & SYMBOL_MASK);
            i += 1;
            b += 1;
        }
        a[e] = new_freq | (a[e] & SYMBOL_MASK);
        e += 1;
        if e >= last_idx {
            break;
        }
    }
}

/// `compute_length_counts` — derive codeword-length counts from the tree, honoring
/// the length-limited constraint.
fn compute_length_counts(a: &mut [u32], root_idx: usize, len_counts: &mut [u32], max_codeword_len: usize) {
    for v in len_counts.iter_mut().take(max_codeword_len + 1) {
        *v = 0;
    }
    len_counts[1] = 2;

    // Set the root node's depth to 0.
    a[root_idx] &= SYMBOL_MASK;

    let mut node = root_idx as i64 - 1;
    while node >= 0 {
        let n = node as usize;
        let parent = (a[n] >> NUM_SYMBOL_BITS) as usize;
        let parent_depth = a[parent] >> NUM_SYMBOL_BITS;
        let mut depth = parent_depth + 1;

        a[n] = (a[n] & SYMBOL_MASK) | (depth << NUM_SYMBOL_BITS);

        if depth as usize >= max_codeword_len {
            depth = max_codeword_len as u32;
            loop {
                depth -= 1;
                if len_counts[depth as usize] != 0 {
                    break;
                }
            }
        }

        len_counts[depth as usize] -= 1;
        len_counts[depth as usize + 1] += 2;

        node -= 1;
    }
}

/// Compile-time check mirroring C `STATIC_ASSERT(DEFLATE_MAX_CODEWORD_LEN <= 16)`.
const _: () = assert!(DEFLATE_MAX_CODEWORD_LEN <= 16);

/// `reverse_codeword` — bit-reverse a codeword (the table path; `rbit32` is ARM-only).
#[inline(always)]
fn reverse_codeword(codeword: u32, len: u8) -> u32 {
    let c = ((BITREVERSE_TAB[(codeword & 0xff) as usize] as u32) << 8)
        | (BITREVERSE_TAB[(codeword >> 8) as usize] as u32);
    c >> (16 - len as u32)
}

/// `gen_codewords` — assign codeword lengths to symbols and generate codewords.
fn gen_codewords(
    a: &mut [u32],
    lens: &mut [u8],
    len_counts: &[u32],
    max_codeword_len: usize,
    num_syms: usize,
) {
    // Assign lengths in decreasing order to the sorted symbols.
    let mut i = 0usize;
    let mut len = max_codeword_len;
    while len >= 1 {
        let mut count = len_counts[len];
        while count != 0 {
            lens[(a[i] & SYMBOL_MASK) as usize] = len as u8;
            i += 1;
            count -= 1;
        }
        len -= 1;
    }

    // Generate the codewords.
    let mut next_codewords = [0u32; DEFLATE_MAX_CODEWORD_LEN as usize + 1];
    next_codewords[0] = 0;
    next_codewords[1] = 0;
    for len in 2..=max_codeword_len {
        next_codewords[len] = (next_codewords[len - 1] + len_counts[len - 1]) << 1;
    }

    for sym in 0..num_syms {
        let l = lens[sym];
        // DEFLATE requires bit-reversed codewords.
        a[sym] = reverse_codeword(next_codewords[l as usize], l);
        next_codewords[l as usize] += 1;
    }
}

/// `deflate_make_huffman_code` — build a length-limited canonical Huffman code.
///
/// `a` is the codeword output array (shared with sort scratch).  On return,
/// `a[sym]` is the (bit-reversed) codeword and `lens[sym]` its length.
fn deflate_make_huffman_code(
    num_syms: usize,
    max_codeword_len: usize,
    freqs: &[u32],
    lens: &mut [u8],
    codewords: &mut [u32],
) {
    let a = codewords;
    let num_used_syms = sort_symbols(num_syms, freqs, lens, a);

    // A complete Huffman code needs >= 2 codewords (libdeflate's "practice over spec"
    // handling: emit two length-1 codewords).
    if num_used_syms < 2 {
        let sym = if num_used_syms != 0 {
            (a[0] & SYMBOL_MASK) as usize
        } else {
            0
        };
        let nonzero_idx = if sym != 0 { sym } else { 1 };
        a[0] = 0;
        lens[0] = 1;
        a[nonzero_idx] = 1;
        lens[nonzero_idx] = 1;
        return;
    }

    build_tree(a, num_used_syms);

    let mut len_counts = [0u32; DEFLATE_MAX_CODEWORD_LEN as usize + 1];
    compute_length_counts(a, num_used_syms - 2, &mut len_counts, max_codeword_len);
    gen_codewords(a, lens, &len_counts, max_codeword_len, num_syms);
}

// ============================================================================
// Offset-slot lookup (deflate_get_offset_slot)
// ============================================================================

/// `deflate_get_offset_slot` — offset slot for `offset` in 1..=32768, via the small map.
#[inline(always)]
fn deflate_get_offset_slot(offset: u32) -> u32 {
    // n = (256 - offset) >> 29 ; equals 0 for offset<=256, 7 otherwise (offset<=32768).
    let n = (256u32.wrapping_sub(offset)) >> 29;
    debug_assert!((1..=32768).contains(&offset));
    DEFLATE_OFFSET_SLOT[((offset - 1) >> n) as usize] as u32 + (n << 1)
}

// ============================================================================
// The Compressor
// ============================================================================

/// Compression strategy chosen at allocation time.  Mirrors the `c->impl` function
/// pointer dispatch in `libdeflate_alloc_compressor`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    None,        // level 0: deflate_compress_none (stored blocks only)
    Fastest,     // level 1: deflate_compress_fastest (ht matchfinder)
    Greedy,      // levels 2-4
    Lazy,        // levels 5-7
    Lazy2,       // levels 8-9
    NearOptimal, // levels 10-12
}

/// Error returned when constructing a `Compressor` with an invalid level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidLevel(pub i32);

impl core::fmt::Display for InvalidLevel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "deflate: invalid compression level {} (must be 0..=12)", self.0)
    }
}

impl std::error::Error for InvalidLevel {}

/// Per-parser scratch for the precode header phase.  Mirrors the `c->o.precode` union arm.
struct PrecodeScratch {
    freqs: [u32; DEFLATE_NUM_PRECODE_SYMS],
    codewords: [u32; DEFLATE_NUM_PRECODE_SYMS],
    lens: [u8; DEFLATE_NUM_PRECODE_SYMS],
    items: [u32; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS],
    num_litlen_syms: usize,
    num_offset_syms: usize,
    num_explicit_lens: usize,
    num_items: usize,
}

/// "Full" length codewords scratch.  Mirrors the `c->o.length` union arm.
struct LengthScratch {
    codewords: [u32; DEFLATE_MAX_MATCH_LEN as usize + 1],
    lens: [u8; DEFLATE_MAX_MATCH_LEN as usize + 1],
}

/// Near-optimal parser state.  Mirrors the `c->p.n` union arm.
struct NearOptimalState {
    bt_mf: BtMatchfinder,
    /// Cached matches for the current block.  Element type mirrors `struct lz_match`
    /// (`u16 length; u16 offset`); the match-count header and literal byte are stored
    /// in `length`/`offset` of the trailing element of each position's run.
    match_cache: Vec<LzMatch>,
    /// Index just past the current block's cached matches.  Mirrors the C `cache_ptr`
    /// passed to `deflate_find_min_cost_path` (which walks backward from it).
    cache_end: usize,
    optimum_nodes: Vec<DeflateOptimumNode>,
    costs: DeflateCosts,
    costs_saved: DeflateCosts,
    offset_slot_full: Vec<u8>, // [DEFLATE_MAX_MATCH_OFFSET + 1]
    prev_observations: [u32; NUM_OBSERVATION_TYPES],
    prev_num_observations: u32,
    new_match_len_freqs: Vec<u32>, // [DEFLATE_MAX_MATCH_LEN + 1]
    match_len_freqs: Vec<u32>,
    max_optim_passes: u32,
    min_improvement_to_continue: u32,
    min_bits_to_use_nonfinal_path: u32,
    max_len_to_optimize_static_block: u32,
}

/// Parser-specific state, one arm per `Strategy` family, built ONLY for
/// the arm the chosen level actually uses.
///
/// Mirrors C's `union { hc_matchfinder... ; ht_matchfinder... ; bt_matchfinder... }`
/// (`deflate_compress.c` L3877-3913): the C allocator sizes/inits only the
/// union member the level needs. Before this, `Compressor::new` allocated AND
/// zero-filled `hc_mf`/`g_sequences`/`ht_mf`/`f_sequences` together
/// (~848 KiB total) for EVERY level, even though a given level only ever
/// touches one family. `Strategy::None` (level 0) needs no parser state at
/// all, hence `Compressor::state: Option<StrategyState>`.
enum StrategyState {
    /// Level 1 (`Strategy::Fastest`): ht-matchfinder parser.
    Fastest {
        ht_mf: HtMatchfinder,
        f_sequences: Vec<DeflateSequence>, // [FAST_SEQ_STORE_LENGTH + 1]
    },
    /// Levels 2-9 (`Strategy::Greedy`/`Lazy`/`Lazy2`): hc-matchfinder parser.
    Greedy {
        hc_mf: HcMatchfinder,
        g_sequences: Vec<DeflateSequence>, // [SEQ_STORE_LENGTH + 1]
    },
    /// Levels 10-12 (`Strategy::NearOptimal`): bt-matchfinder + optimal-parse state.
    /// Boxed: `NearOptimalState` is ~4.5 KiB (match_cache/optimum_nodes etc.),
    /// versus tens of bytes for the other arms -- clippy::large_enum_variant
    /// would otherwise force every `StrategyState` (including the tiny
    /// `Fastest`/`Greedy` arms) to be sized for the biggest one.
    Near(Box<NearOptimalState>),
}

// These are FREE functions taking `&(mut) Option<StrategyState>`, not
// `&(mut) self` methods, and are always called as `xxx(&mut self.state)` at
// the use site. That is load-bearing, not stylistic: a method taking
// `&mut self` reborrows ALL of `self` for as long as its returned reference
// lives, which breaks the disjoint-field borrows the parser loops need
// (`self.freqs`/`self.split_stats`/`self.max_search_depth` accessed directly
// while `hc_mf`/`g_sequences` etc. are still borrowed). A free function whose
// parameter is `&mut self.state` only ever borrows that one field, so the
// borrow checker can see `self.freqs` etc. as disjoint, exactly as it does
// for two plain struct fields. (Confirmed by trying the method form first:
// it failed to compile with E0499/E0503 borrow conflicts in
// `compress_fastest`/`compress_greedy`/`compress_lazy_generic`.)

/// Borrow the near-optimal parser state, if `state` is the `Near` arm
/// (i.e. `strategy == Strategy::NearOptimal`, levels 10-12). `None` on any
/// other level -- callers on the near-optimal-only code paths treat that as
/// "can't happen" (see the many "self.n is Some by construction" comments
/// below, from before `n`/`hc_mf`/`ht_mf` were folded into one `StrategyState`).
#[inline]
fn near(state: &Option<StrategyState>) -> Option<&NearOptimalState> {
    match state {
        Some(StrategyState::Near(n)) => Some(n.as_ref()),
        _ => None,
    }
}

/// Mutable counterpart of [`near`].
#[inline]
fn near_mut(state: &mut Option<StrategyState>) -> Option<&mut NearOptimalState> {
    match state {
        Some(StrategyState::Near(n)) => Some(n.as_mut()),
        _ => None,
    }
}

/// Borrow the `Fastest` arm's matchfinder + sequence buffer together, if
/// `state` is that arm (i.e. `strategy == Strategy::Fastest`). `None` on any
/// other level, mirroring the "not this level" convention used by [`near`]
/// rather than panicking -- `compress_fastest` (the only caller) treats a
/// `None` here as unreachable-in-practice, since `Compressor::new` always
/// builds `StrategyState::Fastest` exactly when `strategy == Strategy::Fastest`.
#[inline]
fn fastest_mut(state: &mut Option<StrategyState>) -> Option<(&mut HtMatchfinder, &mut Vec<DeflateSequence>)> {
    match state {
        Some(StrategyState::Fastest { ht_mf, f_sequences }) => Some((ht_mf, f_sequences)),
        _ => None,
    }
}

/// Borrow the `Greedy` arm's matchfinder + sequence buffer together, if
/// `state` is that arm (`strategy` is `Greedy`/`Lazy`/`Lazy2`). See
/// [`fastest_mut`] for the `None`-on-other-levels contract.
#[inline]
fn greedy_mut(state: &mut Option<StrategyState>) -> Option<(&mut HcMatchfinder, &mut Vec<DeflateSequence>)> {
    match state {
        Some(StrategyState::Greedy { hc_mf, g_sequences }) => Some((hc_mf, g_sequences)),
        _ => None,
    }
}

/// `deflate_set_costs_from_codes` — set the near-optimal cost model from
/// codeword lengths. Free function (not a `&mut self` method, for the same
/// disjoint-borrow reason as `near`/`near_mut`/etc. above): every call site
/// reads `lens` from a field of `self` (`self.codes.lens` or
/// `self.static_codes.lens`) while also needing `&mut` access to the
/// near-optimal state (`self.state`). With this as a method taking `&mut
/// self`, callers had to `self.codes.lens.clone()` first just to end the
/// immutable borrow before calling `self.set_costs_from_codes(&mut self)` --
/// cloning the whole `DeflateLens` table on every optimal-parse pass (called
/// in a loop, once per pass, at levels 10-12). As a free function taking
/// `&mut NearOptimalState` and `&DeflateLens` separately, a call site can
/// pass `near_mut(&mut self.state)` and `&self.codes.lens` in the same
/// statement with no clone, because they borrow disjoint fields of `self`.
fn set_costs_from_codes(n: &mut NearOptimalState, lens: &DeflateLens) {
    // Literals.
    for i in 0..DEFLATE_NUM_LITERALS as usize {
        let bits = if lens.litlen(i) != 0 {
            lens.litlen(i) as u32
        } else {
            LITERAL_NOSTAT_BITS
        };
        n.costs.literal[i] = bits * BIT_COST;
    }
    // Lengths.
    for i in (DEFLATE_MIN_MATCH_LEN as usize)..=(DEFLATE_MAX_MATCH_LEN as usize) {
        let length_slot = DEFLATE_LENGTH_SLOT[i] as usize;
        let litlen_sym = DEFLATE_FIRST_LEN_SYM as usize + length_slot;
        let mut bits = if lens.litlen(litlen_sym) != 0 {
            lens.litlen(litlen_sym) as u32
        } else {
            LENGTH_NOSTAT_BITS
        };
        bits += DEFLATE_EXTRA_LENGTH_BITS[length_slot] as u32;
        n.costs.length[i] = bits * BIT_COST;
    }
    // Offset slots.
    for i in 0..DEFLATE_OFFSET_SLOT_BASE.len() {
        let mut bits = if lens.offset(i) != 0 {
            lens.offset(i) as u32
        } else {
            OFFSET_NOSTAT_BITS
        };
        bits += DEFLATE_EXTRA_OFFSET_BITS[i] as u32;
        n.costs.offset_slot[i] = bits * BIT_COST;
    }
}

/// The main DEFLATE compressor.  Mirrors `struct libdeflate_compressor`.
///
/// Holds the chosen strategy + tuning (`max_search_depth`, `nice_match_length`),
/// the per-block Huffman state, and the parser-specific matchfinder/scratch.
pub struct Compressor {
    strategy: Strategy,
    compression_level: u32,
    /// Anything this size or smaller is emitted as stored blocks.  `max_passthrough_size`.
    max_passthrough_size: usize,
    max_search_depth: u32,
    nice_match_length: u32,

    freqs: DeflateFreqs,
    split_stats: BlockSplitStats,
    codes: DeflateCodes,
    static_codes: DeflateCodes,

    precode: PrecodeScratch,
    length_scratch: LengthScratch,

    /// Parser-specific state: exactly one of `Fastest`/`Greedy`/`Near`, built
    /// only for the arm `strategy` selects; `None` for `Strategy::None`
    /// (level 0, which never runs a matchfinder). See [`StrategyState`].
    state: Option<StrategyState>,
}

impl Compressor {
    /// Create a new compressor for the given level (0..=12).
    /// Mirrors `libdeflate_alloc_compressor_ex` (`_ex` with defaults).
    ///
    /// libdeflate 1.26+: for similarity with zlib's API, `-1` is accepted as an
    /// alias for the default compression level (6).
    /// <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/deflate_compress.c#L3885-L3891>
    pub fn new(level: i32) -> Result<Self, InvalidLevel> {
        let level = if level == -1 { 6 } else { level };
        if !(0..=12).contains(&level) {
            return Err(InvalidLevel(level));
        }
        let compression_level = level as u32;

        // max_passthrough_size = 55 - level*4 (overridden to SIZE_MAX for level 0).
        let mut max_passthrough_size = 55usize.wrapping_sub((compression_level as usize) * 4);

        let strategy;
        let mut max_search_depth = 0u32;
        let mut nice_match_length = 0u32;
        // Near-optimal tuning (only used for levels 10-12).
        let mut max_optim_passes = 0u32;
        let mut min_improvement_to_continue = 0u32;
        let mut min_bits_to_use_nonfinal_path = 0u32;
        let mut max_len_to_optimize_static_block = 0u32;

        match level {
            0 => {
                max_passthrough_size = usize::MAX;
                strategy = Strategy::None;
            }
            1 => {
                strategy = Strategy::Fastest;
                nice_match_length = 32;
            }
            2 => {
                strategy = Strategy::Greedy;
                max_search_depth = 6;
                nice_match_length = 10;
            }
            3 => {
                strategy = Strategy::Greedy;
                max_search_depth = 12;
                nice_match_length = 14;
            }
            4 => {
                strategy = Strategy::Greedy;
                max_search_depth = 16;
                nice_match_length = 30;
            }
            5 => {
                strategy = Strategy::Lazy;
                max_search_depth = 16;
                nice_match_length = 30;
            }
            6 => {
                strategy = Strategy::Lazy;
                max_search_depth = 35;
                nice_match_length = 65;
            }
            7 => {
                strategy = Strategy::Lazy;
                max_search_depth = 100;
                nice_match_length = 130;
            }
            8 => {
                strategy = Strategy::Lazy2;
                max_search_depth = 300;
                nice_match_length = DEFLATE_MAX_MATCH_LEN;
            }
            9 => {
                strategy = Strategy::Lazy2;
                max_search_depth = 600;
                nice_match_length = DEFLATE_MAX_MATCH_LEN;
            }
            10 => {
                strategy = Strategy::NearOptimal;
                max_search_depth = 35;
                nice_match_length = 75;
                max_optim_passes = 2;
                min_improvement_to_continue = 32;
                min_bits_to_use_nonfinal_path = 32;
                max_len_to_optimize_static_block = 0;
            }
            11 => {
                strategy = Strategy::NearOptimal;
                max_search_depth = 100;
                nice_match_length = 150;
                max_optim_passes = 4;
                min_improvement_to_continue = 16;
                min_bits_to_use_nonfinal_path = 16;
                max_len_to_optimize_static_block = 1000;
            }
            _ /* 12 */ => {
                strategy = Strategy::NearOptimal;
                max_search_depth = 300;
                nice_match_length = DEFLATE_MAX_MATCH_LEN;
                max_optim_passes = 10;
                min_improvement_to_continue = 1;
                min_bits_to_use_nonfinal_path = 1;
                max_len_to_optimize_static_block = 10000;
            }
        }

        // Build ONLY the StrategyState arm this level actually uses --
        // mirrors C's union allocation (deflate_compress.c L3877-3913), which
        // sizes/inits only the member the level needs. `new_uninit()` skips
        // the matchfinder's own `init()` sweep because `compress_fastest`/
        // `compress_greedy`/`compress_lazy_generic` all call `.init()`
        // themselves right before first use -- doing it here too would be a
        // second, wasted sweep over the same hash tables.
        let state = match strategy {
            Strategy::None => None,
            Strategy::Fastest => Some(StrategyState::Fastest {
                ht_mf: HtMatchfinder::new_uninit(),
                f_sequences: vec![DeflateSequence::default(); FAST_SEQ_STORE_LENGTH + 1],
            }),
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => Some(StrategyState::Greedy {
                hc_mf: HcMatchfinder::new_uninit(),
                g_sequences: vec![DeflateSequence::default(); SEQ_STORE_LENGTH + 1],
            }),
            Strategy::NearOptimal => {
                let mut st = NearOptimalState {
                    bt_mf: BtMatchfinder::new(),
                    // Extra slots absorb the worst-case overflow (see C comment on the
                    // match_cache array size).
                    match_cache: vec![
                        LzMatch { length: 0, offset: 0 };
                        MATCH_CACHE_LENGTH + MAX_MATCHES_PER_POS + DEFLATE_MAX_MATCH_LEN as usize - 1
                    ],
                    cache_end: 0,
                    optimum_nodes: vec![DeflateOptimumNode::default(); MAX_BLOCK_LENGTH + 1],
                    costs: DeflateCosts::new(),
                    costs_saved: DeflateCosts::new(),
                    offset_slot_full: vec![0u8; DEFLATE_MAX_MATCH_OFFSET as usize + 1],
                    prev_observations: [0; NUM_OBSERVATION_TYPES],
                    prev_num_observations: 0,
                    new_match_len_freqs: vec![0u32; DEFLATE_MAX_MATCH_LEN as usize + 1],
                    match_len_freqs: vec![0u32; DEFLATE_MAX_MATCH_LEN as usize + 1],
                    max_optim_passes,
                    min_improvement_to_continue,
                    min_bits_to_use_nonfinal_path,
                    max_len_to_optimize_static_block,
                };
                deflate_init_offset_slot_full(&mut st.offset_slot_full);
                Some(StrategyState::Near(Box::new(st)))
            }
        };

        let mut c = Compressor {
            strategy,
            compression_level,
            max_passthrough_size,
            max_search_depth,
            nice_match_length,
            freqs: DeflateFreqs::new(),
            split_stats: BlockSplitStats::new(),
            codes: DeflateCodes::new(),
            static_codes: DeflateCodes::new(),
            precode: PrecodeScratch {
                freqs: [0; DEFLATE_NUM_PRECODE_SYMS],
                codewords: [0; DEFLATE_NUM_PRECODE_SYMS],
                lens: [0; DEFLATE_NUM_PRECODE_SYMS],
                items: [0; DEFLATE_NUM_LITLEN_SYMS + DEFLATE_NUM_OFFSET_SYMS],
                num_litlen_syms: 0,
                num_offset_syms: 0,
                num_explicit_lens: 0,
                num_items: 0,
            },
            length_scratch: LengthScratch {
                codewords: [0; DEFLATE_MAX_MATCH_LEN as usize + 1],
                lens: [0; DEFLATE_MAX_MATCH_LEN as usize + 1],
            },
            state,
        };

        c.init_static_codes();
        Ok(c)
    }

    /// Return the compression level this compressor was created with.
    /// Mirrors `libdeflate_get_compression_level`.
    pub fn compression_level(&self) -> u32 {
        self.compression_level
    }

    /// Compress `input` as a raw DEFLATE stream into `out`.
    /// Returns the compressed length in bytes, or 0 if it doesn't fit.
    /// Mirrors `libdeflate_deflate_compress`.
    pub fn deflate_compress(&mut self, input: &[u8], out: &mut [u8]) -> usize {
        // For extremely short inputs, or level 0, just output uncompressed blocks.
        if input.len() <= self.max_passthrough_size {
            return deflate_compress_none(input, out);
        }

        let mut os = OutputBitstream::new(out);

        // Dispatch to the chosen implementation.
        match self.strategy {
            Strategy::None => {
                // Unreachable in practice (level 0 has max_passthrough_size = MAX),
                // but handle defensively to match C's `impl = NULL` not being called.
                return deflate_compress_none(input, os.out);
            }
            Strategy::Fastest => self.compress_fastest(input, &mut os),
            Strategy::Greedy => self.compress_greedy(input, &mut os),
            Strategy::Lazy => self.compress_lazy_generic(input, &mut os, false),
            Strategy::Lazy2 => self.compress_lazy_generic(input, &mut os, true),
            Strategy::NearOptimal => self.compress_near_optimal(input, &mut os),
        }

        if os.overflow {
            return 0;
        }

        // Write the final partial byte if needed.
        debug_assert!(os.bitcount <= 7);
        if os.bitcount != 0 {
            if os.next >= os.end() {
                return 0;
            }
            os.out[os.next] = os.bitbuf as u8;
            os.next += 1;
        }

        os.next
    }

    // ------------------------------------------------------------------------
    // Huffman header / static codes
    // ------------------------------------------------------------------------

    /// `deflate_init_static_codes` — build `static_codes` from the DEFLATE fixed code.
    fn init_static_codes(&mut self) {
        // The fixed-Huffman code tables must be built
        // ONCE per `Compressor` (at construction, `Compressor::new` calls
        // this exactly once) and REUSED for every fixed-Huffman block a
        // later `compress`/`deflate_compress` call emits -- proves the
        // static-codes cache is a cache, not silently rebuilt per block.
        #[cfg(any(test, feature = "probe"))]
        crate::probe::hit(crate::probe::Probe::StaticCodesBuild);

        let mut i = 0usize;
        while i < 144 {
            self.freqs.litlen[i] = 1 << (9 - 8);
            i += 1;
        }
        while i < 256 {
            self.freqs.litlen[i] = 1 << (9 - 9);
            i += 1;
        }
        while i < 280 {
            self.freqs.litlen[i] = 1 << (9 - 7);
            i += 1;
        }
        while i < 288 {
            self.freqs.litlen[i] = 1 << (9 - 8);
            i += 1;
        }
        for i in 0..32 {
            self.freqs.offset[i] = 1 << (5 - 5);
        }
        let codes = make_huffman_codes(&self.freqs);
        self.static_codes = codes;
    }

    /// `deflate_reset_symbol_frequencies`.
    fn reset_symbol_frequencies(&mut self) {
        self.freqs.reset();
    }

    // ------------------------------------------------------------------------
    // Precode / header precompute  (deflate_precompute_huffman_header)
    // ------------------------------------------------------------------------

    /// `deflate_precompute_huffman_header` — compute precode items + build precode.
    ///
    /// Operates on `self.codes.lens` and fills `self.precode`.  The C code temporarily
    /// `memmove`s the offset lengths so litlen+offset lengths are contiguous; our
    /// `DeflateLens.all` array is already contiguous, so we just compute the count and
    /// the contiguous run boundaries directly (no move needed when num_litlen==288,
    /// otherwise we move within `all`).
    fn precompute_huffman_header(&mut self) {
        // Compute how many litlen and offset symbols are needed.
        let mut num_litlen_syms = DEFLATE_NUM_LITLEN_SYMS;
        while num_litlen_syms > 257 {
            if self.codes.lens.litlen(num_litlen_syms - 1) != 0 {
                break;
            }
            num_litlen_syms -= 1;
        }
        let mut num_offset_syms = DEFLATE_NUM_OFFSET_SYMS;
        while num_offset_syms > 1 {
            if self.codes.lens.offset(num_offset_syms - 1) != 0 {
                break;
            }
            num_offset_syms -= 1;
        }
        self.precode.num_litlen_syms = num_litlen_syms;
        self.precode.num_offset_syms = num_offset_syms;

        // Make litlen+offset lengths contiguous starting at `num_litlen_syms`.
        // C moves the offset lengths down to abut the litlen lengths.
        let off_base = DeflateLens::OFFSET_BASE;
        if num_litlen_syms != DEFLATE_NUM_LITLEN_SYMS {
            // memmove(&lens + num_litlen_syms, &lens + 288, num_offset_syms)
            for k in 0..num_offset_syms {
                self.codes.lens.all[num_litlen_syms + k] = self.codes.lens.all[off_base + k];
            }
        }

        // Compute precode items over the contiguous lengths.
        let total = num_litlen_syms + num_offset_syms;
        self.precode.num_items = deflate_compute_precode_items(
            &self.codes.lens.all[..total],
            &mut self.precode.freqs,
            &mut self.precode.items,
        );

        // Build the precode.
        deflate_make_huffman_code(
            DEFLATE_NUM_PRECODE_SYMS,
            MAX_PRE_CODEWORD_LEN as usize,
            &self.precode.freqs,
            &mut self.precode.lens,
            &mut self.precode.codewords,
        );

        // Count how many precode lengths we actually need to output.
        let mut num_explicit_lens = DEFLATE_NUM_PRECODE_SYMS;
        while num_explicit_lens > 4 {
            if self.precode.lens[DEFLATE_PRECODE_LENS_PERMUTATION[num_explicit_lens - 1] as usize]
                != 0
            {
                break;
            }
            num_explicit_lens -= 1;
        }
        self.precode.num_explicit_lens = num_explicit_lens;

        // Restore the offset codeword lengths if we moved them.
        if num_litlen_syms != DEFLATE_NUM_LITLEN_SYMS {
            for k in (0..num_offset_syms).rev() {
                self.codes.lens.all[off_base + k] = self.codes.lens.all[num_litlen_syms + k];
            }
        }
    }

    /// `deflate_compute_full_len_codewords` — precompute "full" match-length codewords.
    fn compute_full_len_codewords(&mut self, codes_is_static: bool) {
        let codes = if codes_is_static {
            &self.static_codes
        } else {
            &self.codes
        };
        for len in (DEFLATE_MIN_MATCH_LEN as usize)..=(DEFLATE_MAX_MATCH_LEN as usize) {
            let slot = DEFLATE_LENGTH_SLOT[len] as usize;
            let litlen_sym = DEFLATE_FIRST_LEN_SYM as usize + slot;
            let extra_bits = len as u32 - DEFLATE_LENGTH_SLOT_BASE[slot];
            self.length_scratch.codewords[len] = codes.codewords.litlen[litlen_sym]
                | (extra_bits << codes.lens.litlen(litlen_sym) as u32);
            self.length_scratch.lens[len] =
                codes.lens.litlen(litlen_sym) + DEFLATE_EXTRA_LENGTH_BITS[slot];
        }
    }
}

/// Free function: build litlen+offset Huffman codes from frequencies.
/// `deflate_make_huffman_codes`.
fn make_huffman_codes(freqs: &DeflateFreqs) -> DeflateCodes {
    let mut codes = DeflateCodes::new();
    // litlen
    {
        let (litlen_lens, _) = codes.lens.all.split_at_mut(DEFLATE_NUM_LITLEN_SYMS);
        deflate_make_huffman_code(
            DEFLATE_NUM_LITLEN_SYMS,
            MAX_LITLEN_CODEWORD_LEN as usize,
            &freqs.litlen,
            litlen_lens,
            &mut codes.codewords.litlen,
        );
    }
    // offset
    {
        let (_, offset_lens) = codes.lens.all.split_at_mut(DEFLATE_NUM_LITLEN_SYMS);
        deflate_make_huffman_code(
            DEFLATE_NUM_OFFSET_SYMS,
            MAX_OFFSET_CODEWORD_LEN as usize,
            &freqs.offset,
            offset_lens,
            &mut codes.codewords.offset,
        );
    }
    codes
}

/// `deflate_compute_precode_items` — RLE-encode the codeword lengths into precode items.
fn deflate_compute_precode_items(
    lens: &[u8],
    precode_freqs: &mut [u32; DEFLATE_NUM_PRECODE_SYMS],
    precode_items: &mut [u32],
) -> usize {
    let num_lens = lens.len();
    *precode_freqs = [0; DEFLATE_NUM_PRECODE_SYMS];

    let mut itemptr = 0usize;
    let mut run_start = 0usize;
    loop {
        let len = lens[run_start];

        // Extend the run.
        let mut run_end = run_start;
        loop {
            run_end += 1;
            if run_end == num_lens || len != lens[run_end] {
                break;
            }
        }

        if len == 0 {
            // Run of zeroes.
            // Symbol 18: RLE 11..138 zeroes at a time.
            while (run_end - run_start) >= 11 {
                let extra_bits = core::cmp::min((run_end - run_start) - 11, 0x7F);
                precode_freqs[18] += 1;
                precode_items[itemptr] = 18 | ((extra_bits as u32) << 5);
                itemptr += 1;
                run_start += 11 + extra_bits;
            }
            // Symbol 17: RLE 3..10 zeroes at a time.
            if (run_end - run_start) >= 3 {
                let extra_bits = core::cmp::min((run_end - run_start) - 3, 0x7);
                precode_freqs[17] += 1;
                precode_items[itemptr] = 17 | ((extra_bits as u32) << 5);
                itemptr += 1;
                run_start += 3 + extra_bits;
            }
        } else {
            // A run of nonzero lengths.
            // Symbol 16: RLE 3..6 of the previous length.
            if (run_end - run_start) >= 4 {
                precode_freqs[len as usize] += 1;
                precode_items[itemptr] = len as u32;
                itemptr += 1;
                run_start += 1;
                loop {
                    let extra_bits = core::cmp::min((run_end - run_start) - 3, 0x3);
                    precode_freqs[16] += 1;
                    precode_items[itemptr] = 16 | ((extra_bits as u32) << 5);
                    itemptr += 1;
                    run_start += 3 + extra_bits;
                    if (run_end - run_start) < 3 {
                        break;
                    }
                }
            }
        }

        // Output any remaining lengths without RLE.
        while run_start != run_end {
            precode_freqs[len as usize] += 1;
            precode_items[itemptr] = len as u32;
            itemptr += 1;
            run_start += 1;
        }

        if run_start == num_lens {
            break;
        }
    }

    itemptr
}

// ============================================================================
// Block flushing  (deflate_flush_block / deflate_finish_block)
// ============================================================================

impl Compressor {
    /// `deflate_flush_block` — choose the cheapest block type and output it.
    ///
    /// `sequences` is `Some` for greedy/lazy/fastest (and for the "use only literals"
    /// near-optimal case), or `None` to use `self.n.optimum_nodes` (near-optimal path).
    /// `self.freqs` and `self.codes` must already be set for the block's symbols.
    #[allow(clippy::too_many_arguments)]
    fn flush_block(
        &mut self,
        os: &mut OutputBitstream,
        input: &[u8],
        block_begin: usize,
        block_length: u32,
        sequences: Option<&[DeflateSequence]>,
        is_final_block: bool,
    ) {
        let mut bitbuf = os.bitbuf;
        let mut bitcount = os.bitcount;
        let mut out_next = os.next;
        // Computed ONCE per block, like C's `u8 * const out_fast_end` in
        // `deflate_flush_block` — see `flush_bits!`'s doc comment.
        let out_fast_end = os.end() - core::cmp::min(WORDBYTES - 1, os.end() - out_next);

        let in_end = block_begin + block_length as usize;

        let mut dynamic_cost: u32 = 3;
        let mut static_cost: u32 = 3;
        let mut uncompressed_cost: u32 = 3;

        // Precompute the precode items and build the precode.
        self.precompute_huffman_header();

        // Cost of encoding the dynamic Huffman codes (header).
        dynamic_cost += 5 + 5 + 4 + (3 * self.precode.num_explicit_lens as u32);
        for sym in 0..DEFLATE_NUM_PRECODE_SYMS {
            let extra = DEFLATE_EXTRA_PRECODE_BITS[sym] as u32;
            dynamic_cost += self.precode.freqs[sym] * (extra + self.precode.lens[sym] as u32);
        }

        // Cost of encoding literals.
        for sym in 0..144 {
            dynamic_cost += self.freqs.litlen[sym] * self.codes.lens.litlen(sym) as u32;
            static_cost += self.freqs.litlen[sym] * 8;
        }
        for sym in 144..256 {
            dynamic_cost += self.freqs.litlen[sym] * self.codes.lens.litlen(sym) as u32;
            static_cost += self.freqs.litlen[sym] * 9;
        }

        // Cost of the end-of-block symbol.
        dynamic_cost += self.codes.lens.litlen(DEFLATE_END_OF_BLOCK as usize) as u32;
        static_cost += 7;

        // Cost of encoding lengths.
        for sym in (DEFLATE_FIRST_LEN_SYM as usize)
            ..(DEFLATE_FIRST_LEN_SYM as usize + DEFLATE_EXTRA_LENGTH_BITS.len())
        {
            let extra = DEFLATE_EXTRA_LENGTH_BITS[sym - DEFLATE_FIRST_LEN_SYM as usize] as u32;
            dynamic_cost += self.freqs.litlen[sym] * (extra + self.codes.lens.litlen(sym) as u32);
            static_cost +=
                self.freqs.litlen[sym] * (extra + self.static_codes.lens.litlen(sym) as u32);
        }

        // Cost of encoding offsets.
        for sym in 0..DEFLATE_EXTRA_OFFSET_BITS.len() {
            let extra = DEFLATE_EXTRA_OFFSET_BITS[sym] as u32;
            dynamic_cost += self.freqs.offset[sym] * (extra + self.codes.lens.offset(sym) as u32);
            static_cost += self.freqs.offset[sym] * (extra + 5);
        }

        // Cost of using uncompressed blocks.
        uncompressed_cost += ((bitcount as i32 + 3).wrapping_neg() & 7) as u32
            + 32
            + (40 * (div_round_up_u32(block_length, u16::MAX as u32) - 1))
            + (8 * block_length);

        // Choose the cheapest block type. Ties: uncompressed, then static, then dynamic.
        let best_cost = dynamic_cost.min(static_cost.min(uncompressed_cost));

        // If the block isn't going to fit, stop early.
        if div_round_up_u32(bitcount + best_cost, 8) as usize > os.end() - out_next {
            os.overflow = true;
            return;
        }

        if best_cost == uncompressed_cost {
            // Uncompressed block(s).
            let mut in_next = block_begin;
            loop {
                let mut bfinal = 0u8;
                let mut len = u16::MAX as usize;
                if in_end - in_next <= u16::MAX as usize {
                    bfinal = is_final_block as u8;
                    len = in_end - in_next;
                }
                // Output BFINAL + BTYPE (UNCOMPRESSED == 0), then align to byte.
                os.out[out_next] = (bfinal << bitcount) | (bitbuf as u8);
                out_next += 1;
                if bitcount > 5 {
                    os.out[out_next] = 0;
                    out_next += 1;
                }
                bitbuf = 0;
                bitcount = 0;
                // LEN and NLEN, then data.
                put_unaligned_le16(len as u16, &mut os.out[out_next..]);
                out_next += 2;
                put_unaligned_le16(!(len as u16), &mut os.out[out_next..]);
                out_next += 2;
                os.out[out_next..out_next + len].copy_from_slice(&input[in_next..in_next + len]);
                out_next += len;
                in_next += len;
                if in_next == in_end {
                    break;
                }
            }
            os.bitbuf = bitbuf;
            os.bitcount = bitcount;
            os.next = out_next;
            return;
        }

        let use_static = best_cost == static_cost;

        if use_static {
            // Static Huffman block.
            add_bits!(bitbuf, bitcount, is_final_block as u32, 1);
            add_bits!(bitbuf, bitcount, DEFLATE_BLOCKTYPE_STATIC_HUFFMAN, 2);
            flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
        } else {
            // Dynamic Huffman block.
            let num_explicit_lens = self.precode.num_explicit_lens;
            let num_precode_items = self.precode.num_items;

            add_bits!(bitbuf, bitcount, is_final_block as u32, 1);
            add_bits!(bitbuf, bitcount, DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, 2);
            add_bits!(bitbuf, bitcount, (self.precode.num_litlen_syms - 257) as u32, 5);
            add_bits!(bitbuf, bitcount, (self.precode.num_offset_syms - 1) as u32, 5);
            add_bits!(bitbuf, bitcount, (num_explicit_lens - 4) as u32, 4);

            // Output precode codeword lengths.
            if can_buffer(3 * (DEFLATE_NUM_PRECODE_SYMS as u32 - 1)) {
                // Merge first len with the previous fields (64-bit path).
                let precode_sym = DEFLATE_PRECODE_LENS_PERMUTATION[0] as usize;
                add_bits!(bitbuf, bitcount, self.precode.lens[precode_sym] as u32, 3);
                flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                let mut i = 1usize;
                loop {
                    let precode_sym = DEFLATE_PRECODE_LENS_PERMUTATION[i] as usize;
                    add_bits!(bitbuf, bitcount, self.precode.lens[precode_sym] as u32, 3);
                    i += 1;
                    if i >= num_explicit_lens {
                        break;
                    }
                }
                flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
            } else {
                flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                let mut i = 0usize;
                loop {
                    let precode_sym = DEFLATE_PRECODE_LENS_PERMUTATION[i] as usize;
                    add_bits!(bitbuf, bitcount, self.precode.lens[precode_sym] as u32, 3);
                    flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                    i += 1;
                    if i >= num_explicit_lens {
                        break;
                    }
                }
            }

            // Output the litlen+offset codeword lengths, encoded by the precode.
            let mut i = 0usize;
            loop {
                let precode_item = self.precode.items[i];
                let precode_sym = (precode_item & 0x1F) as usize;
                add_bits!(
                    bitbuf,
                    bitcount,
                    self.precode.codewords[precode_sym],
                    self.precode.lens[precode_sym] as u32
                );
                add_bits!(
                    bitbuf,
                    bitcount,
                    precode_item >> 5,
                    DEFLATE_EXTRA_PRECODE_BITS[precode_sym] as u32
                );
                flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                i += 1;
                if i >= num_precode_items {
                    break;
                }
            }
        }

        // Output the literals and matches.
        self.compute_full_len_codewords(use_static);

        // Borrow the relevant codes immutably while writing.  To avoid borrow conflicts
        // with `self.length_scratch`, copy out the small codes references via locals.
        // We index `self.codes`/`self.static_codes` and `self.length_scratch` directly.
        match sequences {
            None => {
                // Near-optimal path: output from the minimum-cost path.
                // self.n is Some by construction when strategy == NearOptimal;
                // None only if flush_block were somehow called on a non-near-optimal
                // compressor with sequences==None, which never happens in this codebase.
                let Some(n) = near(&self.state) else { return; };
                let mut node_idx = 0usize;
                let end_idx = block_length as usize;
                loop {
                    let item = n.optimum_nodes[node_idx].item;
                    let length = item & OPTIMUM_LEN_MASK;
                    let offset = item >> OPTIMUM_OFFSET_SHIFT;
                    if length == 1 {
                        // Literal.
                        let (cw, ln) = self.litlen_code(use_static, offset as usize);
                        add_bits!(bitbuf, bitcount, cw, ln);
                        flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                    } else {
                        // Match.
                        let offset_slot = n.offset_slot_full[offset as usize] as usize;
                        let (bb, bc, on) = self.write_match(
                            use_static,
                            length,
                            offset,
                            offset_slot,
                            bitbuf,
                            bitcount,
                            os.out,
                            out_next,
                            out_fast_end,
                        );
                        bitbuf = bb;
                        bitcount = bc;
                        out_next = on;
                    }
                    node_idx += length as usize;
                    if node_idx == end_idx {
                        break;
                    }
                }
            }
            Some(seqs) => {
                // Output literals and matches from the sequences list.
                let mut in_next = block_begin;
                let mut si = 0usize;
                loop {
                    let seq = seqs[si];
                    let mut litrunlen = seq.litrunlen_and_length & SEQ_LITRUNLEN_MASK;
                    let length = seq.litrunlen_and_length >> SEQ_LENGTH_SHIFT;

                    // Output a run of literals (4-at-a-time fast path mirrors C).
                    if can_buffer(4 * MAX_LITLEN_CODEWORD_LEN) {
                        while litrunlen >= 4 {
                            let l0 = input[in_next] as usize; in_next += 1;
                            let (cw, ln) = self.litlen_code(use_static, l0);
                            add_bits!(bitbuf, bitcount, cw, ln);
                            let l1 = input[in_next] as usize; in_next += 1;
                            let (cw, ln) = self.litlen_code(use_static, l1);
                            add_bits!(bitbuf, bitcount, cw, ln);
                            let l2 = input[in_next] as usize; in_next += 1;
                            let (cw, ln) = self.litlen_code(use_static, l2);
                            add_bits!(bitbuf, bitcount, cw, ln);
                            let l3 = input[in_next] as usize; in_next += 1;
                            let (cw, ln) = self.litlen_code(use_static, l3);
                            add_bits!(bitbuf, bitcount, cw, ln);
                            flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                            litrunlen -= 4;
                        }
                        if litrunlen != 0 {
                            litrunlen -= 1;
                            let l = input[in_next] as usize; in_next += 1;
                            let (cw, ln) = self.litlen_code(use_static, l);
                            add_bits!(bitbuf, bitcount, cw, ln);
                            if litrunlen != 0 {
                                litrunlen -= 1;
                                let l = input[in_next] as usize; in_next += 1;
                                let (cw, ln) = self.litlen_code(use_static, l);
                                add_bits!(bitbuf, bitcount, cw, ln);
                                if litrunlen != 0 {
                                    let l = input[in_next] as usize; in_next += 1;
                                    let (cw, ln) = self.litlen_code(use_static, l);
                                    add_bits!(bitbuf, bitcount, cw, ln);
                                }
                            }
                            flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                        }
                    } else {
                        while litrunlen != 0 {
                            let l = input[in_next] as usize; in_next += 1;
                            let (cw, ln) = self.litlen_code(use_static, l);
                            add_bits!(bitbuf, bitcount, cw, ln);
                            flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);
                            litrunlen -= 1;
                        }
                    }

                    if length == 0 {
                        // Last sequence.
                        debug_assert!(in_next == in_end);
                        break;
                    }

                    // Output a match.
                    let (bb, bc, on) = self.write_match(
                        use_static,
                        length,
                        seq.offset as u32,
                        seq.offset_slot as usize,
                        bitbuf,
                        bitcount,
                        os.out,
                        out_next,
                        out_fast_end,
                    );
                    bitbuf = bb;
                    bitcount = bc;
                    out_next = on;
                    in_next += length as usize;
                    si += 1;
                }
            }
        }

        // Output the end-of-block symbol.
        let (cw, ln) = self.litlen_code(use_static, DEFLATE_END_OF_BLOCK as usize);
        add_bits!(bitbuf, bitcount, cw, ln);
        flush_bits!(bitbuf, bitcount, os.out, out_next, out_fast_end);

        os.bitbuf = bitbuf;
        os.bitcount = bitcount;
        os.next = out_next;
    }

    /// Return `(codeword, len)` for litlen symbol `sym` from the active code.
    #[inline(always)]
    fn litlen_code(&self, use_static: bool, sym: usize) -> (u32, u32) {
        let codes = if use_static { &self.static_codes } else { &self.codes };
        (codes.codewords.litlen[sym], codes.lens.litlen(sym) as u32)
    }

    /// `WRITE_MATCH` — emit a match (litlen+extra length, offset sym, extra offset bits).
    ///
    /// Takes the bitstream state by value and returns the updated
    /// `(bitbuf, bitcount, out_next)` so the `add_bits!`/`flush_bits!` macros (which
    /// require plain `ident` lvalues) can operate on locals.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn write_match(
        &self,
        use_static: bool,
        length: u32,
        offset: u32,
        offset_slot: usize,
        mut bitbuf: BitBuf,
        mut bitcount: u32,
        out: &mut [u8],
        mut out_next: usize,
        out_fast_end: usize,
    ) -> (BitBuf, u32, usize) {
        let codes = if use_static { &self.static_codes } else { &self.codes };

        // Litlen symbol + extra length bits (precomputed "full" length codeword).
        add_bits!(
            bitbuf,
            bitcount,
            self.length_scratch.codewords[length as usize],
            self.length_scratch.lens[length as usize] as u32
        );

        if !can_buffer(
            MAX_LITLEN_CODEWORD_LEN
                + DEFLATE_MAX_EXTRA_LENGTH_BITS
                + MAX_OFFSET_CODEWORD_LEN
                + DEFLATE_MAX_EXTRA_OFFSET_BITS,
        ) {
            flush_bits!(bitbuf, bitcount, out, out_next, out_fast_end);
        }

        // Offset symbol.
        add_bits!(
            bitbuf,
            bitcount,
            codes.codewords.offset[offset_slot],
            codes.lens.offset(offset_slot) as u32
        );

        if !can_buffer(MAX_OFFSET_CODEWORD_LEN + DEFLATE_MAX_EXTRA_OFFSET_BITS) {
            flush_bits!(bitbuf, bitcount, out, out_next, out_fast_end);
        }

        // Extra offset bits.
        add_bits!(
            bitbuf,
            bitcount,
            offset - DEFLATE_OFFSET_SLOT_BASE[offset_slot],
            DEFLATE_EXTRA_OFFSET_BITS[offset_slot] as u32
        );

        flush_bits!(bitbuf, bitcount, out, out_next, out_fast_end);
        (bitbuf, bitcount, out_next)
    }

    /// `deflate_finish_block` — tally EOB, build codes, then flush.
    fn finish_block(
        &mut self,
        os: &mut OutputBitstream,
        input: &[u8],
        block_begin: usize,
        block_length: u32,
        use_sequences: bool,
        is_final_block: bool,
    ) {
        self.freqs.litlen[DEFLATE_END_OF_BLOCK as usize] += 1;
        self.codes = make_huffman_codes(&self.freqs);
        if use_sequences {
            // SAFETY of borrows: clone the sequences slice region into the call.
            // We pass an index range so flush_block can re-borrow `input` and `self`.
            // To avoid borrowing both &self.X and &mut self, take the sequences out.
            let seqs = std::mem::take(&mut self.seq_buf_for_flush());
            self.flush_block(os, input, block_begin, block_length, Some(&seqs), is_final_block);
            self.restore_seq_buf(seqs);
        } else {
            self.flush_block(os, input, block_begin, block_length, None, is_final_block);
        }
    }

    /// Helper: which sequence buffer is active for the current strategy.
    /// Returns an owned Vec swapped out of `self` to satisfy the borrow checker
    /// during flushing; `restore_seq_buf` puts it back.
    fn seq_buf_for_flush(&mut self) -> Vec<DeflateSequence> {
        match self.state.as_mut() {
            Some(StrategyState::Fastest { f_sequences, .. }) => std::mem::take(f_sequences),
            Some(StrategyState::Greedy { g_sequences, .. }) => std::mem::take(g_sequences),
            _ => {
                // Only called with use_sequences==true, which only happens on
                // the Fastest/Greedy(-family) paths -- see `finish_block`.
                debug_assert!(false, "seq_buf_for_flush called without an active sequence-based strategy");
                Vec::new()
            }
        }
    }

    fn restore_seq_buf(&mut self, seqs: Vec<DeflateSequence>) {
        match self.state.as_mut() {
            Some(StrategyState::Fastest { f_sequences, .. }) => *f_sequences = seqs,
            Some(StrategyState::Greedy { g_sequences, .. }) => *g_sequences = seqs,
            _ => debug_assert!(false, "restore_seq_buf called without an active sequence-based strategy"),
        }
    }
}

/// `div_round_up` for `u32` operands (header arithmetic uses 32-bit values).
#[inline(always)]
fn div_round_up_u32(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}

// ============================================================================
// Block split statistics  (init / observe / merge / do_end_block_check)
// ============================================================================

/// `init_block_split_stats`.
fn init_block_split_stats(stats: &mut BlockSplitStats) {
    for i in 0..NUM_OBSERVATION_TYPES {
        stats.new_observations[i] = 0;
        stats.observations[i] = 0;
    }
    stats.num_new_observations = 0;
    stats.num_observations = 0;
}

/// `observe_literal` — top 2 bits + low 1 bit → 8 literal observation types.
#[inline(always)]
fn observe_literal(stats: &mut BlockSplitStats, lit: u8) {
    let idx = (((lit >> 5) & 0x6) | (lit & 1)) as usize;
    stats.new_observations[idx] += 1;
    stats.num_new_observations += 1;
}

/// `observe_match` — short vs long match (length >= 9) → 2 match observation types.
#[inline(always)]
fn observe_match(stats: &mut BlockSplitStats, length: u32) {
    let idx = NUM_LITERAL_OBSERVATION_TYPES + (length >= 9) as usize;
    stats.new_observations[idx] += 1;
    stats.num_new_observations += 1;
}

/// `merge_new_observations`.
fn merge_new_observations(stats: &mut BlockSplitStats) {
    for i in 0..NUM_OBSERVATION_TYPES {
        stats.observations[i] += stats.new_observations[i];
        stats.new_observations[i] = 0;
    }
    stats.num_observations += stats.num_new_observations;
    stats.num_new_observations = 0;
}

/// `do_end_block_check` — decide whether the probability distribution shifted enough.
fn do_end_block_check(stats: &mut BlockSplitStats, block_length: u32) -> bool {
    if stats.num_observations > 0 {
        let mut total_delta: u32 = 0;
        for i in 0..NUM_OBSERVATION_TYPES {
            let expected = stats.observations[i] * stats.num_new_observations;
            let actual = stats.new_observations[i] * stats.num_observations;
            // |actual - expected| (matches the C branchy abs; abs_diff is identical).
            total_delta += actual.abs_diff(expected);
        }

        let num_items = stats.num_observations + stats.num_new_observations;
        let mut cutoff = stats.num_new_observations * 200 / 512 * stats.num_observations;

        if block_length < 10000 && num_items < 8192 {
            cutoff += (cutoff as u64 * (8192 - num_items) as u64 / 8192) as u32;
        }

        if total_delta + (block_length / 4096) * stats.num_observations >= cutoff {
            return true;
        }
    }
    merge_new_observations(stats);
    false
}

/// `ready_to_check_block`.
#[inline]
fn ready_to_check_block(
    stats: &BlockSplitStats,
    in_block_begin: usize,
    in_next: usize,
    in_end: usize,
) -> bool {
    stats.num_new_observations >= NUM_OBSERVATIONS_PER_BLOCK_CHECK
        && (in_next - in_block_begin) >= MIN_BLOCK_LENGTH
        && (in_end - in_next) >= MIN_BLOCK_LENGTH
}

/// `should_end_block`.
#[inline]
fn should_end_block(
    stats: &mut BlockSplitStats,
    in_block_begin: usize,
    in_next: usize,
    in_end: usize,
) -> bool {
    if !ready_to_check_block(stats, in_block_begin, in_next, in_end) {
        return false;
    }
    do_end_block_check(stats, (in_next - in_block_begin) as u32)
}

// ============================================================================
// Sequence helpers  (deflate_begin_sequences / choose_literal / choose_match)
// ============================================================================

/// `deflate_choose_literal` — tally a literal into freqs + (optionally) split stats,
/// and extend the current sequence's literal run.
#[inline(always)]
fn choose_literal(
    freqs: &mut DeflateFreqs,
    split_stats: &mut BlockSplitStats,
    literal: usize,
    gather_split_stats: bool,
    seq: &mut DeflateSequence,
) {
    freqs.litlen[literal] += 1;
    if gather_split_stats {
        observe_literal(split_stats, literal as u8);
    }
    seq.litrunlen_and_length += 1;
}

/// `deflate_choose_match` — tally a match, finalize the current sequence, advance to next.
/// Returns the new sequence index (caller advances its `seq` pointer).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn choose_match(
    freqs: &mut DeflateFreqs,
    split_stats: &mut BlockSplitStats,
    length: u32,
    offset: u32,
    gather_split_stats: bool,
    sequences: &mut [DeflateSequence],
    seq_idx: &mut usize,
) {
    let length_slot = DEFLATE_LENGTH_SLOT[length as usize] as usize;
    let offset_slot = deflate_get_offset_slot(offset);

    freqs.litlen[DEFLATE_FIRST_LEN_SYM as usize + length_slot] += 1;
    freqs.offset[offset_slot as usize] += 1;
    if gather_split_stats {
        observe_match(split_stats, length);
    }

    let i = *seq_idx;
    sequences[i].litrunlen_and_length |= length << SEQ_LENGTH_SHIFT;
    sequences[i].offset = offset as u16;
    sequences[i].offset_slot = offset_slot as u16;

    sequences[i + 1].litrunlen_and_length = 0;
    *seq_idx = i + 1;
}

/// `adjust_max_and_nice_len`.
#[inline(always)]
fn adjust_max_and_nice_len(max_len: &mut u32, nice_len: &mut u32, remaining: usize) {
    if remaining < DEFLATE_MAX_MATCH_LEN as usize {
        *max_len = remaining as u32;
        *nice_len = core::cmp::min(*nice_len, *max_len);
    }
}

/// `choose_min_match_len`.
fn choose_min_match_len(num_used_literals: u32, max_search_depth: u32) -> u32 {
    // min_lens[num_used_literals]; the rest (>= len) is implicitly 3.
    const MIN_LENS: [u8; 80] = [
        9, 9, 9, 9, 9, 9, 8, 8, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ];
    if num_used_literals as usize >= MIN_LENS.len() {
        return 3;
    }
    let mut min_len = MIN_LENS[num_used_literals as usize] as u32;
    if max_search_depth < 16 {
        if max_search_depth < 5 {
            min_len = core::cmp::min(min_len, 4);
        } else if max_search_depth < 10 {
            min_len = core::cmp::min(min_len, 5);
        } else {
            min_len = core::cmp::min(min_len, 7);
        }
    }
    min_len
}

/// `calculate_min_match_len`.
fn calculate_min_match_len(data: &[u8], data_len: usize, max_search_depth: u32) -> u32 {
    if data_len < 512 {
        return DEFLATE_MIN_MATCH_LEN;
    }
    let scan = core::cmp::min(data_len, 4096);
    let mut used = [0u8; 256];
    for &b in &data[..scan] {
        used[b as usize] = 1;
    }
    let mut num_used_literals = 0u32;
    for &u in used.iter() {
        num_used_literals += u as u32;
    }
    choose_min_match_len(num_used_literals, max_search_depth)
}

/// `recalculate_min_match_len`.
fn recalculate_min_match_len(freqs: &DeflateFreqs, max_search_depth: u32) -> u32 {
    let mut literal_freq = 0u32;
    for i in 0..DEFLATE_NUM_LITERALS as usize {
        literal_freq += freqs.litlen[i];
    }
    let cutoff = literal_freq >> 10;
    let mut num_used_literals = 0u32;
    for i in 0..DEFLATE_NUM_LITERALS as usize {
        if freqs.litlen[i] > cutoff {
            num_used_literals += 1;
        }
    }
    choose_min_match_len(num_used_literals, max_search_depth)
}

/// `choose_max_block_end`.
#[inline]
fn choose_max_block_end(in_block_begin: usize, in_end: usize, soft_max_len: usize) -> usize {
    if in_end - in_block_begin < soft_max_len + MIN_BLOCK_LENGTH {
        in_end
    } else {
        in_block_begin + soft_max_len
    }
}

// ============================================================================
// Level 0: deflate_compress_none  (stored blocks)
// ============================================================================

/// `deflate_compress_none` — output only uncompressed blocks.
/// Returns the compressed length, or 0 if the output buffer is too small.
fn deflate_compress_none(input: &[u8], out: &mut [u8]) -> usize {
    let in_nbytes = input.len();
    let out_avail = out.len();
    let mut out_next = 0usize;

    // Zero-length input still needs one (empty) block.
    if in_nbytes == 0 {
        if out_avail < 5 {
            return 0;
        }
        out[0] = 1 | ((DEFLATE_BLOCKTYPE_UNCOMPRESSED as u8) << 1);
        put_unaligned_le32(0xFFFF0000, &mut out[1..]);
        return 5;
    }

    let mut in_next = 0usize;
    let in_end = in_nbytes;
    loop {
        let mut bfinal = 0u8;
        let mut len = u16::MAX as usize;
        if in_end - in_next <= u16::MAX as usize {
            bfinal = 1;
            len = in_end - in_next;
        }
        if out_avail - out_next < 5 + len {
            return 0;
        }
        out[out_next] = bfinal | ((DEFLATE_BLOCKTYPE_UNCOMPRESSED as u8) << 1);
        out_next += 1;
        put_unaligned_le16(len as u16, &mut out[out_next..]);
        out_next += 2;
        put_unaligned_le16(!(len as u16), &mut out[out_next..]);
        out_next += 2;
        out[out_next..out_next + len].copy_from_slice(&input[in_next..in_next + len]);
        out_next += len;
        in_next += len;
        if in_next == in_end {
            break;
        }
    }
    out_next
}

// ============================================================================
// Level 1: deflate_compress_fastest
// ============================================================================

impl Compressor {
    /// `deflate_compress_fastest` — ht-matchfinder, fixed-length blocks, no splitting.
    fn compress_fastest(&mut self, input: &[u8], os: &mut OutputBitstream) {
        let in_end = input.len();
        let mut in_next = 0usize;
        let mut in_base_pos = 0usize;
        let mut max_len = DEFLATE_MAX_MATCH_LEN;
        let mut nice_len = core::cmp::min(self.nice_match_length, max_len);
        let mut next_hash: u32 = 0;

        let Some((ht_mf, _)) = fastest_mut(&mut self.state) else { return; };
        ht_mf.init();

        loop {
            let in_block_begin = in_next;
            let in_max_block_end =
                choose_max_block_end(in_next, in_end, FAST_SOFT_MAX_BLOCK_LENGTH);

            // Begin sequences.
            self.reset_symbol_frequencies();
            // Re-borrow each iteration: `finish_block` below needs `&mut self`
            // as a whole, so `ht_mf`/`f_sequences` (borrowed from `self.state`)
            // must not still be live across that call -- see the free-function
            // accessors' comment above `near`.
            let Some((ht_mf, f_sequences)) = fastest_mut(&mut self.state) else { return; };
            let mut seq_idx = 0usize;
            f_sequences[0].litrunlen_and_length = 0;

            // Note: like libdeflate, `next_hash` is NOT precomputed here; it starts at
            // 0 and the first matchfinder call uses hash 0 for the block's first
            // position, then fills in the real hash.  This is deterministic and the
            // decoder is agnostic to which hash the encoder used.

            loop {
                let remaining = in_end - in_next;
                if remaining < DEFLATE_MAX_MATCH_LEN as usize {
                    max_len = remaining as u32;
                    if max_len < HT_MATCHFINDER_REQUIRED_NBYTES as u32 {
                        // Emit remaining bytes as literals.
                        let mut left = max_len;
                        while left != 0 {
                            let lit = input[in_next] as usize;
                            in_next += 1;
                            choose_literal(
                                &mut self.freqs,
                                &mut self.split_stats,
                                lit,
                                false,
                                &mut f_sequences[seq_idx],
                            );
                            left -= 1;
                        }
                        break;
                    }
                    nice_len = core::cmp::min(nice_len, max_len);
                }

                let mut offset = 0u32;
                let length = ht_matchfinder_longest_match(
                    ht_mf,
                    input,
                    &mut in_base_pos,
                    in_next,
                    max_len,
                    nice_len,
                    &mut next_hash,
                    &mut offset,
                );

                if length != 0 {
                    choose_match(
                        &mut self.freqs,
                        &mut self.split_stats,
                        length,
                        offset,
                        false,
                        f_sequences,
                        &mut seq_idx,
                    );
                    ht_matchfinder_skip_bytes(
                        ht_mf,
                        input,
                        &mut in_base_pos,
                        in_next + 1,
                        in_end,
                        length - 1,
                        &mut next_hash,
                    );
                    in_next += length as usize;
                } else {
                    let lit = input[in_next] as usize;
                    in_next += 1;
                    choose_literal(
                        &mut self.freqs,
                        &mut self.split_stats,
                        lit,
                        false,
                        &mut f_sequences[seq_idx],
                    );
                }

                if !(in_next < in_max_block_end && seq_idx < FAST_SEQ_STORE_LENGTH) {
                    break;
                }
            }

            let block_length = (in_next - in_block_begin) as u32;
            let is_final = in_next == in_end;
            self.finish_block(os, input, in_block_begin, block_length, true, is_final);

            // Note (mirrors libdeflate): max_len/nice_len are NOT reset per block; they
            // are only ever decreased near end-of-input by adjust_max_and_nice_len, after
            // which the outer loop will not start another block.

            if in_next == in_end || os.overflow {
                break;
            }
        }
    }
}

// ============================================================================
// Levels 2-4: deflate_compress_greedy
// ============================================================================

impl Compressor {
    /// `deflate_compress_greedy` — hc-matchfinder, always choose the longest match.
    fn compress_greedy(&mut self, input: &[u8], os: &mut OutputBitstream) {
        let in_end = input.len();
        let mut in_next = 0usize;
        let mut in_base_pos = 0usize;
        let mut max_len = DEFLATE_MAX_MATCH_LEN;
        let mut nice_len = core::cmp::min(self.nice_match_length, max_len);
        let mut next_hashes = [0u32; 2];

        let Some((hc_mf, _)) = greedy_mut(&mut self.state) else { return; };
        hc_mf.init();

        loop {
            let in_block_begin = in_next;
            let in_max_block_end = choose_max_block_end(in_next, in_end, SOFT_MAX_BLOCK_LENGTH);

            init_block_split_stats(&mut self.split_stats);
            self.reset_symbol_frequencies();
            // Re-borrow each iteration -- see the comment in `compress_fastest`.
            let Some((hc_mf, g_sequences)) = greedy_mut(&mut self.state) else { return; };
            let mut seq_idx = 0usize;
            g_sequences[0].litrunlen_and_length = 0;

            // next_hashes is NOT precomputed here (mirrors libdeflate, which keeps the
            // {0,0} value carried across blocks); the first matchfinder call fills it in.

            let min_len =
                calculate_min_match_len(&input[in_next..], in_max_block_end - in_next, self.max_search_depth);

            loop {
                adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                let mut offset = 0u32;
                let length = hc_matchfinder_longest_match(
                    hc_mf,
                    input,
                    &mut in_base_pos,
                    in_next,
                    min_len - 1,
                    max_len,
                    nice_len,
                    self.max_search_depth,
                    &mut next_hashes,
                    &mut offset,
                );

                if length >= min_len && (length > DEFLATE_MIN_MATCH_LEN || offset <= 4096) {
                    choose_match(
                        &mut self.freqs,
                        &mut self.split_stats,
                        length,
                        offset,
                        true,
                        g_sequences,
                        &mut seq_idx,
                    );
                    hc_matchfinder_skip_bytes(
                        hc_mf,
                        input,
                        &mut in_base_pos,
                        in_next + 1,
                        in_end,
                        length - 1,
                        &mut next_hashes,
                    );
                    in_next += length as usize;
                } else {
                    let lit = input[in_next] as usize;
                    in_next += 1;
                    choose_literal(
                        &mut self.freqs,
                        &mut self.split_stats,
                        lit,
                        true,
                        &mut g_sequences[seq_idx],
                    );
                }

                if !(in_next < in_max_block_end
                    && seq_idx < SEQ_STORE_LENGTH
                    && !should_end_block(&mut self.split_stats, in_block_begin, in_next, in_end))
                {
                    break;
                }
            }

            let block_length = (in_next - in_block_begin) as u32;
            let is_final = in_next == in_end;
            self.finish_block(os, input, in_block_begin, block_length, true, is_final);

            // max_len/nice_len carry over (mirrors libdeflate; only decreased near EOF).

            if in_next == in_end || os.overflow {
                break;
            }
        }
    }
}

// ============================================================================
// Levels 5-9: deflate_compress_lazy / lazy2
// ============================================================================

impl Compressor {
    /// `deflate_compress_lazy_generic` — lazy (and lazy2) parsing.
    fn compress_lazy_generic(&mut self, input: &[u8], os: &mut OutputBitstream, lazy2: bool) {
        let in_end = input.len();
        let mut in_next = 0usize;
        let mut in_base_pos = 0usize;
        let mut max_len = DEFLATE_MAX_MATCH_LEN;
        let mut nice_len = core::cmp::min(self.nice_match_length, max_len);
        let mut next_hashes = [0u32; 2];

        let Some((hc_mf, _)) = greedy_mut(&mut self.state) else { return; };
        hc_mf.init();

        loop {
            let in_block_begin = in_next;
            let in_max_block_end = choose_max_block_end(in_next, in_end, SOFT_MAX_BLOCK_LENGTH);
            let mut next_recalc_min_len = in_next + core::cmp::min(in_end - in_next, 10000);

            init_block_split_stats(&mut self.split_stats);
            self.reset_symbol_frequencies();
            // Re-borrow each iteration -- see the comment in `compress_fastest`.
            let Some((hc_mf, g_sequences)) = greedy_mut(&mut self.state) else { return; };
            let mut seq_idx = 0usize;
            g_sequences[0].litrunlen_and_length = 0;

            // next_hashes is NOT precomputed (mirrors libdeflate).

            let mut min_len =
                calculate_min_match_len(&input[in_next..], in_max_block_end - in_next, self.max_search_depth);

            loop {
                // Recalculate min_len periodically.
                if in_next >= next_recalc_min_len {
                    min_len = recalculate_min_match_len(&self.freqs, self.max_search_depth);
                    next_recalc_min_len += core::cmp::min(
                        in_end - next_recalc_min_len,
                        in_next - in_block_begin,
                    );
                }

                // Find the longest match at the current position.
                adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                let mut cur_offset = 0u32;
                let mut cur_len = hc_matchfinder_longest_match(
                    hc_mf,
                    input,
                    &mut in_base_pos,
                    in_next,
                    min_len - 1,
                    max_len,
                    nice_len,
                    self.max_search_depth,
                    &mut next_hashes,
                    &mut cur_offset,
                );
                if cur_len < min_len
                    || (cur_len == DEFLATE_MIN_MATCH_LEN && cur_offset > 8192)
                {
                    // No (good) match — emit a literal.
                    let lit = input[in_next] as usize;
                    in_next += 1;
                    choose_literal(
                        &mut self.freqs,
                        &mut self.split_stats,
                        lit,
                        true,
                        &mut g_sequences[seq_idx],
                    );
                    if !(in_next < in_max_block_end
                        && seq_idx < SEQ_STORE_LENGTH
                        && !should_end_block(&mut self.split_stats, in_block_begin, in_next, in_end))
                    {
                        break;
                    }
                    continue;
                }
                in_next += 1;

                // have_cur_match:
                'have_cur_match: loop {
                    // If the match is very long, choose it immediately.
                    if cur_len >= nice_len {
                        choose_match(
                            &mut self.freqs,
                            &mut self.split_stats,
                            cur_len,
                            cur_offset,
                            true,
                            g_sequences,
                            &mut seq_idx,
                        );
                        hc_matchfinder_skip_bytes(
                            hc_mf,
                            input,
                            &mut in_base_pos,
                            in_next,
                            in_end,
                            cur_len - 1,
                            &mut next_hashes,
                        );
                        in_next += (cur_len - 1) as usize;
                        break 'have_cur_match;
                    }

                    // Try to find a better match at the next position (half depth).
                    adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                    let mut next_offset = 0u32;
                    let next_len = hc_matchfinder_longest_match(
                        hc_mf,
                        input,
                        &mut in_base_pos,
                        in_next,
                        cur_len - 1,
                        max_len,
                        nice_len,
                        self.max_search_depth >> 1,
                        &mut next_hashes,
                        &mut next_offset,
                    );
                    in_next += 1;
                    if next_len >= cur_len
                        && 4 * (next_len as i32 - cur_len as i32)
                            + (bsr32(cur_offset) as i32 - bsr32(next_offset) as i32)
                            > 2
                    {
                        // Better match at the next position: output a literal, advance.
                        let lit = input[in_next - 2] as usize;
                        choose_literal(
                            &mut self.freqs,
                            &mut self.split_stats,
                            lit,
                            true,
                            &mut g_sequences[seq_idx],
                        );
                        cur_len = next_len;
                        cur_offset = next_offset;
                        continue 'have_cur_match;
                    }

                    if lazy2 {
                        // lazy2: look ahead another position (quarter depth).
                        adjust_max_and_nice_len(&mut max_len, &mut nice_len, in_end - in_next);
                        let mut next_offset2 = 0u32;
                        let next_len2 = hc_matchfinder_longest_match(
                            hc_mf,
                            input,
                            &mut in_base_pos,
                            in_next,
                            cur_len - 1,
                            max_len,
                            nice_len,
                            self.max_search_depth >> 2,
                            &mut next_hashes,
                            &mut next_offset2,
                        );
                        in_next += 1;
                        if next_len2 >= cur_len
                            && 4 * (next_len2 as i32 - cur_len as i32)
                                + (bsr32(cur_offset) as i32 - bsr32(next_offset2) as i32)
                                > 6
                        {
                            // Much better match two positions ahead: use two literals.
                            let l0 = input[in_next - 3] as usize;
                            choose_literal(
                                &mut self.freqs,
                                &mut self.split_stats,
                                l0,
                                true,
                                &mut g_sequences[seq_idx],
                            );
                            let l1 = input[in_next - 2] as usize;
                            choose_literal(
                                &mut self.freqs,
                                &mut self.split_stats,
                                l1,
                                true,
                                &mut g_sequences[seq_idx],
                            );
                            cur_len = next_len2;
                            cur_offset = next_offset2;
                            continue 'have_cur_match;
                        }
                        // No better match: output the current match.
                        choose_match(
                            &mut self.freqs,
                            &mut self.split_stats,
                            cur_len,
                            cur_offset,
                            true,
                            g_sequences,
                            &mut seq_idx,
                        );
                        if cur_len > 3 {
                            hc_matchfinder_skip_bytes(
                                hc_mf,
                                input,
                                &mut in_base_pos,
                                in_next,
                                in_end,
                                cur_len - 3,
                                &mut next_hashes,
                            );
                            in_next += (cur_len - 3) as usize;
                        }
                    } else {
                        // !lazy2: output the current match.
                        choose_match(
                            &mut self.freqs,
                            &mut self.split_stats,
                            cur_len,
                            cur_offset,
                            true,
                            g_sequences,
                            &mut seq_idx,
                        );
                        hc_matchfinder_skip_bytes(
                            hc_mf,
                            input,
                            &mut in_base_pos,
                            in_next,
                            in_end,
                            cur_len - 2,
                            &mut next_hashes,
                        );
                        in_next += (cur_len - 2) as usize;
                    }
                    break 'have_cur_match;
                }

                if !(in_next < in_max_block_end
                    && seq_idx < SEQ_STORE_LENGTH
                    && !should_end_block(&mut self.split_stats, in_block_begin, in_next, in_end))
                {
                    break;
                }
            }

            let block_length = (in_next - in_block_begin) as u32;
            let is_final = in_next == in_end;
            self.finish_block(os, input, in_block_begin, block_length, true, is_final);

            // max_len/nice_len carry over (mirrors libdeflate; only decreased near EOF).

            if in_next == in_end || os.overflow {
                break;
            }
        }
    }
}

// ============================================================================
// Near-optimal parser  (levels 10-12)
// ============================================================================

/// `deflate_init_offset_slot_full` — fill the full offset→slot map.
fn deflate_init_offset_slot_full(offset_slot_full: &mut [u8]) {
    for offset_slot in 0..DEFLATE_OFFSET_SLOT_BASE.len() {
        let offset = DEFLATE_OFFSET_SLOT_BASE[offset_slot];
        let offset_end = offset + (1 << DEFLATE_EXTRA_OFFSET_BITS[offset_slot]);
        let mut o = offset;
        loop {
            offset_slot_full[o as usize] = offset_slot as u8;
            o += 1;
            if o == offset_end {
                break;
            }
        }
    }
}

impl Compressor {
    /// `deflate_choose_default_litlen_costs`.
    fn choose_default_litlen_costs(
        &mut self,
        input: &[u8],
        block_begin: usize,
        block_length: u32,
    ) -> (u32, u32) {
        let mut num_used_literals = 0u32;
        let mut literal_freq = block_length;

        // Count distinct literals in the block.
        for v in self.freqs.litlen[..DEFLATE_NUM_LITERALS as usize].iter_mut() {
            *v = 0;
        }
        let cutoff = literal_freq >> 11;
        for i in 0..block_length as usize {
            self.freqs.litlen[input[block_begin + i] as usize] += 1;
        }
        for i in 0..DEFLATE_NUM_LITERALS as usize {
            if self.freqs.litlen[i] > cutoff {
                num_used_literals += 1;
            }
        }
        if num_used_literals == 0 {
            num_used_literals = 1;
        }

        // Estimate relative match/literal frequency from the greedy parse stats.
        let mut match_freq = 0u32;
        {
            // self.n is Some by construction for all callers (near-optimal path only).
            let Some(n) = near(&self.state) else { return (0, 0); };
            let start = choose_min_match_len(num_used_literals, self.max_search_depth) as usize;
            for i in start..n.match_len_freqs.len() {
                match_freq += n.match_len_freqs[i];
                literal_freq = literal_freq.wrapping_sub(i as u32 * n.match_len_freqs[i]);
            }
        }
        if (literal_freq as i32) < 0 {
            literal_freq = 0;
        }

        let i = if match_freq > literal_freq {
            2 // many matches
        } else if match_freq * 4 > literal_freq {
            1 // neutral
        } else {
            0 // few matches
        };

        let lit_cost = DEFAULT_LITLEN_COSTS[i].used_lits_to_lit_cost[num_used_literals as usize] as u32;
        let len_sym_cost = DEFAULT_LITLEN_COSTS[i].len_sym_cost as u32;
        (lit_cost, len_sym_cost)
    }

    /// `deflate_set_default_costs`.
    fn set_default_costs(&mut self, lit_cost: u32, len_sym_cost: u32) {
        // self.n is Some by construction for all callers (near-optimal path only).
        let Some(n) = near_mut(&mut self.state) else { return; };
        for i in 0..DEFLATE_NUM_LITERALS as usize {
            n.costs.literal[i] = lit_cost;
        }
        for i in (DEFLATE_MIN_MATCH_LEN as usize)..=(DEFLATE_MAX_MATCH_LEN as usize) {
            n.costs.length[i] = deflate_default_length_cost(i as u32, len_sym_cost);
        }
        for i in 0..DEFLATE_OFFSET_SLOT_BASE.len() {
            n.costs.offset_slot[i] = deflate_default_offset_slot_cost(i);
        }
    }

    /// `deflate_adjust_costs`.
    fn adjust_costs(&mut self, lit_cost: u32, len_sym_cost: u32) {
        let mut total_delta: u64 = 0;
        {
            // self.n is Some by construction for all callers (near-optimal path only).
            let Some(n) = near(&self.state) else { return; };
            for i in 0..NUM_OBSERVATION_TYPES {
                let prev = n.prev_observations[i] as u64 * self.split_stats.num_observations as u64;
                let cur = self.split_stats.observations[i] as u64 * n.prev_num_observations as u64;
                total_delta += prev.abs_diff(cur);
            }
        }
        let cutoff = {
            // self.n is Some by construction; already checked above.
            let Some(n) = near(&self.state) else { return; };
            (n.prev_num_observations as u64 * self.split_stats.num_observations as u64 * 200) / 512
        };

        if total_delta > 3 * cutoff {
            self.set_default_costs(lit_cost, len_sym_cost);
        } else if 4 * total_delta > 9 * cutoff {
            self.adjust_costs_impl(lit_cost, len_sym_cost, 3);
        } else if 2 * total_delta > 3 * cutoff {
            self.adjust_costs_impl(lit_cost, len_sym_cost, 2);
        } else if 2 * total_delta > cutoff {
            self.adjust_costs_impl(lit_cost, len_sym_cost, 1);
        } else {
            self.adjust_costs_impl(lit_cost, len_sym_cost, 0);
        }
    }

    /// `deflate_adjust_costs_impl`.
    fn adjust_costs_impl(&mut self, lit_cost: u32, len_sym_cost: u32, change_amount: i32) {
        // self.n is Some by construction for all callers (near-optimal path only).
        let Some(n) = near_mut(&mut self.state) else { return; };
        for i in 0..DEFLATE_NUM_LITERALS as usize {
            deflate_adjust_cost(&mut n.costs.literal[i], lit_cost, change_amount);
        }
        for i in (DEFLATE_MIN_MATCH_LEN as usize)..=(DEFLATE_MAX_MATCH_LEN as usize) {
            let dc = deflate_default_length_cost(i as u32, len_sym_cost);
            deflate_adjust_cost(&mut n.costs.length[i], dc, change_amount);
        }
        for i in 0..DEFLATE_OFFSET_SLOT_BASE.len() {
            let dc = deflate_default_offset_slot_cost(i);
            deflate_adjust_cost(&mut n.costs.offset_slot[i], dc, change_amount);
        }
    }

    /// `deflate_set_initial_costs`.
    fn set_initial_costs(
        &mut self,
        input: &[u8],
        block_begin: usize,
        block_length: u32,
        is_first_block: bool,
    ) {
        let (lit_cost, len_sym_cost) =
            self.choose_default_litlen_costs(input, block_begin, block_length);
        if is_first_block {
            self.set_default_costs(lit_cost, len_sym_cost);
        } else {
            self.adjust_costs(lit_cost, len_sym_cost);
        }
    }

    /// `deflate_tally_item_list` — tally the min-cost path into freqs.
    fn tally_item_list(&mut self, block_length: u32) {
        let mut node_idx = 0usize;
        let end_idx = block_length as usize;
        loop {
            // Re-borrow `n` immutably for each read so the subsequent `self.freqs`
            // mutation does not conflict with a long-lived borrow.
            let (length, offset) = {
                // self.n is Some by construction; tally_item_list is only called on
                // the near-optimal path.
                let Some(n) = near(&self.state) else { break; };
                let item = n.optimum_nodes[node_idx].item;
                (item & OPTIMUM_LEN_MASK, item >> OPTIMUM_OFFSET_SHIFT)
            };
            if length == 1 {
                self.freqs.litlen[offset as usize] += 1;
            } else {
                let ls = DEFLATE_LENGTH_SLOT[length as usize] as usize;
                self.freqs.litlen[DEFLATE_FIRST_LEN_SYM as usize + ls] += 1;
                // self.n is Some (same loop iteration already checked above).
                let os = if let Some(n) = near(&self.state) { n.offset_slot_full[offset as usize] as usize } else { break; };
                self.freqs.offset[os] += 1;
            }
            node_idx += length as usize;
            if node_idx == end_idx {
                break;
            }
        }
        self.freqs.litlen[DEFLATE_END_OF_BLOCK as usize] += 1;
    }

    /// `deflate_choose_all_literals`.
    fn choose_all_literals(&mut self, input: &[u8], block_begin: usize, block_length: u32) {
        self.reset_symbol_frequencies();
        for i in 0..block_length as usize {
            self.freqs.litlen[input[block_begin + i] as usize] += 1;
        }
        self.freqs.litlen[DEFLATE_END_OF_BLOCK as usize] += 1;
        self.codes = make_huffman_codes(&self.freqs);
    }

    /// `deflate_compute_true_cost` — exact dynamic-block cost in whole bits.
    fn compute_true_cost(&mut self) -> u32 {
        let mut cost = 0u32;
        self.precompute_huffman_header();

        // Zero out litlen lengths beyond num_litlen_syms (C memsets them).
        let num_litlen = self.precode.num_litlen_syms;
        for i in num_litlen..DEFLATE_NUM_LITLEN_SYMS {
            self.codes.lens.all[i] = 0;
        }

        cost += 5 + 5 + 4 + (3 * self.precode.num_explicit_lens as u32);
        for sym in 0..DEFLATE_NUM_PRECODE_SYMS {
            cost += self.precode.freqs[sym]
                * (self.precode.lens[sym] as u32 + DEFLATE_EXTRA_PRECODE_BITS[sym] as u32);
        }

        for sym in 0..DEFLATE_FIRST_LEN_SYM as usize {
            cost += self.freqs.litlen[sym] * self.codes.lens.litlen(sym) as u32;
        }
        for sym in (DEFLATE_FIRST_LEN_SYM as usize)
            ..(DEFLATE_FIRST_LEN_SYM as usize + DEFLATE_EXTRA_LENGTH_BITS.len())
        {
            cost += self.freqs.litlen[sym]
                * (self.codes.lens.litlen(sym) as u32
                    + DEFLATE_EXTRA_LENGTH_BITS[sym - DEFLATE_FIRST_LEN_SYM as usize] as u32);
        }
        for sym in 0..DEFLATE_EXTRA_OFFSET_BITS.len() {
            cost += self.freqs.offset[sym]
                * (self.codes.lens.offset(sym) as u32 + DEFLATE_EXTRA_OFFSET_BITS[sym] as u32);
        }
        cost
    }

    /// `deflate_find_min_cost_path` — backward min-cost path search; sets freqs+codes.
    fn find_min_cost_path(&mut self, block_length: u32) {
        {
            // self.n is Some by construction for all callers (near-optimal path only).
            let Some(n) = near_mut(&mut self.state) else { return; };
            let end = block_length as usize;
            n.optimum_nodes[end].cost_to_end = 0;

            // cache_ptr starts just past the matches for position block_length-1...
            // In C cache_ptr is decremented from the end. We track a cache index.
            // The cache layout: for each position there is a run of `num_matches`
            // LzMatch entries followed by a header element whose `length` = num_matches
            // and `offset` = literal byte.  cache_ptr in C points just past the last
            // position's header at entry.  We mirror this with `cache_idx`.
            let mut cache_idx = n.cache_end;

            let mut cur_idx = end;
            loop {
                cur_idx -= 1;
                cache_idx -= 1;

                let num_matches = n.match_cache[cache_idx].length as usize;
                let literal = n.match_cache[cache_idx].offset as u32;

                // Literal is always possible.
                let mut best_cost_to_end =
                    n.costs.literal[literal as usize] + n.optimum_nodes[cur_idx + 1].cost_to_end;
                n.optimum_nodes[cur_idx].item = (literal << OPTIMUM_OFFSET_SHIFT) | 1;

                if num_matches != 0 {
                    // matches occupy cache[cache_idx - num_matches .. cache_idx].
                    let matches_base = cache_idx - num_matches;
                    let mut len = DEFLATE_MIN_MATCH_LEN;
                    let mut mi = 0usize;
                    loop {
                        let m = n.match_cache[matches_base + mi];
                        let offset = m.offset as u32;
                        let offset_slot = n.offset_slot_full[offset as usize] as usize;
                        let offset_cost = n.costs.offset_slot[offset_slot];
                        loop {
                            let cost_to_end = offset_cost
                                + n.costs.length[len as usize]
                                + n.optimum_nodes[cur_idx + len as usize].cost_to_end;
                            if cost_to_end < best_cost_to_end {
                                best_cost_to_end = cost_to_end;
                                n.optimum_nodes[cur_idx].item =
                                    len | (offset << OPTIMUM_OFFSET_SHIFT);
                            }
                            len += 1;
                            if len > m.length as u32 {
                                break;
                            }
                        }
                        mi += 1;
                        if mi == num_matches {
                            break;
                        }
                    }
                    cache_idx -= num_matches;
                }
                n.optimum_nodes[cur_idx].cost_to_end = best_cost_to_end;
                if cur_idx == 0 {
                    break;
                }
            }
        }

        self.reset_symbol_frequencies();
        self.tally_item_list(block_length);
        self.codes = make_huffman_codes(&self.freqs);
    }

    /// `deflate_optimize_and_flush_block`.
    #[allow(clippy::too_many_arguments)]
    fn optimize_and_flush_block(
        &mut self,
        os: &mut OutputBitstream,
        input: &[u8],
        block_begin: usize,
        block_length: u32,
        is_first_block: bool,
        is_final_block: bool,
    ) -> bool {
        // self.n is Some by construction; optimize_and_flush_block is only called
        // from compress_near_optimal, which is only entered when strategy == NearOptimal.
        let (max_optim_passes, min_improvement_to_continue,
             min_bits_to_use_nonfinal_path, max_len_to_optimize_static_block) = {
            let Some(n) = near(&self.state) else { return false; };
            (n.max_optim_passes, n.min_improvement_to_continue,
             n.min_bits_to_use_nonfinal_path, n.max_len_to_optimize_static_block)
        };

        let mut best_true_cost = u32::MAX;
        // `true_cost` is always assigned in the optimization loop (which runs at least
        // once) before any read, so no initializer is needed.
        let mut true_cost: u32;
        let mut static_cost = u32::MAX;

        // Consider using only literals.
        self.choose_all_literals(input, block_begin, block_length);
        let only_lits_cost = self.compute_true_cost();

        // Force the block to end at block_length even if matches extend past it.
        {
            // self.n is Some — already verified at function entry above.
            let Some(n) = near_mut(&mut self.state) else { return false; };
            let nodes_len = n.optimum_nodes.len();
            let hi = core::cmp::min(
                block_length as usize - 1 + DEFLATE_MAX_MATCH_LEN as usize,
                nodes_len - 1,
            );
            for i in block_length as usize..=hi {
                n.optimum_nodes[i].cost_to_end = 0x8000_0000;
            }
        }

        // If the block is small, find the optimal static-block solution and its cost.
        if block_length <= max_len_to_optimize_static_block {
            // self.n is Some — verified at function entry above.
            if let Some(n) = near_mut(&mut self.state) {
                n.costs_saved = n.costs.clone();
                set_costs_from_codes(n, &self.static_codes.lens);
            }
            self.find_min_cost_path(block_length);
            if let Some(n) = near(&self.state) {
                static_cost = n.optimum_nodes[0].cost_to_end / BIT_COST;
                static_cost += 7;
            }
            if let Some(n) = near_mut(&mut self.state) {
                n.costs = n.costs_saved.clone();
            }
        }

        // Initialize costs with default costs.
        self.set_initial_costs(input, block_begin, block_length, is_first_block);

        let mut num_passes_remaining = max_optim_passes;
        loop {
            self.find_min_cost_path(block_length);
            true_cost = self.compute_true_cost();

            if true_cost + min_improvement_to_continue > best_true_cost {
                break;
            }
            best_true_cost = true_cost;
            // self.n is Some — verified at function entry above.
            if let Some(n) = near_mut(&mut self.state) {
                n.costs_saved = n.costs.clone();
                set_costs_from_codes(n, &self.codes.lens);
            }

            num_passes_remaining -= 1;
            if num_passes_remaining == 0 {
                break;
            }
        }

        let mut used_only_literals = false;
        let mut use_sequences_seq: Option<DeflateSequence> = None;

        if core::cmp::min(only_lits_cost, static_cost) < best_true_cost {
            if only_lits_cost < static_cost {
                // Using only literals was best.
                self.choose_all_literals(input, block_begin, block_length);
                if let Some(n) = near_mut(&mut self.state) {
                    set_costs_from_codes(n, &self.codes.lens);
                }
                let seq = DeflateSequence {
                    litrunlen_and_length: block_length,
                    ..DeflateSequence::default()
                };
                use_sequences_seq = Some(seq);
                used_only_literals = true;
            } else {
                // Static block was best.
                if let Some(n) = near_mut(&mut self.state) {
                    set_costs_from_codes(n, &self.static_codes.lens);
                }
                self.find_min_cost_path(block_length);
            }
        } else if true_cost >= best_true_cost + min_bits_to_use_nonfinal_path {
            // The best solution was from a non-final pass; recover it.
            // self.n is Some — verified at function entry above.
            if let Some(n) = near_mut(&mut self.state) { n.costs = n.costs_saved.clone(); }
            self.find_min_cost_path(block_length);
            if let Some(n) = near_mut(&mut self.state) {
                set_costs_from_codes(n, &self.codes.lens);
            }
        }

        match use_sequences_seq {
            Some(seq) => {
                // Output as a single all-literals sequence.
                let seqs = [seq];
                self.flush_block(os, input, block_begin, block_length, Some(&seqs), is_final_block);
            }
            None => {
                self.flush_block(os, input, block_begin, block_length, None, is_final_block);
            }
        }

        used_only_literals
    }

    /// `deflate_near_optimal_init_stats`.
    fn near_optimal_init_stats(&mut self) {
        init_block_split_stats(&mut self.split_stats);
        // self.n is Some by construction; this method is only called on the near-optimal path.
        let Some(n) = near_mut(&mut self.state) else { return; };
        for v in n.new_match_len_freqs.iter_mut() {
            *v = 0;
        }
        for v in n.match_len_freqs.iter_mut() {
            *v = 0;
        }
    }

    /// `deflate_near_optimal_merge_stats`.
    fn near_optimal_merge_stats(&mut self) {
        merge_new_observations(&mut self.split_stats);
        // self.n is Some by construction; this method is only called on the near-optimal path.
        let Some(n) = near_mut(&mut self.state) else { return; };
        for i in 0..n.match_len_freqs.len() {
            n.match_len_freqs[i] += n.new_match_len_freqs[i];
            n.new_match_len_freqs[i] = 0;
        }
    }

    /// `deflate_near_optimal_save_stats`.
    fn near_optimal_save_stats(&mut self) {
        let num_obs = self.split_stats.num_observations;
        let obs = self.split_stats.observations;
        // self.n is Some by construction; this method is only called on the near-optimal path.
        let Some(n) = near_mut(&mut self.state) else { return; };
        n.prev_observations[..NUM_OBSERVATION_TYPES]
            .copy_from_slice(&obs[..NUM_OBSERVATION_TYPES]);
        n.prev_num_observations = num_obs;
    }

    /// `deflate_near_optimal_clear_old_stats`.
    fn near_optimal_clear_old_stats(&mut self) {
        for i in 0..NUM_OBSERVATION_TYPES {
            self.split_stats.observations[i] = 0;
        }
        self.split_stats.num_observations = 0;
        // self.n is Some by construction; this method is only called on the near-optimal path.
        let Some(n) = near_mut(&mut self.state) else { return; };
        for v in n.match_len_freqs.iter_mut() {
            *v = 0;
        }
    }

    /// `deflate_compress_near_optimal`.
    fn compress_near_optimal(&mut self, input: &[u8], os: &mut OutputBitstream) {
        let in_end = input.len();
        let mut in_next = 0usize;
        let mut in_block_begin = 0usize;
        let mut in_base_pos = 0usize;
        let mut in_next_slide =
            core::cmp::min(in_end, MATCHFINDER_WINDOW_SIZE);
        let mut max_len = DEFLATE_MAX_MATCH_LEN;
        let mut nice_len = core::cmp::min(self.nice_match_length, max_len);
        let mut next_hashes = [0u32; 2];
        let mut prev_block_used_only_literals = false;

        // self.n is Some by construction; compress_near_optimal is only called when
        // strategy == NearOptimal, which always initialises self.n in Compressor::new.
        let Some(n_init) = near_mut(&mut self.state) else { return; };
        n_init.bt_mf.init();
        // cache_ptr = match_cache[0]
        n_init.cache_end = 0;
        let mut cache_idx = 0usize;
        self.near_optimal_init_stats();

        // next_hashes is NOT precomputed (mirrors libdeflate's {0,0} initialization).

        loop {
            // Starting a new block.
            let in_max_block_end =
                choose_max_block_end(in_block_begin, in_end, SOFT_MAX_BLOCK_LENGTH);
            let mut prev_end_block_check: Option<usize> = None;
            let mut change_detected = false;
            let mut next_observation = in_next;

            let min_len = if prev_block_used_only_literals {
                DEFLATE_MAX_MATCH_LEN + 1
            } else {
                calculate_min_match_len(
                    &input[in_block_begin..],
                    in_max_block_end - in_block_begin,
                    self.max_search_depth,
                )
            };

            // Find matches until we decide to end the block.
            loop {
                let mut remaining = in_end - in_next;

                // Slide window if needed.
                if in_next == in_next_slide {
                    if let Some(n) = near_mut(&mut self.state) { n.bt_mf.slide_window(); }
                    in_base_pos = in_next;
                    in_next_slide = in_next + core::cmp::min(remaining, MATCHFINDER_WINDOW_SIZE);
                }

                let mut best_len = 0u32;
                adjust_max_and_nice_len(&mut max_len, &mut nice_len, remaining);

                let matches_start = cache_idx;
                if max_len >= BT_MATCHFINDER_REQUIRED_NBYTES as u32 {
                    // Write directly into the caller-owned match
                    // cache (disjoint field borrow of `n.bt_mf` and
                    // `n.match_cache[cache_idx..]` -- no intermediate `Vec`).
                    // This used to allocate a fresh `Vec` here and copy out
                    // of it, once per input byte in this near-optimal loop
                    // (levels 10-12) -- the single worst allocation hotspot
                    // in the compressor per two independent audits.
                    // self.n is Some — verified at function entry above.
                    if let Some(n) = near_mut(&mut self.state) {
                        let n_matches = bt_matchfinder_get_matches(
                            &mut n.bt_mf,
                            input,
                            in_base_pos,
                            in_next - in_base_pos,
                            max_len,
                            nice_len,
                            self.max_search_depth,
                            &mut next_hashes,
                            &mut n.match_cache[cache_idx..],
                        );
                        cache_idx += n_matches;
                        if cache_idx > matches_start {
                            best_len = n.match_cache[cache_idx - 1].length as u32;
                        }
                    }
                }

                if in_next >= next_observation {
                    if best_len >= min_len {
                        observe_match(&mut self.split_stats, best_len);
                        next_observation = in_next + best_len as usize;
                        if let Some(n) = near_mut(&mut self.state) { n.new_match_len_freqs[best_len as usize] += 1; }
                    } else {
                        observe_literal(&mut self.split_stats, input[in_next]);
                        next_observation = in_next + 1;
                    }
                }

                // Write the match-count header + literal byte for this position.
                {
                    let num_matches = cache_idx - matches_start;
                    if let Some(n) = near_mut(&mut self.state) {
                        n.match_cache[cache_idx].length = num_matches as u16;
                        n.match_cache[cache_idx].offset = input[in_next] as u16;
                        cache_idx += 1;
                    }
                }
                in_next += 1;

                // If a very long match was found, skip caching matches it covers.
                if best_len >= DEFLATE_MIN_MATCH_LEN && best_len >= nice_len {
                    let mut bl = best_len - 1;
                    loop {
                        remaining = in_end - in_next;
                        if in_next == in_next_slide {
                            if let Some(n) = near_mut(&mut self.state) { n.bt_mf.slide_window(); }
                            in_base_pos = in_next;
                            in_next_slide =
                                in_next + core::cmp::min(remaining, MATCHFINDER_WINDOW_SIZE);
                        }
                        adjust_max_and_nice_len(&mut max_len, &mut nice_len, remaining);
                        if max_len >= BT_MATCHFINDER_REQUIRED_NBYTES as u32
                            && let Some(n) = near_mut(&mut self.state)
                        {
                            bt_matchfinder_skip_byte(
                                &mut n.bt_mf,
                                input,
                                in_base_pos,
                                in_next - in_base_pos,
                                nice_len,
                                self.max_search_depth,
                                &mut next_hashes,
                            );
                        }
                        if let Some(n) = near_mut(&mut self.state) {
                            n.match_cache[cache_idx].length = 0;
                            n.match_cache[cache_idx].offset = input[in_next] as u16;
                            cache_idx += 1;
                        }
                        in_next += 1;
                        bl -= 1;
                        if bl == 0 {
                            break;
                        }
                    }
                }

                // Maximum block length or end of input reached?
                if in_next >= in_max_block_end {
                    break;
                }
                // Match cache overflowed?
                if cache_idx >= MATCH_CACHE_LENGTH {
                    break;
                }
                // Not ready to check block?
                if !ready_to_check_block(&self.split_stats, in_block_begin, in_next, in_end) {
                    continue;
                }
                // Worthwhile to end the block here?
                if do_end_block_check(&mut self.split_stats, (in_next - in_block_begin) as u32) {
                    change_detected = true;
                    break;
                }
                self.near_optimal_merge_stats();
                prev_end_block_check = Some(in_next);
            }

            // Now choose the precise end of the block and flush it.
            if let (true, Some(in_block_end)) = (change_detected, prev_end_block_check) {
                let orig_cache_idx = cache_idx;
                let block_length = (in_block_end - in_block_begin) as u32;
                let is_first = in_block_begin == 0;
                let is_final = false;
                let mut num_bytes_to_rewind = (in_next - in_block_end) as u32;

                // Rewind the match cache.
                // self.n is Some — verified at function entry above.
                if let Some(n) = near(&self.state) {
                    let mut ci = cache_idx;
                    loop {
                        ci -= 1;
                        ci -= n.match_cache[ci].length as usize;
                        num_bytes_to_rewind -= 1;
                        if num_bytes_to_rewind == 0 {
                            break;
                        }
                    }
                    cache_idx = ci;
                }
                let cache_len_rewound = orig_cache_idx - cache_idx;

                // cache_end marks the position just past this block's matches.
                if let Some(n) = near_mut(&mut self.state) { n.cache_end = cache_idx; }
                prev_block_used_only_literals = self.optimize_and_flush_block(
                    os,
                    input,
                    in_block_begin,
                    block_length,
                    is_first,
                    is_final,
                );

                // memmove(match_cache, cache_ptr, cache_len_rewound)
                if let Some(n) = near_mut(&mut self.state) {
                    n.match_cache.copy_within(cache_idx..cache_idx + cache_len_rewound, 0);
                }
                cache_idx = cache_len_rewound;
                self.near_optimal_save_stats();
                self.near_optimal_clear_old_stats();
                in_block_begin = in_block_end;
            } else {
                let block_length = (in_next - in_block_begin) as u32;
                let is_first = in_block_begin == 0;
                let is_final = in_next == in_end;

                self.near_optimal_merge_stats();
                if let Some(n) = near_mut(&mut self.state) { n.cache_end = cache_idx; }
                prev_block_used_only_literals = self.optimize_and_flush_block(
                    os,
                    input,
                    in_block_begin,
                    block_length,
                    is_first,
                    is_final,
                );
                cache_idx = 0;
                self.near_optimal_save_stats();
                self.near_optimal_init_stats();
                in_block_begin = in_next;
            }

            if in_next == in_end || os.overflow {
                break;
            }
        }
    }
}

// ============================================================================
// Near-optimal cost-model helper free functions
// ============================================================================

/// `deflate_default_length_cost`.
#[inline]
fn deflate_default_length_cost(len: u32, len_sym_cost: u32) -> u32 {
    let slot = DEFLATE_LENGTH_SLOT[len as usize] as usize;
    let num_extra_bits = DEFLATE_EXTRA_LENGTH_BITS[slot] as u32;
    len_sym_cost + (num_extra_bits * BIT_COST)
}

/// `deflate_default_offset_slot_cost`.
#[inline]
fn deflate_default_offset_slot_cost(slot: usize) -> u32 {
    let num_extra_bits = DEFLATE_EXTRA_OFFSET_BITS[slot] as u32;
    // offset_sym_cost = int(-log2(1/30)*BIT_COST), encoded as 4*BIT_COST + 907*BIT_COST/1000.
    let offset_sym_cost = 4 * BIT_COST + (907 * BIT_COST) / 1000;
    offset_sym_cost + (num_extra_bits * BIT_COST)
}

/// `deflate_adjust_cost` — blend a current cost toward a default by `change_amount`.
#[inline]
fn deflate_adjust_cost(cost_p: &mut u32, default_cost: u32, change_amount: i32) {
    match change_amount {
        0 => *cost_p = (default_cost + 3 * *cost_p) / 4,
        1 => *cost_p = (default_cost + *cost_p) / 2,
        2 => *cost_p = (5 * default_cost + 3 * *cost_p) / 8,
        _ => *cost_p = (3 * default_cost + *cost_p) / 4,
    }
}

// ============================================================================
// Public bound function
// ============================================================================

/// `libdeflate_deflate_compress_bound` — worst-case compressed size for `n` input bytes.
pub fn deflate_compress_bound(n: usize) -> usize {
    let max_blocks = core::cmp::max(n.div_ceil(MIN_BLOCK_LENGTH), 1);
    (5 * max_blocks) + n
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate_decompress::{deflate_decompress, Decompressor};

    /// Round-trip via our own decompressor.
    fn roundtrip(level: i32, data: &[u8]) {
        let mut c = Compressor::new(level).unwrap();
        let bound = deflate_compress_bound(data.len());
        let mut out = vec![0u8; bound];
        let n = c.deflate_compress(data, &mut out);
        assert!(n > 0 || data.is_empty() && n == 0, "level {level}: compress returned 0 for {} bytes", data.len());
        // Decompress.
        let mut d = Decompressor::new();
        let mut decoded = vec![0u8; data.len().max(1)];
        let got = deflate_decompress(&mut d, &out[..n], &mut decoded)
            .unwrap_or_else(|e| panic!("level {level}: decompress failed: {e:?}"));
        assert_eq!(&decoded[..got], data, "level {level}: round-trip mismatch ({} bytes)", data.len());
    }

    fn test_inputs() -> Vec<(String, Vec<u8>)> {
        // ~64KB deterministic pseudo-random (LCG).
        let mut rng: u32 = 0x12345678;
        let mut prng = Vec::with_capacity(65536);
        for _ in 0..65536 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            prng.push((rng >> 16) as u8);
        }
        vec![
            ("empty".into(), vec![]),
            ("short".into(), b"hello, world!".to_vec()),
            ("repetitive".into(), b"abcabcabcabc".repeat(1000)),
            ("all_same".into(), vec![0x5Au8; 70000]),
            ("prng64k".into(), prng),
            ("text".into(), b"The quick brown fox jumps over the lazy dog. ".repeat(2000)),
        ]
    }

    #[test]
    fn roundtrip_all_levels_all_inputs() {
        for level in 0..=12i32 {
            for (name, data) in test_inputs() {
                let mut c = Compressor::new(level).unwrap();
                let bound = deflate_compress_bound(data.len());
                let mut out = vec![0u8; bound];
                let n = c.deflate_compress(&data, &mut out);
                let mut d = Decompressor::new();
                let mut decoded = vec![0u8; data.len().max(1)];
                let got = deflate_decompress(&mut d, &out[..n], &mut decoded).unwrap_or_else(|e| {
                    panic!("level {level} input {name}: decompress failed: {e:?}")
                });
                assert_eq!(
                    &decoded[..got],
                    &data[..],
                    "level {level} input {name}: round-trip mismatch"
                );
            }
        }
    }

    #[test]
    fn roundtrip_each_level_smoke() {
        for level in 0..=12i32 {
            roundtrip(level, b"the quick brown fox the quick brown fox");
        }
    }

    #[test]
    fn bound_is_sane() {
        assert_eq!(deflate_compress_bound(0), 5);
        assert!(deflate_compress_bound(100000) >= 100000);
    }

    #[test]
    fn invalid_level_rejected() {
        assert!(Compressor::new(-2).is_err());
        assert!(Compressor::new(13).is_err());
        assert!(Compressor::new(0).is_ok());
        assert!(Compressor::new(12).is_ok());
    }

    /// libdeflate 1.26: `-1` aliases the default level (6).
    /// <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/deflate_compress.c#L3885-L3891>
    #[test]
    fn level_minus_one_aliases_default() {
        assert!(Compressor::new(-1).is_ok());
        // Same output as the explicit default level for identical input.
        let mut a = Compressor::new(-1).unwrap();
        let mut b = Compressor::new(6).unwrap();
        let input = b"the quick brown fox jumps over the lazy dog, repeated. ".repeat(20);
        let mut out_a = vec![0u8; deflate_compress_bound(input.len())];
        let mut out_b = vec![0u8; deflate_compress_bound(input.len())];
        let n_a = a.deflate_compress(&input, &mut out_a);
        let n_b = b.deflate_compress(&input, &mut out_b);
        assert_eq!(&out_a[..n_a], &out_b[..n_b]);
    }

    /// Counts static-codes table builds: one build must serve two
    /// fixed-Huffman blocks. `pangram_short` (a 3x-repeated pangram,
    /// small enough that a dynamic Huffman header's own overhead isn't worth
    /// it) reliably makes `flush_block`'s own cost comparison pick the
    /// STATIC/fixed Huffman block type at levels 1-6 (confirmed via a
    /// temporary `eprintln!` in `flush_block`'s `use_static` branch,
    /// reverted after confirming). Two SEPARATE `deflate_compress` calls on
    /// the SAME `Compressor` each independently choose a static block --
    /// proving `init_static_codes` (which only ever runs once, at
    /// `Compressor::new`) really is a build-once, reuse-forever cache and
    /// not silently rebuilt per block/per call.
    ///
    /// Mutation check: temporarily added a `self.init_static_codes()` call
    /// at the top of `deflate_compress` (simulating a rebuild-per-call bug)
    /// -- this test failed immediately (`left: 3, right: 1`, one build at
    /// construction plus one per each of the two compress calls), reverted.
    #[test]
    fn static_codes_built_once_serves_two_fixed_huffman_blocks() {
        let pangram_short = b"The quick brown fox jumps over the lazy dog.".repeat(3);

        for level in [1i32, 2, 3, 4, 6] {
            crate::probe::reset(crate::probe::Probe::StaticCodesBuild);
            let mut c = Compressor::new(level).unwrap();
            assert_eq!(
                crate::probe::get(crate::probe::Probe::StaticCodesBuild),
                1,
                "level={level}: Compressor::new must build the static codes exactly once"
            );

            let mut out1 = vec![0u8; deflate_compress_bound(pangram_short.len())];
            let n1 = c.deflate_compress(&pangram_short, &mut out1);
            let mut out2 = vec![0u8; deflate_compress_bound(pangram_short.len())];
            let n2 = c.deflate_compress(&pangram_short, &mut out2);

            // Sanity: both really are static (BTYPE == 1) blocks, not
            // stored/dynamic -- otherwise this test would vacuously pass
            // without exercising the static-codes cache at all.
            let block_type = |compressed: &[u8]| (compressed[0] >> 1) & 3;
            assert_eq!(block_type(&out1[..n1]), 1, "level={level}: block 1 must be static Huffman (BTYPE=1)");
            assert_eq!(block_type(&out2[..n2]), 1, "level={level}: block 2 must be static Huffman (BTYPE=1)");

            assert_eq!(
                crate::probe::get(crate::probe::Probe::StaticCodesBuild),
                1,
                "level={level}: the static-codes table must NOT be rebuilt for the second \
                 static-Huffman block/call -- exactly 1 build must serve both"
            );
        }
    }
}
