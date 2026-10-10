//! Port of libdeflate `lib/bt_matchfinder.h` — binary-tree matchfinder.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/bt_matchfinder.h
//!
//! # Algorithm
//! Binary-Trees (BT) matchfinder.  Two hash tables + a binary tree per bucket:
//! - `hash3_tab` (order 16, 2 ways): direct-mapped with 2 slots for length-3.
//! - `hash4_tab` (order 16): root of a per-bucket binary search tree.
//! - `child_tab` (2 * WINDOW_SIZE entries): left/right children of each tree node,
//!   indexed as `child_tab[pos * 2]` (left) and `child_tab[pos * 2 + 1]` (right).
//!
//! Used by near-optimal compression levels.
//!
//! # Pointer → index mapping
//! C uses pointer arithmetic throughout (`const u8 *in_base`, `ptrdiff_t cur_pos`).
//! Rust uses `in_buf: &[u8]` and `in_base_pos: usize`.
//! - `cur_pos = in_next_pos - in_base_pos`
//! - `&in_base[cur_node]` → `in_buf[(in_base_pos as isize + cur_node as isize) as usize]`
//!   (signed: after a window slide `cur_node` is negative, referring to the previous window)
//! - `in_next - matchptr` → `in_next_pos - match_abs`
//!
//! # `bt_matchfinder_advance_one_byte` inlining
//! The C file has one internal `advance_one_byte` with a `record_matches` bool
//! template parameter (compile-time constant).  Here we use a single generic
//! function with an `RECORD: bool` const-generic, which the compiler optimises
//! identically to the C template specialisation.
//!
//! # N/A
//! - `prefetchw`: no-op (performance hint only).
//! - `STATIC_ASSERT` size checks: replaced by `const` assertions.
//! - `MATCHFINDER_ALIGNED`: Vec heap allocation handles alignment.
//! - `lz_match` C struct: exposed as `LzMatch` Rust struct.

use crate::common::get_unaligned_le32;
use crate::matchfinder_common::{
    lz_extend, lz_hash, load_u24_unaligned, matchfinder_init, matchfinder_rebase, AlignedTab,
    MfPos, MATCHFINDER_INITVAL, MATCHFINDER_WINDOW_SIZE,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Hash table order for length-3 matches.  `BT_MATCHFINDER_HASH3_ORDER`.
pub const BT_MATCHFINDER_HASH3_ORDER: u32 = 16;
/// Number of ways (slots) per hash3 bucket.  `BT_MATCHFINDER_HASH3_WAYS`.
pub const BT_MATCHFINDER_HASH3_WAYS: usize = 2;
/// Hash table order for length-4+ matches.  `BT_MATCHFINDER_HASH4_ORDER`.
pub const BT_MATCHFINDER_HASH4_ORDER: u32 = 16;

const HASH3_SIZE: usize = 1 << BT_MATCHFINDER_HASH3_ORDER; // 65 536
const HASH4_SIZE: usize = 1 << BT_MATCHFINDER_HASH4_ORDER; // 65 536

/// Minimum number of bytes remaining in input for `bt_matchfinder_get_matches`
/// or `bt_matchfinder_skip_byte`.  `BT_MATCHFINDER_REQUIRED_NBYTES`.
pub const BT_MATCHFINDER_REQUIRED_NBYTES: usize = 5;

/// Total hash-table byte size.
/// `BT_MATCHFINDER_TOTAL_HASH_SIZE` = (65536*2 + 65536) * 2 = 393 216 bytes.
pub const BT_MATCHFINDER_TOTAL_HASH_SIZE: usize =
    (HASH3_SIZE * BT_MATCHFINDER_HASH3_WAYS + HASH4_SIZE) * core::mem::size_of::<MfPos>();

const _: () = assert!(
    BT_MATCHFINDER_TOTAL_HASH_SIZE.is_multiple_of(1024),
    "BT_MATCHFINDER_TOTAL_HASH_SIZE must be a multiple of MATCHFINDER_SIZE_ALIGNMENT"
);

// Total entries: hash3_tab (2 ways) + hash4_tab + child_tab (2 * WINDOW_SIZE).
const BT_MF_TOTAL_ENTRIES: usize =
    HASH3_SIZE * BT_MATCHFINDER_HASH3_WAYS + HASH4_SIZE + 2 * MATCHFINDER_WINDOW_SIZE;

const _BT_SIZE_CHECK: () = assert!(
    (BT_MF_TOTAL_ENTRIES * core::mem::size_of::<MfPos>()).is_multiple_of(1024),
    "sizeof(bt_matchfinder) must be a multiple of MATCHFINDER_SIZE_ALIGNMENT"
);

// ---------------------------------------------------------------------------
// LzMatch — mirrors `struct lz_match { u16 length; u16 offset; }`
// ---------------------------------------------------------------------------

/// A Lempel-Ziv match found by the BT matchfinder.
/// Mirrors `struct lz_match` in `bt_matchfinder.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LzMatch {
    /// Number of bytes matched.
    pub length: u16,
    /// Back-offset from the current position to the start of the match.
    pub offset: u16,
}

// ---------------------------------------------------------------------------
// BtMatchfinder struct
//
// Memory layout of flat `table` Vec:
//   [0 .. HASH3_SIZE*WAYS)                      → hash3_tab (2D: [bucket][way])
//   [HASH3_SIZE*WAYS .. HASH3_SIZE*WAYS+HASH4_SIZE) → hash4_tab
//   [HASH3_SIZE*WAYS+HASH4_SIZE .. end)          → child_tab  (left=even, right=odd)
// ---------------------------------------------------------------------------

/// Binary-tree matchfinder.  Mirrors `struct bt_matchfinder`.
///
/// Data-layout task: `table` is `AlignedTab<BT_MF_TOTAL_ENTRIES>`
/// (32-byte aligned, matching C's `MATCHFINDER_ALIGNED`) -- see
/// `AlignedTab` in `matchfinder_common.rs`, shared with
/// `HcMatchfinder`/`HtMatchfinder`.
pub struct BtMatchfinder {
    /// Flat backing store: `hash3_tab | hash4_tab | child_tab`.
    table: AlignedTab<BT_MF_TOTAL_ENTRIES>,
}

impl BtMatchfinder {
    const HASH3_OFF: usize = 0;
    // hash3_tab is 2D [HASH3_SIZE][WAYS]: entry = HASH3_OFF + bucket*WAYS + way
    const HASH4_OFF: usize = HASH3_SIZE * BT_MATCHFINDER_HASH3_WAYS;
    const CHILD_OFF: usize = Self::HASH4_OFF + HASH4_SIZE;

    /// Create and initialise a new `BtMatchfinder`.  Mirrors `bt_matchfinder_init`.
    pub fn new() -> Self {
        let mut mf = BtMatchfinder {
            table: AlignedTab::zeroed(),
        };
        mf.init();
        mf
    }

    /// Reinitialise hash tables for a new input buffer.  Mirrors `bt_matchfinder_init`.
    /// Only the hash tables are initialised (not child_tab; filled lazily during use).
    #[inline]
    pub fn init(&mut self) {
        let hash_entries = HASH3_SIZE * BT_MATCHFINDER_HASH3_WAYS + HASH4_SIZE;
        matchfinder_init(&mut self.table[..hash_entries]);
    }

    /// Slide window by `MATCHFINDER_WINDOW_SIZE`.  Mirrors `bt_matchfinder_slide_window`
    /// which calls `matchfinder_rebase` over the entire struct.
    #[inline]
    pub fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.table);
    }

    // --- Internal accessors ---

    /// `hash3_tab[bucket][way]`
    #[inline(always)]
    fn hash3(&self, bucket: usize, way: usize) -> MfPos {
        self.table[Self::HASH3_OFF + bucket * BT_MATCHFINDER_HASH3_WAYS + way]
    }
    #[inline(always)]
    fn hash3_set(&mut self, bucket: usize, way: usize, v: MfPos) {
        self.table[Self::HASH3_OFF + bucket * BT_MATCHFINDER_HASH3_WAYS + way] = v;
    }

    /// `hash4_tab[bucket]`
    #[inline(always)]
    fn hash4(&self, bucket: usize) -> MfPos {
        self.table[Self::HASH4_OFF + bucket]
    }
    #[inline(always)]
    fn hash4_set(&mut self, bucket: usize, v: MfPos) {
        self.table[Self::HASH4_OFF + bucket] = v;
    }

    /// Left child of tree node at `pos`.
    /// Mirrors `bt_left_child(mf, node)` = `&child_tab[2*(node & (WINDOW-1)) + 0]`.
    #[inline(always)]
    fn left_child(&self, pos: usize) -> MfPos {
        self.table[Self::CHILD_OFF + 2 * (pos & (MATCHFINDER_WINDOW_SIZE - 1))]
    }

    /// Right child of tree node at `pos`.
    /// Mirrors `bt_right_child(mf, node)` = `&child_tab[2*(node & (WINDOW-1)) + 1]`.
    #[inline(always)]
    fn right_child(&self, pos: usize) -> MfPos {
        self.table[Self::CHILD_OFF + 2 * (pos & (MATCHFINDER_WINDOW_SIZE - 1)) + 1]
    }
}

impl Default for BtMatchfinder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// bt_matchfinder_advance_one_byte  (internal)
// Mirrors the C template function with `record_matches` bool parameter.
// RECORD is a const generic bool — the compiler eliminates the dead branch.
//
// Returns the number of matches appended to `matches` (only used when RECORD=true).
// ---------------------------------------------------------------------------

/// Internal: advance by one byte, optionally recording matches.
/// Mirrors `bt_matchfinder_advance_one_byte`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn advance_one_byte<const RECORD: bool>(
    mf: &mut BtMatchfinder,
    in_buf: &[u8],
    in_base_pos: usize,
    cur_pos: usize,        // relative to in_base_pos (mirrors ptrdiff_t cur_pos)
    max_len: u32,
    nice_len: u32,
    max_search_depth: u32,
    next_hashes: &mut [u32; 2],
    // A caller-owned slice (the compressor's match cache) written
    // in place, plus a running count -- not a `Vec` allocated per call site.
    // Ignored (never indexed) when RECORD=false.
    matches: &mut [LzMatch],
    match_count: &mut usize,
) {
    // Precondition (mirrors the doc comment on `bt_matchfinder_get_matches`
    // and `BT_MATCHFINDER_REQUIRED_NBYTES`'s own doc comment): the BST-descent
    // loop below indexes `in_buf[in_next_pos + len]` / `in_buf[match_abs +
    // len]` directly (no bounds check, matching C's raw pointer indexing),
    // relying on the caller having guaranteed enough trailing bytes.
    //
    // `debug_assert!`, not `assert!`: mirrors C's `ASSERT()` (lib_common.h),
    // which expands to nothing unless built with `LIBDEFLATE_ENABLE_ASSERTIONS`
    // — this is a 1:1-port fidelity choice, NOT a way to "avoid a panic in
    // release": if the precondition is actually violated, the raw indexing a
    // few lines below (`in_buf[in_next_pos + len]` / `in_buf[match_abs +
    // len]`) panics on out-of-bounds regardless of whether this assert ran.
    // The precondition is guaranteed by construction at every real call site
    // (`deflate_compress.rs`'s `adjust_max_and_nice_len` clamps `max_len`
    // before every BT call, and the call sites additionally gate on `max_len
    // >= BT_MATCHFINDER_REQUIRED_NBYTES` before calling at all), so neither
    // this assert nor the indexing panic ever fires in production; the assert
    // exists purely to give a debug build an immediate, precise diagnostic
    // at the actual contract boundary instead of a bounds-check panic
    // somewhere downstream (the bug this fixes: a `.get()`-based guard let
    // `None == None` fabricate a match at the buffer tail instead of
    // surfacing the violation at all, in any build).
    debug_assert!(
        max_len >= BT_MATCHFINDER_REQUIRED_NBYTES as u32,
        "bt_matchfinder: max_len ({max_len}) < BT_MATCHFINDER_REQUIRED_NBYTES \
         ({BT_MATCHFINDER_REQUIRED_NBYTES}) — caller must guarantee enough \
         trailing input bytes before calling the BT matchfinder"
    );

    let in_next_pos = in_base_pos + cur_pos;
    let mut depth_remaining = max_search_depth;
    // cutoff: signed — any node index <= cutoff is outside the window.
    let cutoff: i32 = cur_pos as i32 - MATCHFINDER_WINDOW_SIZE as i32;

    // Precompute next hashes (for in_next_pos + 1).
    let next_hashseq = get_unaligned_le32(&in_buf[in_next_pos + 1..]);
    let hash3 = next_hashes[0] as usize;
    let hash4 = next_hashes[1] as usize;
    next_hashes[0] = lz_hash(next_hashseq & 0x00FF_FFFF, BT_MATCHFINDER_HASH3_ORDER);
    next_hashes[1] = lz_hash(next_hashseq, BT_MATCHFINDER_HASH4_ORDER);
    // prefetchw omitted (no-op).

    // --- Length-3 match check via hash3_tab (2-way) ---
    let cur_node_h3_0 = mf.hash3(hash3, 0);
    mf.hash3_set(hash3, 0, cur_pos as MfPos);
    let cur_node_h3_1 = mf.hash3(hash3, 1);
    mf.hash3_set(hash3, 1, cur_node_h3_0);

    if RECORD && cur_node_h3_0 as i32 > cutoff {
        let seq3 = load_u24_unaligned(&in_buf[in_next_pos..]);
        // Signed arithmetic: after a window slide the node is negative (previous window).
        let match_abs_0 = (in_base_pos as isize + cur_node_h3_0 as isize) as usize;
        if seq3 == load_u24_unaligned(&in_buf[match_abs_0..]) {
            matches[*match_count] = LzMatch {
                length: 3,
                offset: (in_next_pos - match_abs_0) as u16,
            };
            *match_count += 1;
        } else if cur_node_h3_1 as i32 > cutoff {
            let match_abs_1 = (in_base_pos as isize + cur_node_h3_1 as isize) as usize;
            if seq3 == load_u24_unaligned(&in_buf[match_abs_1..]) {
                matches[*match_count] = LzMatch {
                    length: 3,
                    offset: (in_next_pos - match_abs_1) as u16,
                };
                *match_count += 1;
            }
        }
    }

    // --- Length-4+ match search via hash4_tab binary tree ---
    let mut cur_node: i32 = mf.hash4(hash4) as i32;
    mf.hash4_set(hash4, cur_pos as MfPos);

    // pending_lt / pending_gt: indices into child_tab that we will write to.
    // We track them as (pos, is_right): after the loop we write INITVAL into them.
    // Use a "pending pointer" approach: remember the last node that needs its
    // child pointer updated.  We store the (parent_pos, side) pair.
    // This mirrors the C mf_pos_t* pointers that are updated in-place.

    // We'll track pending writes as (parent_pos, side: 0=left 1=right).
    // -1 means "write into a temporary that we discard" (shouldn't happen if we
    //  initialise pending properly, but we use Option to be safe).
    // Instead of raw pointer aliasing, we track pending writes by noting
    // which slot of which node needs updating.
    // pending_lt_slot = slot index in mf.table that currently plays the role of
    //   `pending_lt_ptr` in C — i.e., the child_tab slot for the left/right
    //   child of the current node in the tree being rebuilt.
    // We encode the slot directly as an absolute index into mf.table.

    // Initialize pending_lt_ptr = bt_left_child(mf, cur_pos) and
    //            pending_gt_ptr = bt_right_child(mf, cur_pos).
    // These are the children slots of the NEW node being inserted (cur_pos).
    let mut pending_lt_slot = BtMatchfinder::CHILD_OFF + 2 * (cur_pos & (MATCHFINDER_WINDOW_SIZE - 1));
    let mut pending_gt_slot = BtMatchfinder::CHILD_OFF + 2 * (cur_pos & (MATCHFINDER_WINDOW_SIZE - 1)) + 1;

    if cur_node <= cutoff {
        // No in-window node: set both children to INITVAL and return.
        mf.table[pending_lt_slot] = MATCHFINDER_INITVAL;
        mf.table[pending_gt_slot] = MATCHFINDER_INITVAL;
        return;
    }

    let mut best_lt_len: u32 = 0;
    let mut best_gt_len: u32 = 0;
    let mut len: u32 = 0;
    // best_len tracks the longest match found (for record_matches filtering).
    let mut best_len: u32 = 3;

    loop {
        // `cur_node` is a signed i32 node index; after a window slide it can be
        // negative (referring to the previous window), so add it with signed maths.
        let match_abs = (in_base_pos as isize + cur_node as isize) as usize;

        // Try to extend from current len (reuse common prefix). Mirrors C's raw
        // `matchptr[len] == in_next[len]` — direct indexing, relying on the
        // caller's `max_len >= BT_MATCHFINDER_REQUIRED_NBYTES` precondition
        // (`debug_assert!`ed at the top of `advance_one_byte`, this function's
        // caller — not here, and not in `bt_matchfinder_get_matches`, which
        // is just a thin public wrapper around `advance_one_byte`) to keep
        // `in_next_pos + len` and `match_abs + len` in bounds.
        //
        // This used to compare `.get(..) == .get(..)` so it wouldn't
        // panic if `len` ran past the buffer end. That silently accepted
        // `None == None` as a match whenever BOTH sides were out of bounds —
        // fabricating a match at the buffer tail instead of surfacing the
        // precondition violation. Masked in practice because production
        // callers always respected `REQUIRED_NBYTES`; asserting the
        // precondition (rather than silently degrading) is the project rule
        // for latent invariant violations.
        let can_extend = in_buf[match_abs + len as usize] == in_buf[in_next_pos + len as usize];
        if can_extend {
            len = lz_extend(in_buf, in_next_pos, match_abs, len + 1, max_len);

            if !RECORD || len > best_len {
                if RECORD {
                    best_len = len;
                    matches[*match_count] = LzMatch {
                        length: len as u16,
                        offset: (in_next_pos - match_abs) as u16,
                    };
                    *match_count += 1;
                }
                if len >= nice_len {
                    // Link remaining subtree and return.
                    mf.table[pending_lt_slot] = mf.left_child(cur_node as usize);
                    mf.table[pending_gt_slot] = mf.right_child(cur_node as usize);
                    return;
                }
            }
        }

        // Navigate the BST based on lexicographic order at position `len`.
        // Mirrors C's raw `matchptr[len]` / `in_next[len]` — direct indexing;
        // see the note above `can_extend` for why the previous
        // `.get(..).unwrap_or(0)` guard was a silent invariant-violation
        // fallback rather than a real fix.
        let str_byte = in_buf[in_next_pos + len as usize];
        let match_byte = in_buf[match_abs + len as usize];
        if match_byte < str_byte {
            // matchptr < in_next at position len → go right of cur_node.
            mf.table[pending_lt_slot] = cur_node as MfPos;
            // pending_lt_ptr = bt_right_child(mf, cur_node)
            pending_lt_slot = BtMatchfinder::CHILD_OFF + 2 * (cur_node as usize & (MATCHFINDER_WINDOW_SIZE - 1)) + 1;
            cur_node = mf.table[pending_lt_slot] as i32;
            best_lt_len = len;
            if best_gt_len < len {
                len = best_gt_len;
            }
        } else {
            // matchptr > in_next at position len → go left of cur_node.
            mf.table[pending_gt_slot] = cur_node as MfPos;
            // pending_gt_ptr = bt_left_child(mf, cur_node)
            pending_gt_slot = BtMatchfinder::CHILD_OFF + 2 * (cur_node as usize & (MATCHFINDER_WINDOW_SIZE - 1));
            cur_node = mf.table[pending_gt_slot] as i32;
            best_gt_len = len;
            if best_lt_len < len {
                len = best_lt_len;
            }
        }

        depth_remaining -= 1;
        if cur_node <= cutoff || depth_remaining == 0 {
            mf.table[pending_lt_slot] = MATCHFINDER_INITVAL;
            mf.table[pending_gt_slot] = MATCHFINDER_INITVAL;
            return;
        }
    }

}

// ---------------------------------------------------------------------------
// bt_matchfinder_get_matches — public API, RECORD=true
// Mirrors `bt_matchfinder_get_matches`.
// ---------------------------------------------------------------------------

/// Retrieve matches at `cur_pos` relative to `in_base_pos`.
/// Mirrors `bt_matchfinder_get_matches`.
///
/// Matches are appended to `matches` in order of strictly increasing length.
/// The caller must ensure `max_len >= BT_MATCHFINDER_REQUIRED_NBYTES`.
#[inline]
#[allow(clippy::too_many_arguments)]
/// Writes matches into `matches` starting at index 0 and returns how many
/// were written (`matches` must be at least `nice_len - 2` long, the C-
/// documented maximum -- the compressor's match cache is always sized for
/// the whole block, comfortably larger). This used to take
/// `&mut Vec<LzMatch>` and `.push()` into it, forcing every call site to own
/// (and typically freshly allocate) a `Vec` just to immediately copy its
/// contents into the real match cache. Writing directly into a caller-owned
/// slice — here, a sub-slice of that same cache — removes the per-position
/// allocation entirely (this function runs once per input byte at levels
/// 10-12's near-optimal parser).
pub fn bt_matchfinder_get_matches(
    mf: &mut BtMatchfinder,
    in_buf: &[u8],
    in_base_pos: usize,
    cur_pos: usize,
    max_len: u32,
    nice_len: u32,
    max_search_depth: u32,
    next_hashes: &mut [u32; 2],
    matches: &mut [LzMatch],
) -> usize {
    let mut match_count = 0usize;
    advance_one_byte::<true>(
        mf,
        in_buf,
        in_base_pos,
        cur_pos,
        max_len,
        nice_len,
        max_search_depth,
        next_hashes,
        matches,
        &mut match_count,
    );
    match_count
}

// ---------------------------------------------------------------------------
// bt_matchfinder_skip_byte — public API, RECORD=false
// Mirrors `bt_matchfinder_skip_byte`.
// ---------------------------------------------------------------------------

/// Advance the matchfinder without recording matches.
/// Mirrors `bt_matchfinder_skip_byte`.
#[inline]
pub fn bt_matchfinder_skip_byte(
    mf: &mut BtMatchfinder,
    in_buf: &[u8],
    in_base_pos: usize,
    cur_pos: usize,
    nice_len: u32,
    max_search_depth: u32,
    next_hashes: &mut [u32; 2],
) {
    // RECORD=false never indexes `matches`/`match_count` (both are behind
    // `if RECORD` branches the compiler prunes for this monomorphization) --
    // an empty slice and a throwaway counter are enough.
    let mut unused_count = 0usize;
    advance_one_byte::<false>(
        mf,
        in_buf,
        in_base_pos,
        cur_pos,
        nice_len, // max_len = nice_len in skip path (mirrors C)
        nice_len,
        max_search_depth,
        next_hashes,
        &mut [],
        &mut unused_count,
    );
}

// ---------------------------------------------------------------------------
// bt_matchfinder_skip_bytes — skip multiple bytes (convenience wrapper)
// Not in the original header but called in the same pattern as hc.
// ---------------------------------------------------------------------------

/// Advance the matchfinder by `count` bytes without recording matches.
/// Applies slide-window logic when the window fills.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn bt_matchfinder_skip_bytes(
    mf: &mut BtMatchfinder,
    in_buf: &[u8],
    in_base_pos: &mut usize,
    in_next_pos: usize,
    count: usize,
    nice_len: u32,
    max_search_depth: u32,
    next_hashes: &mut [u32; 2],
) {
    for i in 0..count {
        let abs_pos = in_next_pos + i;
        let mut cur_pos = abs_pos - *in_base_pos;
        if cur_pos == MATCHFINDER_WINDOW_SIZE {
            mf.slide_window();
            *in_base_pos += MATCHFINDER_WINDOW_SIZE;
            cur_pos = 0;
        }
        bt_matchfinder_skip_byte(
            mf,
            in_buf,
            *in_base_pos,
            cur_pos,
            nice_len,
            max_search_depth,
            next_hashes,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_initial_hashes(in_buf: &[u8], pos: usize) -> [u32; 2] {
        // next_hashes entering a call must hold the hash of in_buf[pos..pos+4].
        let seq = get_unaligned_le32(&in_buf[pos..]);
        [
            lz_hash(seq & 0x00FF_FFFF, BT_MATCHFINDER_HASH3_ORDER),
            lz_hash(seq, BT_MATCHFINDER_HASH4_ORDER),
        ]
    }

    #[test]
    fn bt_finds_match_in_repeated_buffer() {
        // "abcdeabcde________" — second "abcde" should match first.
        let input = b"abcdeabcde________";
        let mut mf = BtMatchfinder::new();
        let in_base_pos = 0usize;
        let mut next_hashes = make_initial_hashes(input, 0);

        // Advance positions 0..5 (first "abcde") without recording.
        for pos in 0..5usize {
            bt_matchfinder_skip_byte(
                &mut mf,
                input,
                in_base_pos,
                pos,
                258,
                64,
                &mut next_hashes,
            );
        }

        // Now get matches at position 5 (start of second "abcde").
        let mut matches = [LzMatch { length: 0, offset: 0 }; 32];
        let n = bt_matchfinder_get_matches(
            &mut mf,
            input,
            in_base_pos,
            5,
            (input.len() - 5) as u32,
            258,
            64,
            &mut next_hashes,
            &mut matches,
        );

        // Should find a match of length >= 5 at offset 5.
        let best = matches[..n].iter().max_by_key(|m| m.length);
        assert!(best.is_some(), "no matches found");
        let best = best.unwrap();
        assert!(best.length >= 5, "expected length >= 5, got {}", best.length);
        assert_eq!(best.offset, 5, "expected offset 5, got {}", best.offset);
    }

    /// Regression: `bt_matchfinder_get_matches` with `max_len <
    /// BT_MATCHFINDER_REQUIRED_NBYTES` must panic on the asserted
    /// precondition, not silently degrade. Before the fix, the BST-descent
    /// loop's `.get(..) == .get(..)` comparison let `None == None` fabricate
    /// a match whenever both `in_next_pos + len` and `match_abs + len` ran
    /// past the buffer end — exactly the scenario a too-small `max_len`
    /// invites.
    ///
    /// `debug_assert!` only fires with `cfg(debug_assertions)` (off in a
    /// plain `--release` build, mirroring C's `ASSERT()`), so this test is
    /// gated the same way — under `--release` it doesn't run at all rather
    /// than failing to observe a panic that was never supposed to happen
    /// there. Mutation checked: reverting the `debug_assert!` back to a
    /// no-op (or deleting it) makes this test fail with "test completed
    /// successfully despite being marked `should_panic`" — confirmed by
    /// hand while developing the fix.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "max_len")]
    fn bt_get_matches_panics_on_too_small_max_len() {
        let input = b"abcdeabcde________";
        let mut mf = BtMatchfinder::new();
        let in_base_pos = 0usize;
        let mut next_hashes = make_initial_hashes(input, 0);
        for pos in 0..5usize {
            bt_matchfinder_skip_byte(&mut mf, input, in_base_pos, pos, 258, 64, &mut next_hashes);
        }
        let mut matches = Vec::new();
        bt_matchfinder_get_matches(
            &mut mf,
            input,
            in_base_pos,
            5,
            (BT_MATCHFINDER_REQUIRED_NBYTES - 1) as u32, // violates the precondition
            258,
            64,
            &mut next_hashes,
            &mut matches,
        );
    }

    #[test]
    fn bt_skip_advances_tree() {
        // After skipping, the tree should have entries (hash4_tab not all INITVAL).
        let input = b"abcdefghijklmnopqrstuvwxyz______";
        let mut mf = BtMatchfinder::new();
        let in_base_pos = 0usize;
        let mut next_hashes = make_initial_hashes(input, 0);

        for pos in 0..10usize {
            bt_matchfinder_skip_byte(
                &mut mf,
                input,
                in_base_pos,
                pos,
                258,
                64,
                &mut next_hashes,
            );
        }

        // At least one hash4_tab entry should be non-INITVAL.
        let initval = MATCHFINDER_INITVAL;
        let any_set = mf.table[BtMatchfinder::HASH4_OFF..BtMatchfinder::HASH4_OFF + HASH4_SIZE]
            .iter()
            .any(|&v| v != initval);
        assert!(any_set, "hash4_tab should have some non-INITVAL entries after skip");
    }

    #[test]
    fn bt_matches_sorted_by_increasing_length() {
        // A buffer with a long repeated run to produce multiple matches. Long
        // enough that `nice_len` (258) never exceeds the remaining bytes at
        // any position used below — `max_len` (and, for
        // `bt_matchfinder_skip_byte`, `nice_len` doubling as `max_len`) must
        // never exceed the buffer's remaining bytes, exactly like production
        // (`deflate_compress.rs`'s `adjust_max_and_nice_len`) always ensures.
        let input: Vec<u8> = (0u32..320).map(|i| (i % 17) as u8).collect();
        let mut mf = BtMatchfinder::new();
        let in_base_pos = 0usize;
        let mut next_hashes = make_initial_hashes(&input, 0);

        // Advance first 50 bytes.
        for pos in 0..50usize {
            let nice_len = core::cmp::min(258, (input.len() - pos) as u32);
            bt_matchfinder_skip_byte(
                &mut mf,
                &input,
                in_base_pos,
                pos,
                nice_len,
                64,
                &mut next_hashes,
            );
        }

        let mut matches = [LzMatch { length: 0, offset: 0 }; 32];
        let max_len = core::cmp::min(258, (input.len() - 50) as u32);
        let n = bt_matchfinder_get_matches(
            &mut mf,
            &input,
            in_base_pos,
            50,
            max_len,
            258,
            64,
            &mut next_hashes,
            &mut matches,
        );
        let matches = &matches[..n];

        // Lengths must be strictly increasing.
        for w in matches.windows(2) {
            assert!(
                w[0].length < w[1].length,
                "matches not strictly increasing: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn bt_init_resets_hash_tables() {
        let input = b"abcdefghij_________";
        let mut mf = BtMatchfinder::new();
        let in_base_pos = 0usize;
        let mut next_hashes = make_initial_hashes(input, 0);
        for pos in 0..5usize {
            bt_matchfinder_skip_byte(
                &mut mf,
                input,
                in_base_pos,
                pos,
                258,
                64,
                &mut next_hashes,
            );
        }
        mf.init();
        let hash_entries = HASH3_SIZE * BT_MATCHFINDER_HASH3_WAYS + HASH4_SIZE;
        for &v in &mf.table[..hash_entries] {
            assert_eq!(v, MATCHFINDER_INITVAL);
        }
    }

    #[test]
    fn bt_max_search_depth_limits_work() {
        // depth=1 must not panic or produce incorrect lengths. `nice_len`
        // (used as `max_len` by `bt_matchfinder_skip_byte`) is capped to the
        // buffer's remaining bytes at each position — see the precondition
        // comment in `bt_matches_sorted_by_increasing_length` above.
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; // 32 'a's
        let mut mf = BtMatchfinder::new();
        let in_base_pos = 0usize;
        let mut next_hashes = make_initial_hashes(input, 0);
        for pos in 0..10usize {
            let nice_len = core::cmp::min(258, (input.len() - pos) as u32);
            bt_matchfinder_skip_byte(
                &mut mf,
                input,
                in_base_pos,
                pos,
                nice_len,
                1, // depth=1
                &mut next_hashes,
            );
        }
        let mut matches = [LzMatch { length: 0, offset: 0 }; 32];
        // Mirrors `adjust_max_and_nice_len` (deflate_compress.c:2271-2277):
        // whenever `max_len` gets clamped to the remaining bytes, `nice_len`
        // must be clamped to `<= max_len` too, so the loop's `len >= nice_len`
        // early-return fires at `len == max_len` — before the BST-navigation
        // step would index `in_buf[in_next_pos + max_len]` (== buffer length,
        // out of bounds).
        let max_len = core::cmp::min(258, (input.len() - 10) as u32);
        let nice_len = core::cmp::min(258, max_len);
        let n = bt_matchfinder_get_matches(
            &mut mf,
            input,
            in_base_pos,
            10,
            max_len,
            nice_len,
            1, // depth=1
            &mut next_hashes,
            &mut matches,
        );
        // With depth=1, we may or may not find a match, but we must not panic.
        for m in &matches[..n] {
            assert!(m.length >= 3);
            assert!(m.offset > 0);
        }
    }

    /// At least one input > `MATCHFINDER_WINDOW_SIZE` (32 KiB),
    /// actually crossing a `slide_window()` call, with the match verified
    /// byte-for-byte. See the identical rationale on
    /// `hc_matchfinder::tests::hc_finds_correct_match_across_window_slide`.
    ///
    /// MUTATION CHECKED: temporarily changed this function's
    /// `offset: (in_next_pos - match_abs) as u16,` (the match-recording
    /// site in `advance_one_byte`) to `... + 1` and reran this test — it
    /// went RED with a byte-mismatch panic. Reverted immediately.
    #[test]
    fn bt_finds_correct_match_across_window_slide() {
        const PERIOD: usize = 37;
        let pattern: Vec<u8> = (0..PERIOD as u8).collect();
        let len = 40_000usize;
        let input: Vec<u8> = (0..len).map(|i| pattern[i % PERIOD]).collect();

        let mut mf = BtMatchfinder::new();
        let mut in_base_pos = 0usize;
        let mut next_hashes = make_initial_hashes(&input, 0);

        let search_pos = len - 40;
        // nice_len=32 stays <= the remaining bytes at every skipped position
        // (minimum remaining during the skip loop is len - (search_pos - 1)
        // = 41), satisfying `bt_matchfinder_skip_byte`'s `max_len == nice_len
        // <= remaining` precondition throughout the whole slide.
        bt_matchfinder_skip_bytes(
            &mut mf,
            &input,
            &mut in_base_pos,
            0,
            search_pos,
            32,
            64,
            &mut next_hashes,
        );
        assert!(
            in_base_pos > 0,
            "test buffer (40000 B) must actually cross a MATCHFINDER_WINDOW_SIZE \
             (32768 B) slide; in_base_pos stayed 0"
        );

        // Mirrors adjust_max_and_nice_len: clamp both to the remaining bytes.
        let max_len = core::cmp::min(258, (len - search_pos) as u32);
        let nice_len = core::cmp::min(258, max_len);
        let mut matches = [LzMatch { length: 0, offset: 0 }; 32];
        let n = bt_matchfinder_get_matches(
            &mut mf,
            &input,
            in_base_pos,
            search_pos - in_base_pos,
            max_len,
            nice_len,
            64,
            &mut next_hashes,
            &mut matches,
        );

        let best = matches[..n]
            .iter()
            .max_by_key(|m| m.length)
            .expect("expected a match after the window slide, found none");
        assert!(
            best.offset > 0 && (best.offset as usize) <= search_pos,
            "offset out of range: {}",
            best.offset
        );
        let src = search_pos - best.offset as usize;
        assert_eq!(
            &input[src..src + best.length as usize],
            &input[search_pos..search_pos + best.length as usize],
            "matchfinder reported a match that is not byte-identical (offset={}, len={})",
            best.offset,
            best.length
        );
    }
}
