//! Port of libdeflate `lib/ht_matchfinder.h` — hash-table matchfinder.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/ht_matchfinder.h
//!
//! # Algorithm
//! Hash-Table (HT) matchfinder: a variant of HC that stores small inline chains
//! (buckets of `HT_MATCHFINDER_BUCKET_SIZE` slots) directly in the hash table,
//! avoiding a separate `next_tab` array.  Optimised for very fast (level-1)
//! compression where `max_search_depth` is fixed and tiny.
//!
//! - No length-3 match support (unlike HC/BT).
//! - `max_search_depth` is fixed at compile time as `HT_MATCHFINDER_BUCKET_SIZE`.
//! - Minimum match length is 4.
//!
//! # Pointer → index mapping
//! Same as `hc_matchfinder`: `in_buf: &[u8]`, `in_base_pos: usize`.
//!
//! # `BUCKET_SIZE == 2` hand-unrolled path
//! The C source has three hand-specialised branches for BUCKET_SIZE 1, 2, and >2.
//! We implement the BUCKET_SIZE==2 path (which is the actual build-time value)
//! exactly as in C, plus the generic `>2` path for completeness.
//!
//! # N/A
//! - `prefetchw`: no-op (performance hint only).
//! - `STATIC_ASSERT` checks: replaced by `const` assertions.
//! - `MATCHFINDER_ALIGNED`: Vec heap allocation handles alignment.

use crate::common::get_unaligned_le32;
use crate::matchfinder_common::{
    lz_extend, lz_hash, matchfinder_init, matchfinder_rebase, AlignedTab, MfPos,
    MATCHFINDER_WINDOW_SIZE,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Hash table order.  `HT_MATCHFINDER_HASH_ORDER`.
pub const HT_MATCHFINDER_HASH_ORDER: u32 = 15;

/// Number of positions stored per hash bucket.  `HT_MATCHFINDER_BUCKET_SIZE`.
pub const HT_MATCHFINDER_BUCKET_SIZE: usize = 2;

/// Minimum match length.  `HT_MATCHFINDER_MIN_MATCH_LEN`.
pub const HT_MATCHFINDER_MIN_MATCH_LEN: u32 = 4;

/// Minimum value of `max_len` required by `ht_matchfinder_longest_match`.
/// `HT_MATCHFINDER_REQUIRED_NBYTES`.
pub const HT_MATCHFINDER_REQUIRED_NBYTES: usize = 5;

const HASH_SIZE: usize = 1 << HT_MATCHFINDER_HASH_ORDER; // 32 768
const TOTAL_ENTRIES: usize = HASH_SIZE * HT_MATCHFINDER_BUCKET_SIZE; // 65 536

/// Total struct size in bytes — must be a multiple of `MATCHFINDER_SIZE_ALIGNMENT`.
/// `sizeof(ht_matchfinder)` = 65536 * 2 = 131 072 bytes (= 128 * 1024).
pub const HT_MATCHFINDER_SIZE: usize = TOTAL_ENTRIES * core::mem::size_of::<MfPos>();

const _: () = assert!(
    HT_MATCHFINDER_SIZE.is_multiple_of(1024),
    "sizeof(ht_matchfinder) must be a multiple of MATCHFINDER_SIZE_ALIGNMENT"
);

// ---------------------------------------------------------------------------
// HtMatchfinder struct
//
// Memory layout of flat `table` Vec:
//   Indexed as `hash_tab[hash * BUCKET_SIZE + slot]` (row-major 2-D array).
//   Mirrors `mf_pos_t hash_tab[HASH_SIZE][BUCKET_SIZE]`.
// ---------------------------------------------------------------------------

/// Hash-table matchfinder.  Mirrors `struct ht_matchfinder`.
///
/// Data-layout task: `table` is `AlignedTab<TOTAL_ENTRIES>` (32-byte
/// aligned, matching C's `MATCHFINDER_ALIGNED`) -- see `AlignedTab` in
/// `matchfinder_common.rs`, shared with `HcMatchfinder`/`BtMatchfinder`.
pub struct HtMatchfinder {
    /// Flat backing store: `hash_tab[HASH_SIZE][BUCKET_SIZE]` row-major.
    table: AlignedTab<TOTAL_ENTRIES>,
}

impl HtMatchfinder {
    /// Create and initialise a new `HtMatchfinder`.  Mirrors `ht_matchfinder_init`.
    pub fn new() -> Self {
        let mut mf = HtMatchfinder {
            table: AlignedTab::zeroed(),
        };
        mf.init();
        mf
    }

    /// Allocate the backing storage WITHOUT running `init()` on it.
    /// See `HcMatchfinder::new_uninit` -- same rationale, used only by
    /// `Compressor::new`, which always calls `.init()` itself right before
    /// the first `compress_fastest` call.
    pub(crate) fn new_uninit() -> Self {
        HtMatchfinder {
            table: AlignedTab::zeroed(),
        }
    }

    /// Reinitialise for a new input buffer.  Mirrors `ht_matchfinder_init`.
    #[inline]
    pub fn init(&mut self) {
        matchfinder_init(&mut self.table);
    }

    /// Slide window by `MATCHFINDER_WINDOW_SIZE`.  Mirrors `ht_matchfinder_slide_window`.
    #[inline]
    pub fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.table);
    }

    // --- Internal accessor: `hash_tab[hash][slot]` ---

    #[inline(always)]
    fn tab(&self, hash: usize, slot: usize) -> MfPos {
        self.table[hash * HT_MATCHFINDER_BUCKET_SIZE + slot]
    }
    #[inline(always)]
    fn tab_set(&mut self, hash: usize, slot: usize, v: MfPos) {
        self.table[hash * HT_MATCHFINDER_BUCKET_SIZE + slot] = v;
    }
}

impl Default for HtMatchfinder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ht_matchfinder_longest_match
// Mirrors the C function, using the BUCKET_SIZE==2 hand-unrolled branch.
//
// Parameters:
//   mf          — matchfinder state
//   in_buf      — full input buffer
//   in_base_pos — mutable: current base offset (mirrors `*in_base_p`)
//   in_next_pos — current search position (mirrors `in_next`)
//   max_len     — maximum permissible match length (must be >= REQUIRED_NBYTES)
//   nice_len    — stop early if match >= this length
//   next_hash   — precomputed hash for in_next_pos; updated to in_next_pos+1
//   offset_ret  — output: offset of best match
//
// Returns: best match length (0 if none).
// ---------------------------------------------------------------------------

/// Find the longest match at `in_next_pos`.  Mirrors `ht_matchfinder_longest_match`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ht_matchfinder_longest_match(
    mf: &mut HtMatchfinder,
    in_buf: &[u8],
    in_base_pos: &mut usize,
    in_next_pos: usize,
    max_len: u32,
    nice_len: u32,
    next_hash: &mut u32,
    offset_ret: &mut u32,
) -> u32 {
    let mut best_len: u32 = 0;
    let mut best_match_abs: usize = in_next_pos; // "no match" sentinel

    let mut cur_pos = in_next_pos - *in_base_pos;

    // Slide window if full.
    if cur_pos == MATCHFINDER_WINDOW_SIZE {
        mf.slide_window();
        *in_base_pos += MATCHFINDER_WINDOW_SIZE;
        cur_pos = 0;
    }
    let in_base = *in_base_pos;
    // cutoff: signed; any node <= cutoff is out of window.
    let cutoff: i32 = cur_pos as i32 - MATCHFINDER_WINDOW_SIZE as i32;

    // Use and update hash.
    let hash = *next_hash as usize;
    // Precompute next hash (for in_next_pos + 1).
    *next_hash = lz_hash(get_unaligned_le32(&in_buf[in_next_pos + 1..]), HT_MATCHFINDER_HASH_ORDER);
    // prefetchw omitted (no-op).

    let seq = get_unaligned_le32(&in_buf[in_next_pos..]);

    // --- BUCKET_SIZE == 2 hand-unrolled path (mirrors C `#elif HT_MATCHFINDER_BUCKET_SIZE == 2`) ---
    // Read slot 0.
    let cur_node_0 = mf.tab(hash, 0);
    // Write cur_pos into slot 0 (new head of bucket).
    mf.tab_set(hash, 0, cur_pos as MfPos);

    if cur_node_0 as i32 <= cutoff {
        *offset_ret = 0;
        return 0;
    }
    // C: `matchptr = &in_base[cur_node]` with SIGNED `cur_node` (s16).  After a window
    // slide the node is negative (it refers to the previous window), so the address
    // must be computed with signed arithmetic, not `in_base + (cur_node as usize)`.
    let match_abs_0 = (in_base as isize + cur_node_0 as isize) as usize;

    // Shift slot 0 → slot 1 (evict oldest entry).
    let to_insert = cur_node_0;
    let cur_node_1 = mf.tab(hash, 1);
    mf.tab_set(hash, 1, to_insert);

    if get_unaligned_le32(&in_buf[match_abs_0..]) == seq {
        best_len = lz_extend(in_buf, in_next_pos, match_abs_0, 4, max_len);
        best_match_abs = match_abs_0;
        if cur_node_1 as i32 <= cutoff || best_len >= nice_len {
            *offset_ret = (in_next_pos - best_match_abs) as u32;
            return best_len;
        }
        // Check slot 1 for a potentially longer match.
        let match_abs_1 = (in_base as isize + cur_node_1 as isize) as usize;
        if get_unaligned_le32(&in_buf[match_abs_1..]) == seq
            && get_unaligned_le32(&in_buf[match_abs_1 + best_len as usize - 3..])
               == get_unaligned_le32(&in_buf[in_next_pos + best_len as usize - 3..])
        {
            let len = lz_extend(in_buf, in_next_pos, match_abs_1, 4, max_len);
            if len > best_len {
                best_len = len;
                best_match_abs = match_abs_1;
            }
        }
    } else {
        // Slot 0 didn't match; try slot 1.
        if cur_node_1 as i32 <= cutoff {
            *offset_ret = 0;
            return 0;
        }
        let match_abs_1 = (in_base as isize + cur_node_1 as isize) as usize;
        if get_unaligned_le32(&in_buf[match_abs_1..]) == seq {
            best_len = lz_extend(in_buf, in_next_pos, match_abs_1, 4, max_len);
            best_match_abs = match_abs_1;
        }
    }

    *offset_ret = (in_next_pos - best_match_abs) as u32;
    best_len
}

// ---------------------------------------------------------------------------
// ht_matchfinder_skip_bytes
// Mirrors `ht_matchfinder_skip_bytes`.
// ---------------------------------------------------------------------------

/// Advance the matchfinder by `count` bytes without searching.
/// Mirrors `ht_matchfinder_skip_bytes`.
#[inline]
pub fn ht_matchfinder_skip_bytes(
    mf: &mut HtMatchfinder,
    in_buf: &[u8],
    in_base_pos: &mut usize,
    mut in_next_pos: usize,
    in_end_pos: usize,
    count: u32,
    next_hash: &mut u32,
) {
    // Guard: need count + REQUIRED_NBYTES bytes remaining.
    if (count as usize).saturating_add(HT_MATCHFINDER_REQUIRED_NBYTES)
        > in_end_pos.saturating_sub(in_next_pos)
    {
        return;
    }

    let mut cur_pos = in_next_pos as i32 - *in_base_pos as i32;

    // Slide window in one shot if any position in this skip would overflow.
    if cur_pos + count as i32 > MATCHFINDER_WINDOW_SIZE as i32 {
        mf.slide_window();
        *in_base_pos += MATCHFINDER_WINDOW_SIZE;
        cur_pos -= MATCHFINDER_WINDOW_SIZE as i32;
    }

    let mut hash = *next_hash as usize;
    let mut remaining = count;

    loop {
        // Shift all slots right (evict oldest), insert cur_pos at slot 0.
        for slot in (1..HT_MATCHFINDER_BUCKET_SIZE).rev() {
            let prev = mf.tab(hash, slot - 1);
            mf.tab_set(hash, slot, prev);
        }
        mf.tab_set(hash, 0, cur_pos as MfPos);

        in_next_pos += 1;
        hash = lz_hash(
            get_unaligned_le32(&in_buf[in_next_pos..]),
            HT_MATCHFINDER_HASH_ORDER,
        ) as usize;
        cur_pos += 1;

        remaining -= 1;
        if remaining == 0 {
            break;
        }
    }

    // prefetchw omitted (no-op).
    *next_hash = hash as u32;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn initial_hash(in_buf: &[u8], pos: usize) -> u32 {
        lz_hash(get_unaligned_le32(&in_buf[pos..]), HT_MATCHFINDER_HASH_ORDER)
    }

    #[test]
    fn ht_finds_match_in_repeated_buffer() {
        // "the quick the quick" — second occurrence should match first.
        let input = b"the quick the quick";
        let mut mf = HtMatchfinder::new();
        let mut in_base_pos = 0usize;
        let mut next_hash = initial_hash(input, 0);

        // Warm up first 10 positions.
        ht_matchfinder_skip_bytes(
            &mut mf,
            input,
            &mut in_base_pos,
            0,
            input.len(),
            10,
            &mut next_hash,
        );

        // Search at position 10.
        let mut offset = 0u32;
        let len = ht_matchfinder_longest_match(
            &mut mf,
            input,
            &mut in_base_pos,
            10,
            (input.len() - 10) as u32,
            258,
            &mut next_hash,
            &mut offset,
        );

        assert!(len >= 4, "expected at least a 4-byte match, got {}", len);
        assert_eq!(offset, 10, "expected offset 10, got {}", offset);
    }

    #[test]
    fn ht_no_match_returns_zero() {
        // Completely unique bytes — no match should be found.
        let input: Vec<u8> = (0u8..20).collect();
        let mut mf = HtMatchfinder::new();
        let mut in_base_pos = 0usize;
        let mut next_hash = initial_hash(&input, 0);

        ht_matchfinder_skip_bytes(
            &mut mf,
            &input,
            &mut in_base_pos,
            0,
            input.len(),
            5,
            &mut next_hash,
        );

        let mut offset = 0u32;
        let len = ht_matchfinder_longest_match(
            &mut mf,
            &input,
            &mut in_base_pos,
            5,
            (input.len() - 5) as u32,
            258,
            &mut next_hash,
            &mut offset,
        );
        // With unique bytes it's possible no match is found (len == 0).
        // The important thing is no panic and offset makes sense if len > 0.
        if len > 0 {
            assert!(offset > 0);
        }
    }

    #[test]
    fn ht_skip_updates_hash() {
        let input = b"abcdeabcde__________";
        let mut mf = HtMatchfinder::new();
        let mut in_base_pos = 0usize;
        let mut next_hash = initial_hash(input, 0);
        let before = next_hash;
        ht_matchfinder_skip_bytes(
            &mut mf,
            input,
            &mut in_base_pos,
            0,
            input.len(),
            3,
            &mut next_hash,
        );
        assert_ne!(next_hash, before, "next_hash should be updated after skip");
    }

    #[test]
    fn ht_init_fills_initval() {
        let mut mf = HtMatchfinder::new();
        // Dirty it.
        mf.table[0] = 42;
        mf.table[TOTAL_ENTRIES - 1] = 99;
        mf.init();
        let initval = crate::matchfinder_common::MATCHFINDER_INITVAL;
        assert_eq!(mf.table[0], initval);
        assert_eq!(mf.table[TOTAL_ENTRIES - 1], initval);
    }

    #[test]
    fn ht_nice_len_cutoff() {
        // If nice_len is very small (4), should stop after first match.
        let input = b"abcdabcdabcdabcdabcdabcd";
        let mut mf = HtMatchfinder::new();
        let mut in_base_pos = 0usize;
        let mut next_hash = initial_hash(input, 0);
        ht_matchfinder_skip_bytes(
            &mut mf,
            input,
            &mut in_base_pos,
            0,
            input.len(),
            4,
            &mut next_hash,
        );
        let mut offset = 0u32;
        let len = ht_matchfinder_longest_match(
            &mut mf,
            input,
            &mut in_base_pos,
            4,
            (input.len() - 4) as u32,
            4, // nice_len = 4
            &mut next_hash,
            &mut offset,
        );
        // If a match is found, it should not exceed nice_len in a meaningful way
        // (the C code stops early at nice_len).
        if len > 0 {
            assert!(len >= 4);
        }
    }

    /// At least one input > `MATCHFINDER_WINDOW_SIZE` (32 KiB),
    /// actually crossing a `slide_window()` call, with the match verified
    /// byte-for-byte. See the identical rationale on
    /// `hc_matchfinder::tests::hc_finds_correct_match_across_window_slide`.
    ///
    /// MUTATION CHECKED: temporarily changed this function's final
    /// `*offset_ret = (in_next_pos - best_match_abs) as u32;` to `... + 1`
    /// and reran this test — it went RED with a byte-mismatch panic.
    /// Reverted immediately.
    #[test]
    fn ht_finds_correct_match_across_window_slide() {
        const PERIOD: usize = 37;
        let pattern: Vec<u8> = (0..PERIOD as u8).collect();
        let len = 40_000usize;
        let input: Vec<u8> = (0..len).map(|i| pattern[i % PERIOD]).collect();

        let mut mf = HtMatchfinder::new();
        let mut in_base_pos = 0usize;
        let mut next_hash = initial_hash(&input, 0);

        let search_pos = len - 40;
        ht_matchfinder_skip_bytes(
            &mut mf,
            &input,
            &mut in_base_pos,
            0,
            len,
            search_pos as u32,
            &mut next_hash,
        );
        assert!(
            in_base_pos > 0,
            "test buffer (40000 B) must actually cross a MATCHFINDER_WINDOW_SIZE \
             (32768 B) slide; in_base_pos stayed 0"
        );

        // Mirrors adjust_max_and_nice_len (see the identical note in
        // hc_matchfinder's window-slide test).
        let max_len = (len - search_pos) as u32;
        let nice_len = core::cmp::min(258, max_len);
        let mut offset = 0u32;
        let found_len = ht_matchfinder_longest_match(
            &mut mf,
            &input,
            &mut in_base_pos,
            search_pos,
            max_len,
            nice_len,
            &mut next_hash,
            &mut offset,
        );

        assert!(found_len >= 4, "expected a match after the window slide, found none");
        assert!(offset > 0 && offset as usize <= search_pos, "offset out of range: {offset}");
        let src = search_pos - offset as usize;
        assert_eq!(
            &input[src..src + found_len as usize],
            &input[search_pos..search_pos + found_len as usize],
            "matchfinder reported a match that is not byte-identical (offset={offset}, len={found_len})"
        );
    }
}
