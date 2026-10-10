//! Port of libdeflate `lib/hc_matchfinder.h` — hash-chain matchfinder.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/hc_matchfinder.h
//!
//! # Algorithm
//! Hash-chains (HC) matchfinder.  Two hash tables:
//! - `hash3_tab` (order 15): singleton per bucket, finds length-3 matches.
//! - `hash4_tab` (order 16): head of linked list, finds length-4+ matches.
//! - `next_tab` (size = window): singly-linked chain via position index.
//!
//! Used by greedy and lazy compression levels.
//!
//! # Pointer → index mapping
//! C uses `const u8 *in_base` (pointer to window base) and pointer arithmetic.
//! Rust uses `in_buf: &[u8]` (the full input) and `in_base_pos: usize` (offset
//! of the current window base within `in_buf`).
//! - `cur_pos = in_next_pos - in_base_pos`   (relative position in window)
//! - `&in_base[node]` → `in_buf[in_base_pos + node as usize]`
//! - `in_next - best_matchptr` → `in_next_pos - best_match_abs` (absolute positions)
//!
//! # N/A
//! - `prefetchw`: no-op (performance hint only, no correctness impact).
//! - `STATIC_ASSERT(... % MATCHFINDER_SIZE_ALIGNMENT == 0)`: sizes are fixed
//!   constants; verified by `const` assertions at compile time.
//! - `MATCHFINDER_ALIGNED`: heap allocation via `Vec` does not require it.
//! - `UNALIGNED_ACCESS_IS_FAST` fast-path in the inner loop: we use the simpler
//!   `last-byte` check (`matchptr[best_len] == in_next[best_len]`), which is the
//!   portable branch that the C code falls back to on platforms without fast
//!   unaligned access.  The `lz_extend` word-at-a-time path handles the actual
//!   extension quickly.

use crate::common::get_unaligned_le32;
use crate::matchfinder_common::{
    lz_extend, lz_hash, load_u24_unaligned, loaded_u32_to_u24, matchfinder_init,
    matchfinder_rebase, AlignedTab, MfPos, MATCHFINDER_WINDOW_SIZE,
};

// ---------------------------------------------------------------------------
// Constants — mirrors C `#define` values in hc_matchfinder.h
// ---------------------------------------------------------------------------

/// Hash table order for length-3 matches.  `HC_MATCHFINDER_HASH3_ORDER`.
pub const HC_MATCHFINDER_HASH3_ORDER: u32 = 15;
/// Hash table order for length-4+ matches.  `HC_MATCHFINDER_HASH4_ORDER`.
pub const HC_MATCHFINDER_HASH4_ORDER: u32 = 16;

const HASH3_SIZE: usize = 1 << HC_MATCHFINDER_HASH3_ORDER; // 32 768
const HASH4_SIZE: usize = 1 << HC_MATCHFINDER_HASH4_ORDER; // 65 536

/// Total hash-table byte size for the two hash tables (excludes `next_tab`).
/// `HC_MATCHFINDER_TOTAL_HASH_SIZE` = (32768 + 65536) * 2 = 196 608 bytes.
/// Must be a multiple of `MATCHFINDER_SIZE_ALIGNMENT` (1024) — verified below.
pub const HC_MATCHFINDER_TOTAL_HASH_SIZE: usize =
    (HASH3_SIZE + HASH4_SIZE) * core::mem::size_of::<MfPos>();

const _: () = assert!(
    HC_MATCHFINDER_TOTAL_HASH_SIZE.is_multiple_of(1024),
    "HC_MATCHFINDER_TOTAL_HASH_SIZE must be a multiple of MATCHFINDER_SIZE_ALIGNMENT"
);

// sizeof(hc_matchfinder) includes hash3_tab + hash4_tab + next_tab.
// In C: (32768 + 65536 + 32768) * 2 = 262 144 bytes, which is 1024-aligned.
const HC_MF_TOTAL_ENTRIES: usize = HASH3_SIZE + HASH4_SIZE + MATCHFINDER_WINDOW_SIZE;
const _HC_SIZE_CHECK: () = assert!(
    (HC_MF_TOTAL_ENTRIES * core::mem::size_of::<MfPos>()).is_multiple_of(1024),
    "sizeof(hc_matchfinder) must be a multiple of MATCHFINDER_SIZE_ALIGNMENT"
);

// ---------------------------------------------------------------------------
// HcMatchfinder struct
// Mirrors `struct hc_matchfinder { mf_pos_t hash3_tab[32768]; hash4_tab[65536]; next_tab[32768]; }`.
// Fields are stored in a single flat Vec for cache locality (same layout as C),
// sliced out as needed.
// ---------------------------------------------------------------------------

/// Hash-chain matchfinder.  Mirrors `struct hc_matchfinder`.
///
/// Memory layout (flat `table`):
/// - `[0 .. HASH3_SIZE)` → `hash3_tab`
/// - `[HASH3_SIZE .. HASH3_SIZE + HASH4_SIZE)` → `hash4_tab`
/// - `[HASH3_SIZE + HASH4_SIZE .. HC_MF_TOTAL_ENTRIES)` → `next_tab`
///
/// Data-layout task: `table` is `AlignedTab<HC_MF_TOTAL_ENTRIES>`
/// (32-byte aligned, matching C's `MATCHFINDER_MEM_ALIGNMENT` /
/// `MATCHFINDER_ALIGNED`, `matchfinder_common.h:66,86`) instead of a plain
/// `Vec<MfPos>` -- see [`AlignedTab`] in `matchfinder_common.rs`, shared
/// with `HtMatchfinder`/`BtMatchfinder` (one generic type, not three
/// copies).
pub struct HcMatchfinder {
    /// Flat backing store: `hash3_tab | hash4_tab | next_tab`.
    table: AlignedTab<HC_MF_TOTAL_ENTRIES>,
}

impl HcMatchfinder {
    // Index offsets into `self.table`.
    const HASH3_OFF: usize = 0;
    const HASH4_OFF: usize = HASH3_SIZE;
    const NEXT_OFF: usize = HASH3_SIZE + HASH4_SIZE;

    /// Create and initialise a new `HcMatchfinder`.  Mirrors `hc_matchfinder_init`.
    pub fn new() -> Self {
        let mut mf = HcMatchfinder {
            table: AlignedTab::zeroed(),
        };
        mf.init();
        mf
    }

    /// Allocate the backing storage WITHOUT running `init()` on it.
    ///
    /// `Compressor::new` (the allocator) always calls `.init()` itself
    /// right before the first `deflate_compress_greedy`/`lazy_generic` call
    /// on the compressor it just built, so calling the self-initialising
    /// `new()` there ran `matchfinder_init` (a full sweep filling
    /// `hash3_tab`/`hash4_tab` with `MATCHFINDER_INITVAL`) twice for every
    /// compressor, once at construction and once discarded a moment later.
    /// C's allocator (`deflate_compress.c` union arm) never runs
    /// `hc_matchfinder_init` at alloc time -- only right before use, same as
    /// this crate's `compress_greedy`/`compress_lazy_generic` already do.
    /// Not exposed outside the crate: any caller besides `Compressor::new`
    /// that wants a ready-to-use matchfinder should keep using `new()`.
    pub(crate) fn new_uninit() -> Self {
        HcMatchfinder {
            table: AlignedTab::zeroed(),
        }
    }

    /// Reinitialise the hash tables (but not `next_tab`) for a new input buffer.
    /// Mirrors `hc_matchfinder_init` which only initialises the hash portion.
    #[inline]
    pub fn init(&mut self) {
        // Only the hash tables are initialised; `next_tab` gets overwritten lazily.
        matchfinder_init(&mut self.table[Self::HASH3_OFF..Self::HASH3_OFF + HASH3_SIZE]);
        matchfinder_init(&mut self.table[Self::HASH4_OFF..Self::HASH4_OFF + HASH4_SIZE]);
    }

    /// Slide the window by `MATCHFINDER_WINDOW_SIZE`.
    /// Mirrors `hc_matchfinder_slide_window` which calls `matchfinder_rebase`
    /// over the ENTIRE struct (hash tables + next_tab).
    #[inline]
    pub fn slide_window(&mut self) {
        matchfinder_rebase(&mut self.table);
    }

    // --- Internal table accessors (inline, zero-cost) ---

    #[inline(always)]
    fn hash3(&self, h: usize) -> MfPos {
        self.table[Self::HASH3_OFF + h]
    }
    #[inline(always)]
    fn hash3_set(&mut self, h: usize, v: MfPos) {
        self.table[Self::HASH3_OFF + h] = v;
    }
    #[inline(always)]
    fn hash4(&self, h: usize) -> MfPos {
        self.table[Self::HASH4_OFF + h]
    }
    #[inline(always)]
    fn hash4_set(&mut self, h: usize, v: MfPos) {
        self.table[Self::HASH4_OFF + h] = v;
    }
    #[inline(always)]
    fn next(&self, pos: usize) -> MfPos {
        // Mirrors `mf->next_tab[cur_node & (MATCHFINDER_WINDOW_SIZE - 1)]`
        self.table[Self::NEXT_OFF + (pos & (MATCHFINDER_WINDOW_SIZE - 1))]
    }
    #[inline(always)]
    fn next_set(&mut self, pos: usize, v: MfPos) {
        self.table[Self::NEXT_OFF + (pos & (MATCHFINDER_WINDOW_SIZE - 1))] = v;
    }
}

impl Default for HcMatchfinder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// hc_matchfinder_longest_match
// Mirrors the C function exactly, using index arithmetic instead of pointers.
//
// Parameters:
//   mf              — matchfinder state
//   in_buf          — full input buffer (immutable slice)
//   in_base_pos     — mutable: current base offset into in_buf (mirrors `*in_base_p`)
//   in_next_pos     — current search position in in_buf (mirrors `in_next`)
//   best_len        — already-known match length; find something longer
//   max_len         — maximum permissible match length
//   nice_len        — stop early if match >= this length
//   max_search_depth — max chain nodes to visit
//   next_hashes     — [hash3, hash4] precomputed for in_next_pos; updated to in_next_pos+1
//   offset_ret      — output: offset of best match found
//
// Returns the new best_len (unchanged if no longer match found).
// ---------------------------------------------------------------------------

/// Find the longest match longer than `best_len` at `in_next_pos`.
/// Mirrors `hc_matchfinder_longest_match`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn hc_matchfinder_longest_match(
    mf: &mut HcMatchfinder,
    in_buf: &[u8],
    in_base_pos: &mut usize,
    in_next_pos: usize,
    mut best_len: u32,
    max_len: u32,
    nice_len: u32,
    max_search_depth: u32,
    next_hashes: &mut [u32; 2],
    offset_ret: &mut u32,
) -> u32 {
    let mut depth_remaining = max_search_depth;

    // Absolute position of the best match found so far (initialised to in_next = no match).
    let mut best_match_abs: usize = in_next_pos;

    // cur_pos: position relative to current window base — mirrors `in_next - *in_base_p`.
    let mut cur_pos = in_next_pos - *in_base_pos;

    // Slide window if we've filled it.
    if cur_pos == MATCHFINDER_WINDOW_SIZE {
        mf.slide_window();
        *in_base_pos += MATCHFINDER_WINDOW_SIZE;
        cur_pos = 0;
    }

    let in_base = *in_base_pos;
    // cutoff: any node <= cutoff is out of the current window.
    // Mirrors `mf_pos_t cutoff = cur_pos - MATCHFINDER_WINDOW_SIZE` (signed arithmetic).
    let cutoff: i32 = cur_pos as i32 - MATCHFINDER_WINDOW_SIZE as i32;

    if max_len < 5 {
        // Cannot read 4 bytes from in_next + 1; bail immediately.
        *offset_ret = 0;
        return best_len;
    }

    // Get precomputed hash codes.
    let hash3 = next_hashes[0] as usize;
    let hash4 = next_hashes[1] as usize;

    // Read head nodes from hash tables before updating them.
    let cur_node3 = mf.hash3(hash3);
    let mut cur_node4 = mf.hash4(hash4);

    // Update hash3: replace with current position (singleton bucket).
    mf.hash3_set(hash3, cur_pos as MfPos);
    // Update hash4: prepend current position to chain.
    mf.hash4_set(hash4, cur_pos as MfPos);
    mf.next_set(cur_pos, cur_node4);

    // Precompute next hashes for in_next_pos + 1.
    let next_hashseq = get_unaligned_le32(&in_buf[in_next_pos + 1..]);
    next_hashes[0] = lz_hash(next_hashseq & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER);
    next_hashes[1] = lz_hash(next_hashseq, HC_MATCHFINDER_HASH4_ORDER);
    // prefetchw omitted (no-op, see module doc).

    if best_len < 4 {
        // No match of length >= 4 found yet.

        // Check for length-3 match.
        if cur_node3 as i32 > cutoff {
            let seq4 = get_unaligned_le32(&in_buf[in_next_pos..]);

            if best_len < 3 {
                // C: `matchptr = &in_base[cur_node3]` — SIGNED node indexing (no mask).
                // The mask `& (WINDOW-1)` is used only for `next_tab` chain lookups.
                let match_abs = (in_base as isize + cur_node3 as isize) as usize;
                if load_u24_unaligned(&in_buf[match_abs..]) == loaded_u32_to_u24(seq4) {
                    best_len = 3;
                    best_match_abs = match_abs;
                }
            }

            // Check for length-4 match by walking the chain.
            if cur_node4 as i32 <= cutoff {
                *offset_ret = (in_next_pos - best_match_abs) as u32;
                return best_len;
            }

            let seq4_u32 = seq4;
            loop {
                // Check first 4 bytes.
                let match_abs = (in_base as isize + cur_node4 as isize) as usize;
                if get_unaligned_le32(&in_buf[match_abs..]) == seq4_u32 {
                    break; // Found length >= 4.
                }
                // Advance chain.
                cur_node4 = mf.next(cur_node4 as usize);
                depth_remaining -= 1;
                if cur_node4 as i32 <= cutoff || depth_remaining == 0 {
                    *offset_ret = (in_next_pos - best_match_abs) as u32;
                    return best_len;
                }
            }

            // Extend the length-4 match to its full length.
            let match_abs = (in_base as isize + cur_node4 as isize) as usize;
            best_match_abs = match_abs;
            best_len = lz_extend(in_buf, in_next_pos, match_abs, 4, max_len);
            if best_len >= nice_len {
                *offset_ret = (in_next_pos - best_match_abs) as u32;
                return best_len;
            }
            cur_node4 = mf.next(cur_node4 as usize);
            depth_remaining -= 1;
            if cur_node4 as i32 <= cutoff || depth_remaining == 0 {
                *offset_ret = (in_next_pos - best_match_abs) as u32;
                return best_len;
            }
        } else {
            // cur_node3 <= cutoff: no length-3 match possible.
            *offset_ret = 0;
            return best_len;
        }
    } else {
        // Already have a length-4+ match.
        if cur_node4 as i32 <= cutoff || best_len >= nice_len {
            *offset_ret = 0;
            return best_len;
        }
    }

    // Search for length >= 5 matches.
    //
    // Mirrors C's UNALIGNED_ACCESS_IS_FAST fast filter
    // (hc_matchfinder.h:296-306), not the portable 1-byte-check fallback
    // (`matchptr[best_len] == in_next[best_len]`) this used to port instead.
    // The fast filter compares TWO u32s: the 4 bytes ending at `best_len`
    // (the byte that would extend the match past the current best, plus 3
    // bytes of context -- "the last byte... is the most important") AND the
    // first 4 bytes (needed because a chain candidate hasn't necessarily
    // been re-verified there -- only the very first length-4 candidate was).
    // Filtering on both cuts the number of `lz_extend` calls on candidates
    // that can't possibly beat `best_len`, without changing which match
    // wins: `lz_extend` always recomputes the TRUE length from scratch, so
    // a filter that lets through occasional false positives (a hash
    // collision on both u32 windows) or is merely a fast pre-check can never
    // change the actual output, only how much work is spent getting there.
    //
    // Safety of the 4-byte reads at `+ best_len - 3`: reachable only with
    // `best_len < nice_len` (we'd have returned via the `best_len >=
    // nice_len` check otherwise) and `nice_len <= max_len <= remaining`
    // (the latter always holds: `adjust_max_and_nice_len` clamps `max_len`
    // to `remaining` and `nice_len` to `<= max_len` whenever `remaining <
    // DEFLATE_MAX_MATCH_LEN`, and `nice_len <= max_len` holds by
    // configuration otherwise) -- so `best_len < remaining`, keeping
    // `in_next_pos + best_len` (the last byte either 4-byte read touches)
    // in bounds. The identical invariant chain already justifies the same
    // read pattern in `ht_matchfinder.rs`'s `longest_match`.
    'outer: loop {
        loop {
            let match_abs = (in_base as isize + cur_node4 as isize) as usize;
            let tail_ok = get_unaligned_le32(&in_buf[match_abs + best_len as usize - 3..])
                == get_unaligned_le32(&in_buf[in_next_pos + best_len as usize - 3..]);
            let head_ok = get_unaligned_le32(&in_buf[match_abs..])
                == get_unaligned_le32(&in_buf[in_next_pos..]);
            if tail_ok && head_ok {
                break; // Potential improvement — do full extension.
            }
            cur_node4 = mf.next(cur_node4 as usize);
            depth_remaining -= 1;
            if cur_node4 as i32 <= cutoff || depth_remaining == 0 {
                break 'outer;
            }
        }

        let match_abs = (in_base as isize + cur_node4 as isize) as usize;
        // Start extension from 4: the head_ok check above already verified
        // the first 4 bytes match, mirroring C's `len = 4;` under
        // UNALIGNED_ACCESS_IS_FAST (vs. `len = 0` on the portable path this
        // replaces). `lz_extend` still recomputes the true length; this only
        // skips 4 redundant byte comparisons it would otherwise repeat.
        let len = lz_extend(in_buf, in_next_pos, match_abs, 4, max_len);
        if len > best_len {
            best_len = len;
            best_match_abs = match_abs;
            if best_len >= nice_len {
                break 'outer;
            }
        }

        cur_node4 = mf.next(cur_node4 as usize);
        depth_remaining -= 1;
        if cur_node4 as i32 <= cutoff || depth_remaining == 0 {
            break 'outer;
        }
    }

    *offset_ret = (in_next_pos - best_match_abs) as u32;
    best_len
}

// ---------------------------------------------------------------------------
// hc_matchfinder_skip_bytes
// Mirrors `hc_matchfinder_skip_bytes` — advance without searching.
// ---------------------------------------------------------------------------

/// Advance the matchfinder by `count` bytes without searching for matches.
/// Mirrors `hc_matchfinder_skip_bytes`.
#[inline]
pub fn hc_matchfinder_skip_bytes(
    mf: &mut HcMatchfinder,
    in_buf: &[u8],
    in_base_pos: &mut usize,
    mut in_next_pos: usize,
    in_end_pos: usize,
    count: u32,
    next_hashes: &mut [u32; 2],
) {
    // Guard: need count + 5 bytes remaining.
    if (count as usize).saturating_add(5) > in_end_pos.saturating_sub(in_next_pos) {
        return;
    }

    let mut cur_pos = in_next_pos - *in_base_pos;
    let mut hash3 = next_hashes[0] as usize;
    let mut hash4 = next_hashes[1] as usize;
    let mut remaining = count;

    loop {
        if cur_pos == MATCHFINDER_WINDOW_SIZE {
            mf.slide_window();
            *in_base_pos += MATCHFINDER_WINDOW_SIZE;
            cur_pos = 0;
        }

        mf.hash3_set(hash3, cur_pos as MfPos);
        let old_head = mf.hash4(hash4);
        mf.next_set(cur_pos, old_head);
        mf.hash4_set(hash4, cur_pos as MfPos);

        in_next_pos += 1;
        let next_hashseq = get_unaligned_le32(&in_buf[in_next_pos..]);
        hash3 = lz_hash(next_hashseq & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER) as usize;
        hash4 = lz_hash(next_hashseq, HC_MATCHFINDER_HASH4_ORDER) as usize;
        cur_pos += 1;

        remaining -= 1;
        if remaining == 0 {
            break;
        }
    }

    // prefetchw omitted (no-op).
    next_hashes[0] = hash3 as u32;
    next_hashes[1] = hash4 as u32;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a test buffer with a known repeated substring and verify that the
    /// HC matchfinder finds the expected match.
    #[test]
    fn hc_finds_match_in_repeated_buffer() {
        // "the quick the quick" — the second "the quick" should match the first.
        let input = b"the quick the quick";
        let mut mf = HcMatchfinder::new();
        let mut in_base_pos: usize = 0;

        // Pre-compute initial hashes for position 0 from input[0..4].
        let seq0 = get_unaligned_le32(&input[0..]);
        let mut next_hashes = [
            lz_hash(seq0 & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER),
            lz_hash(seq0, HC_MATCHFINDER_HASH4_ORDER),
        ];

        // Warm up positions 0..10 (first "the quick ") via skip.
        hc_matchfinder_skip_bytes(
            &mut mf,
            input,
            &mut in_base_pos,
            0,
            input.len(),
            10,
            &mut next_hashes,
        );

        // Now search at position 10 — should find match at offset 10 (back to pos 0).
        let mut offset = 0u32;
        let len = hc_matchfinder_longest_match(
            &mut mf,
            input,
            &mut in_base_pos,
            10,
            0,
            (input.len() - 10) as u32,
            258,
            32,
            &mut next_hashes,
            &mut offset,
        );

        // The match "the quick" is 9 bytes long and at offset 10.
        assert!(len >= 9, "expected len >= 9, got {}", len);
        assert_eq!(offset, 10, "expected offset 10, got {}", offset);
    }

    #[test]
    fn hc_skip_bytes_updates_hashes() {
        // Use a buffer with no period-5 pattern to avoid hash collisions.
        let input = b"abcdefghijklmnopqrst"; // 20 distinct bytes
        let mut mf = HcMatchfinder::new();
        let mut in_base_pos = 0usize;
        let seq0 = get_unaligned_le32(&input[0..]);
        let mut next_hashes = [
            lz_hash(seq0 & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER),
            lz_hash(seq0, HC_MATCHFINDER_HASH4_ORDER),
        ];
        let hashes_before = next_hashes;
        hc_matchfinder_skip_bytes(
            &mut mf,
            input,
            &mut in_base_pos,
            0,
            input.len(),
            5,
            &mut next_hashes,
        );
        // Hashes must have been updated (now reflect position 5 "fghij").
        assert_ne!(next_hashes[0], hashes_before[0]);
        assert_ne!(next_hashes[1], hashes_before[1]);
    }

    #[test]
    fn hc_short_buffer_no_panic() {
        // max_len < 5 → should return best_len unchanged and not panic.
        let input = b"abcd";
        let mut mf = HcMatchfinder::new();
        let mut in_base_pos = 0usize;
        let seq0 = get_unaligned_le32(input);
        let mut next_hashes = [
            lz_hash(seq0 & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER),
            lz_hash(seq0, HC_MATCHFINDER_HASH4_ORDER),
        ];
        let mut offset = 0u32;
        let len = hc_matchfinder_longest_match(
            &mut mf,
            input,
            &mut in_base_pos,
            0,
            0,
            4, // max_len < 5
            4,
            32,
            &mut next_hashes,
            &mut offset,
        );
        assert_eq!(len, 0);
    }

    #[test]
    fn hc_max_search_depth_cutoff() {
        // A buffer with many repeated 4-byte sequences: depth=1 should still find something.
        let input: Vec<u8> = b"abcdabcdabcdabcdabcdabcdabcdabcd".to_vec();
        let mut mf = HcMatchfinder::new();
        let mut in_base_pos = 0usize;
        let seq0 = get_unaligned_le32(&input[0..]);
        let mut next_hashes = [
            lz_hash(seq0 & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER),
            lz_hash(seq0, HC_MATCHFINDER_HASH4_ORDER),
        ];
        // Skip first 4 bytes to build chain.
        hc_matchfinder_skip_bytes(
            &mut mf,
            &input,
            &mut in_base_pos,
            0,
            input.len(),
            4,
            &mut next_hashes,
        );
        let mut offset = 0u32;
        let len = hc_matchfinder_longest_match(
            &mut mf,
            &input,
            &mut in_base_pos,
            4,
            0,
            (input.len() - 4) as u32,
            258,
            1, // depth = 1
            &mut next_hashes,
            &mut offset,
        );
        assert!(len >= 4, "expected at least a 4-byte match, got {}", len);
    }

    #[test]
    fn hc_init_clears_hash_tables() {
        let mut mf = HcMatchfinder::new();
        // Warm up a few positions.
        let input = b"abcdefghij_________";
        let mut in_base_pos = 0usize;
        let seq0 = get_unaligned_le32(&input[0..]);
        let mut next_hashes = [
            lz_hash(seq0 & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER),
            lz_hash(seq0, HC_MATCHFINDER_HASH4_ORDER),
        ];
        hc_matchfinder_skip_bytes(
            &mut mf,
            input,
            &mut in_base_pos,
            0,
            input.len(),
            5,
            &mut next_hashes,
        );
        // Re-init should reset hash tables.
        mf.init();
        // All hash entries should be MATCHFINDER_INITVAL again.
        let initval = crate::matchfinder_common::MATCHFINDER_INITVAL;
        for &v in &mf.table[..HASH3_SIZE + HASH4_SIZE] {
            assert_eq!(v, initval);
        }
    }

    /// At least one input > `MATCHFINDER_WINDOW_SIZE` (32 KiB) per
    /// matchfinder test, actually crossing a `slide_window()` call, with the
    /// match verified byte-for-byte (not just "something was found") —
    /// current unit tests here never exceeded a few hundred bytes, so a
    /// matchfinder bug that only manifests after a real window slide (stale
    /// `next_tab`/hash entries referencing pre-slide positions, wrong
    /// `cutoff` sign handling, etc.) had zero unit-level coverage.
    ///
    /// MUTATION CHECKED (the byte comparison must catch a real wrong answer, not just "no
    /// panic"): temporarily changed the length-4 match's `best_match_abs =
    /// match_abs;` (this file, in the length-4-then-extend branch) to
    /// `match_abs + 1` and reran this test — it went RED with "matchfinder
    /// reported a match that is not byte-identical (offset=..., len=...)",
    /// confirming the assertion is load-bearing. Reverted immediately. The
    /// analogous mutation was independently checked for `bt` and `ht` on
    /// their own `*_finds_correct_match_across_window_slide` tests below /
    /// in `ht_matchfinder.rs`, with the same result.
    #[test]
    fn hc_finds_correct_match_across_window_slide() {
        // A 37-byte-period cycle (co-prime-ish length vs. the hash orders)
        // repeated past MATCHFINDER_WINDOW_SIZE (32768), so a real match
        // exists both before and after the slide crosses at position 32768.
        const PERIOD: usize = 37;
        let pattern: Vec<u8> = (0..PERIOD as u8).collect();
        let len = 40_000usize;
        let input: Vec<u8> = (0..len).map(|i| pattern[i % PERIOD]).collect();

        let mut mf = HcMatchfinder::new();
        let mut in_base_pos = 0usize;
        let seq0 = get_unaligned_le32(&input[0..]);
        let mut next_hashes = [
            lz_hash(seq0 & 0x00FF_FFFF, HC_MATCHFINDER_HASH3_ORDER),
            lz_hash(seq0, HC_MATCHFINDER_HASH4_ORDER),
        ];

        // Skip through most of the buffer in one call, crossing the slide at
        // position MATCHFINDER_WINDOW_SIZE (32768) — hc_matchfinder_skip_bytes
        // handles the slide internally (mirrors production compressor use).
        let search_pos = len - 40; // stop early enough to leave room to search
        hc_matchfinder_skip_bytes(
            &mut mf,
            &input,
            &mut in_base_pos,
            0,
            len,
            search_pos as u32,
            &mut next_hashes,
        );
        assert!(
            in_base_pos > 0,
            "test buffer (40000 B) must actually cross a MATCHFINDER_WINDOW_SIZE \
             (32768 B) slide; in_base_pos stayed 0"
        );

        // Mirrors adjust_max_and_nice_len (deflate_compress.c:2271-2277):
        // nice_len must be clamped to <= max_len whenever max_len is clamped
        // to the remaining bytes, so the loop's nice_len early-exit fires at
        // best_len == max_len instead of indexing in_buf[in_next_pos+max_len]
        // (== buffer length, out of bounds).
        let max_len = (len - search_pos) as u32;
        let nice_len = core::cmp::min(258, max_len);
        let mut offset = 0u32;
        let found_len = hc_matchfinder_longest_match(
            &mut mf,
            &input,
            &mut in_base_pos,
            search_pos,
            0,
            max_len,
            nice_len,
            32,
            &mut next_hashes,
            &mut offset,
        );

        // The periodic pattern guarantees a length->=PERIOD match at some
        // multiple-of-PERIOD offset is findable; assert it's found AND that
        // the claimed match is byte-for-byte real (the actual correctness
        // property — a matchfinder returning a plausible-looking but wrong
        // offset/length is exactly the class of bug window slides can hide).
        assert!(found_len >= 3, "expected a match after the window slide, found none");
        assert!(offset > 0 && offset as usize <= search_pos, "offset out of range: {offset}");
        let src = search_pos - offset as usize;
        assert_eq!(
            &input[src..src + found_len as usize],
            &input[search_pos..search_pos + found_len as usize],
            "matchfinder reported a match that is not byte-identical (offset={offset}, len={found_len})"
        );
    }
}
