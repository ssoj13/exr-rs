//! Port of libdeflate `lib/deflate_constants.h` — DEFLATE format constants.
//! https://github.com/ebiggers/libdeflate/blob/v1.26/lib/deflate_constants.h
//!
//! All symbols are ported 1:1 as Rust `const`s.

// ---------------------------------------------------------------------------
// Block types
// ---------------------------------------------------------------------------

/// Uncompressed (stored) block. `DEFLATE_BLOCKTYPE_UNCOMPRESSED`.
pub const DEFLATE_BLOCKTYPE_UNCOMPRESSED: u32 = 0;
/// Static Huffman block. `DEFLATE_BLOCKTYPE_STATIC_HUFFMAN`.
pub const DEFLATE_BLOCKTYPE_STATIC_HUFFMAN: u32 = 1;
/// Dynamic Huffman block. `DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN`.
pub const DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN: u32 = 2;

// ---------------------------------------------------------------------------
// Match length limits
// ---------------------------------------------------------------------------

/// Minimum match length in bytes. `DEFLATE_MIN_MATCH_LEN`.
pub const DEFLATE_MIN_MATCH_LEN: u32 = 3;
/// Maximum match length in bytes. `DEFLATE_MAX_MATCH_LEN`.
pub const DEFLATE_MAX_MATCH_LEN: u32 = 258;

// ---------------------------------------------------------------------------
// Match offset / window
// ---------------------------------------------------------------------------

/// Maximum match offset (sliding window size) in bytes. `DEFLATE_MAX_MATCH_OFFSET`.
pub const DEFLATE_MAX_MATCH_OFFSET: u32 = 32768;
/// log2 of `DEFLATE_MAX_MATCH_OFFSET`. `DEFLATE_WINDOW_ORDER`.
pub const DEFLATE_WINDOW_ORDER: u32 = 15;

// ---------------------------------------------------------------------------
// Huffman symbol counts
// ---------------------------------------------------------------------------

/// Number of precode (code-length) symbols. `DEFLATE_NUM_PRECODE_SYMS`.
pub const DEFLATE_NUM_PRECODE_SYMS: usize = 19;
/// Maximum number of literal/length symbols. `DEFLATE_NUM_LITLEN_SYMS`.
pub const DEFLATE_NUM_LITLEN_SYMS: usize = 288;
/// Maximum number of offset symbols. `DEFLATE_NUM_OFFSET_SYMS`.
pub const DEFLATE_NUM_OFFSET_SYMS: usize = 32;
/// Maximum number of symbols across all codes. `DEFLATE_MAX_NUM_SYMS`.
pub const DEFLATE_MAX_NUM_SYMS: usize = 288;

// ---------------------------------------------------------------------------
// Literal/length symbol divisions
// ---------------------------------------------------------------------------

/// Number of literal symbols (byte values 0–255). `DEFLATE_NUM_LITERALS`.
pub const DEFLATE_NUM_LITERALS: u32 = 256;
/// End-of-block symbol. `DEFLATE_END_OF_BLOCK`.
pub const DEFLATE_END_OF_BLOCK: u32 = 256;
/// First length symbol in the litlen alphabet. `DEFLATE_FIRST_LEN_SYM`.
pub const DEFLATE_FIRST_LEN_SYM: u32 = 257;

// ---------------------------------------------------------------------------
// Maximum codeword lengths
// ---------------------------------------------------------------------------

/// Max codeword length for the precode. `DEFLATE_MAX_PRE_CODEWORD_LEN`.
pub const DEFLATE_MAX_PRE_CODEWORD_LEN: u32 = 7;
/// Max codeword length for the literal/length code. `DEFLATE_MAX_LITLEN_CODEWORD_LEN`.
pub const DEFLATE_MAX_LITLEN_CODEWORD_LEN: u32 = 15;
/// Max codeword length for the offset code. `DEFLATE_MAX_OFFSET_CODEWORD_LEN`.
pub const DEFLATE_MAX_OFFSET_CODEWORD_LEN: u32 = 15;
/// Max codeword length across all codes. `DEFLATE_MAX_CODEWORD_LEN`.
pub const DEFLATE_MAX_CODEWORD_LEN: u32 = 15;

// ---------------------------------------------------------------------------
// Miscellaneous
// ---------------------------------------------------------------------------

/// Maximum possible overrun when decoding codeword lengths. `DEFLATE_MAX_LENS_OVERRUN`.
pub const DEFLATE_MAX_LENS_OVERRUN: u32 = 137;
/// Max extra bits for a match length. `DEFLATE_MAX_EXTRA_LENGTH_BITS`.
pub const DEFLATE_MAX_EXTRA_LENGTH_BITS: u32 = 5;
/// Max extra bits for a match offset. `DEFLATE_MAX_EXTRA_OFFSET_BITS`.
pub const DEFLATE_MAX_EXTRA_OFFSET_BITS: u32 = 13;

// ---------------------------------------------------------------------------
// Tests — trivially verify a representative cross-section of values
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_types() {
        assert_eq!(DEFLATE_BLOCKTYPE_UNCOMPRESSED, 0);
        assert_eq!(DEFLATE_BLOCKTYPE_STATIC_HUFFMAN, 1);
        assert_eq!(DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN, 2);
    }

    #[test]
    fn match_limits() {
        assert_eq!(DEFLATE_MIN_MATCH_LEN, 3);
        assert_eq!(DEFLATE_MAX_MATCH_LEN, 258);
        assert_eq!(DEFLATE_MAX_MATCH_OFFSET, 32768);
        // Window order: 2^15 = 32768
        assert_eq!(1u32 << DEFLATE_WINDOW_ORDER, DEFLATE_MAX_MATCH_OFFSET);
    }

    #[test]
    fn symbol_counts() {
        assert_eq!(DEFLATE_NUM_PRECODE_SYMS, 19);
        assert_eq!(DEFLATE_NUM_LITLEN_SYMS, 288);
        assert_eq!(DEFLATE_NUM_OFFSET_SYMS, 32);
        assert_eq!(DEFLATE_MAX_NUM_SYMS, 288);
    }

    #[test]
    fn litlen_divisions() {
        assert_eq!(DEFLATE_NUM_LITERALS, 256);
        assert_eq!(DEFLATE_END_OF_BLOCK, 256);
        assert_eq!(DEFLATE_FIRST_LEN_SYM, 257);
    }

    #[test]
    fn codeword_lengths() {
        assert_eq!(DEFLATE_MAX_PRE_CODEWORD_LEN, 7);
        assert_eq!(DEFLATE_MAX_LITLEN_CODEWORD_LEN, 15);
        assert_eq!(DEFLATE_MAX_OFFSET_CODEWORD_LEN, 15);
        assert_eq!(DEFLATE_MAX_CODEWORD_LEN, 15);
    }
}
