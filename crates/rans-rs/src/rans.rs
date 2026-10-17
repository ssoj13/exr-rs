//! `rans-rs` — integer interleaved rANS: the entropy foundation of EXV.
//!
//! # What and why
//! The single entropy layer of the format. Replaces serial Huffman/DEFLATE/MQ with
//! **integer** rANS using independent lanes: the result is bit-exact deterministic on any
//! CPU/GPU and parallelizes (N lanes ↔ N GPU threads, lockstep, no interaction).
//!
//! # Format invariants (see `docs/archive/PLAN.md` §3.1)
//! - 32-bit state · `M = 2^table_log` (power of two) · alphabet ≤ 256 · N independent lanes.
//! - [`RansConfig`] = the CPU↔WGSL contract; **locked after Spike A**.
//! - All operations are `u32` (no 64-bit) → ports to WGSL 1:1.
//!
//! # Modules
//! - [`model`] — frequency model (normalization, cum, slot→sym).
//! - [`byte_coder`] — canonical rANS byte-coder (`rans_byte.h`).
//! - [`codec`] — interleaved wrapper + self-contained container ([`encode`]/[`decode_capped`]).
//!
//! # Status
//! Spike A / Phase 1: CPU reference + tests. Next — golden vectors and WGSL decode (`exv-gpu`),
//! checked bit-exact against this reference.

mod byte_coder;
mod codec;
mod model;

pub use codec::{
    EncodeError, MAGIC, MAX_DECODE_SYMBOLS, Prepared, RansConfig, build_table, decode_block, decode_block_into,
    decode_capped, decode_capped_into, decode_into, encode, encode_block, encode_block_into, encode_into, parse, parse_ref, parse_span,
    prepare, with_model,
};
/// The CPU<->WGSL byte-coder contract constants (valid state range `[RANS_L, RANS_UPPER)`, bounded renorm
/// loop). Exported so GPU kernels generate their shader literals from THESE, never re-declare them.
pub use byte_coder::{RANS_L, RANS_UPPER, RENORM_GUARD};
/// The normalized frequency model — public for the EXV-RT shared-table block codec (build once over the
/// pooled tile symbols, encode/decode each block against it). Interleaved container users don't need it.
pub use model::Model;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-side decode with the library's absolute ceiling (production callers pass the expected length).
    /// Bytes of the container's frequency table (256 x u16).
    const ALPHABET_BYTES: usize = 512;

    fn decode(d: &[u8]) -> Option<Vec<u8>> {
        decode_capped(d, MAX_DECODE_SYMBOLS)
    }

    /// Deterministic xorshift64 for reproducible test data (no `rand`).
    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// Full round-trip: encode→decode must return the original bytes.
    fn round_trip(data: &[u8], cfg: RansConfig) {
        let enc = encode(data, cfg).expect("encode");
        let dec = decode(&enc).expect("decode must succeed on our own stream");
        assert_eq!(dec, data, "round-trip mismatch (cfg={cfg:?}, len={})", data.len());
    }

    #[test]
    fn rt_empty() {
        round_trip(&[], RansConfig::default());
    }

    #[test]
    fn rt_single_byte() {
        round_trip(&[42], RansConfig::default());
        round_trip(&[0], RansConfig::new(1, 8));
        round_trip(&[255], RansConfig::new(8, 15));
    }

    #[test]
    fn rt_all_same() {
        let data = vec![7u8; 5000];
        round_trip(&data, RansConfig::new(1, 12));
        round_trip(&data, RansConfig::new(32, 12));
    }

    #[test]
    fn rt_two_symbols_skewed() {
        // 90% zeros, 10% ones — exercises the fractional bits of rANS.
        let mut seed = 0x1234_5678_9abc_def0u64;
        let data: Vec<u8> = (0..20_000)
            .map(|_| if xorshift(&mut seed).is_multiple_of(10) { 1u8 } else { 0u8 })
            .collect();
        for lanes in [1u8, 4, 16, 32] {
            for tl in [8u8, 12, 15] {
                round_trip(&data, RansConfig::new(lanes, tl));
            }
        }
    }

    #[test]
    fn rt_full_alphabet_random() {
        let mut seed = 0xdead_beef_cafe_babeu64;
        let data: Vec<u8> = (0..30_000).map(|_| (xorshift(&mut seed) & 0xff) as u8).collect();
        for lanes in [1u8, 3, 16, 32, 64] {
            for tl in [8u8, 11, 14] {
                round_trip(&data, RansConfig::new(lanes, tl));
            }
        }
    }

    #[test]
    fn rt_lengths_not_multiple_of_lanes() {
        // edge cases: length not a multiple of the lane count (last lanes are shorter).
        for n in [1usize, 7, 31, 33, 100, 1023] {
            let data: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
            round_trip(&data, RansConfig::new(32, 12));
        }
    }

    /// Encode is deterministic: the same input → a byte-for-byte identical container.
    #[test]
    fn encode_is_deterministic() {
        let mut seed = 0x0f0f_0f0f_1111_2222u64;
        let data: Vec<u8> = (0..8000).map(|_| (xorshift(&mut seed) & 0xff) as u8).collect();
        let cfg = RansConfig::default();
        assert_eq!(encode(&data, cfg).unwrap(), encode(&data, cfg).unwrap(), "encode must be deterministic");
    }

    /// Compression actually works on skewed data (container smaller than the input).
    #[test]
    fn compresses_skewed_data() {
        let data = vec![0u8; 50_000]; // maximally compressible
        let enc = encode(&data, RansConfig::new(32, 12)).unwrap();
        assert!(enc.len() < data.len() / 4, "expected strong compression, got {} bytes", enc.len());
    }

    /// FNV-1a 64 for golden vectors (freeze the bitstream as a single number).
    fn fnv1a(d: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in d {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Deterministic test vector (fixed input).
    fn golden_input() -> Vec<u8> {
        (0..512u32).flat_map(|i| [((i * 73 + 11) % 256) as u8, ((i * 73 + 11) % 256) as u8 ^ 0x5a]).collect()
    }

    /// Golden vector: fixed input+config → frozen bitstream (FNV-1a).
    /// Any change to the rANS container layout breaks this test on purpose.
    #[test]
    fn golden_vector() {
        let input = golden_input();
        let cfg = RansConfig::new(8, 12);
        let enc = encode(&input, cfg).unwrap();
        let h = fnv1a(&enc);
        // Frozen on RansConfig{lanes:8,table_log:12} + golden_input(): any shift in the
        // container/rANS layout will break this test (bit-exact format stability).
        const GOLDEN: u64 = 0x5f9f_a5cf_e4f3_2f83;
        assert_eq!(enc.len(), 1610, "golden container length changed");
        assert_eq!(h, GOLDEN, "rANS bitstream layout changed (golden vector)");
        assert_eq!(decode(&enc).unwrap(), input, "golden round-trip");
    }

    /// Corrupt stream → None, without panicking.
    #[test]
    fn decode_garbage_is_none() {
        assert!(decode(&[]).is_none());
        assert!(decode(&[0, 1, 2, 3]).is_none()); // wrong MAGIC
        let mut ok = encode(&[1, 2, 3, 4, 5], RansConfig::default()).unwrap();
        ok.truncate(ok.len() - 1); // truncated tail: the last lane slice runs past the buffer
        assert!(decode(&ok).is_none());
    }

    /// A container whose lane length was rewritten to match a truncated / padded stream is structurally
    /// valid, but the lane does not end in state `RANS_L` at its last byte → hard `None`, never garbage.
    #[test]
    fn decode_rejects_forged_lane_ends() {
        let data: Vec<u8> = (0..400u32).map(|i| (i * 7 % 13) as u8).collect();
        let enc = encode(&data, RansConfig::new(1, 12)).unwrap();
        assert_eq!(decode(&enc).as_deref(), Some(&data[..]));
        const LEN_AT: usize = 4 + 1 + 1 + 4 + 512; // single lane_len entry
        let len = u32::from_le_bytes(enc[LEN_AT..LEN_AT + 4].try_into().unwrap()) as usize;
        // Truncated lane (length consistently reduced): the decoder over-reads past the end.
        let mut cut = enc.clone();
        cut.pop();
        cut[LEN_AT..LEN_AT + 4].copy_from_slice(&((len - 1) as u32).to_le_bytes());
        assert!(decode(&cut).is_none(), "truncated lane");
        // Padded lane: unread trailing bytes.
        let mut pad = enc.clone();
        pad.push(0);
        pad[LEN_AT..LEN_AT + 4].copy_from_slice(&((len + 1) as u32).to_le_bytes());
        assert!(decode(&pad).is_none(), "padded lane");
        // Bit flip inside the stream: the final state no longer returns to RANS_L.
        let mut flip = enc.clone();
        let mid = LEN_AT + 4 + len / 2;
        flip[mid] ^= 0x55;
        assert!(decode(&flip).is_none(), "forged lane byte");
    }

    /// Shared-table blocks apply the same exact-end rule (and still reject trailing bytes).
    #[test]
    fn block_rejects_truncated_and_padded() {
        let blk: Vec<u8> = (0..300u32).map(|i| (i * 5 % 11) as u8).collect();
        let table = build_table(&blk, 12);
        let sub = encode_block(&blk, &table, 1).unwrap();
        assert_eq!(decode_block(&sub, &table, blk.len(), 1).as_deref(), Some(&blk[..]));
        let mut short = sub.clone();
        short.pop();
        assert!(decode_block(&short, &table, blk.len(), 1).is_none());
        let mut long = sub;
        long.push(0);
        assert!(decode_block(&long, &table, blk.len(), 1).is_none());
    }

    /// Shared-table per-block round-trip (EXV-RT shape): ONE table over the pool, N headerless blocks,
    /// each decoded INDEPENDENTLY against a table rebuilt from its serialized 512-byte form, reassembled
    /// == original. Exercises lanes=1 (serial) and lanes=4 (GPU sub-lanes) + varying block sizes.
    #[test]
    fn shared_table_blocks_round_trip() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let blocks: Vec<Vec<u8>> = (0..6usize)
            .map(|bi| (0..64 + bi * 40).map(|_| (xorshift(&mut seed) & 0xff) as u8).collect())
            .collect();
        let pool: Vec<u8> = blocks.iter().flatten().copied().collect();
        for lanes in [1u8, 4] {
            let table = build_table(&pool, 12);
            // Decode side rebuilds the table from its serialized freqs (the real CPU/GPU path).
            let rebuilt = Model::from_freqs(&table.freqs_le(), 12).expect("from_freqs");
            for blk in &blocks {
                let sub = encode_block(blk, &table, lanes).unwrap();
                let dec = decode_block(&sub, &rebuilt, blk.len(), lanes).expect("decode_block");
                assert_eq!(&dec, blk, "shared-table block round-trip (lanes={lanes})");
            }
        }
    }

    /// Regression: a structurally valid forged container with state=0 → None, WITHOUT hanging
    /// (the renorm loop used to read zeros forever). If this test hangs, the regression is back.
    #[test]
    fn decode_rejects_zero_state_no_hang() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.push(12); // table_log (M=4096)
        buf.push(1); // lanes
        buf.extend_from_slice(&1u32.to_le_bytes()); // n=1
        buf.extend_from_slice(&4096u16.to_le_bytes()); // freq[0]=M
        buf.extend_from_slice(&[0u8; 510]); // freq[1..256]=0 (sum==M → from_freqs passes)
        buf.extend_from_slice(&4u32.to_le_bytes()); // lane_len[0]=4
        buf.extend_from_slice(&[0u8; 4]); // lane stream: state=0
        assert!(decode(&buf).is_none());
    }

    /// A symbol absent from the shared model is a typed error, not a divide-by-zero panic.
    #[test]
    fn encode_block_rejects_symbol_not_in_model() {
        let table = build_table(&[1, 2, 3], 12);
        assert_eq!(encode_block(&[1, 2, 9], &table, 1), Err(EncodeError::SymbolNotInModel(9)));
        assert_eq!(encode_block(&[1, 2, 3], &table, 2).map(|_| ()), Ok(()));
    }

    /// `table_log` outside 8..=15 is clamped (never a shift overflow / unreachable normalization panic).
    #[test]
    fn model_clamps_table_log() {
        let mut counts = [0u32; 256];
        for (i, c) in counts.iter_mut().enumerate() {
            *c = i as u32 + 1;
        }
        for tl in [0u32, 3, 7, 16, 40] {
            let m = Model::from_counts(&counts, tl);
            assert!((8..=15).contains(&m.table_log()), "tl={tl} -> {}", m.table_log());
            assert_eq!(m.freq().iter().map(|&f| f as u32).sum::<u32>(), 1 << m.table_log());
            assert!(m.freq().iter().all(|&f| f >= 1), "every present symbol keeps freq >= 1");
        }
    }

    /// A container declaring more than `max_n` symbols is refused before allocation; a forged larger `n`
    /// over a short stream is a hard `None` (the lane runs out of bytes and aborts at the first read past
    /// its end instead of finishing `n` zero-fed symbols).
    #[test]
    fn forged_n_is_bounded_by_cap_and_stream() {
        let data: Vec<u8> = (0..64u32).map(|i| (i % 7) as u8).collect();
        let mut enc = encode(&data, RansConfig::new(1, 12)).unwrap();
        assert!(decode_capped(&enc, data.len() - 1).is_none(), "n over the caller cap");
        enc[6..10].copy_from_slice(&(1u32 << 24).to_le_bytes());
        assert!(decode_capped(&enc, MAX_DECODE_SYMBOLS).is_none());
    }

    /// The per-thread decode model is rebuilt IN PLACE per blob: back-to-back decodes with different `table_log` /
    /// lane counts / symbol skews on one thread all round-trip, a corrupt table (rejected before any write) leaves it
    /// usable, and `decode_capped_into` REPLACES its buffer (capacity reused).
    #[test]
    fn decode_reuses_the_thread_model() {
        let mut seed = 7u64;
        let blobs: Vec<(Vec<u8>, Vec<u8>)> = [(12u8, 8u8, 5000usize, 5u64), (9, 3, 777, 200), (15, 1, 4096, 64), (8, 128, 33, 3), (10, 255, 3000, 40)]
            .iter()
            .map(|&(tl, lanes, n, m)| {
                let data: Vec<u8> = (0..n).map(|_| (xorshift(&mut seed) % m) as u8).collect();
                (encode(&data, RansConfig::new(lanes, tl)).unwrap(), data)
            })
            .collect();
        let mut buf = vec![0xAAu8; 10_000];
        for _ in 0..2 {
            for (enc, data) in &blobs {
                let mut out = vec![0u8; data.len()];
                assert_eq!(decode_into(enc, &mut out), Some(()));
                assert_eq!(&out, data);
                assert_eq!(decode_capped_into(enc, data.len(), &mut buf), Some(()));
                assert_eq!(&buf, data, "the scratch Vec is replaced, not appended to");
                let mut bad = enc.clone();
                bad[10] ^= 1; // freq[0]: the table no longer sums to M
                assert!(decode_capped(&bad, data.len()).is_none());
            }
        }
        let p = parse_ref(&blobs[1].0).expect("parse_ref after the other decodes");
        assert_eq!((p.cfg, p.n), (RansConfig::new(3, 9), blobs[1].1.len()));
        assert_eq!(p.slot_to_sym.len(), 1 << 9, "slot table sized by THIS blob's table_log, not a stale one");
    }

    /// The decode cursors live on the stack (`[Lane; 255]`): every lane count up to the `u8` maximum round-trips, for the
    /// container and for a shared-table block, and trailing garbage after the last block lane is still rejected.
    #[test]
    fn stack_cursors_cover_every_lane_count() {
        let mut seed = 11u64;
        let data: Vec<u8> = (0..4000).map(|_| (xorshift(&mut seed) % 30) as u8).collect();
        let table = build_table(&data, 12);
        for lanes in [1u8, 2, 31, 128, 254, 255] {
            let enc = encode(&data, RansConfig::new(lanes, 12)).expect("encode");
            assert_eq!(decode_capped(&enc, data.len()).as_deref(), Some(data.as_slice()), "container lanes={lanes}");
            let mut blk = encode_block(&data, &table, lanes).expect("encode_block");
            let mut out = vec![0u8; data.len()];
            assert_eq!(decode_block_into(&blk, &table, &mut out, lanes), Some(()), "block lanes={lanes}");
            assert_eq!(out, data);
            if lanes > 1 {
                blk.push(0);
                assert!(decode_block_into(&blk, &table, &mut out, lanes).is_none(), "trailing byte, lanes={lanes}");
            }
        }
    }

    /// `parse_span`: the span is exactly the concatenation of `lane_streams` (what the GPU host memcpys), the lanes are
    /// adjacent sub-slices of it, and it accepts/rejects the SAME inputs as `parse_ref` (forged lane length, truncation,
    /// bad state, bad magic).
    #[test]
    fn parse_span_is_the_lane_concat_and_matches_parse_ref() {
        let mut seed = 7u64;
        let data: Vec<u8> = (0..5000).map(|_| (xorshift(&mut seed) % 40) as u8).collect();
        let enc = encode(&data, RansConfig::new(5, 10)).expect("encode");
        let (p, span) = parse_span(&enc).expect("parse_span");
        let r = parse_ref(&enc).expect("parse_ref");
        assert_eq!((p.cfg, p.n), (r.cfg, r.n));
        let cat: Vec<u8> = p.lane_streams.iter().flat_map(|l| l.iter().copied()).collect();
        assert_eq!(span, &cat[..]);
        let mut at = span.as_ptr() as usize;
        for l in &p.lane_streams {
            assert_eq!(l.as_ptr() as usize, at, "lanes are adjacent sub-slices of the span");
            at += l.len();
        }
        // Forged / truncated variants: identical verdicts.
        let mut forged = enc.clone();
        let len_off = 4 + 1 + 1 + 4 + ALPHABET_BYTES; // first lane length word
        forged[len_off..len_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut bad_state = enc.clone();
        let first_lane = len_off + 5 * 4;
        bad_state[first_lane..first_lane + 4].copy_from_slice(&0u32.to_le_bytes());
        let mut bad_magic = enc.clone();
        bad_magic[0] ^= 0xFF;
        for v in [forged, bad_state, bad_magic, enc[..enc.len() - 3].to_vec(), enc[..20].to_vec()] {
            assert_eq!(parse_span(&v).is_some(), parse_ref(&v).is_some());
        }
        assert!(parse_span(&{
            let mut f = enc.clone();
            f[len_off..len_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            f
        })
        .is_none());
    }
}
