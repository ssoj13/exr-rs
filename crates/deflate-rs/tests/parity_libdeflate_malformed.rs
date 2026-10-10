//! Malformed-input differential vs the live C libdeflate oracle.
//!
//! `tests/parity_libdeflate.rs` only ever feeds VALID compressed streams to
//! both sides. This test feeds corrupted, truncated and bit-flipped zlib
//! streams to the real C libdeflate (via the oracle's `zdx <out_cap>` CLI
//! mode, which calls `libdeflate_zlib_decompress_ex`) and asserts our
//! `zlib_decompress_ex` reports the SAME `enum libdeflate_result` code and
//! the same `actual_in`/`actual_out` on success.
//!
//! Ignored by default (spawns a subprocess per case); run with
//! `cargo test -p deflate-rs --release --test parity_libdeflate_malformed -- --ignored`.
//! Requires `cargo xtask parity-deflate` to have been run at least once.

mod support;

use deflate_rs::Compressor;
use deflate_rs::deflate_decompress::Decompressor;
use deflate_rs::zlib_decompress::{zlib_decompress_ex, ZlibDecompressError};

/// `enum libdeflate_result` (libdeflate.h:194-207) — pulled out here (rather
/// than reused from `parity_libdeflate.rs`, a separate test binary with its
/// own crate root) since integration tests can't share items across crate
/// roots without a `tests/support` module (which now exists for the
/// oracle-plumbing helpers below); three named consts are cheap enough to
/// duplicate rather than route through `support` for.
const LIBDEFLATE_SUCCESS: i64 = 0;
const LIBDEFLATE_BAD_DATA: i64 = 1;
const LIBDEFLATE_INSUFFICIENT_SPACE: i64 = 3;

/// Our `zlib_decompress_ex` mapped onto the same `enum libdeflate_result`
/// numbering the oracle emits, for a direct comparison. `zlib_decompress_ex`
/// is the "counted" form (like the oracle's `_ex` call with a non-NULL
/// `actual_out_nbytes_ret`): it never returns `ShortOutput`, matching C's
/// documented behaviour for that call shape.
fn ours_zlib_decompress_ex(out_cap: usize, compressed: &[u8]) -> (i64, i64, i64) {
    let mut d = Decompressor::new();
    let mut out = vec![0u8; out_cap];
    match zlib_decompress_ex(&mut d, compressed, &mut out) {
        Ok((actual_out, actual_in)) => (LIBDEFLATE_SUCCESS, actual_in as i64, actual_out as i64),
        Err(ZlibDecompressError::BadHeader) => (LIBDEFLATE_BAD_DATA, -1, -1),
        Err(ZlibDecompressError::AdlerMismatch) => (LIBDEFLATE_BAD_DATA, -1, -1),
        Err(ZlibDecompressError::BadData) => (LIBDEFLATE_BAD_DATA, -1, -1),
        Err(ZlibDecompressError::InsufficientSpace) => (LIBDEFLATE_INSUFFICIENT_SPACE, -1, -1),
        Err(ZlibDecompressError::ShortOutput) => {
            panic!("zlib_decompress_ex (counted form) must never return ShortOutput")
        }
    }
}

fn assert_matches_oracle(oracle: &std::path::Path, label: &str, out_cap: usize, compressed: &[u8]) {
    let want = support::oracle_zlib_decompress_ex(oracle, out_cap, compressed);
    let got = ours_zlib_decompress_ex(out_cap, compressed);
    assert_eq!(
        got, want,
        "{label}: (result, actual_in, actual_out) mismatch vs C libdeflate \
         (ours={got:?} lib={want:?}, out_cap={out_cap}, compressed={} bytes)",
        compressed.len()
    );
}

#[test]
#[ignore] // spawns a subprocess per case; run explicitly
fn malformed_input_matches_libdeflate() {
    let Some(oracle) = support::oracle_or_skip("malformed_input_matches_libdeflate") else {
        return;
    };

    // A few valid streams to corrupt, at different levels (different block
    // structures / bit-alignments) and sizes. Kept small: this test spawns
    // one subprocess PER CASE (no in-process oracle), so input size and the
    // sampling step below directly bound the wall-clock cost -- unlike the
    // in-process fastloop fuzz in `deflate_decompress.rs`, which can afford exhaustive per-byte
    // coverage for free.
    let base_inputs: Vec<(&str, Vec<u8>)> = vec![
        ("short", b"hello, deflate-rs vs libdeflate! this needs to compress reasonably.".to_vec()),
        ("repetitive", b"abcabcabcabc".repeat(500)),
        ("prng_small", support::lcg_bytes(0xABCD_1234, 512)),
    ];

    let mut cases = 0usize;

    for level in [1i32, 6] {
        for (name, data) in &base_inputs {
            let mut c = Compressor::new(level).expect("valid level");
            let compressed = c.zlib_compress(data);
            let out_cap = data.len().max(1);

            // 1. The valid stream itself: SUCCESS on both sides.
            assert_matches_oracle(&oracle, &format!("{name}/L{level}/valid"), out_cap, &compressed);
            cases += 1;

            // 2. Truncated at ~20 sampled byte boundaries spanning the whole
            //    stream (header-only, mid-block, one-byte-short-of-footer),
            //    not every byte -- each case here is a real subprocess spawn.
            let step = (compressed.len() / 20).max(1);
            for cut in (0..compressed.len()).step_by(step) {
                assert_matches_oracle(
                    &oracle,
                    &format!("{name}/L{level}/trunc{cut}"),
                    out_cap,
                    &compressed[..cut],
                );
                cases += 1;
            }

            // 3. Single-bit flips at several offsets across the stream
            //    (header, body, trailer) -- BAD_DATA (checksum or stream
            //    corruption) on both sides, or occasionally a still-valid
            //    stream if the flipped bit didn't change anything semantic
            //    (e.g. an unused reserved bit) -- whichever it is, both
            //    implementations must agree.
            for &byte_idx in &[0usize, 1, compressed.len() / 2, compressed.len().saturating_sub(1)] {
                if byte_idx >= compressed.len() {
                    continue;
                }
                for bit in 0..8u8 {
                    let mut corrupted = compressed.clone();
                    corrupted[byte_idx] ^= 1 << bit;
                    assert_matches_oracle(
                        &oracle,
                        &format!("{name}/L{level}/bitflip@{byte_idx}.{bit}"),
                        out_cap,
                        &corrupted,
                    );
                    cases += 1;
                }
            }

            // 4. Output buffer too small (INSUFFICIENT_SPACE) -- confirms
            //    the FDICT/header/CRC-adjacent cases don't shadow a plain
            //    undersized-buffer report.
            if out_cap > 1 {
                assert_matches_oracle(
                    &oracle,
                    &format!("{name}/L{level}/tight"),
                    out_cap - 1,
                    &compressed,
                );
                cases += 1;
            }
        }
    }

    // 5. Header-specific malformations, each isolating ONE zlib-header check
    // (`zlib_decompress_ex` validates FCHECK, then CM, then CINFO, then
    // FDICT in that order -- see `zlib_decompress.rs`). An audit of an
    // earlier version of this test found both header cases here had a bad
    // FCHECK, so they only ever exercised the FCHECK branch and never
    // actually reached the CM/FDICT checks they claimed to test. Every
    // header below is verified by hand to have `(CMF*256+FLG) % 31 == 0`
    // (a VALID FCHECK) so the intended later check is the one that fires:
    //   - `cm_not_8`: CMF=0x09 (CINFO=0, CM=9 != 8), FLG=0x15.
    //     hdr=0x0915=2325=31*75 (FCHECK OK); CM=9 triggers the CM check.
    //   - `fdict_set`: CMF=0x78 (CINFO=7, CM=8, both valid), FLG=0x20
    //     (FDICT bit set, FCHECK bits 0, FLEVEL bits 0).
    //     hdr=0x7820=30752=31*992 (FCHECK OK); CM/CINFO both valid, so
    //     FDICT is the one check that fires.
    //   - `fcheck_fail`: CMF=0x78, FLG=0x00. hdr=0x7800=30720, NOT a
    //     multiple of 31 -- a genuine FCHECK failure, for contrast with the
    //     two cases above (which deliberately have a VALID FCHECK).
    let header_cases: &[(&str, [u8; 2])] = &[
        ("cm_not_8", [0x09, 0x15]),
        ("fdict_set", [0x78, 0x20]),
        ("fcheck_fail", [0x78, 0x00]),
    ];
    for (name, hdr) in header_cases {
        debug_assert_eq!(
            (u16::from_be_bytes(*hdr)) % 31 == 0,
            *name != "fcheck_fail",
            "header_cases[{name}]: FCHECK property doesn't match this case's intent -- fix the hand-picked bytes"
        );
        let mut stream = hdr.to_vec();
        stream.extend_from_slice(&[0, 0, 0, 0]); // dummy trailer bytes
        assert_matches_oracle(&oracle, &format!("header/{name}"), 16, &stream);
        cases += 1;
    }
    // An empty input and a 1-byte input (shorter than the minimum overhead).
    assert_matches_oracle(&oracle, "header/empty", 16, &[]);
    cases += 1;
    assert_matches_oracle(&oracle, "header/one_byte", 16, &[0x78]);
    cases += 1;

    eprintln!("malformed_input_matches_libdeflate: {cases} cases agree with C libdeflate");
}
