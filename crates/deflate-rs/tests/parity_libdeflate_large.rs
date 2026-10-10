//! Byte-exact parity vs C libdeflate on realistic EXR-scale
//! inputs (100 KiB - 8 MiB), not just the 6 small (max 70000 B) fixed
//! inputs `tests/parity_libdeflate.rs` uses.
//!
//! `tests/parity_libdeflate.rs` compares against a COMMITTED golden file
//! (`tests/parity/golden.txt`); baking megabyte-scale compressed output into
//! that file as hex would roughly double its size on disk and in git history
//! per input added. Instead this test spawns the oracle binary LIVE (same
//! `oracle.exe` `cargo xtask parity-deflate` already builds) in a small CLI
//! mode added alongside its golden generator:
//!
//!   `oracle.exe zc <level>` — stdin = raw input bytes, stdout (binary) =
//!   `libdeflate_zlib_compress` output.
//!
//! Ignored by default (spawns a subprocess per case); run with
//! `cargo test -p deflate-rs --release --test parity_libdeflate_large -- --ignored`.
//! Requires `cargo xtask parity-deflate` to have been run at least once
//! (builds `tests/parity/oracle.exe`).

mod support;

use deflate_rs::{Compressor, Decompressor};

/// EXR-scale synthetic inputs at realistic sizes (100 KiB - 8 MiB): a
/// 16k-wide-row-like repeating pattern (mimics a packed scanline channel),
/// pure PRNG (worst case for compression, exercises every litlen/offset
/// subtable heavily), and a large single run (repeated max-length matches
/// back-to-back, at MB scale rather than the existing 70000 B case).
fn large_inputs() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("prng_128k", support::lcg_bytes(0xC0FF_EE01, 131_072)),
        (
            "row16k_pattern_1m",
            // A 16384-byte "row" (mimics one packed scanline of a 16k-wide
            // image) repeated 64x = 1 MiB, with a slowly-drifting byte so
            // rows aren't bit-identical (closer to real image data than a
            // pure repeat).
            (0..64usize)
                .flat_map(|row| (0..16384usize).map(move |i| ((i + row * 7) % 256) as u8))
                .collect(),
        ),
        ("prng_4m", support::lcg_bytes(0x1357_9BDF, 4 * 1024 * 1024)),
        ("run_8m", vec![0x3Cu8; 8 * 1024 * 1024]),
    ]
}

/// Compress is byte-identical to C at every level, for every large input; the
/// decode round-trip (canonical) is checked with our OWN decompressor, same
/// pattern as `parity_libdeflate.rs`'s golden-based test.
#[test]
#[ignore] // spawns a subprocess per (level, input) case; run explicitly
fn large_inputs_byte_identical_to_libdeflate() {
    let Some(oracle) = support::oracle_or_skip("large_inputs_byte_identical_to_libdeflate") else {
        return;
    };

    let inputs = large_inputs();
    let mut total = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for level in [1i32, 4, 6, 9, 12] {
        for (name, data) in &inputs {
            total += 1;
            let want = support::oracle_zlib_compress(&oracle, level, data);

            let mut c = Compressor::new(level).expect("valid level");
            let got = c.zlib_compress(data);

            if got != want {
                let first_diff = got
                    .iter()
                    .zip(want.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(want.len()));
                mismatches.push(format!(
                    "  level={level} input={name} ({} B): lens ours={} lib={}, first diff @ offset {first_diff}",
                    data.len(),
                    got.len(),
                    want.len()
                ));
                continue; // still worth checking the other cases
            }

            // Decode round-trip of the (now proven byte-identical) compressed
            // stream with our own decompressor -- canonical correctness check.
            let mut d = Decompressor::new();
            let recovered = d.zlib_decompress(&want, data.len()).unwrap_or_else(|e| {
                panic!("level={level} input={name}: our zlib_decompress REJECTED a valid stream: {e:?}")
            });
            assert_eq!(&recovered, data, "level={level} input={name}: round-trip mismatch");
        }
    }

    eprintln!(
        "large_inputs_byte_identical_to_libdeflate: {}/{} cases byte-identical",
        total - mismatches.len(),
        total
    );
    if !mismatches.is_empty() {
        panic!(
            "deflate-rs zlib_compress diverged from libdeflate on {}/{} large-input cases:\n{}",
            mismatches.len(),
            total,
            mismatches.join("\n")
        );
    }
}
