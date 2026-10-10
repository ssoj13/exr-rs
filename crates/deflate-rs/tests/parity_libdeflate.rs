//! C libdeflate byte-exact golden-parity test for `deflate-rs`.
//!
//! Reads the committed `tests/parity/golden.txt` (produced by the C libdeflate
//! oracle via `cargo run -p xtask -- parity-deflate`), reproduces the SAME
//! deterministic inputs, runs `deflate-rs`'s `zlib_compress` at every level
//! 1..=12, and asserts the compressed bytes are **BYTE-IDENTICAL** to what the
//! real C libdeflate produced.
//!
//! This is the gate that proves EXR ZIP/ZIPS output will match C++ OpenEXR:
//! OpenEXR's ZIP codec calls `libdeflate_zlib_compress`, so byte-identity here
//! means byte-identical EXR pixel chunks.
//!
//! ## Two assertions per case
//! 1. **Compress is byte-identical**: `deflate_rs::Compressor::zlib_compress`
//!    output == libdeflate's output, byte-for-byte. DEFLATE streams are NOT
//!    canonical, so this only holds because the port matches libdeflate's exact
//!    heuristics; any divergence is a real port bug (do NOT loosen this check).
//! 2. **Decode round-trips**: our `zlib_decompress` of *libdeflate's* output
//!    recovers the original input. Decode IS canonical (any valid DEFLATE stream
//!    must decode correctly), so this is a correctness check independent of (1).
//!
//! ## Input reproduction is self-verified
//! The oracle also emits each input's exact bytes as `input.<name> = <hex>`.
//! Before comparing compressed output, we assert our locally-reproduced input
//! matches the oracle's `input.<name>` — so any input-drift fails loudly here
//! rather than masquerading as a compression mismatch.
//!
//! If `golden.txt` is missing (a machine without the C build), the test prints a
//! skip note and passes, so `cargo test` works standalone everywhere; running
//! `cargo run -p xtask -- parity-deflate` regenerates it on Windows + vcpkg.

use std::collections::BTreeMap;
use std::path::PathBuf;

use deflate_rs::{crc32, Compressor, Decompressor, ZlibDecompressError};
use deflate_rs::gzip_decompress::gzip_decompress;

/// Names of the fixed inputs, in oracle order. Must match `oracle.c`.
const INPUT_NAMES: &[&str] = &[
    "empty",
    "short_str",
    "repetitive",
    "all_same",
    "pseudo_random",
    "structured",
];

// ===========================================================================
// Input generators — byte-for-byte identical to oracle.c's definitions.
// ===========================================================================

/// xorshift32 PRNG, seed `0x1234_5678`. Emits the low byte of each new state.
/// Mirrors `fill_pseudo_random` in `oracle.c`.
fn fill_pseudo_random(n: usize) -> Vec<u8> {
    let mut x: u32 = 0x1234_5678;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        out.push((x & 0xFF) as u8);
    }
    out
}

/// 1000 formatted text lines. Mirrors `fill_structured` in `oracle.c`.
fn fill_structured() -> Vec<u8> {
    let mut s = String::new();
    for i in 0..1000 {
        s.push_str(&format!(
            "row {i}: the quick brown fox jumps over the lazy dog ({i})\n"
        ));
    }
    s.into_bytes()
}

/// Build the named input exactly as `oracle.c` does.
fn make_input(name: &str) -> Vec<u8> {
    match name {
        "empty" => Vec::new(),
        // NOTE: the trailing "\n" is part of the string; no NUL is included.
        "short_str" => b"hello, deflate-rs vs libdeflate!\n".to_vec(),
        // 4096 bytes of the 3-byte cycle 'a','b','c': buf[i] = b'a' + (i % 3).
        "repetitive" => (0..4096usize).map(|i| b'a' + (i % 3) as u8).collect(),
        // 8192 bytes all 0x5A.
        "all_same" => vec![0x5Au8; 8192],
        // 65536 bytes from the documented xorshift32 PRNG.
        "pseudo_random" => fill_pseudo_random(65536),
        // ~64 KiB of formatted text lines.
        "structured" => fill_structured(),
        other => panic!("unknown input name `{other}` (test/oracle out of sync)"),
    }
}

// ===========================================================================
// Golden-file ingestion
// ===========================================================================

/// Parse `key = hex` lines (ignoring `#` comments / blanks) into a map.
fn parse_golden(text: &str) -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, hex)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_string();
        let hex = hex.trim();
        let bytes = hex_decode(hex)
            .unwrap_or_else(|| panic!("golden.txt: bad hex for key `{key}`"));
        map.insert(key, bytes);
    }
    map
}

/// Decode a lowercase/uppercase hex string into bytes; `None` on odd length or
/// non-hex digit.
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let nib = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut i = 0;
    while i < bytes.len() {
        out.push((nib(bytes[i])? << 4) | nib(bytes[i + 1])?);
        i += 2;
    }
    Some(out)
}

/// Lowercase hex of `data` (for diagnostics).
fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for &b in data {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// `enum libdeflate_result` values (libdeflate.h:193-207), as the oracle emits them.
const LIBDEFLATE_SUCCESS: u8 = 0;
const LIBDEFLATE_SHORT_OUTPUT: u8 = 2;
const LIBDEFLATE_INSUFFICIENT_SPACE: u8 = 3;

/// Read a one-byte `enum libdeflate_result` value from the golden map.
fn result_code(golden: &BTreeMap<String, Vec<u8>>, key: &str) -> u8 {
    let v = golden
        .get(key)
        .unwrap_or_else(|| panic!("golden.txt missing `{key}` -- regenerate the golden"));
    assert_eq!(v.len(), 1, "`{key}`: expected one byte from %02x");
    v[0]
}

/// Map `Decompressor::zlib_decompress_exact` onto the C result codes so the two
/// APIs can be compared directly.
fn zlib_exact_code(d: &mut Decompressor, compressed: &[u8], expected: usize) -> u8 {
    match d.zlib_decompress_exact(compressed, expected) {
        Ok(_) => LIBDEFLATE_SUCCESS,
        Err(ZlibDecompressError::ShortOutput) => LIBDEFLATE_SHORT_OUTPUT,
        Err(ZlibDecompressError::InsufficientSpace) => LIBDEFLATE_INSUFFICIENT_SPACE,
        Err(other) => panic!("unexpected zlib error {other}"),
    }
}

/// Path to the committed golden file.
fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("golden.txt")
}

// ===========================================================================
// The parity test
// ===========================================================================

#[test]
fn parity_libdeflate_byte_exact() {
    let path = golden_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            // A parity gate that reports success because it never ran is worse
            // than no gate at all. Opting out has to be explicit.
            if std::env::var_os("EXR_ALLOW_SKIP").is_some_and(|v| v == "1") {
                eprintln!("SKIP (EXR_ALLOW_SKIP=1): {} not found", path.display());
                return;
            }
            panic!(
                "{} could not be read ({e}) -- the C libdeflate parity gate did NOT run. \
                 Run `cargo run -p xtask -- parity-deflate` (Windows + vcpkg) to generate \
                 it, or set EXR_ALLOW_SKIP=1 to skip it deliberately.",
                path.display()
            );
        }
    };

    let golden = parse_golden(&text);
    assert!(!golden.is_empty(), "golden.txt parsed to zero records");

    // --- Step 1: reproduce each input and verify it matches the oracle's bytes. ---
    let mut inputs: Vec<(&str, Vec<u8>)> = Vec::with_capacity(INPUT_NAMES.len());
    for &name in INPUT_NAMES {
        let ours = make_input(name);
        let key = format!("input.{name}");
        let expected = golden
            .get(&key)
            .unwrap_or_else(|| panic!("golden.txt missing `{key}` (oracle/test out of sync)"));
        assert_eq!(
            &ours, expected,
            "INPUT REPRODUCTION MISMATCH for `{name}`: our {} bytes != oracle {} bytes \
             (the Rust generator drifted from oracle.c — fix the generator)",
            ours.len(),
            expected.len()
        );
        inputs.push((name, ours));
    }

    // --- Step 2: per level x input, compress with deflate-rs and compare. ---
    let mut total_cases = 0usize;
    let mut byte_identical = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    // Per-level identity tally for a compact summary table.
    let mut per_level: BTreeMap<i32, (usize, usize)> = BTreeMap::new();

    for level in 1..=12_i32 {
        for (name, data) in &inputs {
            total_cases += 1;
            let key = format!("level{level}.{name}");
            let want = golden
                .get(&key)
                .unwrap_or_else(|| panic!("golden.txt missing `{key}` (oracle/test out of sync)"));

            let mut c = Compressor::new(level).expect("valid level");
            let got = c.zlib_compress(data);

            let entry = per_level.entry(level).or_insert((0, 0));
            if got == *want {
                byte_identical += 1;
                entry.0 += 1;
            } else {
                entry.1 += 1;
                // Find the first differing offset for a precise report.
                let first_diff = got
                    .iter()
                    .zip(want.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(want.len()));
                let our_byte = got
                    .get(first_diff)
                    .map(|b| format!("{b:02x}"))
                    .unwrap_or_else(|| "<eof>".to_string());
                let lib_byte = want
                    .get(first_diff)
                    .map(|b| format!("{b:02x}"))
                    .unwrap_or_else(|| "<eof>".to_string());
                mismatches.push(format!(
                    "  level={level} input={name}: lens ours={} lib={}, \
                     first diff @ offset {first_diff} (ours={our_byte} lib={lib_byte})",
                    got.len(),
                    want.len()
                ));
            }

            // --- Step 3: decode round-trip of libdeflate's output (canonical). ---
            let mut d = Decompressor::new();
            let recovered = d
                .zlib_decompress(want, data.len())
                .unwrap_or_else(|e| {
                    panic!(
                        "level={level} input={name}: our zlib_decompress REJECTED \
                         libdeflate's valid stream: {e:?}"
                    )
                });
            assert_eq!(
                &recovered, data,
                "level={level} input={name}: round-trip mismatch decoding libdeflate output"
            );
        }
    }

    // --- SHORT_OUTPUT / INSUFFICIENT_SPACE contract (libdeflate.h:193-207). ---
    // The oracle records the real `enum libdeflate_result` C returns for an
    // exact-size, an oversized and an undersized output buffer when
    // `actual_out_nbytes_ret` is NULL. deflate-rs expresses the same three
    // outcomes through `Decompressor::zlib_decompress_exact`.
    {
        let mut checked = 0usize;
        for (name, data) in &inputs {
            let mut c = Compressor::new(6).expect("valid level");
            let compressed = c.zlib_compress(data);
            let mut d = Decompressor::new();

            let key = format!("zlibresult.exact.{name}");
            let want = result_code(&golden, &key);
            let got = zlib_exact_code(&mut d, &compressed, data.len());
            assert_eq!(got, want, "`{key}`: exact-size buffer");
            assert_eq!(want, LIBDEFLATE_SUCCESS, "`{key}`: C must succeed here");
            checked += 1;

            let key = format!("zlibresult.short.{name}");
            let want = result_code(&golden, &key);
            let got = zlib_exact_code(&mut d, &compressed, data.len() + 16);
            assert_eq!(got, want, "`{key}`: oversized buffer must be SHORT_OUTPUT");
            assert_eq!(want, LIBDEFLATE_SHORT_OUTPUT, "`{key}`: C must report SHORT_OUTPUT");
            checked += 1;

            if !data.is_empty() {
                let key = format!("zlibresult.tight.{name}");
                let want = result_code(&golden, &key);
                let got = zlib_exact_code(&mut d, &compressed, data.len() - 1);
                assert_eq!(got, want, "`{key}`: undersized buffer");
                assert_eq!(
                    want, LIBDEFLATE_INSUFFICIENT_SPACE,
                    "`{key}`: C must report INSUFFICIENT_SPACE"
                );
                checked += 1;
            }

            // The counted form keeps the old, C-equivalent behaviour: a short
            // stream is success plus a short buffer.
            let counted = d
                .zlib_decompress(&compressed, data.len() + 16)
                .expect("counted form succeeds on an oversized buffer");
            assert_eq!(&counted, data, "counted zlib_decompress lost data for `{name}`");
        }
        eprintln!("SHORT_OUTPUT: {checked} result codes match C libdeflate  [PASS]");
    }

    // --- CRC-32 parity: our crc32 == libdeflate_crc32 for every input. ---
    // Keys in golden.txt: `crc32.<name> = <8 hex digits>`
    // MANDATORY: a golden regenerated without these keys must fail, not quietly
    // drop the coverage.
    let mut crc32_checked = 0usize;
    for (name, data) in &inputs {
        let key = format!("crc32.{name}");
        {
            let want_bytes = golden.get(&key).unwrap_or_else(|| {
                panic!("golden.txt missing `{key}` -- regenerate with `xtask parity-deflate`")
            });
            // The oracle emits `%08x` which is 8 hex chars decoded to 4 raw bytes
            // by parse_golden.  Interpret as big-endian u32 (most-significant digit first).
            assert_eq!(want_bytes.len(), 4, "crc32.{name}: expected 4 bytes from %08x");
            let want_u32 = u32::from_be_bytes(want_bytes[..4].try_into().unwrap());
            let got_u32 = crc32(0, data);
            assert_eq!(
                got_u32, want_u32,
                "CRC-32 MISMATCH for input `{name}`: ours=0x{got_u32:08x} lib=0x{want_u32:08x}"
            );
            crc32_checked += 1;
        }
    }
    assert_eq!(crc32_checked, inputs.len(), "every input must have a crc32 golden");
    eprintln!("CRC-32: {crc32_checked}/{} inputs byte-exact vs libdeflate  [PASS]", inputs.len());

    // --- Gzip parity: our gzip_compress byte-identical, our gzip_decompress round-trips. ---
    let mut gzip_total = 0usize;
    let mut gzip_identical = 0usize;
    let mut gzip_mismatches: Vec<String> = Vec::new();
    let mut gzip_per_level: BTreeMap<i32, (usize, usize)> = BTreeMap::new();

    for level in 1..=12_i32 {
        for (name, data) in &inputs {
            let key = format!("gzip{level}.{name}");
            // MANDATORY: gzip goldens are committed, so a missing key means the
            // golden drifted -- fail instead of silently skipping the level.
            let want = golden.get(&key).unwrap_or_else(|| {
                panic!("golden.txt missing `{key}` -- regenerate with `xtask parity-deflate`")
            });
            gzip_total += 1;

            // Compress with our gzip_compress via the facade.
            let mut c = Compressor::new(level).expect("valid level");
            let out = c.gzip_compress(data);

            let entry = gzip_per_level.entry(level).or_insert((0, 0));
            if out == *want {
                gzip_identical += 1;
                entry.0 += 1;
            } else {
                entry.1 += 1;
                let first_diff = out.iter().zip(want.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(out.len().min(want.len()));
                gzip_mismatches.push(format!(
                    "  level={level} input={name}: lens ours={} lib={}, \
                     first diff @ offset {first_diff}",
                    out.len(), want.len()
                ));
            }

            // Decode round-trip of libdeflate's gzip output with our decompressor.
            let mut recovered = vec![0u8; data.len()];
            gzip_decompress(want, &mut recovered)
                .unwrap_or_else(|e| panic!(
                    "level={level} input={name}: our gzip_decompress REJECTED \
                     libdeflate's gzip stream: {e:?}"
                ));
            assert_eq!(
                &recovered, data,
                "level={level} input={name}: gzip round-trip mismatch"
            );
        }
    }

    if gzip_total > 0 {
        eprintln!("\n=== deflate-rs vs C libdeflate gzip byte-identity (per level) ===");
        for (level, (pass, fail)) in &gzip_per_level {
            let status = if *fail == 0 { "PASS" } else { "FAIL" };
            eprintln!("  level{level:>2}: {pass}/{} byte-identical  [{status}]", pass + fail);
        }
        eprintln!("GZIP TOTAL: {gzip_identical}/{gzip_total} cases byte-identical to libdeflate");
        if !gzip_mismatches.is_empty() {
            panic!(
                "deflate-rs gzip_compress is NOT byte-identical to libdeflate in \
                 {}/{} cases:\n{}",
                gzip_mismatches.len(), gzip_total,
                gzip_mismatches.join("\n")
            );
        }
    }

    // --- Compact per-level summary (always printed). ---
    eprintln!("\n=== deflate-rs vs C libdeflate byte-identity (per level) ===");
    for (level, (pass, fail)) in &per_level {
        let status = if *fail == 0 { "PASS" } else { "FAIL" };
        eprintln!(
            "  level{level:>2}: {pass}/{} byte-identical  [{status}]",
            pass + fail
        );
    }
    eprintln!(
        "TOTAL: {byte_identical}/{total_cases} cases byte-identical to libdeflate\n"
    );

    if !mismatches.is_empty() {
        // Dump the full hex of the first mismatching case to aid diagnosis.
        let detail = mismatches.join("\n");
        panic!(
            "deflate-rs zlib_compress is NOT byte-identical to libdeflate in \
             {}/{} cases:\n{}\n\nDecode round-trip PASSED for all cases (our stream \
             reader is correct); the divergence is in the COMPRESSOR. Diagnose \
             against libdeflate v1.25 lib/deflate_compress.c \
             (https://github.com/ebiggers/libdeflate/blob/v1.25/lib/deflate_compress.c) \
             — do NOT loosen this test.",
            mismatches.len(),
            total_cases,
            detail
        );
    }
}

/// Standalone smoke test that input generators are deterministic and the
/// expected sizes hold, even when `golden.txt` is absent.
#[test]
fn input_generators_are_deterministic() {
    assert_eq!(make_input("empty").len(), 0);
    assert_eq!(make_input("short_str"), b"hello, deflate-rs vs libdeflate!\n");
    assert_eq!(make_input("repetitive").len(), 4096);
    assert_eq!(make_input("all_same"), vec![0x5Au8; 8192]);
    assert_eq!(make_input("pseudo_random").len(), 65536);
    // Two calls must produce identical bytes (no hidden state).
    assert_eq!(make_input("pseudo_random"), make_input("pseudo_random"));
    // structured is non-empty text.
    assert!(make_input("structured").len() > 1000);
    // hex round-trip sanity.
    assert_eq!(hex_decode(&hex_encode(&[0, 1, 254, 255])).unwrap(), vec![0, 1, 254, 255]);
}
