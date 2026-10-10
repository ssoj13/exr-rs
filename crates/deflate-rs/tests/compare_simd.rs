//! W-B SIMD parity test — the PUBLIC `crc32` / `adler32` / compressor must
//! produce results independent of which SIMD tier the dispatcher selects.
//!
//! The active tier is controlled by the `DEFLATE_RS_SIMD` env var
//! (`scalar|sse2|avx2|pclmul`, default = auto-detect). Because that override is
//! read once per process via a `OnceLock`, this test self-respawns child
//! processes with each tier value set and checks they all agree.
//!
//! In each child (selected by the `SIMD_CHILD` env marker) we verify:
//!  1. `crc32` over varied data == an independent naive bit-by-bit CRC-32.
//!  2. `adler32` over varied data == an independent naive Adler-32.
//!  3. `Compressor::zlib_compress` output is identical to the SCALAR child's
//!     output (passed via a golden hash on stdout) — proving the matchfinder
//!     SIMD init/rebase did not perturb the compressed stream.
//!
//! The parent process drives the children and asserts cross-tier agreement.
//! Each child also reports the tiers its dispatchers actually selected, and
//! the parent checks them against the cap and this CPU's features, so a
//! dispatcher that silently stays scalar (or ignores the cap) fails. A child
//! that cannot be spawned is a failure, never a skip.

use std::process::Command;

use deflate_rs::{adler32, crc32, Compressor};
use deflate_rs::cpu::active_tiers_report;

/// Deterministic varied test data (xorshift32), `n` bytes.
fn gen_data(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        out.push((x & 0xFF) as u8);
    }
    out
}

/// Independent naive CRC-32 (gzip, reflected poly 0xEDB88320), bit-by-bit.
/// Deliberately unrelated to the table/SIMD implementations under test.
fn naive_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let m = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & m);
        }
    }
    !crc
}

/// Independent naive Adler-32.
fn naive_adler32(data: &[u8]) -> u32 {
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &b in data {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    (s2 << 16) | s1
}

/// FNV-1a hash over a byte slice — compact stdout-friendly fingerprint of the
/// compressed stream, used to compare child outputs cross-tier.
fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The exact lengths exercised: edge cases around every SIMD branch boundary
/// plus large sizes hitting the main loops and the multi-chunk Adler path.
fn test_lengths() -> Vec<usize> {
    let mut v = vec![
        0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 100, 127, 128, 129, 255,
        256, 257, 1023, 1024, 4095, 4096, 4097, 5503, 5504, 5505, 8191, 8192,
    ];
    v.push(40000); // > MAX_CHUNK_LEN for adler multi-chunk
    v
}

/// Run all in-process checks; return an FNV fingerprint of every compressed
/// stream concatenated, so the parent can compare it across tiers.
fn run_checks() -> u64 {
    let mut acc: u64 = 0;
    for &len in &test_lengths() {
        let data = gen_data(len, 0x1234_5678 ^ len as u32);

        // (1) crc32 vs naive
        assert_eq!(crc32(0, &data), naive_crc32(&data), "crc32 mismatch len={len}");
        // (2) adler32 vs naive
        assert_eq!(adler32(1, &data), naive_adler32(&data), "adler32 mismatch len={len}");

        // (3) compress fingerprint (matchfinder SIMD must not perturb output)
        for level in [1i32, 6, 9, 12] {
            let mut c = Compressor::new(level).unwrap();
            let z = c.zlib_compress(&data);
            acc ^= fnv1a(&z).rotate_left((level as u32 * 7 + len as u32) & 63);
        }
    }
    acc
}

/// CHILD entry: run checks, print the fingerprint as the last stdout line.
fn child_main() {
    let fp = run_checks();
    println!("TIERS={}", active_tiers_report());
    println!("FINGERPRINT={fp:#018x}");
}

/// The tiers a child capped at `cap` must report on this CPU: the best tier
/// the CPU supports, lowered to the cap (mirrors `cpu.rs`'s selection rules,
/// written out independently).
fn expected_tiers(cap: &str) -> Vec<String> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let rank = |c: &str| ["scalar", "sse2", "avx2", "pclmul"].iter().position(|&t| t == c).unwrap_or(3);
        let r = rank(cap);
        let vec_tier = if r >= 2 && std::is_x86_feature_detected!("avx2") {
            "Avx2"
        } else if r >= 1 && std::is_x86_feature_detected!("sse2") {
            "Sse2"
        } else {
            "Scalar"
        };
        let crc = if r >= 3 && std::is_x86_feature_detected!("pclmulqdq") && std::is_x86_feature_detected!("sse4.1") {
            "Pclmul"
        } else {
            "Scalar"
        };
        vec![format!("adler32={vec_tier} "), format!("crc32={crc} "), format!("matchfinder={vec_tier} ")]
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = cap;
        vec!["crc32=Scalar ".to_string()]
    }
}

#[test]
fn simd_tiers_agree_with_scalar_and_naive() {
    // If we are the child, just run and print.
    if std::env::var("SIMD_CHILD").is_ok() {
        child_main();
        return;
    }

    // Parent: always run the checks in-process under whatever tier is active.
    let parent_fp = run_checks();

    // Then respawn one child per tier and assert identical fingerprints.
    let exe = std::env::current_exe().expect("current_exe");
    let tiers = ["scalar", "sse2", "avx2", "pclmul"];
    let mut fingerprints = Vec::new();
    for tier in tiers {
        let out = Command::new(&exe)
            .args(["--exact", "simd_tiers_agree_with_scalar_and_naive", "--nocapture"])
            .env("SIMD_CHILD", "1")
            .env("DEFLATE_RS_SIMD", tier)
            .output();
        let out = out.unwrap_or_else(|e| panic!("could not spawn the {tier} child: {e}"));
        assert!(
            out.status.success(),
            "child tier={tier} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let fp = stdout
            .lines()
            .find_map(|l| l.strip_prefix("FINGERPRINT="))
            .and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok())
            .unwrap_or_else(|| panic!("tier {tier}: no FINGERPRINT in:\n{stdout}"));
        let tiers = stdout
            .lines()
            .find_map(|l| l.strip_prefix("TIERS="))
            .unwrap_or_else(|| panic!("tier {tier}: no TIERS in:
{stdout}"));
        for want in expected_tiers(tier) {
            assert!(tiers.contains(&want), "cap {tier}: expected `{want}` in `{tiers}`");
        }
        eprintln!("tier {tier}: {tiers} fingerprint={fp:#018x}");
        fingerprints.push((tier, fp));
    }

    // Every tier that ran must match the parent (and thus each other). The
    // `scalar` tier is the source of truth; its presence is guaranteed.
    for (tier, fp) in &fingerprints {
        assert_eq!(
            *fp, parent_fp,
            "tier {tier} fingerprint diverged from parent (SIMD output not byte-identical to scalar)"
        );
    }
    assert!(
        fingerprints.iter().any(|(t, _)| *t == "scalar"),
        "scalar child did not run"
    );
}
