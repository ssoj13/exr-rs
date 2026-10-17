//! Tile-blob integrity: CRC-32C (Castagnoli) of every coded tile blob, stored in the frame tile index
//! (`docs/FORMAT.md` §5) next to the blob's length.
//!
//! # Why
//! A codec stream is not self-checking: a flipped bit inside an rANS / HTJ2K / EXV-RT payload usually decodes
//! "successfully" to wrong pixels. The writer checksums every blob it emits; a reader verifies a blob before the
//! codec (CPU) or the GPU upload sees it, under the caller's [`crate::limits::Verify`] policy.
//!
//! # Why CRC-32C
//! It is the checksum with dedicated instructions on both x86-64 (SSE4.2 `crc32`) and ARMv8 (`crc32c*`). Our x86-64
//! loop (3-way interleaved hardware instructions) reaches tens of GB/s per core, and everything else falls back to the `crc32c`
//! crate's safe slicing-by-8 table, so verification costs a small fraction of any decode. The ONE `unsafe` of this
//! crate is the call into our own SSE4.2 loop (after runtime feature detection), see [`crc32c`].

/// Reflected Castagnoli (`0x82F6_3B78`) byte-by-byte table for software fallback and test verification.
const CRC32C_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut n = 0usize;
    while n < 256 {
        let mut c = n as u32;
        let mut b = 0;
        while b < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
            b += 1;
        }
        t[n] = c;
        n += 1;
    }
    t
};

/// Software CRC-32C (Castagnoli) implementation.
#[inline]
pub fn crc32c_sw(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &byte in data {
        c = (c >> 8) ^ CRC32C_TABLE[((c as u8) ^ byte) as usize];
    }
    !c
}

/// CRC-32C (Castagnoli, reflected, init/xorout `0xFFFF_FFFF`) of `data`. `crc32c(b"") == 0`, so a SKIP tile (no
/// blob) carries `0`.
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.2") {
        // SAFETY: `sse42` only executes SSE4.2 instructions, and the CPU reports SSE4.2 (checked just above).
        return !unsafe { sse42::update(!0, data) };
    }
    crc32c_sw(data)
}

/// The x86-64 hardware path. Why our own loop instead of the `crc32c` crate's: the crate's per-word helpers are
/// `#[target_feature]` functions called from a function WITHOUT the feature, so they cannot inline and every 8 bytes
/// pay a call (measured 3.3-5 GB/s/core). Here the whole loop is compiled with SSE4.2: three independent
/// `crc32` streams (the instruction has a 3-cycle latency, 1/cycle throughput) merged with a table-driven
/// "advance over `BLK` zero bytes" operator. The crate stays the software / other-arch fallback and the test oracle.
#[cfg(target_arch = "x86_64")]
mod sse42 {
    use std::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};
    use std::sync::LazyLock;

    /// Bytes per stream of one 3-way round (`3 * BLK` bytes per round).
    const BLK: usize = 4096;

    /// `shift(x)` = the raw CRC register `x` advanced over `BLK` zero bytes, as 4 byte-indexed tables (the operator is
    /// linear over GF(2), so it is the XOR of its action on each byte of `x`).
    static SHIFT: LazyLock<[[u32; 256]; 4]> = LazyLock::new(|| {
        // Reflected CRC-32C, one zero byte: r = (r >> 8) ^ T[r & 0xff].
        let mut t = [0u32; 256];
        for (n, e) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
            }
            *e = c;
        }
        let zeros = |mut r: u32| {
            for _ in 0..BLK {
                r = (r >> 8) ^ t[(r & 0xff) as usize];
            }
            r
        };
        // The operator on each of the 32 basis bits, then expanded to per-byte tables.
        let basis: Vec<u32> = (0..32).map(|b| zeros(1 << b)).collect();
        let mut out = [[0u32; 256]; 4];
        for (k, tab) in out.iter_mut().enumerate() {
            for (v, e) in tab.iter_mut().enumerate() {
                *e = (0..8).filter(|i| v >> i & 1 != 0).fold(0, |acc, i| acc ^ basis[8 * k + i]);
            }
        }
        out
    });

    #[inline]
    fn shift(tab: &[[u32; 256]; 4], x: u32) -> u32 {
        tab[0][(x & 0xff) as usize] ^ tab[1][(x >> 8 & 0xff) as usize] ^ tab[2][(x >> 16 & 0xff) as usize] ^ tab[3][(x >> 24) as usize]
    }

    #[inline]
    fn word(c: &[u8; 8]) -> u64 {
        u64::from_le_bytes(*c)
    }

    /// Advance the raw CRC-32C register `crc` (no pre/post inversion) over `data`.
    #[target_feature(enable = "sse4.2")]
    pub(super) fn update(crc: u32, data: &[u8]) -> u32 {
        let mut c = crc as u64;
        let (rounds, tail) = data.as_chunks::<{ 3 * BLK }>();
        if !rounds.is_empty() {
            let tab = &*SHIFT;
            for round in rounds {
                let (a, rest) = round.split_at(BLK);
                let (b, d) = rest.split_at(BLK);
                let (mut cb, mut cd) = (0u64, 0u64);
                for ((wa, wb), wd) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0).zip(d.as_chunks::<8>().0) {
                    c = _mm_crc32_u64(c, word(wa));
                    cb = _mm_crc32_u64(cb, word(wb));
                    cd = _mm_crc32_u64(cd, word(wd));
                }
                let ab = shift(tab, c as u32) ^ cb as u32;
                c = (shift(tab, ab) ^ cd as u32) as u64;
            }
        }
        let (words, rest) = tail.as_chunks::<8>();
        for w in words {
            c = _mm_crc32_u64(c, word(w));
        }
        let mut c = c as u32;
        for &byte in rest {
            c = _mm_crc32_u8(c, byte);
        }
        c
    }
}

/// A coded blob whose bytes do not match the checksum recorded in the frame index ([`crate::DecodeError::Checksum`]).
/// `tile` is the row-major tile index inside its frame; the frame is known to the caller that walks frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mismatch {
    /// Row-major tile index within the frame.
    pub tile: u32,
    /// Channel of a per-channel codec's sub-blob; `None` for a multi-component (HTJ2K) tile's single blob.
    pub chan: Option<u16>,
    /// CRC-32C recorded by the writer.
    pub expected: u32,
    /// CRC-32C of the bytes actually read.
    pub got: u32,
}

/// `Ok` when `blob` hashes to `expected`, else the typed [`Mismatch`] of (`tile`, `chan`).
#[inline]
pub fn check(tile: u32, chan: Option<u16>, blob: &[u8], expected: u32) -> Result<(), Mismatch> {
    let got = crc32c(blob);
    if got == expected { Ok(()) } else { Err(Mismatch { tile, chan, expected, got }) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard CRC-32C check value ("123456789" -> 0xE3069283, RFC 3720 B.4) pins the polynomial / reflection.
    #[test]
    fn castagnoli_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        assert_eq!(check(3, None, b"123456789", 0xE306_9283), Ok(()));
        let m = check(3, Some(2), b"123456788", 0xE306_9283).expect_err("one byte differs");
        assert_eq!((m.tile, m.chan, m.expected), (3, Some(2), 0xE306_9283));
    }

    /// The hardware path equals the crate's (software-checked) CRC at every length around the 8-byte word and the
    /// 3-way round boundaries, at every alignment.
    #[test]
    fn hw_path_matches_reference() {
        let data: Vec<u8> = (0..40_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for len in (0..80).chain([4095, 4096, 12_287, 12_288, 12_289, 12_295, 24_576, 24_583, 39_000]) {
            for off in [0usize, 1, 7] {
                let s = &data[off..off + len];
                assert_eq!(crc32c(s), crc32c_sw(s), "len {len} off {off}");
            }
        }
    }
}
