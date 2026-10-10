//! Bit-exact check of `exr_core::rgba::{round12_log, RoundNBit}` against real
//! C++ `Imf::round12log`/`roundNBit` (`ImfLut.cpp`), over EVERY possible
//! `half` bit pattern.
//!
//! `lut_dump.bin` (`oracle.cpp`'s `lut-dump <out> 4` mode) holds, for `i` in
//! `0..65536` in order: `round12log(half::setBits(i))` then
//! `roundNBit(4)(half::setBits(i))`, each as a `u16` LE half-bits pair
//! (`65536 * 4` bytes). `HalfLut`/`RgbaLut` are themselves just a
//! table-lookup wrapper around a `half -> half` function
//! (`imath_rs::HalfFunction`, independently oracle-verified in imath-rs's own
//! parity suite), so exercising `round12_log`/`RoundNBit` this exhaustively
//! covers `HalfLut`/`RgbaLut` built from them too.

use imath_rs::Half;

mod support;

const HOW_TO_GET: &str = "Run `cargo run -p xtask -- parity-exr` to regenerate the C++ fixture.";

#[test]
fn round12_log_and_round_n_bit_match_cpp_over_every_half() {
    use exr_core::rgba::{round12_log, RoundNBit};

    let name = "round12_log_and_round_n_bit_match_cpp_over_every_half";
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("parity").join("lut_dump.bin");
    let Some(path) = support::fixture_or_skip(name, path, HOW_TO_GET) else {
        return;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len(), 65536 * 4, "lut_dump.bin has an unexpected length");

    let rn = RoundNBit::new(4);
    let mut round12log_mismatches = 0usize;
    let mut roundnbit_mismatches = 0usize;
    for i in 0..65536u32 {
        let off = i as usize * 4;
        let want_r12 = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let want_rnb = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]);

        let x = Half::from_bits(i as u16);
        let got_r12 = round12_log(x).bits();
        let got_rnb = rn.apply(x).bits();

        if got_r12 != want_r12 {
            if round12log_mismatches < 5 {
                eprintln!("round12_log: bits {i:#06x} -> ours {got_r12:#06x} cpp {want_r12:#06x}");
            }
            round12log_mismatches += 1;
        }
        if got_rnb != want_rnb {
            if roundnbit_mismatches < 5 {
                eprintln!("RoundNBit(4): bits {i:#06x} -> ours {got_rnb:#06x} cpp {want_rnb:#06x}");
            }
            roundnbit_mismatches += 1;
        }
    }
    assert_eq!(round12log_mismatches, 0, "{round12log_mismatches}/65536 round12_log values differ from C++");
    assert_eq!(roundnbit_mismatches, 0, "{roundnbit_mismatches}/65536 RoundNBit(4) values differ from C++");
}
