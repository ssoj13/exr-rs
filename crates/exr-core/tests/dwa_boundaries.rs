//! Mutation-testing gap-fix tests for DWA's input-hardening boundaries,
//! driven entirely through the PUBLIC `Compressor` trait (`DwaCompressor::new`
//! and `Compressor::{compress,uncompress}`), not the crate-private unit-test
//! helpers — so these keep working across internal refactors of
//! `compressor/dwa/mod.rs` (e.g. `classify_channels`'s signature changing to
//! `Result<Classification>`).
//!
//! Confirmed real gaps: `cargo-mutants` missed these mutants against BOTH the
//! crate's internal unit tests AND the full `parity_exr` oracle-comparison
//! suite (two independent passes). Every
//! test here was verified by hand-applying the exact confirmed mutation in an
//! isolated clone (never the shared working tree) and confirming the test fails — not by
//! inspection alone.
//!
//! `validate_dwa_size`'s contract (cited in its own doc comment) is a direct
//! port of C++ `internal_dwa_compressor.h:691-726` (OpenEXRCore, 3.4.15-only):
//! for one planar scheme (UNKNOWN or RLE), if `required_size > 0` the header's
//! declared `uncompressed_size` must reach it and `compressed_size` must be
//! nonzero; if `required_size == 0` (no channel of that scheme exists) every
//! size the header declares for it must be exactly zero. `required_size`
//! itself (`unknown_raw_total` / `rle_raw_total` in `decompress_dwa`) is the
//! sum of that scheme's channels' own packed byte counts — for a single
//! unsampled channel of `w x h` pixels, that is simply `w * h *
//! bytes_per_sample`, computed independently here from the channel's own
//! declared type, not read back from the crate.

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::compressor::Compressor;
use exr_core::compressor::dwa::{DWAA_SCANLINES, DwaCompressor};
use exr_core::header::Header;
use imath_rs::V2f;

const W: usize = 64;
const H: usize = DWAA_SCANLINES as usize; // one full DWAA chunk, no partial-chunk edge case.

/// Header size in bytes for one DWA chunk (`NUM_SIZES_SINGLE = 11` fields x
/// 8 bytes each, `compressor/dwa/mod.rs`'s private `HEADER_SIZE`/`Field`).
/// Hardcoded here (not imported — those are module-private) because the wire
/// layout is a stable, documented on-disk format, not an implementation
/// detail: `Version`(0), `UnknownUncompressedSize`(1),
/// `UnknownCompressedSize`(2), `AcCompressedSize`(3), `DcCompressedSize`(4),
/// `RleCompressedSize`(5), `RleUncompressedSize`(6), `RleRawSize`(7),
/// `AcUncompressedCount`(8), `DcUncompressedCount`(9), `AcCompression`(10).
const FIELD_UNKNOWN_UNCOMPRESSED: usize = 1;
const FIELD_UNKNOWN_COMPRESSED: usize = 2;
const FIELD_RLE_UNCOMPRESSED: usize = 6; // "extra_size" arg to validate_dwa_size for RLE.
const FIELD_RLE_RAW: usize = 7; // "uncompressed_size" arg to validate_dwa_size for RLE.

fn set_field(chunk: &mut [u8], field: usize, value: u64) {
    chunk[field * 8..field * 8 + 8].copy_from_slice(&value.to_le_bytes());
}

fn header_with_channels(channels: &[(&str, PixelType)]) -> Header {
    let mut h = Header::with_dimensions(
        W as i32,
        H as i32,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::Dwaa,
    )
    .expect("build header");
    let mut cl = ChannelList::new();
    for (name, ty) in channels {
        cl.insert(*name, Channel::new(*ty, 1, 1, false)).expect("insert channel");
    }
    h.insert("channels", ChannelListAttribute::new(cl)).expect("insert channels attr");
    h
}

fn bytes_per_sample(ty: PixelType) -> usize {
    match ty {
        PixelType::Half => 2,
        PixelType::Float | PixelType::Uint => 4,
        PixelType::NumPixelTypes => unreachable!(),
    }
}

/// All-zero pixel data for `channels` (alphabetical order, `H` scanlines of
/// `W` samples each) — maximally compressible, guaranteed to shrink under
/// DWA regardless of channel mix.
fn zero_buffer(channels: &[(&str, PixelType)]) -> Vec<u8> {
    let mut sorted: Vec<&(&str, PixelType)> = channels.iter().collect();
    sorted.sort_by_key(|(name, _)| *name);
    let per_row: usize = sorted.iter().map(|(_, ty)| W * bytes_per_sample(*ty)).sum();
    vec![0u8; per_row * H]
}

fn compress_good(channels: &[(&str, PixelType)]) -> (Header, Vec<u8>) {
    let header = header_with_channels(channels);
    let buf = zero_buffer(channels);
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).expect("construct DwaCompressor");
    let chunk = c.compress(&buf, 0).expect("compress").to_vec();
    (header, chunk)
}

/// A single UNKNOWN-scheme channel ("custom", a standalone `Float` — not
/// part of any R/G/B CSC group and not the alpha-shaped RLE case either) has
/// `unknown_raw_total = W * H * 4` (4 bytes/sample for FLOAT) and no RLE
/// channel at all (`rle_raw_total == 0`).
#[test]
fn validate_dwa_size_unknown_boundary_via_public_api() {
    let channels: &[(&str, PixelType)] = &[("custom", PixelType::Float)];
    let (header, good) = compress_good(channels);
    let required = (W * H * 4) as u64;

    // Sanity: the unpatched chunk still decodes.
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    assert!(c.uncompress(&good, 0, 0).is_ok(), "unpatched chunk must decode");

    // uncompressed_size == required: exactly at the boundary, must be OK
    // (the check is `uncompressed_size < required_size`, so equality passes).
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    let mut chunk = good.clone();
    set_field(&mut chunk, FIELD_UNKNOWN_UNCOMPRESSED, required);
    assert!(c.uncompress(&chunk, 0, 0).is_ok(), "uncompressed_size == required_size must be accepted");

    // uncompressed_size == required - 1: one below the boundary, must error.
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    let mut chunk = good.clone();
    set_field(&mut chunk, FIELD_UNKNOWN_UNCOMPRESSED, required - 1);
    assert!(c.uncompress(&chunk, 0, 0).is_err(), "uncompressed_size == required_size - 1 must be rejected (planar section too small)");

    // compressed_size == 0 while uncompressed_size is otherwise sufficient:
    // must still error (a real compressed section has to be present).
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    let mut chunk = good.clone();
    set_field(&mut chunk, FIELD_UNKNOWN_COMPRESSED, 0);
    assert!(c.uncompress(&chunk, 0, 0).is_err(), "compressed_size == 0 must be rejected even when uncompressed_size is enough");
}

/// No UNKNOWN-scheme channel present (`required_size == 0`): every size the
/// header declares for that scheme must be exactly zero.
#[test]
fn validate_dwa_size_no_matching_channel_via_public_api() {
    // R alone (standalone Half, not part of a CSC group) is LOSSY_DCT, not
    // UNKNOWN or RLE - so both unknown_raw_total and rle_raw_total are 0 for
    // this channel set.
    let channels: &[(&str, PixelType)] = &[("R", PixelType::Half)];
    let (header, good) = compress_good(channels);

    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    assert!(c.uncompress(&good, 0, 0).is_ok(), "unpatched chunk must decode");

    // Declaring a nonzero UnknownCompressedSize with no UNKNOWN channel at
    // all must be rejected.
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    let mut chunk = good.clone();
    set_field(&mut chunk, FIELD_UNKNOWN_COMPRESSED, 1);
    assert!(c.uncompress(&chunk, 0, 0).is_err(), "unknown_compressed > 0 with no UNKNOWN-scheme channel must be rejected");

    // Same for RleUncompressedSize (validate_dwa_size's "extra_size" arg for
    // the RLE scheme) with no RLE channel present.
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    let mut chunk = good.clone();
    set_field(&mut chunk, FIELD_RLE_UNCOMPRESSED, 1);
    assert!(c.uncompress(&chunk, 0, 0).is_err(), "rle_uncompressed > 0 with no RLE-scheme channel must be rejected");
}

/// A single RLE-scheme channel ("A", a standalone `Half` — the classic
/// alpha shape, see `alpha_channel_uses_rle_roundtrip` in the crate's own
/// unit tests) has `rle_raw_total = W * H * 2` (2 bytes/sample for HALF).
#[test]
fn validate_dwa_size_rle_boundary_via_public_api() {
    let channels: &[(&str, PixelType)] = &[("A", PixelType::Half)];
    let (header, good) = compress_good(channels);
    let required = (W * H * 2) as u64;

    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    assert!(c.uncompress(&good, 0, 0).is_ok(), "unpatched chunk must decode");

    // rle_raw (the "uncompressed_size" arg for the RLE scheme) exactly at
    // the boundary must be accepted.
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    let mut chunk = good.clone();
    set_field(&mut chunk, FIELD_RLE_RAW, required);
    assert!(c.uncompress(&chunk, 0, 0).is_ok(), "rle_raw == rle_raw_total must be accepted");

    // One below the boundary must error.
    let mut c = DwaCompressor::new(&header, DWAA_SCANLINES, false).unwrap();
    let mut chunk = good.clone();
    set_field(&mut chunk, FIELD_RLE_RAW, required - 1);
    assert!(c.uncompress(&chunk, 0, 0).is_err(), "rle_raw == rle_raw_total - 1 must be rejected");
}
