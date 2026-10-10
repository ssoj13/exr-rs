//! Regression tests for the **writer-side "store raw" decision**
//! ([`exr_core::compressor::stored_chunk`]).
//!
//! C++ keeps a compressed chunk only when it is *strictly smaller* than the raw
//! block (`ImfOutputFile.cpp:573`, `ImfTiledOutputFile.cpp:783`); at equality it
//! stores the raw bytes.  Readers — ours and C++'s — detect a raw passthrough by
//! `dataSize == uncompressedSize` alone, so a *compressed* chunk written at
//! exactly the uncompressed size is read back scrambled by every implementation.
//!
//! Before the writer seam existed, a 32x32 two-FLOAT B44 file hit that case
//! exactly: our reader **and** real
//! C++ OpenEXR both read it scrambled.  The tests below pin the fix:
//!
//! * `*_equal_size_chunk_is_stored_raw` — coverage guards: the file really does
//!   contain a chunk stored at exactly the raw block size, so the round-trip
//!   tests cannot silently stop exercising the bug.
//! * `*_round_trips_*` — the pixels survive the round trip.

use exr_core::api::{ChannelData, Image};
use exr_core::attr::Compression;
use exr_core::chunk::RawImage;
use exr_core::io::{MemIStream, MemOStream, OStream};
use imath_rs::{Box2i, Half, V2i};

/// Deterministic pseudo-random f32 in `[0, 1)` — a cheap LCG so the test needs
/// no dev-dependency and produces the same (incompressible) bytes everywhere.
fn lcg_floats(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

/// Deterministic pseudo-random halves in `[0, 1)`.
fn lcg_halves(n: usize, seed: u32) -> Vec<Half> {
    lcg_floats(n, seed).into_iter().map(Half::from_f32).collect()
}

fn image(w: i32, h: i32, channels: Vec<(&str, ChannelData)>) -> Image {
    let win = Box2i::from_min_max(V2i::new(0, 0), V2i::new(w - 1, h - 1));
    let mut img = Image::new(win);
    for (name, data) in channels {
        img.insert_channel(name, data).expect("insert_channel");
    }
    img
}

fn write_bytes(img: &Image, codec: Compression) -> Vec<u8> {
    let mut out = MemOStream::new();
    img.write_to(&mut out as &mut dyn OStream, codec)
        .expect("write");
    out.into_data()
}

fn read_back(bytes: Vec<u8>) -> Image {
    let mut input = MemIStream::new(bytes);
    Image::read_from(&mut input).expect("read")
}

fn round_trip(img: &Image, codec: Compression) -> Image {
    read_back(write_bytes(img, codec))
}

/// Stored bytes of every chunk in a single-part file, in file order.
fn chunk_bytes(bytes: Vec<u8>) -> Vec<Vec<u8>> {
    let mut input = MemIStream::new(bytes);
    let raw = RawImage::read(&mut input).expect("RawImage::read");
    assert_eq!(raw.parts.len(), 1, "single-part fixture");
    raw.parts[0].chunks.iter().map(|c| c.data.clone()).collect()
}

fn float_channel(img: &Image, name: &str) -> Vec<f32> {
    match img.channel(name).expect("channel present") {
        ChannelData::Float(v) => v.clone(),
        other => panic!("{name}: expected FLOAT, got {:?}", other.pixel_type()),
    }
}

fn half_channel(img: &Image, name: &str) -> Vec<Half> {
    match img.channel(name).expect("channel present") {
        ChannelData::Half(v) => v.clone(),
        other => panic!("{name}: expected HALF, got {:?}", other.pixel_type()),
    }
}

// ---------------------------------------------------------------------------
// The C1 repro: 32x32, two FLOAT channels, B44
// ---------------------------------------------------------------------------

/// B44 leaves FLOAT channels uncompressed (only HALF is block-encoded), so this
/// image's B44 chunk comes out at exactly the raw block size.
fn b44_float_repro() -> Image {
    let n = 32 * 32;
    image(
        32,
        32,
        vec![
            ("A", ChannelData::Float(lcg_floats(n, 0x1234_5678))),
            ("B", ChannelData::Float(lcg_floats(n, 0x9abc_def0))),
        ],
    )
}

#[test]
fn b44_two_float_32x32_equal_size_chunk_is_stored_raw() {
    let src = b44_float_repro();

    // NONE writes one raw Xdr scanline per chunk; concatenated, those are
    // exactly the bytes the 32-line B44 block starts from.
    let raw_block: Vec<u8> = chunk_bytes(write_bytes(&src, Compression::NoCompression))
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(raw_block.len(), 32 * 32 * 2 * 4, "32 rows * 32 px * 2 FLOAT");

    let stored = chunk_bytes(write_bytes(&src, Compression::B44));
    assert_eq!(stored.len(), 1, "B44 batches 32 scanlines into one chunk");
    // The encoding for this image comes out at exactly the raw size, so the
    // writer must store the RAW block — a same-size *compressed* chunk is
    // indistinguishable from a raw one to every reader.
    assert_eq!(
        stored[0].len(),
        raw_block.len(),
        "the equal-size path must fire here"
    );
    assert_eq!(
        stored[0], raw_block,
        "chunk stored at the uncompressed size must BE the uncompressed bytes"
    );
}

#[test]
fn b44_two_float_32x32_round_trips_bit_exact() {
    let src = b44_float_repro();
    let out = round_trip(&src, Compression::B44);
    for name in ["A", "B"] {
        assert_eq!(
            float_channel(&out, name),
            float_channel(&src, name),
            "channel {name} must survive B44 bit-exactly (FLOAT is passthrough)"
        );
    }
}

#[test]
fn zip_two_float_32x32_round_trips_bit_exact() {
    let src = b44_float_repro();
    let out = round_trip(&src, Compression::Zip);
    for name in ["A", "B"] {
        assert_eq!(float_channel(&out, name), float_channel(&src, name));
    }
}

// ---------------------------------------------------------------------------
// The 7x2 HALF shape
// ---------------------------------------------------------------------------

fn half_7x2() -> Image {
    let n = 7 * 2;
    image(
        7,
        2,
        vec![
            ("A", ChannelData::Half(lcg_halves(n, 0x0bad_c0de))),
            ("B", ChannelData::Half(lcg_halves(n, 0x00c0_ffee))),
        ],
    )
}

#[test]
fn zip_half_7x2_round_trips_bit_exact() {
    let src = half_7x2();
    let out = round_trip(&src, Compression::Zip);
    for name in ["A", "B"] {
        assert_eq!(half_channel(&out, name), half_channel(&src, name));
    }
}

/// B44 is lossy for HALF, so the assertion is "close, and stable", not
/// bit-exact against the source: a chunk mis-read as compressed-vs-raw produces
/// garbage magnitudes (or NaN), which this comfortably catches.  The second
/// round trip must be bit-exact — B44 is idempotent once the values sit on its
/// quantization grid.
#[test]
fn b44_half_7x2_round_trips_within_quantization() {
    let src = half_7x2();
    let once = round_trip(&src, Compression::B44);
    let twice = round_trip(&once, Compression::B44);

    for name in ["A", "B"] {
        let s = half_channel(&src, name);
        let a = half_channel(&once, name);
        let b = half_channel(&twice, name);
        assert_eq!(a, b, "channel {name}: B44 must be idempotent after one pass");
        for (i, (sv, av)) in s.iter().zip(a.iter()).enumerate() {
            let (sf, af) = (sv.to_f32(), av.to_f32());
            assert_b44_approx(name, i, sf, af);
        }
    }
}

/// Same bound the 7x2 test uses, factored out so `half_9x5` can share it.
fn assert_b44_approx(name: &str, i: usize, sf: f32, af: f32) {
    assert!(
        (sf - af).abs() <= 0.05 * sf.abs().max(1e-3),
        "channel {name}[{i}]: {af} is not a B44 approximation of {sf}"
    );
}

/// Mutation-testing gap fix (audit finding): `half_7x2` is only 2 rows tall, so
/// every 4x4 `unpack14` block only ever writes back rows 0-1 (`uncompress`'s
/// "only in-bounds pixels are stored" clipping, `b44/mod.rs`) — rows 2-3 of
/// every block (the `s[8..16]` half of `unpack14`'s output) are computed but
/// never checked by any existing test. `cargo-mutants` confirmed this: mutating
/// `unpack14`'s `<<`/`>>`/`|`/`&` operators that feed `s[8]`/`s[9]`/`s[12]`/`s[13]`
/// (rows 2-3) survived the whole suite. This fixture is 9x5 (>1 full BLOCK=4
/// tall, and a non-multiple-of-4 width so the right-edge column-padding path
/// also gets exercised) so every row of every block is written back and
/// checked.
/// A smooth gradient with a small high-frequency dither, NOT pure `lcg_halves`
/// noise: B44 predicts each block from its own local deltas, so
/// spatially-uncorrelated per-pixel random noise (adjacent deltas as large as
/// the whole [0, 1) range) produces legitimately coarse quantization even
/// with no bug, which made a 5%-relative bound flaky on a fixture this size.
/// A gradient (small adjacent deltas, like the imagery B44 targets) keeps the
/// true quantization error small so the tight bound stays meaningful, while
/// the dither still varies every sample enough to exercise `unpack14`'s bit
/// paths (unlike a pure ramp, which could degenerate to the flat-block
/// `unpack3` path).
fn smooth_halves(w: i32, h: i32, seed: u32) -> Vec<Half> {
    let dither = lcg_halves((w * h) as usize, seed);
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .zip(dither)
        .map(|((x, y), d)| {
            let ramp = (x as f32 / (w - 1).max(1) as f32 + y as f32 / (h - 1).max(1) as f32) * 0.5;
            Half::from_f32((ramp * 0.9 + d.to_f32() * 0.1).clamp(0.0, 1.0))
        })
        .collect()
}

fn half_9x5() -> Image {
    image(
        9,
        5,
        vec![
            ("A", ChannelData::Half(smooth_halves(9, 5, 0x1357_9bdf))),
            ("B", ChannelData::Half(smooth_halves(9, 5, 0x2468_ace0))),
        ],
    )
}

#[test]
fn b44_half_9x5_round_trips_within_quantization() {
    let src = half_9x5();
    let once = round_trip(&src, Compression::B44);
    let twice = round_trip(&once, Compression::B44);

    for name in ["A", "B"] {
        let s = half_channel(&src, name);
        let a = half_channel(&once, name);
        let b = half_channel(&twice, name);
        assert_eq!(a, b, "channel {name}: B44 must be idempotent after one pass");
        for (i, (sv, av)) in s.iter().zip(a.iter()).enumerate() {
            let (sf, af) = (sv.to_f32(), av.to_f32());
            assert_b44_approx(name, i, sf, af);
        }
    }
}
