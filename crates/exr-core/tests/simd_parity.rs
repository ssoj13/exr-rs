//! `assert-bytes` scaffold: scalar codec reference over varied inputs.
//!
//! # Why this exists (and what it will become)
//! SIMD tiers are added to the hot codecs. The non-negotiable rule
//! is that **every SIMD tier is byte-identical to
//! the scalar source-of-truth** — a one-bit difference is a bug, not a tradeoff.
//! This file is the harness that proves it: it runs the codecs over a fuzz-ish
//! set of input images (varied sizes + content patterns) and asserts the
//! lossless codecs round-trip **byte-exact**.
//!
//! exr-core's one SIMD path is the ZIP/RLE decode predictor, via `codec-simd`.
//! Its per-tier differential against scalar lives in `codec-simd`'s own
//! `tests/cross_tier.rs`, and `tests/codec_simd_tier.rs` here pins that the
//! SIMD body is actually selected. This file round-trips the codecs through
//! whatever tier is active. [`roundtrip_scanline`] is the seam — it drives
//! the **real verified writer + reader** (so the codec's size/raw-passthrough
//! contract is honoured exactly as in production), and every input in
//! [`corpus`] becomes an automatic assertion with no new plumbing.
//!
//! These are correctness tests (cheap, CI-able), distinct from `benches/codecs.rs`
//! (timing). They reuse the production scanline path, so a mis-built buffer
//! cannot give a false pass.

use exr_core::attr::{
    Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType,
};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{Header, SCANLINE_IMAGE};
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::{InputFile, OutputFile};
use imath_rs::{Half, V2f};

// ---------------------------------------------------------------------------
// Varied input content (the "fuzz-ish" corpus)
// ---------------------------------------------------------------------------

/// Content patterns that stress different codec stages: a flat field
/// (best-case RLE/predictor), a smooth ramp (best-case wavelet/DCT), pseudo
/// noise (worst-case entropy coder), and a structured mix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    Flat,
    Ramp,
    Noise,
    Mixed,
}

#[inline]
fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

fn value(p: Pattern, c: usize, x: usize, y: usize, w: usize, h: usize) -> f32 {
    match p {
        Pattern::Flat => 0.5 + 0.1 * c as f32,
        Pattern::Ramp => (x as f32 / w as f32 + y as f32 / h as f32) * 0.5,
        Pattern::Noise => hash01(
            (x as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add((y as u32).wrapping_mul(40_503))
                .wrapping_add(c as u32 * 7),
        ),
        Pattern::Mixed => {
            let ramp = (x as f32 / w as f32) * 0.6 + 0.2;
            let n = hash01((x as u32) ^ (y as u32).wrapping_mul(2_246_822_519) ^ (c as u32));
            (ramp + 0.08 * (n - 0.5)).clamp(0.0, 4.0)
        }
    }
}

/// Channel descriptor: name, value-index, pixel type.
type Chan = (&'static str, usize, PixelType);

/// Header order is alphabetical: A, B, G, R.
const RGBA: [Chan; 4] = [
    ("A", 3, PixelType::Float),
    ("B", 2, PixelType::Half),
    ("G", 1, PixelType::Half),
    ("R", 0, PixelType::Half),
];
/// HALF-only RGBA (for PXR24's lossless-on-HALF guarantee).
const RGBA_HALF: [Chan; 4] = [
    ("A", 3, PixelType::Half),
    ("B", 2, PixelType::Half),
    ("G", 1, PixelType::Half),
    ("R", 0, PixelType::Half),
];

fn channel_list(chans: &[Chan]) -> ChannelList {
    let mut cl = ChannelList::new();
    for &(name, _, ty) in chans {
        cl.insert(name, Channel::new(ty, 1, 1, false)).unwrap();
    }
    cl
}

/// Pixel stride (bytes) and per-channel offset within a pixel, interleaved in
/// header order.
fn layout(chans: &[Chan]) -> (usize, Vec<usize>) {
    let mut off = Vec::with_capacity(chans.len());
    let mut cur = 0usize;
    for &(_, _, ty) in chans {
        off.push(cur);
        cur += exr_core::misc::pixel_type_size(ty).unwrap();
    }
    (cur, off)
}

/// Build the interleaved native-endian pixel buffer + matching FrameBuffer for a
/// `w × h` image with `chans`, at origin (0,0).
fn build_image(p: Pattern, chans: &[Chan], w: usize, h: usize) -> (Vec<u8>, FrameBuffer) {
    let (pxb, offs) = layout(chans);
    let row_bytes = pxb * w;
    let mut buf = vec![0u8; row_bytes * h];
    let mut fb = FrameBuffer::new();
    for (ci, &(name, vidx, ty)) in chans.iter().enumerate() {
        let o = offs[ci];
        for y in 0..h {
            for x in 0..w {
                let v = value(p, vidx, x, y, w, h);
                let dst = y * row_bytes + x * pxb + o;
                match ty {
                    PixelType::Half => buf[dst..dst + 2]
                        .copy_from_slice(&Half::from(v).bits().to_ne_bytes()),
                    PixelType::Float => buf[dst..dst + 4].copy_from_slice(&v.to_ne_bytes()),
                    _ => unreachable!(),
                }
            }
        }
        fb.insert(
            name,
            Slice {
                type_: ty,
                base_offset: o as i64,
                x_stride: pxb,
                y_stride: row_bytes,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
    }
    (buf, fb)
}

fn header(chans: &[Chan], w: i32, h: i32, comp: Compression) -> Header {
    let mut hd = Header::with_dimensions(
        w,
        h,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        comp,
    )
    .unwrap();
    hd.insert("channels", ChannelListAttribute::new(channel_list(chans))).unwrap();
    hd.set_type(SCANLINE_IMAGE).unwrap();
    hd
}

/// Drive the **real** scanline writer + reader for `comp` and return the decoded
/// interleaved buffer. This is the production codepath (it carries the chunk
/// size / raw-passthrough context the codecs need), so it is the correct seam
/// for a future scalar-vs-SIMD byte-equality assertion.
fn roundtrip_scanline(
    p: Pattern,
    chans: &[Chan],
    w: usize,
    h: usize,
    comp: Compression,
) -> (Vec<u8>, Vec<u8>) {
    let hd = header(chans, w as i32, h as i32, comp);
    let (pixels, fb) = build_image(p, chans, w, h);

    let mut mem = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut mem, &hd).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, h as i32).unwrap();
    }
    let bytes = mem.into_data();

    let mut out = vec![0u8; pixels.len()];
    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();
    inf.set_frame_buffer(&fb).unwrap();
    inf.read_pixels(0, h as i32 - 1, &mut out).unwrap();
    (pixels, out)
}

/// The fuzz-ish corpus: every (pattern × width × height) combination. Widths and
/// heights are chosen to straddle the codec block boundaries (1/16/32/256) and
/// include awkward small/odd sizes that exercise partial blocks.
fn corpus() -> Vec<(Pattern, usize, usize)> {
    let mut v = Vec::new();
    for p in [Pattern::Flat, Pattern::Ramp, Pattern::Noise, Pattern::Mixed] {
        for &(w, h) in &[(1, 1), (3, 5), (7, 17), (16, 16), (33, 40), (64, 70)] {
            v.push((p, w, h));
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Every fully-lossless codec round-trips every corpus image **byte-exact**
/// through the production scanline path. This pins the scalar reference; a SIMD
/// tier that diverges by one bit will fail here once it is routed through the
/// same path.
#[test]
fn lossless_roundtrip_is_byte_exact() {
    // NONE, RLE, ZIPS, ZIP, PIZ are fully lossless for HALF + FLOAT, and so is
    // HTJ2K: C++ `ImfCompression.cpp` registers htj2k256/htj2k32 with
    // `lossy = false` and `internal_ht.cpp` sets `cod.set_reversible(true)`.
    // HTJ2K used to be absent from this list AND classified lossy, so nothing
    // anywhere asserted that its reversible transform actually reverses.
    let lossless = [
        Compression::NoCompression,
        Compression::Rle,
        Compression::Zips,
        Compression::Zip,
        Compression::Piz,
        Compression::Htj2k256,
        Compression::Htj2k32,
    ];
    for comp in lossless {
        for &(p, w, h) in &corpus() {
            let (orig, back) = roundtrip_scanline(p, &RGBA, w, h, comp);
            assert_eq!(
                orig, back,
                "{comp:?} {p:?} {w}x{h}: lossless round-trip not byte-exact"
            );
        }
    }
}

/// PXR24 is **lossless for HALF/UINT** (only FLOAT is the lossy f24 path), so a
/// HALF-only image must round-trip **byte-exact** through the full scanline path
/// over the whole fuzz corpus — including the odd extents that straddle the 16-row
/// chunk boundary (e.g. 7x17 = one full chunk + a 1-row partial trailing chunk).
/// The FLOAT-bearing set is lossy, so it is only checked structurally (size).
#[test]
fn pxr24_half_roundtrip_is_byte_exact() {
    for &(p, w, h) in &corpus() {
        // FLOAT (f24) is lossy — structural check only.
        let (orig, back) = roundtrip_scanline(p, &RGBA, w, h, Compression::Pxr24);
        assert_eq!(orig.len(), back.len(), "PXR24 {p:?} {w}x{h}: decoded size mismatch");
        // HALF is lossless — must be byte-exact at every extent (partial chunks too).
        let (orig, back) = roundtrip_scanline(p, &RGBA_HALF, w, h, Compression::Pxr24);
        assert_eq!(
            orig, back,
            "PXR24 HALF {p:?} {w}x{h}: lossless round-trip not byte-exact"
        );
    }
}

/// The lossy codecs (B44 / B44A / DWAA / DWAB) over the corpus.
///
/// A size-only check (which is all this used to be) passes on a decoder that
/// returns zeros, so it also asserts what these codecs are *exactly* right
/// about:
///
/// * every decoded sample is finite — no NaN/Inf leaking out of the DCT or the
///   B44 unpack;
/// * a FLAT image round-trips **bit-exact** through B44/B44A. That is the whole
///   point of the flat-block encoding (`internal_b44.c` emits a 3-byte block
///   when all 16 samples are equal, and the decode restores them verbatim), and
///   it is a hard equality, not a tolerance.
#[test]
fn lossy_codecs_roundtrip_without_error() {
    let lossy = [
        Compression::B44,
        Compression::B44a,
        Compression::Dwaa,
        Compression::Dwab,
    ];
    for comp in lossy {
        for &(p, w, h) in &corpus() {
            let (orig, back) = roundtrip_scanline(p, &RGBA, w, h, comp);
            assert_eq!(
                orig.len(),
                back.len(),
                "{comp:?} {p:?} {w}x{h}: decoded size mismatch"
            );
            // A HALF-only image lets the decoded buffer be read as halves, so
            // a decoder that emits NaN/Inf out of the DCT or the B44 unpack is
            // caught instead of passing a size check.
            let (orig_h, back_h) = roundtrip_scanline(p, &RGBA_HALF, w, h, comp);
            assert!(
                back_h.as_chunks::<2>().0.iter().all(|b| {
                    Half::from_bits(u16::from_ne_bytes(*b)).to_f32().is_finite()
                }),
                "{comp:?} {p:?} {w}x{h}: decode produced a non-finite HALF sample"
            );
            if p == Pattern::Flat && matches!(comp, Compression::B44 | Compression::B44a) {
                assert_eq!(
                    orig_h, back_h,
                    "{comp:?} {w}x{h}: a FLAT image must round-trip bit-exact                      (that is what the flat-block encoding is for)"
                );
            }
        }
    }
}

/// Determinism guard: encoding the same image twice yields byte-identical files
/// (the decoded buffers are equal by construction; here we check the *encoded*
/// stream is stable). A future SIMD tier must also be deterministic — this
/// catches uninitialised-scratch / data-dependent-path bugs in either path.
#[test]
fn encode_is_deterministic() {
    fn encode(p: Pattern, w: usize, h: usize, comp: Compression) -> Vec<u8> {
        let hd = header(&RGBA, w as i32, h as i32, comp);
        let (pixels, fb) = build_image(p, &RGBA, w, h);
        let mut mem = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut mem, &hd).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&pixels, h as i32).unwrap();
        }
        mem.into_data()
    }
    for comp in [Compression::Zip, Compression::Piz, Compression::Dwaa] {
        for &(p, w, h) in &corpus() {
            let a = encode(p, w, h, comp);
            let b = encode(p, w, h, comp);
            assert_eq!(a, b, "{comp:?} {p:?} {w}x{h}: encode not deterministic");
        }
    }
}
