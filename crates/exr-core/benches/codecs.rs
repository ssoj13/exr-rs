//! Criterion benchmark harness + baseline profile for exr-core.
//!
//! **MEASURE FIRST.** This bench establishes where the codec time actually goes
//! so performance work (rayon parallel, DWA/ZIP/PIZ SIMD) targets the real hot
//! spots instead of guessing. It does **not** change any codec behaviour — it
//! drives the existing, verified codec / scanline / tiled APIs.
//!
//! # What it measures
//! * Per-codec **compress + uncompress** throughput on one representative chunk
//!   (full image width × the codec's own block height), reported in MB/s on the
//!   *uncompressed* byte size (the standard codec-throughput metric).
//! * A scanline **write + read** round-trip and a tiled **write + read**
//!   round-trip at a couple of sizes (whole-pipeline cost, incl. gather/scatter).
//! * The Imath `Half` ↔ `f32` conversion paths (they underlie every HALF codec).
//!
//! # Representative content (NOT all-zeros)
//! All-zero buffers compress trivially and badly mispredict real timings, so the
//! image generator ([`gen_image`]) lays down a smooth gradient plus a cheap
//! deterministic hash-noise term — the mix of low-frequency structure (which the
//! wavelet / DCT codecs exploit) and high-frequency detail (which stresses the
//! entropy coders) that real renders contain.
//!
//! # Per-codec chunk layout
//! The codecs consume one "line buffer" block: the scanlines of the block laid
//! out **scanline-interleaved**, each line being every channel's samples for
//! that `y` in header (alphabetical) order, encoded in XDR (little-endian) — the
//! exact bytes `OutputFile::write_pixels` hands to `Compressor::compress`
//! (see `bytes_per_line_table` + `offset_in_line_buffer_table` in `misc.rs`).
//! [`build_chunk`] reproduces that layout, and `setup` asserts the lossless
//! codecs round-trip byte-exact so a mis-built buffer cannot silently skew the
//! numbers.
//!
//! Run a quick pass:
//! ```text
//! cargo bench -p exr-core --bench codecs -- --warm-up-time 1 --measurement-time 2
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use exr_core::attr::{
    Channel, ChannelList, ChannelListAttribute, Compression, LevelMode, LevelRoundingMode,
    LineOrder, PixelType, TileDescription, TileDescriptionAttribute,
};
use exr_core::compressor::new_compressor;
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{Header, SCANLINE_IMAGE, TILED_IMAGE};
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::{InputFile, OutputFile};
use exr_core::tiled::{TiledInputFile, TiledOutputFile};
use imath_rs::{Half, V2f};

// ---------------------------------------------------------------------------
// Representative image content
// ---------------------------------------------------------------------------

/// Cheap deterministic hash → `[0,1)` noise (a finalizer-style integer hash).
/// Used to add high-frequency detail on top of the smooth gradient so the
/// entropy coders (PIZ Huffman, ZIP deflate) see realistic, non-degenerate
/// input rather than the trivially-compressible all-zero / pure-ramp cases.
#[inline]
fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

/// One channel's sample value at `(x, y)` for channel index `c`.
///
/// Smooth diagonal gradient (low frequency, exploited by DWA/PIZ) plus a small
/// noise term (high frequency, stresses the entropy stage). Kept in a sane HDR
/// range so HALF stays well-conditioned.
#[inline]
fn sample(c: usize, x: usize, y: usize, w: usize, h: usize) -> f32 {
    let gx = x as f32 / w as f32;
    let gy = y as f32 / h as f32;
    let grad = 0.15 + 0.7 * (gx * 0.6 + gy * 0.4) + 0.1 * (c as f32) * 0.25;
    let n = hash01((x as u32).wrapping_mul(73_856_093) ^ (y as u32).wrapping_mul(19_349_663) ^ (c as u32).wrapping_mul(83_492_791));
    (grad + 0.06 * (n - 0.5)).clamp(0.0, 8.0)
}

/// The benchmark channel set: R, G, B as HALF, A as FLOAT (a common RGBA layout
/// that exercises both the HALF and the FLOAT codec paths in one image).
fn bench_channels() -> ChannelList {
    let mut cl = ChannelList::new();
    cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    cl
}

/// Per-channel typed planes (R,G,B HALF; A FLOAT), in header (alphabetical:
/// A,B,G,R) order, raster-major over `w × h`.
struct ImagePlanes {
    w: usize,
    h: usize,
    /// (name, pixel_type, native-endian plane bytes) in header order.
    planes: Vec<(&'static str, PixelType, Vec<u8>)>,
}

/// Generate the representative `w × h` RGBA image once. Planes are emitted in
/// header (alphabetical) order so they line up with `ChannelList::iter`.
fn gen_image(w: usize, h: usize) -> ImagePlanes {
    gen_image_with(w, h, |c, x, y| sample(c, x, y, w, h))
}

/// [`gen_image`] with an arbitrary per-sample generator `f(channel, x, y)`.
fn gen_image_with(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> f32) -> ImagePlanes {
    // header order is alphabetical: A, B, G, R.
    let order: [(&'static str, usize, PixelType); 4] = [
        ("A", 3, PixelType::Float),
        ("B", 2, PixelType::Half),
        ("G", 1, PixelType::Half),
        ("R", 0, PixelType::Half),
    ];
    let mut planes = Vec::new();
    for (name, c, ty) in order {
        let mut bytes = Vec::with_capacity(w * h * exr_core::misc::pixel_type_size(ty).unwrap());
        for y in 0..h {
            for x in 0..w {
                let v = f(c, x, y);
                match ty {
                    PixelType::Half => bytes.extend_from_slice(&Half::from(v).bits().to_ne_bytes()),
                    PixelType::Float => bytes.extend_from_slice(&v.to_ne_bytes()),
                    PixelType::Uint | PixelType::NumPixelTypes => unreachable!(),
                }
            }
        }
        planes.push((name, ty, bytes));
    }
    ImagePlanes { w, h, planes }
}

// ---------------------------------------------------------------------------
// Per-codec chunk buffer (scanline-interleaved XDR)
// ---------------------------------------------------------------------------

/// Build one codec chunk: rows `0..rows` laid out scanline-interleaved, each row
/// being every channel's `w` samples for that `y` in header order, XDR (LE).
///
/// This is byte-identical to what `OutputFile::write_pixels` gathers into the
/// line buffer before calling `Compressor::compress` (see the module docs).
fn build_chunk(img: &ImagePlanes, rows: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..rows {
        for (_, ty, plane) in &img.planes {
            let sz = exr_core::misc::pixel_type_size(*ty).unwrap();
            let row = &plane[y * img.w * sz..(y + 1) * img.w * sz];
            match ty {
                PixelType::Half => {
                    // native u16 → XDR LE u16.
                    for px in row.as_chunks::<2>().0 {
                        let bits = u16::from_ne_bytes([px[0], px[1]]);
                        out.extend_from_slice(&bits.to_le_bytes());
                    }
                }
                PixelType::Float => {
                    for px in row.as_chunks::<4>().0 {
                        let bits = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
                        out.extend_from_slice(&bits.to_le_bytes());
                    }
                }
                PixelType::Uint | PixelType::NumPixelTypes => unreachable!(),
            }
        }
    }
    out
}

/// Header for a scanline image of `w × h` with the bench channels + `comp`.
fn scan_header(w: i32, h: i32, comp: Compression) -> Header {
    let mut h_ = Header::with_dimensions(
        w,
        h,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        comp,
    )
    .unwrap();
    h_.insert("channels", ChannelListAttribute::new(bench_channels())).unwrap();
    h_.set_type(SCANLINE_IMAGE).unwrap();
    h_
}

/// The codecs to profile, with their per-chunk block height (`numScanLines`).
/// NONE is included as the zero-cost baseline (gather/scatter only).
const CODECS: &[(&str, Compression, usize)] = &[
    ("NONE", Compression::NoCompression, 1),
    ("RLE", Compression::Rle, 1),
    ("ZIPS", Compression::Zips, 1),
    ("ZIP", Compression::Zip, 16),
    ("PIZ", Compression::Piz, 32),
    ("PXR24", Compression::Pxr24, 16),
    ("B44", Compression::B44, 32),
    ("B44A", Compression::B44a, 32),
    ("DWAA", Compression::Dwaa, 32),
    ("DWAB", Compression::Dwab, 256),
    ("HTJ2K256", Compression::Htj2k256, 256),
    ("HTJ2K32", Compression::Htj2k32, 32),
];

// ---------------------------------------------------------------------------
// Per-codec compress / uncompress benchmarks
// ---------------------------------------------------------------------------

fn bench_codecs(c: &mut Criterion) {
    // Image wide enough that a single block is a substantial, realistic chunk.
    // Height = 256 covers the tallest block (DWAB) in one chunk.
    const W: usize = 512;
    const H: usize = 256;
    let img = gen_image(W, H);

    let mut grp = c.benchmark_group("codec_compress");
    for &(name, comp, block) in CODECS {
        let rows = block.min(H);
        let chunk = build_chunk(&img, rows);
        let uncompressed_len = chunk.len() as u64;
        let header = scan_header(W as i32, rows as i32, comp);

        // Pre-flight: compress once, and for lossless codecs assert the round
        // trip is byte-exact so a mis-built chunk cannot skew the numbers.
        let compressed = {
            let mut cmp = new_compressor(comp, W * 14, &header).unwrap();
            match cmp.as_mut() {
                Some(cc) => cc.compress(&chunk, 0).unwrap().to_vec(),
                None => chunk.clone(),
            }
        };
        // Byte-exact round-trip holds only for the fully-lossless codecs over
            // this RGBA(+FLOAT alpha) chunk. PXR24 is lossless for HALF/UINT but
            // *lossy* for FLOAT (it stores 24 of the 32 mantissa+exponent bits), so
            // it does NOT round-trip the FLOAT "A" channel byte-exact — it is still
            // benched, just not asserted here. B44/DWA are lossy by design.
        let lossless = matches!(
            comp,
            Compression::NoCompression
                | Compression::Rle
                | Compression::Zips
                | Compression::Zip
                | Compression::Piz
        );
        if lossless {
            let mut cmp = new_compressor(comp, W * 14, &header).unwrap();
            if let Some(cc) = cmp.as_mut() {
                let back = cc.uncompress(&compressed, 0, chunk.len()).unwrap();
                assert_eq!(back, &chunk[..], "{name}: lossless round-trip mismatch — chunk layout is wrong");
            }
        }

        grp.throughput(Throughput::Bytes(uncompressed_len));
        grp.bench_with_input(BenchmarkId::new("compress", name), &chunk, |b, chunk| {
            let mut cmp = new_compressor(comp, W * 14, &header).unwrap();
            b.iter(|| match cmp.as_mut() {
                Some(cc) => {
                    let out = cc.compress(black_box(chunk), 0).unwrap();
                    black_box(out.len())
                }
                None => black_box(chunk.len()),
            });
        });

        grp.bench_with_input(BenchmarkId::new("uncompress", name), &compressed, |b, compressed| {
            let mut cmp = new_compressor(comp, W * 14, &header).unwrap();
            b.iter(|| match cmp.as_mut() {
                Some(cc) => {
                    let out = cc.uncompress(black_box(compressed), 0, uncompressed_len as usize).unwrap();
                    black_box(out.len())
                }
                None => black_box(compressed.len()),
            });
        });
    }
    grp.finish();
}

// ---------------------------------------------------------------------------
// Half <-> f32 conversion benchmarks
// ---------------------------------------------------------------------------

fn bench_half(c: &mut Criterion) {
    const N: usize = 1 << 16;
    let floats: Vec<f32> = (0..N).map(|i| sample(i & 3, i, i >> 8, 256, 256)).collect();
    let halves: Vec<Half> = floats.iter().map(|&f| Half::from(f)).collect();

    let mut grp = c.benchmark_group("half");
    grp.throughput(Throughput::Elements(N as u64));
    grp.bench_function("f32_to_half", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &f in black_box(&floats) {
                acc = acc.wrapping_add(Half::from(black_box(f)).bits() as u32);
            }
            black_box(acc)
        });
    });
    grp.bench_function("half_to_f32", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &h in black_box(&halves) {
                acc += f32::from(black_box(h));
            }
            black_box(acc)
        });
    });
    grp.finish();
}

// ---------------------------------------------------------------------------
// Scanline write + read round-trip
// ---------------------------------------------------------------------------

/// Interleaved `[R half][G half][B half][A f32]` buffer (12 bytes/pixel) +
/// matching FrameBuffer for a `w × h` image at origin (0,0).
fn interleaved_buffer(img: &ImagePlanes) -> (Vec<u8>, FrameBuffer) {
    let (w, h) = (img.w, img.h);
    // Channel byte offsets within a pixel, in header order A,B,G,R:
    // A f32 (4) @0, B half (2) @4, G half (2) @6, R half (2) @8 → 10 bytes/pixel.
    let pxb = 10usize;
    let row_bytes = pxb * w;
    let mut buf = vec![0u8; row_bytes * h];
    // offsets keyed by name
    let off = |name: &str| -> usize {
        match name {
            "A" => 0,
            "B" => 4,
            "G" => 6,
            "R" => 8,
            _ => unreachable!(),
        }
    };
    for (name, ty, plane) in &img.planes {
        let o = off(name);
        let sz = exr_core::misc::pixel_type_size(*ty).unwrap();
        for y in 0..h {
            for x in 0..w {
                let src = (y * w + x) * sz;
                let dst = y * row_bytes + x * pxb + o;
                buf[dst..dst + sz].copy_from_slice(&plane[src..src + sz]);
            }
        }
    }
    let mut fb = FrameBuffer::new();
    let mk = |ty: PixelType, o: usize| Slice {
        type_: ty,
        base_offset: o as i64,
        x_stride: pxb,
        y_stride: row_bytes,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    };
    fb.insert("A", mk(PixelType::Float, 0)).unwrap();
    fb.insert("B", mk(PixelType::Half, 4)).unwrap();
    fb.insert("G", mk(PixelType::Half, 6)).unwrap();
    fb.insert("R", mk(PixelType::Half, 8)).unwrap();
    (buf, fb)
}

fn write_scanline(header: &Header, fb: &FrameBuffer, pixels: &[u8], h: i32) -> Vec<u8> {
    let mut mem = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut mem, header).unwrap();
        of.set_frame_buffer(fb).unwrap();
        of.write_pixels(pixels, h).unwrap();
    }
    mem.into_data()
}

fn read_scanline(bytes: Vec<u8>, fb: &FrameBuffer, out: &mut [u8], h: i32) {
    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();
    inf.set_frame_buffer(fb).unwrap();
    inf.read_pixels(0, h - 1, out).unwrap();
}

fn bench_scanline_roundtrip(c: &mut Criterion) {
    let mut grp = c.benchmark_group("scanline_roundtrip");
    for &(w, h) in &[(256usize, 256usize), (512, 512)] {
        let img = gen_image(w, h);
        let (pixels, fb) = interleaved_buffer(&img);
        let bytes_total = (pixels.len()) as u64;

        // Profile two representative codecs end-to-end: ZIP (lossless, ubiquitous)
        // and DWAA (the heaviest lossy codec).
        for comp in [Compression::Zip, Compression::Dwaa] {
            let header = scan_header(w as i32, h as i32, comp);
            let label = format!("{}x{}/{:?}", w, h, comp);
            grp.throughput(Throughput::Bytes(bytes_total));
            grp.bench_function(BenchmarkId::new("write", &label), |b| {
                b.iter(|| black_box(write_scanline(&header, &fb, &pixels, h as i32).len()));
            });
            let file = write_scanline(&header, &fb, &pixels, h as i32);
            grp.bench_function(BenchmarkId::new("read", &label), |b| {
                let mut out = vec![0u8; pixels.len()];
                b.iter(|| {
                    read_scanline(black_box(file.clone()), &fb, &mut out, h as i32);
                    black_box(out.len())
                });
            });
        }
    }
    grp.finish();
}

// ---------------------------------------------------------------------------
// Tiled write + read round-trip
// ---------------------------------------------------------------------------

fn tiled_header(w: i32, h: i32, tile: u32, comp: Compression) -> Header {
    let mut h_ = Header::with_dimensions(
        w,
        h,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        comp,
    )
    .unwrap();
    h_.insert("channels", ChannelListAttribute::new(bench_channels())).unwrap();
    h_.set_type(TILED_IMAGE).unwrap();
    h_.insert(
        "tiles",
        TileDescriptionAttribute::new(TileDescription {
            x_size: tile,
            y_size: tile,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }),
    )
    .unwrap();
    h_
}

fn bench_tiled_roundtrip(c: &mut Criterion) {
    let mut grp = c.benchmark_group("tiled_roundtrip");
    for &(w, h) in &[(256usize, 256usize), (512, 512)] {
        let img = gen_image(w, h);
        let (pixels, fb) = interleaved_buffer(&img);
        let bytes_total = pixels.len() as u64;

        // Tiled bench uses ZIP (lossless, the ubiquitous tiled path). The tiled
        // DWAA tile-taller-than-the-32-row-block bug is FIXED (new_tile_compressor
        // now passes the tile height; verified end-to-end by
        // `tiled::input::tests::dwaa_tile_taller_than_dwa_block`). The per-codec
        // DWAA cost is captured in the `codec_compress` group, so the tiled
        // round-trip bench stays on ZIP simply because it's the most common path.
        {
            let comp = Compression::Zip;
            let header = tiled_header(w as i32, h as i32, 64, comp);
            let label = format!("{}x{}/{:?}", w, h, comp);

            // write closure
            let do_write = |header: &Header, fb: &FrameBuffer, pixels: &[u8]| -> Vec<u8> {
                let mut mem = MemOStream::new();
                {
                    let mut tof = TiledOutputFile::new(&mut mem, header).unwrap();
                    tof.set_frame_buffer(fb).unwrap();
                    let nx = tof.num_x_tiles(0).unwrap();
                    let ny = tof.num_y_tiles(0).unwrap();
                    tof.write_tiles(pixels, 0, nx - 1, 0, ny - 1, 0, 0).unwrap();
                }
                mem.into_data()
            };

            grp.throughput(Throughput::Bytes(bytes_total));
            grp.bench_function(BenchmarkId::new("write", &label), |b| {
                b.iter(|| black_box(do_write(&header, &fb, &pixels).len()));
            });

            let file = do_write(&header, &fb, &pixels);
            grp.bench_function(BenchmarkId::new("read", &label), |b| {
                let mut out = vec![0u8; pixels.len()];
                b.iter(|| {
                    let mut is = MemIStream::new(black_box(file.clone()));
                    let mut tif = TiledInputFile::new(&mut is).unwrap();
                    tif.set_frame_buffer(&fb).unwrap();
                    let nx = tif.num_x_tiles(0).unwrap();
                    let ny = tif.num_y_tiles(0).unwrap();
                    tif.read_tiles(0, nx - 1, 0, ny - 1, 0, 0, &mut out).unwrap();
                    black_box(out.len())
                });
            });
        }
    }
    grp.finish();
}

// ---------------------------------------------------------------------------
// PIZ scratch reuse on tiny-entropy chunks, apply_lut A/B
// ---------------------------------------------------------------------------

/// PIZ compress/uncompress of CONSTANT-colour chunks, 512 and 16 pixels wide.
/// Only the 16-wide chunk is tiny enough (`nBits <= 128`) for
/// `huffman::decompress` to take the generic scalar decoder — a 512-wide
/// constant chunk already has ~3000 bits of zero runs and takes FastHuf, like
/// the gradient chunks in `bench_codecs`.
fn bench_piz_const(c: &mut Criterion) {
    const ROWS: usize = 32;
    let mut grp = c.benchmark_group("phase4_piz_const");
    for w in [512usize, 16] {
        let img = gen_image_with(w, ROWS, |ch, _, _| 0.25 + 0.125 * ch as f32);
        let chunk = build_chunk(&img, ROWS);
        let header = scan_header(w as i32, ROWS as i32, Compression::Piz);
        let compressed = {
            let mut cmp = new_compressor(Compression::Piz, w * 14, &header).unwrap();
            cmp.as_mut().unwrap().compress(&chunk, 0).unwrap().to_vec()
        };
        assert!(compressed.len() < chunk.len(), "constant chunk must not be stored raw");
        grp.throughput(Throughput::Bytes(chunk.len() as u64));
        grp.bench_function(BenchmarkId::new("compress", w), |b| {
            let mut cmp = new_compressor(Compression::Piz, w * 14, &header).unwrap();
            b.iter(|| black_box(cmp.as_mut().unwrap().compress(black_box(&chunk), 0).unwrap().len()));
        });
        grp.bench_function(BenchmarkId::new("uncompress", w), |b| {
            let mut cmp = new_compressor(Compression::Piz, w * 14, &header).unwrap();
            b.iter(|| {
                black_box(cmp.as_mut().unwrap().uncompress(black_box(&compressed), 0, chunk.len()).unwrap().len())
            });
        });
    }
    grp.finish();
}

/// `apply_lut` variants for the A/B in `bench_apply_lut_ab` (the real one is
/// private to `compressor::piz`; `a_current` is a verbatim copy of it).
mod lut_variants {
    pub fn a_current(lut: &[u16], data: &mut [u16]) {
        let (chunks, rem) = data.as_chunks_mut::<8>();
        for chunk in chunks {
            let mut dvals = [0u16; 8];
            dvals.copy_from_slice(chunk);
            for v in &mut dvals {
                *v = lut[*v as usize];
            }
            chunk.copy_from_slice(&dvals);
        }
        for v in rem {
            *v = lut[*v as usize];
        }
    }
    /// Plain scalar loop.
    pub fn b_plain(lut: &[u16], data: &mut [u16]) {
        for v in data.iter_mut() {
            *v = lut[*v as usize];
        }
    }
    /// Plain loop over a fixed-size LUT: a `u16` index can never be out of
    /// range, so the per-element bounds check disappears.
    pub fn c_array(lut: &[u16], data: &mut [u16]) {
        let lut: &[u16; 65536] = lut.try_into().unwrap();
        for v in data.iter_mut() {
            *v = lut[*v as usize];
        }
    }
    /// 8-way chunks, in place, fixed-size LUT.
    pub fn d_chunk8_array(lut: &[u16], data: &mut [u16]) {
        let lut: &[u16; 65536] = lut.try_into().unwrap();
        let (chunks, rem) = data.as_chunks_mut::<8>();
        for c in chunks {
            for v in c.iter_mut() {
                *v = lut[*v as usize];
            }
        }
        for v in rem {
            *v = lut[*v as usize];
        }
    }
    /// Explicit 8-way unroll (all loads before all stores), fixed-size LUT.
    pub fn e_unroll8_array(lut: &[u16], data: &mut [u16]) {
        let lut: &[u16; 65536] = lut.try_into().unwrap();
        let (chunks, rem) = data.as_chunks_mut::<8>();
        for c in chunks {
            let (a, b, cc, d, e, f, g, h) = (
                lut[c[0] as usize],
                lut[c[1] as usize],
                lut[c[2] as usize],
                lut[c[3] as usize],
                lut[c[4] as usize],
                lut[c[5] as usize],
                lut[c[6] as usize],
                lut[c[7] as usize],
            );
            c.copy_from_slice(&[a, b, cc, d, e, f, g, h]);
        }
        for v in rem {
            *v = lut[*v as usize];
        }
    }
}

/// Median-of-`REPEATS` interleaved A/B of the `apply_lut` variants on four data
/// shapes. Runs only when the CLI filter contains `apply_lut_ab` (it is not a
/// criterion benchmark: it reports median + min/max over repeats itself).
fn bench_apply_lut_ab(_c: &mut Criterion) {
    if !std::env::args().any(|a| a.contains("apply_lut_ab")) {
        return;
    }
    type F = fn(&[u16], &mut [u16]);
    const REPEATS: usize = 21;
    let variants: [(&str, F); 5] = [
        ("a_current", lut_variants::a_current),
        ("b_plain", lut_variants::b_plain),
        ("c_array", lut_variants::c_array),
        ("d_chunk8_array", lut_variants::d_chunk8_array),
        ("e_unroll8_array", lut_variants::e_unroll8_array),
    ];
    // A forward-style LUT: ~1/4 of the 64K values "used", the others map to 0.
    let lut: Vec<u16> = {
        let mut k = 0u16;
        (0..65536u32)
            .map(|i| {
                if i == 0 || hash01(i * 7 + 1) < 0.25 {
                    let v = k;
                    k = k.wrapping_add(1);
                    v
                } else {
                    0
                }
            })
            .collect()
    };
    // (label, sample count, value generator)
    type Gen = fn(usize) -> u16;
    let shapes: [(&str, usize, Gen); 4] = [
        ("chunk 82k wide-values", 512 * 32 * 5, |i| (hash01(i as u32) * 65535.0) as u16),
        ("chunk 82k small-values", 512 * 32 * 5, |i| (hash01(i as u32) * 4000.0) as u16),
        ("tile 1.3M wide-values", 16384 * 16 * 5, |i| (hash01(i as u32) * 65535.0) as u16),
        ("tile 1.3M small-values", 16384 * 16 * 5, |i| (hash01(i as u32) * 4000.0) as u16),
    ];
    for (label, n, gen_v) in shapes {
        let src: Vec<u16> = (0..n).map(gen_v).collect();
        let mut work = vec![0u16; n];
        // Every variant must produce the same values as the current one.
        let mut reference = src.clone();
        lut_variants::a_current(&lut, &mut reference);
        for (name, f) in variants {
            work.copy_from_slice(&src);
            f(&lut, &mut work);
            assert_eq!(work, reference, "{name} diverges from apply_lut");
        }
        // Iterations per repeat: ~30M elements, so one repeat is ~10-30 ms.
        let iters = (30_000_000 / n).max(1);
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
        for _ in 0..REPEATS {
            for (vi, (_, f)) in variants.iter().enumerate() {
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    work.copy_from_slice(black_box(&src));
                    f(black_box(&lut), black_box(&mut work));
                }
                black_box(&work);
                times[vi].push(t.elapsed().as_secs_f64() * 1e9 / (iters * n) as f64);
            }
        }
        println!("apply_lut A/B [{label}] n={n} iters={iters} repeats={REPEATS} (ns/element incl. common copy)");
        let base_med = median(&mut times[0].clone());
        for (vi, (name, _)) in variants.iter().enumerate() {
            let mut t = times[vi].clone();
            let med = median(&mut t);
            let (mn, mx) = (t[0], t[t.len() - 1]);
            println!(
                "  {name:<16} median {med:.4}  min {mn:.4}  max {mx:.4}  spread {:.1}%  vs a_current {:+.1}%",
                (mx - mn) / med * 100.0,
                (med / base_med - 1.0) * 100.0
            );
        }
    }
}

/// Median of `v` (sorts it in place).
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

criterion_group!(
    benches,
    bench_codecs,
    bench_half,
    bench_scanline_roundtrip,
    bench_tiled_roundtrip,
    bench_piz_const,
    bench_apply_lut_ab
);
criterion_main!(benches);
