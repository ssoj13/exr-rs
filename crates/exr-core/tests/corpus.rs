//! **Corpus verification runner** (OpenEXR `src/test` images + official
//! `openexr-images`).
//!
//! The real-world confidence gate: run REAL C++-written EXR files through *our*
//! readers and assert that
//!
//! * **valid** files decode without panicking and their pixels come through
//!   (and, for a curated lossless subset, round-trip write→read bit-exact);
//! * **known-invalid** fixtures (corrupt / fuzzed / bad-magic) error *gracefully*
//!   — our reader returns `Err`, never panics;
//! * a residual set whose intent is genuinely ambiguous after reading the C++
//!   tests survives with the weakest assertion: **no panic** (Ok or Err both OK).
//!
//! A panic on ANY corpus file is treated as a **real bug** in our readers
//! (a stray `unwrap`/index/slice that should have been an `Err`): the runner
//! records it, prints it, and the test fails at the end — it does *not* mask it.
//!
//! # Corpora (two roots, scanned together)
//! Both come from the test image set (`exr-test-data`); the test runs with `--features test-data`.
//! 1. **Vendored**: OpenEXR's `src/test` images, v3.4.15 (overridable via
//!    `EXR_CORPUS_DIR`).  Mix of valid codec images + deliberately corrupt
//!    fixtures; each file's expected outcome comes from the C++ test that
//!    consumes it (see [`expected_outcome`]).
//! 2. **Official `openexr-images`**, e38ffb0.  **Every** `.exr` here is a VALID real image (the corrupt
//!    fuzzer fixtures in its `Damaged/` dir carry no `.exr` extension and are
//!    therefore not collected) → expected outcome **Valid**.  These exercise
//!    paths the vendored set never touches: multi-resolution MIP/RIP tiled,
//!    single-part **multiView**, **chromaticities**, multipart (incl. deep
//!    multipart), and **luminance-chroma subsampled** channels (`xSampling` /
//!    `ySampling` != 1) — see the planar frame-buffer builder below.
//!
//! # Subsampling (the `openexr-images` edge case)
//! Six openexr-images files store chroma at half resolution
//! (`BY`/`RY` channels `xSampling==ySampling==2`, `Y` at 1×1):
//! `Chromaticities/{Rec709_YC,XYZ_YC}.exr`,
//! `LuminanceChroma/{CrissyField,Flowers,MtTamNorth,StarField}.exr`.
//! The interleaved single-record frame buffer used for the (all-1×1) vendored
//! corpus cannot describe these.  The openexr-images path therefore builds a
//! **planar, per-channel** frame buffer: each channel owns a region sized by its
//! *subsampled* extent (`divp(w,xs) * divp(h,ys) * pixelSize`), exactly the
//! addressing our scanline/tiled readers use
//! (`base + divp(y,ys)*yStride + divp(x,xs)*xStride`).  This decodes the chroma
//! channels at their true stored resolution — no guessing.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use exr_core::attr::{Compression, PixelType};
use exr_core::container::version;
use exr_core::deep::frame_buffer::DeepSlice;
use exr_core::deep::{DeepFrameBuffer, DeepScanLineInputFile, DeepTiledInputFile};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{self as hdr, Header};
use exr_core::multipart::{DeepScanLineInputPartReader, DeepTiledInputPartReader};
use exr_core::scanline::{InputFile, OutputFile};
use exr_core::tiled::{TiledInputFile, TiledInputPartReader, TiledOutputFile};
use exr_core::{MemIStream, MemOStream, MultiPartInputFile, StdIfStream};
use imath_rs::Box2i;

// ---------------------------------------------------------------------------
// Small reader-agnostic traits (container caching):
// `MultiPartInputFile::{tiled_input_part, deep_scanline_input_part,
// deep_tiled_input_part}` return borrowing views over cached per-part state
// (`TiledInputPartReader` / `DeepScanLineInputPartReader` /
// `DeepTiledInputPartReader`) instead of a freshly-built owned `TiledInputFile`
// / `DeepScanLineInputFile` / `DeepTiledInputFile`. These traits let
// `decode_flat_tiled_part` / `decode_deep_scanline` / `decode_deep_tiled` stay
// ONE function each, driven by either the standalone reader or the cached
// multipart view, exactly the shape `scanline::GenericInputReader` already
// uses for the flat scanline path.
// ---------------------------------------------------------------------------

/// Shared surface of `TiledInputFile` and `TiledInputPartReader` needed by
/// [`decode_flat_tiled_part`].
trait TiledReaderLike {
    fn header(&self) -> &Header;
    fn num_x_levels(&self) -> i32;
    fn num_y_levels(&self) -> i32;
    fn is_valid_level(&self, lx: i32, ly: i32) -> bool;
    fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i, exr_core::ExrError>;
    fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<(), exr_core::ExrError>;
    fn num_x_tiles(&self, lx: i32) -> Result<i32, exr_core::ExrError>;
    fn num_y_tiles(&self, ly: i32) -> Result<i32, exr_core::ExrError>;
    #[allow(clippy::too_many_arguments)]
    fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        out: &mut [u8],
    ) -> Result<(), exr_core::ExrError>;
}

macro_rules! impl_tiled_reader_like {
    ($t:ty) => {
        impl TiledReaderLike for $t {
            fn header(&self) -> &Header {
                Self::header(self)
            }
            fn num_x_levels(&self) -> i32 {
                Self::num_x_levels(self)
            }
            fn num_y_levels(&self) -> i32 {
                Self::num_y_levels(self)
            }
            fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
                Self::is_valid_level(self, lx, ly)
            }
            fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i, exr_core::ExrError> {
                Self::data_window_for_level(self, lx, ly)
            }
            fn set_frame_buffer(
                &mut self,
                frame_buffer: &FrameBuffer,
            ) -> Result<(), exr_core::ExrError> {
                Self::set_frame_buffer(self, frame_buffer)
            }
            fn num_x_tiles(&self, lx: i32) -> Result<i32, exr_core::ExrError> {
                Self::num_x_tiles(self, lx)
            }
            fn num_y_tiles(&self, ly: i32) -> Result<i32, exr_core::ExrError> {
                Self::num_y_tiles(self, ly)
            }
            fn read_tiles(
                &mut self,
                dx1: i32,
                dx2: i32,
                dy1: i32,
                dy2: i32,
                lx: i32,
                ly: i32,
                out: &mut [u8],
            ) -> Result<(), exr_core::ExrError> {
                Self::read_tiles(self, dx1, dx2, dy1, dy2, lx, ly, out)
            }
        }
    };
}

impl_tiled_reader_like!(TiledInputFile<'_>);
impl_tiled_reader_like!(TiledInputPartReader<'_>);

/// Shared surface of `DeepScanLineInputFile` and `DeepScanLineInputPartReader`
/// needed by [`decode_deep_scanline`].
trait DeepScanLineReaderLike {
    fn header(&self) -> &Header;
    fn set_frame_buffer(&mut self, fb: &DeepFrameBuffer) -> Result<(), exr_core::ExrError>;
    fn read_pixel_sample_counts(
        &mut self,
        s1: i32,
        s2: i32,
        counts: &mut [u8],
    ) -> Result<(), exr_core::ExrError>;
    fn read_pixels(
        &mut self,
        s1: i32,
        s2: i32,
        counts: &[u8],
        data: &mut [u8],
    ) -> Result<(), exr_core::ExrError>;
}

macro_rules! impl_deep_scanline_reader_like {
    ($t:ty) => {
        impl DeepScanLineReaderLike for $t {
            fn header(&self) -> &Header {
                Self::header(self)
            }
            fn set_frame_buffer(&mut self, fb: &DeepFrameBuffer) -> Result<(), exr_core::ExrError> {
                Self::set_frame_buffer(self, fb)
            }
            fn read_pixel_sample_counts(
                &mut self,
                s1: i32,
                s2: i32,
                counts: &mut [u8],
            ) -> Result<(), exr_core::ExrError> {
                Self::read_pixel_sample_counts(self, s1, s2, counts)
            }
            fn read_pixels(
                &mut self,
                s1: i32,
                s2: i32,
                counts: &[u8],
                data: &mut [u8],
            ) -> Result<(), exr_core::ExrError> {
                Self::read_pixels(self, s1, s2, counts, data)
            }
        }
    };
}

impl_deep_scanline_reader_like!(DeepScanLineInputFile<'_>);
impl_deep_scanline_reader_like!(DeepScanLineInputPartReader<'_>);

/// Shared surface of `DeepTiledInputFile` and `DeepTiledInputPartReader`
/// needed by [`decode_deep_tiled`].
trait DeepTiledReaderLike {
    fn header(&self) -> &Header;
    fn num_x_levels(&self) -> i32;
    fn num_y_levels(&self) -> i32;
    fn is_valid_level(&self, lx: i32, ly: i32) -> bool;
    fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i, exr_core::ExrError>;
    fn num_x_tiles(&self, lx: i32) -> Result<i32, exr_core::ExrError>;
    fn num_y_tiles(&self, ly: i32) -> Result<i32, exr_core::ExrError>;
    fn set_frame_buffer(&mut self, fb: &DeepFrameBuffer) -> Result<(), exr_core::ExrError>;
    #[allow(clippy::too_many_arguments)]
    fn read_pixel_sample_counts(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        counts: &mut [u8],
    ) -> Result<(), exr_core::ExrError>;
    #[allow(clippy::too_many_arguments)]
    fn read_tiles(
        &mut self,
        dx1: i32,
        dx2: i32,
        dy1: i32,
        dy2: i32,
        lx: i32,
        ly: i32,
        counts: &[u8],
        data: &mut [u8],
    ) -> Result<(), exr_core::ExrError>;
}

macro_rules! impl_deep_tiled_reader_like {
    ($t:ty) => {
        impl DeepTiledReaderLike for $t {
            fn header(&self) -> &Header {
                Self::header(self)
            }
            fn num_x_levels(&self) -> i32 {
                Self::num_x_levels(self)
            }
            fn num_y_levels(&self) -> i32 {
                Self::num_y_levels(self)
            }
            fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
                Self::is_valid_level(self, lx, ly)
            }
            fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i, exr_core::ExrError> {
                Self::data_window_for_level(self, lx, ly)
            }
            fn num_x_tiles(&self, lx: i32) -> Result<i32, exr_core::ExrError> {
                Self::num_x_tiles(self, lx)
            }
            fn num_y_tiles(&self, ly: i32) -> Result<i32, exr_core::ExrError> {
                Self::num_y_tiles(self, ly)
            }
            fn set_frame_buffer(&mut self, fb: &DeepFrameBuffer) -> Result<(), exr_core::ExrError> {
                Self::set_frame_buffer(self, fb)
            }
            fn read_pixel_sample_counts(
                &mut self,
                dx1: i32,
                dx2: i32,
                dy1: i32,
                dy2: i32,
                lx: i32,
                ly: i32,
                counts: &mut [u8],
            ) -> Result<(), exr_core::ExrError> {
                Self::read_pixel_sample_counts(self, dx1, dx2, dy1, dy2, lx, ly, counts)
            }
            fn read_tiles(
                &mut self,
                dx1: i32,
                dx2: i32,
                dy1: i32,
                dy2: i32,
                lx: i32,
                ly: i32,
                counts: &[u8],
                data: &mut [u8],
            ) -> Result<(), exr_core::ExrError> {
                Self::read_tiles(self, dx1, dx2, dy1, dy2, lx, ly, counts, data)
            }
        }
    };
}

impl_deep_tiled_reader_like!(DeepTiledInputFile<'_>);
impl_deep_tiled_reader_like!(DeepTiledInputPartReader<'_>);

mod support;

// ---------------------------------------------------------------------------
// `divp` — the C++ ceil-style sample-index division used everywhere a
// subsampled coordinate is mapped to a stored sample.  `divp(x, n)` is the
// number of the highest sample whose coordinate is <= x, mirroring
// `Imf::divp` (round toward -inf for negatives).  Used to size planar channel
// regions so chroma channels land at their true (subsampled) resolution.
// ---------------------------------------------------------------------------
fn divp(x: i32, y: i32) -> i32 {
    if x >= 0 {
        if y >= 0 { x / y } else { -(x / -y) }
    } else if y >= 0 {
        -((-x + y - 1) / y)
    } else {
        (-x + -y - 1) / -y
    }
}

/// Count of stored samples of a channel with sampling `s` across the closed
/// coordinate range `[min, max]` — `divp(max,s) - divp(min,s) + 1`, the C++
/// `numSamples` count used to size a planar per-channel buffer.
fn num_samples(s: i32, min: i32, max: i32) -> usize {
    (divp(max, s) - divp(min, s) + 1).max(0) as usize
}

// ---------------------------------------------------------------------------
// Expected outcome model
// ---------------------------------------------------------------------------

/// The expected result of running one corpus file through our readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// A valid, decodable image: our reader must return `Ok` and read pixels.
    Valid,
    /// A corrupt / invalid fixture: our reader must return `Err` (graceful),
    /// and crucially must NOT panic.
    Invalid,
    /// A file C++ only ever *opens* (header + chunk/offset-table parse) and
    /// never decodes — our reader must **open** it successfully (the C++
    /// `exr_start_read` / `isOpenExrFile` contract), but a subsequent pixel
    /// read may fail gracefully because the on-disk pixel data is absent.
    OpenOnly,
    /// Intent ambiguous (or behaviour intentionally diverges from C++): the
    /// only guarantee is *no panic* — `Ok` or `Err` are both acceptable.
    NoPanicOnly,
}

/// The per-file expected-outcome table for the **vendored** corpus, justified
/// by the C++ test that consumes each filename.  Returns `None` for files not
/// explicitly listed (those get the safe [`Outcome::NoPanicOnly`] default and
/// are flagged in the summary).
///
/// Justifications (all under `openexr/src/test/OpenEXRTest/`):
/// * `comp_*.exr` valid codec images — `testSampleImages.cpp` /
///   `testCompression*.cpp` read them as ordinary images.
/// * `comp_short_decode_piz` / `comp_invalid_unknown` / `comp_early_eof_piz` —
///   `testMalformedImages.cpp` asserts `readImage` THROWS (BaseExc / IoExc /
///   InputExc): corrupt → **Invalid**.
/// * `comp_bad_*bounds_*` — `testMalformedImages.cpp` notes their header is
///   *valid*; only the test's own framebuffer-pointer math is bad, so C++ reads
///   ONLY the header (`readBadBoundsImage`) and never the pixels.  Our reader may
///   succeed or error on the pixel read → **NoPanicOnly** (must not panic).
/// * `invalid.exr` — `testMagic.cpp`: `isImfMagic` is false (not an EXR file) →
///   our reader rejects on bad magic → **Invalid**.
/// * `invalid_shared_attrs_multipart.exr` — `testMultiPartSharedAttributes.cpp`
///   asserts `MultiPartInputFile` THROWS InputExc → **Invalid**.
/// * `{scanline,tiled}_with_*_type.exr` — `testBadTypeAttributes.cpp`: VALID
///   files whose `type` attribute is wrong; C++ *fixes* it on read.  We do not
///   port that fixup (our readers reject a mismatched `type`), so our outcome is
///   a graceful `Err` → **NoPanicOnly** (the file is not corrupt; must not panic).
/// * `tiled.exr`, `v1.7.test.tiled.exr` — tiled sample images (`testMagic.cpp`,
///   `testBackwardCompatibility.cpp`) → **Valid** (tiled) / **OpenOnly** (stub).
/// * `v1.7.test.{1,interleaved,planar}.exr`, `lineOrder_*`, `test_native*`,
///   `comp_none/rle/zip/zips/piz` — valid scanline sample images
///   (`testBackwardCompatibility.cpp`, `testSampleImages.cpp`,
///   `testNativeFormat.cpp`) → **Valid** (scanline).
/// * `*.deep.exr` — real deep-scanline sample images → **Valid** (deep).
fn expected_outcome(file_name: &str) -> Option<Outcome> {
    use Outcome::*;
    let o = match file_name {
        // --- valid scanline codec images (testSampleImages / testCompression) ---
        "comp_none.exr" | "comp_rle.exr" | "comp_zip.exr" | "comp_zips.exr"
        | "comp_piz.exr" | "comp_b44.exr" | "comp_b44_piz.exr" | "comp_dwaa_v1.exr"
        | "comp_dwaa_v2.exr" | "comp_dwaa_piz.exr" | "comp_dwab_v1.exr"
        | "comp_dwab_v2.exr" | "comp_dwab_piz.exr" => Valid,

        // --- valid scanline sample images ---
        "lineOrder_increasing.exr" | "lineOrder_decreasing.exr" => Valid,
        "test_native1.exr" | "test_native2.exr" => Valid,
        "v1.7.test.1.exr" | "v1.7.test.interleaved.exr" | "v1.7.test.planar.exr" => Valid,

        // --- valid tiled sample image (full 255 KB image) ---
        "v1.7.test.tiled.exr" => Valid,

        // --- open-only tiled stub (480-byte truncated tiled file) -------------
        // Header + tile flag parse fine (ONE_LEVEL, 20x20 tiles, ZIP), but its
        // stored tile-offset table is bogus/truncated: only tile (0,0) lies
        // inside the file.  Every entry is NON-zero, so neither C++
        // `ImfTileOffsets::readFrom` nor our 1:1 port triggers reconstruction.
        // The ONLY C++ tests that consume it just OPEN it (`testMagic.cpp`
        // `isOpenExrFile`/`isTiledOpenExrFile`; `OpenEXRCoreTest read.cpp
        // testOpenTiles` does `exr_start_read` + `exr_finish`).  So OpenOnly:
        // our `TiledInputFile::new` must succeed, a pixel read errors gracefully.
        "tiled.exr" => OpenOnly,

        // --- real deep-scanline sample images ---
        "11.deep.exr" | "42.deep.exr" | "64.deep.exr" | "multivariate.deep.exr"
        | "objectid.deep.exr" => Valid,

        // --- known-invalid (corrupt / fuzzed) — testMalformedImages.cpp ---
        "comp_short_decode_piz.exr" | "comp_invalid_unknown.exr"
        | "comp_early_eof_piz.exr" => Invalid,
        // --- bad magic — testMagic.cpp ---
        "invalid.exr" => Invalid,
        // --- malformed multipart — testMultiPartSharedAttributes.cpp ---
        "invalid_shared_attrs_multipart.exr" => Invalid,

        // --- valid-but-headers-only; we must not panic (testMalformedImages) ---
        "comp_bad_bounds_piz.exr" | "comp_bad_pos_bounds_piz.exr"
        | "comp_bad_pos_bounds_pxr24.exr" | "comp_bad_neg_bounds_pxr24.exr" => NoPanicOnly,

        // --- type-attr-fixup files: valid in C++, we reject gracefully ---
        "scanline_with_tiledimage_type.exr" | "scanline_with_deeptiled_type.exr"
        | "scanline_with_deepscanline_type.exr" | "tiled_with_scanlineimage_type.exr"
        | "tiled_with_deepscanline_type.exr" | "tiled_with_deeptile_type.exr" => NoPanicOnly,

        _ => return None,
    };
    Some(o)
}

// ---------------------------------------------------------------------------
// File-kind detection (mirrors the C++ generic-open dispatch)
// ---------------------------------------------------------------------------

/// The on-disk kind of a single-part file, derived from the version flags plus
/// the first header's `type` attribute — the same signals C++'s generic open
/// uses to pick a reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Scanline,
    Tiled,
    DeepScanline,
    DeepTiled,
    MultiPart,
    /// Magic/flags rejected before any header could be classified.
    Unreadable,
}

/// Peek the magic + version word + first header `type` to classify the file
/// without committing to a reader.  Returns `Unreadable` for bad magic / unknown
/// flags (the file will then be exercised through the reader that errors on it).
fn detect_kind(path: &Path) -> Kind {
    let mut is = match StdIfStream::open(path.to_str().unwrap_or_default()) {
        Ok(s) => s,
        Err(_) => return Kind::Unreadable,
    };
    let (_header, version_word) = match Header::read_from(&mut is) {
        Ok(v) => v,
        Err(_) => return Kind::Unreadable,
    };
    let flags = version::get_flags(version_word as u32);

    if version::is_multi_part(flags) {
        return Kind::MultiPart;
    }
    if version::is_non_image(flags) {
        return if version::is_tiled(flags) {
            Kind::DeepTiled
        } else {
            Kind::DeepScanline
        };
    }
    if version::is_tiled(flags) {
        Kind::Tiled
    } else {
        Kind::Scanline
    }
}

// ---------------------------------------------------------------------------
// Generic frame-buffer construction (any channel layout)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Planar (per-channel) frame buffer — handles subsampling
//
// The vendored (all-1×1) corpus and the openexr-images corpus both use this
// single planar builder: for full-resolution channels it degenerates to a
// per-channel contiguous plane, and for subsampled chroma channels it sizes
// each region by its true (divp-reduced) extent.  One layout path, no special
// cases.
// ---------------------------------------------------------------------------

/// One channel's planar placement: its byte region `[base, base+len)` inside the
/// combined buffer, its pixel type and (possibly subsampled) sampling factors.
struct PlanarChan {
    name: String,
    type_: PixelType,
    x_sampling: i32,
    y_sampling: i32,
    base: usize,
    x_stride: usize,
    y_stride: usize,
}

/// Compute a **planar** layout for the data window `dw`: each channel owns a
/// contiguous region sized by its *subsampled* extent
/// (`numSamples(xs,minX,maxX) * numSamples(ys,minY,maxY) * pixelSize`).  Returns
/// the per-channel placements and the total combined buffer length.  This is the
/// only layout that correctly describes luminance-chroma subsampled files.
fn planar_layout(
    header: &Header,
    dw: Box2i,
) -> Result<(Vec<PlanarChan>, usize), exr_core::ExrError> {
    let channels = header.channels()?;
    let mut chans = Vec::new();
    let mut total = 0usize;
    for (name, ch) in channels.iter() {
        let cols = num_samples(ch.x_sampling, dw.min.x, dw.max.x);
        let rows = num_samples(ch.y_sampling, dw.min.y, dw.max.y);
        let px = exr_core::misc::pixel_type_size(ch.type_)?;
        let x_stride = px;
        let y_stride = cols * px;
        let len = rows * y_stride;
        chans.push(PlanarChan {
            name: name.to_string(),
            type_: ch.type_,
            x_sampling: ch.x_sampling,
            y_sampling: ch.y_sampling,
            base: total,
            x_stride,
            y_stride,
        });
        total += len.max(px); // never zero-size a present channel
    }
    Ok((chans, total))
}

/// Build a planar [`FrameBuffer`] from a [`planar_layout`].  Each channel's
/// `Slice::make` folds the data-window origin into the base offset exactly as
/// C++ does (`base - divp(origin.x,xs)*xStride - divp(origin.y,ys)*yStride`),
/// so the reader scatters subsampled samples into the right region.
fn make_planar_frame_buffer(
    dw: Box2i,
    chans: &[PlanarChan],
) -> Result<FrameBuffer, exr_core::ExrError> {
    let w = dw.max.x as i64 - dw.min.x as i64 + 1;
    let mut fb = FrameBuffer::new();
    for c in chans {
        let slice = Slice::make(
            c.type_,
            c.base,
            dw.min,
            w,
            c.x_stride,
            c.y_stride,
            c.x_sampling,
            c.y_sampling,
            0.0,
            false,
            false,
        )?;
        fb.insert(c.name.clone(), slice)?;
    }
    Ok(fb)
}

// ---------------------------------------------------------------------------
// Per-kind decode + (where supported) round-trip
// ---------------------------------------------------------------------------

/// The lossless codecs we round-trip (write→read must be byte-exact).  Lossy
/// codecs (PXR24/B44/B44A/DWAA/DWAB) are decode-only.
fn is_lossless(c: Compression) -> bool {
    matches!(
        c,
        Compression::NoCompression
            | Compression::Rle
            | Compression::Zip
            | Compression::Zips
            | Compression::Piz
    )
}

/// What a reader did with a file: whether it round-tripped (write→read→compare
/// bit-exact) in addition to decoding.  Recorded in the per-file summary so the
/// gate is auditable.
#[derive(Clone, Copy, Debug, Default)]
struct Report {
    round_tripped: bool,
}

/// Decode a single-part **scanline** file via the planar (subsampling-aware)
/// frame buffer, reading the whole data window.  When `round_trip` is set and
/// the file is lossless + full-resolution, also write→read→compare bit-exact.
fn run_scanline(path: &Path, round_trip: bool) -> Result<Report, exr_core::ExrError> {
    let mut is = StdIfStream::open(path.to_str().unwrap_or_default())?;
    let mut inf = InputFile::new(&mut is)?;
    let header = inf.header().clone();
    let dw = header.data_window()?;
    let (chans, total) = planar_layout(&header, dw)?;

    let mut out = vec![0u8; total];
    let fb = make_planar_frame_buffer(dw, &chans)?;
    inf.set_frame_buffer(&fb)?;
    inf.read_pixels(dw.min.y, dw.max.y, &mut out)?;
    drop(inf);

    let comp = header.compression()?;
    let full_res = chans.iter().all(|c| c.x_sampling == 1 && c.y_sampling == 1);
    if !round_trip || !is_lossless(comp) || !full_res {
        return Ok(Report::default());
    }

    let h = (dw.max.y as i64 - dw.min.y as i64 + 1) as i32;
    let mut os = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut os, &header)?;
        let wfb = make_planar_frame_buffer(dw, &chans)?;
        of.set_frame_buffer(&wfb)?;
        of.write_pixels(&out, h)?;
    }
    let mut ris = MemIStream::new(os.data().to_vec());
    let mut rinf = InputFile::new(&mut ris)?;
    let mut back = vec![0u8; total];
    let rfb = make_planar_frame_buffer(dw, &chans)?;
    rinf.set_frame_buffer(&rfb)?;
    rinf.read_pixels(dw.min.y, dw.max.y, &mut back)?;
    if back != out {
        return Err(exr_core::ExrError::Input(format!(
            "round-trip mismatch for {}",
            path.display()
        )));
    }
    Ok(Report { round_tripped: true })
}

/// Decode a single-part **tiled** file: read every tile of every level via the
/// planar frame buffer.  When `round_trip` is set and the file is lossless +
/// full-resolution, write all levels back and compare bit-exact.
fn run_tiled(path: &Path, round_trip: bool) -> Result<Report, exr_core::ExrError> {
    let mut is = StdIfStream::open(path.to_str().unwrap_or_default())?;
    let mut tif = TiledInputFile::new(&mut is)?;
    let header = tif.header().clone();
    let comp = tif.compression()?;

    // Probe full-res via the level-0 channels.
    let full_res = header.channels()?.iter().all(|(_, c)| c.x_sampling == 1 && c.y_sampling == 1);
    let do_rt = round_trip && is_lossless(comp) && full_res;

    // (lx, ly, level data window, planar layout, decoded bytes) per valid level.
    type Level = (i32, i32, Box2i, Vec<PlanarChan>, Vec<u8>);
    let mut levels: Vec<Level> = Vec::new();
    for lx in 0..tif.num_x_levels() {
        for ly in 0..tif.num_y_levels() {
            if !tif.is_valid_level(lx, ly) {
                continue;
            }
            let lvl = tif.data_window_for_level(lx, ly)?;
            let (chans, total) = planar_layout(&header, lvl)?;
            let mut buf = vec![0u8; total];
            let fb = make_planar_frame_buffer(lvl, &chans)?;
            tif.set_frame_buffer(&fb)?;
            let nx = tif.num_x_tiles(lx)?;
            let ny = tif.num_y_tiles(ly)?;
            tif.read_tiles(0, nx - 1, 0, ny - 1, lx, ly, &mut buf)?;
            levels.push((lx, ly, lvl, chans, buf));
        }
    }
    drop(tif);

    if !do_rt {
        return Ok(Report::default());
    }

    let mut os = MemOStream::new();
    {
        let mut tof = TiledOutputFile::new(&mut os, &header)?;
        for (lx, ly, lvl, chans, buf) in &levels {
            let nx = tof.num_x_tiles(*lx)?;
            let ny = tof.num_y_tiles(*ly)?;
            let fb = make_planar_frame_buffer(*lvl, chans)?;
            tof.set_frame_buffer(&fb)?;
            tof.write_tiles(buf, 0, nx - 1, 0, ny - 1, *lx, *ly)?;
        }
    }
    let mut ris = MemIStream::new(os.data().to_vec());
    let mut rtif = TiledInputFile::new(&mut ris)?;
    for (lx, ly, lvl, chans, want) in &levels {
        let mut back = vec![0u8; want.len()];
        let fb = make_planar_frame_buffer(*lvl, chans)?;
        rtif.set_frame_buffer(&fb)?;
        let nx = rtif.num_x_tiles(*lx)?;
        let ny = rtif.num_y_tiles(*ly)?;
        rtif.read_tiles(0, nx - 1, 0, ny - 1, *lx, *ly, &mut back)?;
        if &back != want {
            return Err(exr_core::ExrError::Input(format!(
                "tiled round-trip mismatch (level {lx},{ly}) for {}",
                path.display()
            )));
        }
    }
    Ok(Report { round_tripped: true })
}

/// Build a deep frame buffer over ONE flat sample heap (the C++/our "flat
/// model"): a buffer-relative sample-count slice plus one [`DeepSlice`] per
/// channel whose `base_offset` is the channel's byte origin within the combined
/// heap.  Returns `(frame_buffer, per_channel_bases)`.
fn deep_frame_buffer(
    chan: &[(String, PixelType)],
    total_samples: u64,
    grid_w: i64,
    sc_x: usize,
    sc_y: usize,
) -> Result<(DeepFrameBuffer, Vec<i64>), exr_core::ExrError> {
    let mut fb = DeepFrameBuffer::new();
    fb.insert_sample_count_slice(Slice::new(
        PixelType::Uint,
        0,
        sc_x,
        sc_y,
        1,
        1,
        0.0,
        false,
        false,
    )?)?;
    let mut bases = Vec::with_capacity(chan.len());
    let mut base = 0i64;
    for (name, t) in chan {
        bases.push(base);
        fb.insert(
            name.clone(),
            DeepSlice::new(*t, base, 8, 8 * grid_w as usize, exr_core::misc::pixel_type_size(*t)?, 1, 1, 0.0, false, false)?,
        )?;
        base += total_samples as i64 * exr_core::misc::pixel_type_size(*t)? as i64;
    }
    Ok((fb, bases))
}

/// Sum every u32 sample-count in `counts`.
fn total_samples(counts: &[u8]) -> u64 {
    (0..counts.len() / 4)
        .map(|i| u32::from_le_bytes(counts[i * 4..i * 4 + 4].try_into().unwrap_or([0; 4])) as u64)
        .sum()
}

/// Decode a **deep-scanline** reader (single-part or a multipart part): read the
/// per-pixel sample counts then the deep samples for the whole data window.
fn decode_deep_scanline<T: DeepScanLineReaderLike>(inf: &mut T) -> Result<(), exr_core::ExrError> {
    let header = inf.header().clone();
    let dw = header.data_window()?;
    let chan: Vec<(String, PixelType)> = header
        .channels()?
        .iter()
        .map(|(n, c)| (n.to_string(), c.type_))
        .collect();

    let abs_w = (dw.max.x.max(0) + 1) as i64;
    let abs_h = (dw.max.y.max(0) + 1) as usize;
    let sc_x = 4usize;
    let sc_y = abs_w as usize * sc_x;
    let mut counts = vec![0u8; abs_h * sc_y];

    let (cfb, _) = deep_frame_buffer(&[], 0, abs_w, sc_x, sc_y)?;
    inf.set_frame_buffer(&cfb)?;
    inf.read_pixel_sample_counts(dw.min.y, dw.max.y, &mut counts)?;

    let total = total_samples(&counts);
    let (dfb, bases) = deep_frame_buffer(&chan, total, abs_w, sc_x, sc_y)?;
    let heap_len = bases
        .last()
        .map(|&b| b as usize + total as usize * chan.last().map(|(_, t)| exr_core::misc::pixel_type_size(*t).unwrap_or(0)).unwrap_or(0))
        .unwrap_or(0);
    let mut heap = vec![0u8; heap_len];
    inf.set_frame_buffer(&dfb)?;
    inf.read_pixels(dw.min.y, dw.max.y, &counts, &mut heap)?;
    Ok(())
}

/// Decode a **deep-tiled** reader (single-part or a multipart part): every tile
/// of every level, addressed in level-0 grid raster order.
fn decode_deep_tiled<T: DeepTiledReaderLike>(inf: &mut T) -> Result<(), exr_core::ExrError> {
    let header = inf.header().clone();
    let chan: Vec<(String, PixelType)> = header
        .channels()?
        .iter()
        .map(|(n, c)| (n.to_string(), c.type_))
        .collect();

    let l0 = inf.data_window_for_level(0, 0)?;
    let grid_w = (l0.max.x - l0.min.x + 1) as i64;

    for lx in 0..inf.num_x_levels() {
        for ly in 0..inf.num_y_levels() {
            if !inf.is_valid_level(lx, ly) {
                continue;
            }
            let lvl = inf.data_window_for_level(lx, ly)?;
            let lh = (lvl.max.y - lvl.min.y + 1) as usize;
            let sc_x = 4usize;
            let sc_y = grid_w as usize * sc_x;
            let mut counts = vec![0u8; lh * sc_y];

            let nx = inf.num_x_tiles(lx)?;
            let ny = inf.num_y_tiles(ly)?;

            let (cfb, _) = deep_frame_buffer(&[], 0, grid_w, sc_x, sc_y)?;
            inf.set_frame_buffer(&cfb)?;
            inf.read_pixel_sample_counts(0, nx - 1, 0, ny - 1, lx, ly, &mut counts)?;

            let total = total_samples(&counts);
            let (dfb, bases) = deep_frame_buffer(&chan, total, grid_w, sc_x, sc_y)?;
            let heap_len = bases
                .last()
                .map(|&b| {
                    b as usize + total as usize * chan.last().map(|(_, t)| exr_core::misc::pixel_type_size(*t).unwrap_or(0)).unwrap_or(0)
                })
                .unwrap_or(0);
            let mut heap = vec![0u8; heap_len];
            inf.set_frame_buffer(&dfb)?;
            inf.read_tiles(0, nx - 1, 0, ny - 1, lx, ly, &counts, &mut heap)?;
        }
    }
    Ok(())
}

/// Decode a single-part **deep-scanline** file.
fn run_deep_scanline(path: &Path) -> Result<Report, exr_core::ExrError> {
    let mut is = StdIfStream::open(path.to_str().unwrap_or_default())?;
    let mut inf = DeepScanLineInputFile::new(&mut is)?;
    decode_deep_scanline(&mut inf)?;
    Ok(Report::default())
}

/// Decode a single-part **deep-tiled** file.
fn run_deep_tiled(path: &Path) -> Result<Report, exr_core::ExrError> {
    let mut is = StdIfStream::open(path.to_str().unwrap_or_default())?;
    let mut inf = DeepTiledInputFile::new(&mut is)?;
    decode_deep_tiled(&mut inf)?;
    Ok(Report::default())
}

/// Decode a flat (scanline / tiled) **multipart part** through the planar frame
/// buffer.  Shared by [`run_multipart`] for `scanlineimage` / `tiledimage`
/// parts.  `tiled` selects the per-level tile read; otherwise a whole-window
/// scanline read.
fn decode_flat_scanline_part<T: exr_core::scanline::GenericInputReader>(
    inf: &mut T,
) -> Result<(), exr_core::ExrError> {
    let header = inf.header().clone();
    let dw = header.data_window()?;
    let (chans, total) = planar_layout(&header, dw)?;
    let mut out = vec![0u8; total];
    let fb = make_planar_frame_buffer(dw, &chans)?;
    inf.set_frame_buffer(&fb)?;
    inf.read_pixels(dw.min.y, dw.max.y, &mut out)?;
    Ok(())
}

fn decode_flat_tiled_part<T: TiledReaderLike>(tif: &mut T) -> Result<(), exr_core::ExrError> {
    let header = tif.header().clone();
    for lx in 0..tif.num_x_levels() {
        for ly in 0..tif.num_y_levels() {
            if !tif.is_valid_level(lx, ly) {
                continue;
            }
            let lvl = tif.data_window_for_level(lx, ly)?;
            let (chans, total) = planar_layout(&header, lvl)?;
            let mut buf = vec![0u8; total];
            let fb = make_planar_frame_buffer(lvl, &chans)?;
            tif.set_frame_buffer(&fb)?;
            let nx = tif.num_x_tiles(lx)?;
            let ny = tif.num_y_tiles(ly)?;
            tif.read_tiles(0, nx - 1, 0, ny - 1, lx, ly, &mut buf)?;
        }
    }
    Ok(())
}

/// Open a **multipart** file and decode EVERY part through the reader matching
/// its `type` — port of the C++ `MultiPartInputFile` + per-part
/// `getInputPart<T>` dispatch.  Each part type (scanline / tiled / deep
/// scanline / deep tiled) is read in full; the invalid multipart fixture errors
/// gracefully on `new`.  Decode-only (multipart round-trip is out of scope).
fn run_multipart(path: &Path) -> Result<Report, exr_core::ExrError> {
    let mut is = StdIfStream::open(path.to_str().unwrap_or_default())?;
    let mut mp = MultiPartInputFile::new(&mut is)?;
    let n = mp.parts();
    for p in 0..n {
        let ty = mp.part_type(p)?.to_string();
        match ty.as_str() {
            hdr::TILED_IMAGE => {
                let mut tif = mp.tiled_input_part(p)?;
                decode_flat_tiled_part(&mut tif)?;
            }
            hdr::DEEP_SCANLINE => {
                let mut inf = mp.deep_scanline_input_part(p)?;
                decode_deep_scanline(&mut inf)?;
            }
            hdr::DEEP_TILE => {
                let mut inf = mp.deep_tiled_input_part(p)?;
                decode_deep_tiled(&mut inf)?;
            }
            // "scanlineimage" or empty `type` → scanline part.
            _ => {
                let mut inf = mp.input_part(p)?;
                decode_flat_scanline_part(&mut inf)?;
            }
        }
    }
    Ok(Report::default())
}

/// Open a file through the reader matching its [`Kind`] **without decoding any
/// pixels** — the analogue of C++ `exr_start_read` / `isOpenExrFile`.
fn open_only(path: &Path, kind: Kind) -> Result<(), exr_core::ExrError> {
    let mut is = StdIfStream::open(path.to_str().unwrap_or_default())?;
    match kind {
        Kind::Scanline | Kind::Unreadable => {
            InputFile::new(&mut is)?;
        }
        Kind::Tiled => {
            TiledInputFile::new(&mut is)?;
        }
        Kind::DeepScanline => {
            DeepScanLineInputFile::new(&mut is)?;
        }
        Kind::DeepTiled => {
            DeepTiledInputFile::new(&mut is)?;
        }
        Kind::MultiPart => {
            MultiPartInputFile::new(&mut is)?;
        }
    }
    Ok(())
}

/// Dispatch a file to the matching reader by its detected [`Kind`].  `round_trip`
/// is honoured only by the flat scanline/tiled readers (lossless + full-res).
fn read_file(path: &Path, kind: Kind, round_trip: bool) -> Result<Report, exr_core::ExrError> {
    match kind {
        Kind::Scanline | Kind::Unreadable => run_scanline(path, round_trip),
        Kind::Tiled => run_tiled(path, round_trip),
        Kind::DeepScanline => run_deep_scanline(path),
        Kind::DeepTiled => run_deep_tiled(path),
        Kind::MultiPart => run_multipart(path),
    }
}

// ---------------------------------------------------------------------------
// Corpus discovery
// ---------------------------------------------------------------------------

/// A corpus root and how its files are scored.
struct CorpusRoot {
    path: PathBuf,
    /// `true` → use the [`expected_outcome`] table (vendored mix of valid +
    /// corrupt); `false` → every collected `.exr` is expected **Valid**
    /// (official openexr-images).
    use_table: bool,
    label: &'static str,
}

/// One discovered corpus file plus the root context needed to score it.
struct CorpusFile {
    path: PathBuf,
    /// Whether this file's root uses the [`expected_outcome`] table.
    use_table: bool,
    /// The root directory (for the path-relative summary key).
    root: PathBuf,
    /// The root's human label ("vendored" / "openexr-images").
    label: &'static str,
}

/// The two corpus roots, from the test image set (`exr-test-data`): OpenEXR's own
/// `src/test` images (`EXR_CORPUS_DIR` points it at another OpenEXR test tree)
/// and openexr-images.
fn corpus_roots() -> Vec<CorpusRoot> {
    let vendored = match std::env::var("EXR_CORPUS_DIR") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => exr_test_data::openexr_test(),
    };
    vec![
        CorpusRoot { path: vendored, use_table: true, label: "vendored" },
        CorpusRoot { path: exr_test_data::openexr_images(), use_table: false, label: "openexr-images" },
    ]
}

/// Recursively collect every `*.exr` under `dir`.
fn collect_exr(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_exr(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("exr") {
            out.push(p);
        }
    }
}

/// A small, representative subset of **openexr-images** files we round-trip
/// (write→read→bit-exact) rather than merely decoding — covers a lossless
/// scanline, a lossless MIP tiled, a multiView scanline, and an off-origin
/// data-window tiled.  Everything else in openexr-images is decode-only to keep
/// the 268 MB corpus fast.  Matched by the path tail (forward-slash normalised).
fn is_roundtrip_sample(rel: &str) -> bool {
    matches!(
        rel,
        "ScanLines/Tree.exr"            // PIZ scanline, A/B/G/R half (lossless)
            | "Chromaticities/Rec709.exr" // PIZ scanline, B/G/R half (lossless)
            | "MultiView/Balls.exr"       // PIZ multiView scanline (lossless)
            | "TestImages/AllHalfValues.exr" // PIZ, every half value (lossless)
            | "MultiResolution/Bonita.exr"   // ZIP MIP tiled (lossless, multi-level)
            | "MultiResolution/KernerEnvCube.exr" // ZIP MIP tiled cube (lossless)
    )
}

// ---------------------------------------------------------------------------
// The runner
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn corpus_real_cpp_exr_files() {
    let roots = corpus_roots();
    // Gather every file across all roots together with its scoring context.
    let mut files: Vec<CorpusFile> = Vec::new();
    for r in &roots {
        assert!(r.path.is_dir(), "corpus root [{}] {} absent", r.label, r.path.display());
        let mut these = Vec::new();
        collect_exr(&r.path, &mut these);
        these.sort();
        for f in these {
            files.push(CorpusFile {
                path: f,
                use_table: r.use_table,
                root: r.path.clone(),
                label: r.label,
            });
        }
    }
    // An empty corpus would pass while testing nothing.
    assert!(!files.is_empty(), "the corpus roots hold no .exr files");

    let start = std::time::Instant::now();

    let mut n_valid = 0usize;
    let mut n_invalid = 0usize;
    let mut n_nopanic = 0usize;
    let mut n_openonly = 0usize;
    let mut n_round_tripped = 0usize;
    let mut n_decode_only = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut panics: Vec<String> = Vec::new();
    let mut uncategorized: Vec<String> = Vec::new();
    let mut summary: BTreeMap<String, String> = BTreeMap::new();

    for CorpusFile { path, use_table, root, label } in &files {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<?>")
            .to_string();
        // Path relative to its root, forward-slash normalised (for the
        // round-trip sample match + a unique summary key across roots).
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let key = format!("[{label}] {rel}");

        // Vendored files use the C++-justified table; openexr-images files are
        // all VALID real images → expected Valid.
        let expected = if *use_table {
            expected_outcome(&name).unwrap_or_else(|| {
                uncategorized.push(name.clone());
                Outcome::NoPanicOnly
            })
        } else {
            Outcome::Valid
        };

        // openexr-images: round-trip only the curated representative subset.
        let want_round_trip = !*use_table && is_roundtrip_sample(&rel);

        let path_c = path.clone();
        let want_open = expected == Outcome::OpenOnly;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let kind = detect_kind(&path_c);
            let opened = !want_open || open_only(&path_c, kind).is_ok();
            (kind, opened, read_file(&path_c, kind, want_round_trip))
        }));

        match result {
            Err(_) => {
                panics.push(key.clone());
                summary.insert(key.clone(), "PANIC (reader bug)".into());
                continue;
            }
            Ok((kind, opened, read)) => {
                let ok = read.is_ok();
                let round_tripped = read.as_ref().map(|r| r.round_tripped).unwrap_or(false);
                if round_tripped {
                    n_round_tripped += 1;
                } else if ok && expected == Outcome::Valid {
                    n_decode_only += 1;
                }
                let rt = if round_tripped {
                    " + round-trip bit-exact"
                } else if !*use_table && ok {
                    " (decode-only)"
                } else {
                    ""
                };
                let verdict = match expected {
                    Outcome::Valid => {
                        if ok {
                            n_valid += 1;
                            format!("PASS  Valid       [{kind:?}] decoded{rt}")
                        } else {
                            failures.push(format!(
                                "{key}: expected Valid decode, got Err: {}",
                                read.err().map(|e| e.to_string()).unwrap_or_default()
                            ));
                            format!("FAIL  Valid       [{kind:?}] expected Ok, got Err")
                        }
                    }
                    Outcome::Invalid => {
                        if !ok {
                            n_invalid += 1;
                            format!("PASS  Invalid     [{kind:?}] errored gracefully")
                        } else {
                            failures.push(format!(
                                "{key}: expected Invalid (Err), but read returned Ok"
                            ));
                            format!("FAIL  Invalid     [{kind:?}] expected Err, got Ok")
                        }
                    }
                    Outcome::OpenOnly => {
                        if opened {
                            n_openonly += 1;
                            let dec = if ok { "decoded too" } else { "decode errored gracefully" };
                            format!("PASS  OpenOnly    [{kind:?}] opened; {dec}")
                        } else {
                            failures.push(format!(
                                "{key}: expected OpenOnly (file must OPEN), but open failed"
                            ));
                            format!("FAIL  OpenOnly    [{kind:?}] expected open Ok, got Err")
                        }
                    }
                    Outcome::NoPanicOnly => {
                        n_nopanic += 1;
                        let how = if ok { "Ok" } else { "Err" };
                        format!("PASS  NoPanicOnly [{kind:?}] survived ({how})")
                    }
                };
                summary.insert(key.clone(), verdict);
            }
        }
    }

    let elapsed = start.elapsed();

    // --- per-file summary (auditable) --------------------------------------
    eprintln!("\n=== corpus runner: {} files across {} root(s) ===", files.len(), roots.len());
    for (name, verdict) in &summary {
        eprintln!("  {name:<52} {verdict}");
    }
    eprintln!(
        "--- decoded(Valid)={n_valid}  errored(Invalid)={n_invalid}  opened(OpenOnly)={n_openonly}  \
         survived(NoPanicOnly)={n_nopanic}  round-tripped-bit-exact={n_round_tripped}  \
         valid-decode-only={n_decode_only}  ---"
    );
    eprintln!("--- total runtime: {:.2}s ---", elapsed.as_secs_f64());
    if !uncategorized.is_empty() {
        eprintln!(
            "NOTE: {} vendored file(s) not in the expected table (treated NoPanicOnly): {:?}",
            uncategorized.len(),
            uncategorized
        );
    }

    if !panics.is_empty() {
        panic!(
            "{} corpus file(s) PANICKED our readers (real bugs to fix, must propagate as Err): {:?}",
            panics.len(),
            panics
        );
    }
    if !failures.is_empty() {
        panic!(
            "{} corpus expectation failure(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    eprintln!("OK corpus: every file accounted for, zero panics.");
}
