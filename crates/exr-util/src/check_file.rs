//! Port of `ImfCheckFile.{h,cpp}` (OpenEXR v3.4.15,
//! `src/lib/OpenEXRUtil/ImfCheckFile.cpp`): read a file through every OpenEXR read path and report
//! whether any of them failed. This is what `exr-rs check` (OpenEXR's `exrcheck`) runs.
//!
//! Two flavours, like the C++:
//! * the default one exercises the API layer: `MultiPartInputFile`, the `*InputPart` wrappers, and the
//!   single-part `RgbaInputFile` / `InputFile` / `TiledInputFile` / `DeepScanLineInputFile` /
//!   `DeepTiledInputFile` (in the C++ this is `runChecks`);
//! * `core` is `runCoreChecks`, which in C++ uses only the OpenEXRCore C API (chunk info + decoding
//!   pipeline). exr-core has no separate C layer, so it walks the same chunks through the readers
//!   (see `core_checks` for the exact mapping and the known differences).
//!
//! The verdict is a plain `bool`: `true` = the file is **bad**. (The C++ header comment says "returns
//! true if the file reads correctly", the code returns `threw`; `exrcheck` prints `bad` for `true`.)
//!
//! Only the boolean verdict is observable, so where the C++ keeps going after an error only to reach
//! the same `true` later, the port may stop once that verdict is established. Reader creation and
//! repeated chunk errors can otherwise consume work proportional to malformed declared geometry.
//!
//! # Deep images and exr-core's flat sample heap
//! The C++ hands the deep readers one *pointer per pixel* into a per-scanline (or per-tile) buffer.
//! exr-core's deep readers use a flat per-channel sample heap for the requested range. The checks
//! reuse one scanline or tile of counts and a sample heap, reading counts then samples for each unit
//! as C++ does. Memory scales with the largest requested unit, independent of the whole image area.
//! The guards and verdict follow C++; scratch capacities survive between units.
//!
//! # Better than the C++: deep tile requests
//! `readDeepTile` (`ImfCheckFile.cpp:698`) asks for the wrong tiles, so `exrcheck` rejects valid deep tiled
//! files with more tile columns than rows (and deep RIPMAP files with a wide-short level). This port asks
//! for the tile it means to; the difference is pinned in `tests/image_io.rs` and `tests/parity_tool_check.rs`
//! and reported upstream (`.bughunt/upstream_openexr.md` section 4).
//!
//! # Process-global limits
//! `runChecks` temporarily changes process-wide limits (`Header` max image / tile size,
//! `CompositeDeepScanLine` max sample count). A static lock serialises checks in one process and a
//! drop guard restores the limits even if a check unwinds, which the C++ does not do.

use std::sync::{Mutex, PoisonError};

use exr_core::attr::name::Name;
use exr_core::attr::{LevelMode, PixelType};
use exr_core::compressor::num_lines_in_buffer;
use exr_core::deep::composite_scanline::{get_maximum_sample_count, set_maximum_sample_count};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::{Header, DEEP_SCANLINE, DEEP_TILE, SCANLINE_IMAGE, TILED_IMAGE};
use exr_core::io::{IStream, StdIfStream};
use exr_core::scanline::InputFilePartReader;
use exr_core::tiled::misc::calculate_bytes_per_pixel;
use exr_core::tiled::TiledInputPartReader;
use exr_core::{
    DeepFrameBuffer, DeepScanLineInputFile, DeepSlice, DeepTiledInputFile, ExrError, MultiPartInputFile, Result,
    RgbaInputFile,
};
use exr_core::tiled::TiledInputFile;
use exr_core::scanline::InputFile;
use exr_core::multipart::input::{DeepScanLineInputPartReader, DeepTiledInputPartReader};
use imath_rs::Box2i;
use exr_core::rgba::Rgba;

// -- limits for reduceMemory mode (ImfCheckFile.cpp:45-49) ---------------------------------------
const MAX_BYTES_PER_SCANLINE: u64 = 8_000_000;
const MAX_TILE_BYTES_PER_SCANLINE: u64 = 8_000_000;
const MAX_TILE_BYTES: u64 = 1000 * 1000;
const MAX_BYTES_PER_DEEP_PIXEL: u64 = 1000;
const MAX_BYTES_PER_DEEP_SCANLINE: u64 = 1 << 12;

// -- limits for reduceTime mode (ImfCheckFile.cpp:54-55) -----------------------------------------
const TARGET_PIXELS_TO_READ: usize = 1 << 28;
const MAX_SCANLINES_TO_READ: usize = 1 << 20;

/// What to check, mirroring the flags of `checkOpenEXRFile`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckOpts {
    /// Avoid tests or inputs known to take large amounts of memory (may hide errors).
    pub reduce_memory: bool,
    /// After the first error found, skip further work for speed (may hide errors).
    pub reduce_time: bool,
    /// Run the OpenEXRCore-style check (`core_checks`) instead of the API-layer one.
    pub core: bool,
}

/// Where the bytes come from: each read path opens its own reader, like the C++ (a file name is
/// re-opened per reader, the memory stream is rewound with `resetInput`).
#[derive(Clone, Copy)]
enum Src<'a> {
    Path(&'a str),
    Mem(&'a [u8]),
}

impl<'a> Src<'a> {
    fn open(&self) -> Result<Box<dyn IStream + 'a>> {
        match *self {
            Src::Path(p) => Ok(Box::new(StdIfStream::open(p)?)),
            Src::Mem(d) => Ok(Box::new(PtrIStream::new(d))),
        }
    }
}

/// Port of `PtrIStream` (`ImfCheckFile.cpp:992`): an [`IStream`] over a borrowed byte slice.
struct PtrIStream<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PtrIStream<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl IStream for PtrIStream<'_> {
    fn file_name(&self) -> &str {
        "none"
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<bool> {
        let end = self.pos.checked_add(buf.len()).filter(|&e| e <= self.data.len()).ok_or_else(|| {
            ExrError::Input(format!(
                "Early end of file: requesting {} extra bytes after file\n",
                (self.pos + buf.len()).saturating_sub(self.data.len())
            ))
        })?;
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        // C++ returns `current != end`; exr-core's IStream contract is `true` for any exact read.
        Ok(true)
    }

    fn tellg(&mut self) -> Result<u64> {
        Ok(self.pos as u64)
    }

    fn seekg(&mut self, pos: u64) -> Result<()> {
        match usize::try_from(pos) {
            Ok(p) if p <= self.data.len() => {
                self.pos = p;
                Ok(())
            }
            _ => Err(ExrError::Input("Out of range seek requested\n".into())),
        }
    }

    fn size(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
}

/// Check the file at `path`: `true` if it is bad. Port of `checkOpenEXRFile(const char*, ...)`.
pub fn check_file(path: &str, opts: CheckOpts) -> bool {
    run(Src::Path(path), opts)
}

/// Check an in-memory file: `true` if it is bad. Port of `checkOpenEXRFile(const char*, size_t, ...)`.
pub fn check_bytes(data: &[u8], opts: CheckOpts) -> bool {
    run(Src::Mem(data), opts)
}

/// Serialises checks: the limits below are process-global.
static CHECK_LOCK: Mutex<()> = Mutex::new(());

// Test-only work counters prove malformed inputs stop before reader allocation or repeated reads.
#[cfg(test)]
std::thread_local! {
    static FLAT_READER_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FALLBACK_READER_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CORE_CHUNK_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Process-global limits saved on entry and restored on drop.
struct Saved {
    samples: i64,
    image: (i32, i32),
    tile: (i32, i32),
}

impl Saved {
    /// `runChecks`'s prologue (`ImfCheckFile.cpp:1089-1107`): in reduce mode cap the image, tile and
    /// deep sample sizes, otherwise allow tiles up to 16384. `core` skips the deep-sample cap: the C++
    /// core path only sets per-context image/tile limits, and only in reduce mode.
    fn apply(opts: CheckOpts) -> Self {
        let saved = Self {
            samples: get_maximum_sample_count(),
            image: Header::max_image_size(),
            // C++ reads the *image* limit into its tile variables (`getMaxImageSize (maxTileWidth, ..)`,
            // :1095) and restores that, leaving the tile limit changed; here it is restored properly.
            tile: Header::max_tile_size(),
        };
        if opts.reduce_memory || opts.reduce_time {
            if !opts.core {
                set_maximum_sample_count(1 << 20);
            }
            Header::set_max_image_size(2048, 2048);
            Header::set_max_tile_size(512, 512);
        } else if !opts.core {
            Header::set_max_tile_size(16384, 16384);
        }
        saved
    }
}

impl Drop for Saved {
    fn drop(&mut self) {
        set_maximum_sample_count(self.samples);
        Header::set_max_image_size(self.image.0, self.image.1);
        Header::set_max_tile_size(self.tile.0, self.tile.1);
    }
}

fn run(src: Src, opts: CheckOpts) -> bool {
    let _lock = CHECK_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let _saved = Saved::apply(opts);
    if opts.core { core_checks(src, opts) } else { api_checks(src, opts) }
}

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/// An exception the C++ catches: the check reports the file as bad. Logged so `RUST_LOG=debug` says why.
fn failed(e: ExrError) -> bool {
    note(&e);
    true
}

/// Log an error the check swallows (it only matters through the verdict).
fn note(e: &ExrError) {
    log::debug!("check: {e}");
}

/// `static_cast<uint64_t>(max) - static_cast<uint64_t>(min) + 1` on `int` coordinates (sign-extended
/// first, wrapping like C).
fn extent(min: i32, max: i32) -> u64 {
    (max as i64 as u64).wrapping_sub(min as i64 as u64).wrapping_add(1)
}

/// A zero-filled vector whose allocation failure is an error (C++ `bad_alloc`, caught as "threw").
fn zeroed<T: Clone>(n: u64, v: T) -> Result<Vec<T>> {
    let n = usize::try_from(n).map_err(|_| ExrError::Base("allocation too large".into()))?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(n).map_err(|_| ExrError::Base("out of memory".into()))?;
    buf.resize(n, v);
    Ok(buf)
}

/// Channels of a header as `(name, type, x_sampling, y_sampling)`, so no borrow of the header
/// outlives the call that sets the frame buffer.
fn chans(h: &Header) -> Result<Vec<(Name, PixelType, i32, i32)>> {
    Ok(h.channels()?.iter().map(|(n, c)| (n.clone(), c.type_, c.x_sampling, c.y_sampling)).collect())
}

/// `getStep`: row stride that keeps `reduceTime` runs quick (only the Rgba path uses it).
fn get_step(dw: &Box2i, reduce_time: bool) -> usize {
    if !reduce_time {
        return 1;
    }
    let rows = (dw.max.y as i64 - dw.min.y as i64 + 1) as usize;
    let pixels = rows.wrapping_mul((dw.max.x as i64 - dw.min.x as i64 + 1) as usize);
    1.max((pixels / TARGET_PIXELS_TO_READ).max(rows / MAX_SCANLINES_TO_READ))
}

/// `y` values `min..=max` stepping by `step`, without i32 overflow at the top of the range.
fn ys(min: i32, max: i32, step: usize) -> impl Iterator<Item = i32> {
    (min as i64..=max as i64).step_by(step).map(|y| y as i32)
}

/// The line ranges a flat scanline read is issued in: `[y, y + step - 1]` clipped to the window, `y`
/// advancing by `step`.
///
/// C++ reads line by line (`readPixels (y)`) and its `ScanLineInputFile` keeps the decoded chunk of the
/// last line it read, so the `lines_in_buffer - 1` following lines cost nothing. exr-core decodes the
/// chunk again for every call, which made a 256-line DWAB chunk 256 times slower. One call per chunk
/// (`step == lines_in_buffer`, chunks start at the window's first line) touches the same chunks, so the
/// verdict is the same. Returns `(first, last)` pairs.
fn spans(min: i32, max: i32, step: usize) -> impl Iterator<Item = (i32, i32)> {
    ys(min, max, step).map(move |y| (y, (y as i128 + step as i128 - 1).min(max as i128) as i32))
}

/// What a part's `type` says it is. Classified from the raw bytes (`Header::type_bytes`): C++ keeps `type` a
/// `std::string` without an encoding check and OpenEXRCore takes any byte string, an unknown one being
/// kept as an unknown storage (`parse_header.c:1789-1819`), so a non-UTF-8 `type` is an `Other` part
/// here, not an error (`Header::type_` would fail on it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Kind {
    Scan,
    Tiled,
    DeepScan,
    DeepTile,
    #[default]
    Other,
}

impl Kind {
    /// `header.type ()` of a header that has the attribute.
    fn of(h: &Header) -> Result<Self> {
        Ok(match h.type_bytes()? {
            b if b == SCANLINE_IMAGE.as_bytes() => Self::Scan,
            b if b == TILED_IMAGE.as_bytes() => Self::Tiled,
            b if b == DEEP_SCANLINE.as_bytes() => Self::DeepScan,
            b if b == DEEP_TILE.as_bytes() => Self::DeepTile,
            _ => Self::Other,
        })
    }

    /// `isTiled (type)`.
    fn is_tiled(self) -> bool {
        matches!(self, Self::Tiled | Self::DeepTile)
    }
}

/// `size_t(tileW) * tileH * bpp`, with a deep tile costing at least 4 bytes per pixel
/// (`ImfCheckFile.cpp:892-896`, `:1146-1156`).
fn tile_bytes(h: &Header, deep: bool) -> Result<u64> {
    let td = h.tile_description()?;
    let bpp = calculate_bytes_per_pixel(h)? as u64;
    let px = td.x_size as u64 * td.y_size as u64;
    Ok(px.wrapping_mul(if deep { bpp.max(4) } else { bpp }))
}

// ---------------------------------------------------------------------------------------------
// Flat readers: one trait per storage kind so InputFile and InputPart share the C++ template body
// ---------------------------------------------------------------------------------------------

/// What `readScanline<T>` needs from `InputFile` / `InputPart`.
trait ScanIn {
    fn hdr(&self) -> &Header;
    fn set_fb(&mut self, fb: &FrameBuffer) -> Result<()>;
    fn read(&mut self, y1: i32, y2: i32, out: &mut [u8]) -> Result<()>;
    /// Chunk framing check only (`exr_read_scanline_chunk_info` stand-in).
    fn raw(&mut self, y: i32) -> Result<()>;
}

/// What `readTile<T>` needs from `TiledInputFile` / `TiledInputPart`.
trait TileIn {
    fn hdr(&self) -> &Header;
    fn levels(&self) -> (i32, i32);
    fn ntiles(&self, lx: i32, ly: i32) -> Result<(i32, i32)>;
    fn level_dw(&self, lx: i32, ly: i32) -> Result<Box2i>;
    fn set_fb(&mut self, fb: &FrameBuffer) -> Result<()>;
    fn read(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, out: &mut [u8]) -> Result<()>;
    fn raw(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()>;
}

macro_rules! impl_scan {
    ($t:ty) => {
        impl ScanIn for $t {
            fn hdr(&self) -> &Header {
                self.header()
            }
            fn set_fb(&mut self, fb: &FrameBuffer) -> Result<()> {
                self.set_frame_buffer(fb)
            }
            fn read(&mut self, y1: i32, y2: i32, out: &mut [u8]) -> Result<()> {
                self.read_pixels(y1, y2, out)
            }
            fn raw(&mut self, y: i32) -> Result<()> {
                self.raw_pixel_data(y).map(drop)
            }
        }
    };
}
impl_scan!(InputFile<'_>);
impl_scan!(InputFilePartReader<'_>);

macro_rules! impl_tile {
    ($t:ty) => {
        impl TileIn for $t {
            fn hdr(&self) -> &Header {
                self.header()
            }
            fn levels(&self) -> (i32, i32) {
                (self.num_x_levels(), self.num_y_levels())
            }
            fn ntiles(&self, lx: i32, ly: i32) -> Result<(i32, i32)> {
                Ok((self.num_x_tiles(lx)?, self.num_y_tiles(ly)?))
            }
            fn level_dw(&self, lx: i32, ly: i32) -> Result<Box2i> {
                self.data_window_for_level(lx, ly)
            }
            fn set_fb(&mut self, fb: &FrameBuffer) -> Result<()> {
                self.set_frame_buffer(fb)
            }
            fn read(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, out: &mut [u8]) -> Result<()> {
                self.read_tile(dx, dy, lx, ly, out)
            }
            fn raw(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()> {
                self.raw_tile_data(dx, dy, lx, ly).map(drop)
            }
        }
    };
}
impl_tile!(TiledInputFile<'_>);
impl_tile!(TiledInputPartReader<'_>);

/// Holds the `CompositeDeepScanLine` sample limit off for one ranged read, restoring it on drop.
struct NoSampleLimit(i64);

impl NoSampleLimit {
    fn new() -> Self {
        let old = get_maximum_sample_count();
        set_maximum_sample_count(0);
        Self(old)
    }
}

impl Drop for NoSampleLimit {
    fn drop(&mut self) {
        set_maximum_sample_count(self.0);
    }
}

/// Does every line of deep scanline part `part` stay within the `CompositeDeepScanLine` sample limit?
///
/// The flat readers flatten a deep part through `CompositeDeepScanLine`, which refuses a read whose
/// sample total exceeds the limit. C++ reads one line per call, so the limit is per line; exr-core cannot
/// yet read a deep part line by line (the gap noted in [`read_scanline`]), so the ranged read runs with
/// the limit off ([`NoSampleLimit`]) and this restores the per-line verdict from the counts alone.
fn deep_lines_fit(src: Src, part: i32) -> Result<bool> {
    let limit = get_maximum_sample_count();
    if limit <= 0 {
        return Ok(true);
    }
    let mut stream = src.open()?;
    let mut multi = MultiPartInputFile::new(&mut *stream)?;
    let mut rd = multi.deep_scanline_input_part(part)?;
    let dw = rd.header().data_window()?;
    // One row of counts: y_stride 0, so every line is read into the same row.
    let w = extent(dw.min.x, dw.max.x);
    let count = Slice::new(PixelType::Uint, -(dw.min.x as i64) * 4, 4, 0, 1, 1, 0.0, false, false)?;
    rd.set_fb(&deep_fb(&[], &count, 0)?)?;
    let mut row = zeroed(w.wrapping_mul(4), 0u8)?;
    for y in ys(dw.min.y, dw.max.y, 1) {
        rd.counts(y, y, &mut row)?;
        let total: u64 = row.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c) as u64).sum();
        if total as i64 > limit {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `readRgba` (`ImfCheckFile.cpp:82`): every part through the Rgba interface, one line at a time.
/// `Err` is an exception escaping the function (C++ `in.setPart` sits outside the `try`).
fn read_rgba(inp: &mut RgbaInputFile, opts: CheckOpts, fit: &dyn Fn(i32) -> Result<bool>) -> Result<bool> {
    let mut threw = false;
    for part in 0..inp.parts() {
        inp.set_part(part)?;
        let r = (|| -> Result<Option<bool>> {
            let dw = inp.data_window()?;
            let w = extent(dw.min.x, dw.max.x);
            let bpp = calculate_bytes_per_pixel(inp.header()?)? as u64;
            let lines = num_lines_in_buffer(inp.header()?.compression()?)? as u64;
            if opts.reduce_memory && w.wrapping_mul(bpp).wrapping_mul(lines) > MAX_BYTES_PER_SCANLINE {
                return Ok(Some(false));
            }
            let mut pixels = zeroed(w, Rgba::default())?;
            let hdr = inp.header()?;
            let deep = hdr.has_type() && Kind::of(hdr)? == Kind::DeepScan;
            let skip = get_step(&dw, opts.reduce_time);
            // reduceTime skips rows (`getStep`): single rows then; otherwise whole chunks (see `spans`).
            let step = if deep || skip > 1 { skip } else { lines.max(1) as usize };
            let _unlimited = deep.then(NoSampleLimit::new);
            if deep && !fit(part)? {
                threw = true;
                if opts.reduce_time {
                    return Ok(Some(true));
                }
            }
            for (y, y2) in spans(dw.min.y, dw.max.y, step) {
                let y2 = if skip > 1 { y } else { y2 };
                // A failing line does not stop the scan (unless reduceTime): later lines may still
                // fail in their own way.
                if let Err(e) = inp.read_pixels(y, y2, &mut pixels, 1, 0) {
                    note(&e);
                    threw = true;
                    if opts.reduce_time {
                        return Ok(Some(true));
                    }
                }
            }
            Ok(None)
        })();
        match r {
            Ok(Some(v)) => return Ok(v),
            Ok(None) => {}
            Err(_) => threw = true,
        }
    }
    Ok(threw)
}

/// Frame buffer + backing bytes that read "all channels present": channel `i` is HALF, FLOAT or UINT
/// by `i % 3`, into one `w`-wide row per type, with `y_stride == 0` so every line lands in the same
/// row (`ImfCheckFile.cpp:165-227`). `tile_quirk` reproduces `readTile`'s missing `break` (:346-357):
/// its FLOAT case falls through into the UINT case, whose `insert` replaces the FLOAT slice, so
/// channels `i % 3 == 1` end up UINT.
fn flat_fb(h: &Header, w: u64, dx: i32, tile_quirk: bool) -> Result<(FrameBuffer, Vec<u8>)> {
    let mut fb = FrameBuffer::new();
    for (i, (name, _, xs, ys)) in chans(h)?.into_iter().enumerate() {
        let (ty, size, region) = match i % 3 {
            0 => (PixelType::Half, 2i64, 0i64),
            1 if !tile_quirk => (PixelType::Float, 4, 2 * w as i64),
            _ => (PixelType::Uint, 4, 6 * w as i64),
        };
        // C++ `halfData - sizeof(half) * (dx / xSampling)` (scanline) / `&halfChannels[-dx / xSampling]` (tile).
        let idx = if tile_quirk { (-(dx as i64)) / xs as i64 } else { -((dx / xs) as i64) };
        fb.insert(name, Slice::new(ty, region + idx * size, size as usize, 0, xs, ys, 0.0, false, false)?)?;
    }
    Ok((fb, zeroed(10 * w, 0u8)?))
}

/// `readScanline<T>` (`ImfCheckFile.cpp:145`).
fn read_scanline<T: ScanIn>(inp: &mut T, opts: CheckOpts, fit: &dyn Fn() -> Result<bool>) -> bool {
    (|| -> Result<bool> {
        let dw = inp.hdr().data_window()?;
        let w = extent(dw.min.x, dw.max.x);
        let bpp = calculate_bytes_per_pixel(inp.hdr())? as u64;
        let lines = num_lines_in_buffer(inp.hdr().compression()?)? as u64;
        if opts.reduce_memory && w.wrapping_mul(bpp).wrapping_mul(lines) > MAX_BYTES_PER_SCANLINE {
            return Ok(false);
        }
        let (fb, mut buf) = flat_fb(inp.hdr(), w, dw.min.x, false)?;
        inp.set_fb(&fb)?;
        let mut threw = false;
        // Flatten deep parts one row at a time, matching C++ and bounding temporary heaps.
        let deep = inp.hdr().has_type() && Kind::of(inp.hdr())? == Kind::DeepScan;
        let step = if deep { 1 } else { lines.max(1) as usize };
        let _unlimited = deep.then(NoSampleLimit::new);
        if deep && !fit()? {
            threw = true;
            if opts.reduce_time {
                return Ok(true);
            }
        }
        for (y, y2) in spans(dw.min.y, dw.max.y, step) {
            if let Err(e) = inp.read(y, y2, &mut buf) {
                note(&e);
                threw = true;
                if opts.reduce_time {
                    return Ok(true);
                }
            }
        }
        Ok(threw)
    })()
    .unwrap_or_else(failed)
}

/// Is a failure on level `(lx, ly)` a real one? Non-RIPMAP files only have levels with `lx == ly`;
/// the C++ ignores the exceptions of the others (`ImfCheckFile.cpp:400-408`).
fn level_counts(ripmap: bool, lx: i32, ly: i32) -> bool {
    ripmap || lx == ly
}

/// `readTile<T>` (`ImfCheckFile.cpp:298`): every tile of every level, one tile at a time.
fn read_tile<T: TileIn>(inp: &mut T, opts: CheckOpts) -> bool {
    (|| -> Result<bool> {
        let dw = inp.hdr().data_window()?;
        let w = extent(dw.min.x, dw.max.x);
        let (nxl, nyl) = inp.levels();
        let td = inp.hdr().tile_description()?;
        let bpp = calculate_bytes_per_pixel(inp.hdr())? as u64;
        // `td.xSize * td.ySize` is a 32-bit product in C++ before it widens to `bytes`' type.
        if opts.reduce_memory
            && (w.wrapping_mul(bpp) > MAX_BYTES_PER_SCANLINE
                || (td.x_size.wrapping_mul(td.y_size) as u64).wrapping_mul(bpp) > MAX_TILE_BYTES)
        {
            return Ok(false);
        }
        let (fb, mut buf) = flat_fb(inp.hdr(), w, dw.min.x, true)?;
        inp.set_fb(&fb)?;
        let ripmap = td.mode == LevelMode::RipmapLevels;
        let mut threw = false;
        for ly in 0..nyl {
            for lx in 0..nxl {
                let (nx, ny) = inp.ntiles(lx, ly)?;
                for y in 0..ny {
                    for x in 0..nx {
                        if let Err(e) = inp.read(x, y, lx, ly, &mut buf) {
                            note(&e);
                            if level_counts(ripmap, lx, ly) {
                                threw = true;
                                if opts.reduce_time {
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(threw)
    })()
    .unwrap_or_else(failed)
}

// ---------------------------------------------------------------------------------------------
// Deep readers
// ---------------------------------------------------------------------------------------------

/// What the deep scanline checks need from `DeepScanLineInputFile` / `DeepScanLineInputPart`.
trait DeepScanIn {
    fn hdr(&self) -> &Header;
    fn set_fb(&mut self, fb: &DeepFrameBuffer) -> Result<()>;
    fn counts(&mut self, y1: i32, y2: i32, sc: &mut [u8]) -> Result<()>;
    fn data(&mut self, y1: i32, y2: i32, sc: &[u8], heap: &mut [u8]) -> Result<()>;
    fn raw(&mut self, y: i32) -> Result<()>;
    /// The decode of a chunk whose samples are not wanted (`decode_to_ptr == NULL`): the payload is read
    /// and decompressed, nothing is stored.
    fn payload(&mut self, y: i32) -> Result<()>;
}

/// What the deep tiled checks need from `DeepTiledInputFile` / `DeepTiledInputPart`.
trait DeepTileIn {
    fn hdr(&self) -> &Header;
    fn levels(&self) -> (i32, i32);
    fn ntiles(&self, lx: i32, ly: i32) -> Result<(i32, i32)>;
    fn level_dw(&self, lx: i32, ly: i32) -> Result<Box2i>;
    fn set_fb(&mut self, fb: &DeepFrameBuffer) -> Result<()>;
    fn counts(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, sc: &mut [u8]) -> Result<()>;
    fn data(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, sc: &[u8], heap: &mut [u8]) -> Result<()>;
    fn raw(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()>;
    /// See [`DeepScanIn::payload`].
    fn payload(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()>;
}

macro_rules! impl_deep_scan {
    ($t:ty) => {
        impl DeepScanIn for $t {
            fn hdr(&self) -> &Header {
                self.header()
            }
            fn set_fb(&mut self, fb: &DeepFrameBuffer) -> Result<()> {
                self.set_frame_buffer(fb)
            }
            fn counts(&mut self, y1: i32, y2: i32, sc: &mut [u8]) -> Result<()> {
                self.read_pixel_sample_counts(y1, y2, sc)
            }
            fn data(&mut self, y1: i32, y2: i32, sc: &[u8], heap: &mut [u8]) -> Result<()> {
                self.read_pixels(y1, y2, sc, heap)
            }
            fn raw(&mut self, y: i32) -> Result<()> {
                self.raw_pixel_data(y).map(drop)
            }
            fn payload(&mut self, y: i32) -> Result<()> {
                let (y, _counts, data, unpacked) = self.raw_pixel_data(y)?;
                check_payload(self.header(), &data, y, unpacked)
            }
        }
    };
}
impl_deep_scan!(DeepScanLineInputFile<'_>);
impl_deep_scan!(DeepScanLineInputPartReader<'_>);

macro_rules! impl_deep_tile {
    ($t:ty) => {
        impl DeepTileIn for $t {
            fn hdr(&self) -> &Header {
                self.header()
            }
            fn levels(&self) -> (i32, i32) {
                (self.num_x_levels(), self.num_y_levels())
            }
            fn ntiles(&self, lx: i32, ly: i32) -> Result<(i32, i32)> {
                Ok((self.num_x_tiles(lx)?, self.num_y_tiles(ly)?))
            }
            fn level_dw(&self, lx: i32, ly: i32) -> Result<Box2i> {
                self.data_window_for_level(lx, ly)
            }
            fn set_fb(&mut self, fb: &DeepFrameBuffer) -> Result<()> {
                self.set_frame_buffer(fb)
            }
            fn counts(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, sc: &mut [u8]) -> Result<()> {
                self.read_pixel_sample_counts(dx, dx, dy, dy, lx, ly, sc)
            }
            fn data(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, sc: &[u8], heap: &mut [u8]) -> Result<()> {
                self.read_tiles(dx, dx, dy, dy, lx, ly, sc, heap)
            }
            fn raw(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()> {
                self.raw_tile_data(dx, dy, lx, ly).map(drop)
            }
            fn payload(&mut self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()> {
                let raw = self.raw_tile_data(dx, dy, lx, ly)?;
                // Deep data only allows NONE / RLE / ZIPS, which ignore the tile's position.
                check_payload(self.header(), &raw.packed_data, 0, raw.unpacked_data_size)
            }
        }
    };
}
impl_deep_tile!(DeepTiledInputFile<'_>);
impl_deep_tile!(DeepTiledInputPartReader<'_>);

/// `bytes` of `readCoreTiledPart` (`ImfCheckFile.cpp:1561-1575`): one tile of level window `lw`, every
/// channel at its native size; a tile is the nominal size clipped to the level.
fn level_tile_bytes(td: &exr_core::attr::TileDescription, lw: &Box2i, names: &[(Name, PixelType, i32, i32)]) -> u64 {
    let w = (td.x_size as u64).min(extent(lw.min.x, lw.max.x));
    let h = (td.y_size as u64).min(extent(lw.min.y, lw.max.y));
    w.wrapping_mul(h).wrapping_mul(elem_bytes(names))
}

/// Decompress a deep chunk's sample data into a scratch buffer and discard it: what the C++ core check
/// does when `realloc_deepdata` leaves `decode_to_ptr` NULL (`ImfCheckFile.cpp:1313-1322`, for a line
/// or tile with no samples or with `>= gMaxBytesPerDeepScanline` bytes), where `exr_decoding_run`
/// still reads and decompresses the payload and so still finds a corrupt stream. A NONE chunk has no
/// stream; exr-core has already checked that its packed and unpacked sizes agree.
fn check_payload(h: &Header, packed: &[u8], min_y: i32, unpacked: u64) -> Result<()> {
    let size = usize::try_from(unpacked).map_err(|_| ExrError::Input("deep sample data size too large".into()))?;
    if let Some(mut c) = exr_core::compressor::new_compressor(h.compression()?, 0, h)?
        && c.uncompress(packed, min_y, size)?.len() != size
    {
        return Err(ExrError::Input("deep sample data has the wrong size".into()));
    }
    Ok(())
}

/// Which guard decides whether a line / tile of deep data is read.
#[derive(Clone, Copy)]
enum Guard {
    /// `readDeepScanLine` / `readDeepTile`: the `reduceMemory` size limits of the API-layer check.
    Api,
    /// `realloc_deepdata` (`ImfCheckFile.cpp:1313`): data is decoded only if `0 < bytes < 4096`.
    Core,
}

/// Reusable counts raster for one requested scanline or tile.
struct CountRaster {
    bytes: Vec<u8>,
    slice: Slice,
    active_width: usize,
    active_height: usize,
}

impl CountRaster {
    fn new(rect: &Box2i) -> Result<Self> {
        let mut raster = Self {
            bytes: Vec::new(),
            slice: Slice::new(PixelType::Uint, 0, 4, 4, 1, 1, 0.0, false, false)?,
            active_width: 0,
            active_height: 0,
        };
        raster.set_window(rect)?;
        Ok(raster)
    }

    fn set_window(&mut self, rect: &Box2i) -> Result<()> {
        let w = extent(rect.min.x, rect.max.x);
        let h = extent(rect.min.y, rect.max.y);
        let n = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| ExrError::Base("deep count raster too large".into()))?;
        let ys = w
            .checked_mul(4)
            .and_then(|n| i64::try_from(n).ok())
            .ok_or_else(|| ExrError::Base("deep count row too large".into()))?;
        let base = (rect.min.y as i64)
            .checked_mul(ys)
            .and_then(|v| v.checked_add(rect.min.x as i64 * 4))
            .and_then(i64::checked_neg)
            .ok_or_else(|| ExrError::Base("deep count offset too large".into()))?;
        resize_scratch(&mut self.bytes, n)?;
        self.bytes.fill(0);
        self.slice = Slice::new(
            PixelType::Uint,
            base,
            4,
            ys as usize,
            1,
            1,
            0.0,
            false,
            false,
        )?;
        self.active_width = w as usize;
        self.active_height = h as usize;
        Ok(())
    }

    /// Change only the valid cells in a fixed-pitch tile-local count buffer.
    fn set_active_window(&mut self, rect: &Box2i) -> Result<()> {
        let (w, h) = (
            extent(rect.min.x, rect.max.x) as usize,
            extent(rect.min.y, rect.max.y) as usize,
        );
        let pitch = self.slice.y_stride;
        if w > pitch / 4 || h > self.bytes.len() / pitch {
            return Err(ExrError::Base(
                "deep count tile exceeds its scratch layout".into(),
            ));
        }
        self.active_width = w;
        self.active_height = h;
        self.bytes.fill(0);
        Ok(())
    }

    fn counts(&self) -> impl Iterator<Item = u32> + '_ {
        let pitch = if self.slice.y_stride == 0 {
            self.bytes.len()
        } else {
            self.slice.y_stride
        };
        self.bytes
            .chunks_exact(pitch)
            .take(self.active_height)
            .flat_map(|row| row[..self.active_width * 4].as_chunks::<4>().0.iter())
            .map(|c| u32::from_ne_bytes(*c))
    }

    fn total(&self) -> u64 {
        self.counts().map(u64::from).sum()
    }
}

/// Keep scratch capacity across rows, tiles and levels; only grow when needed.
fn resize_scratch(buf: &mut Vec<u8>, size: u64) -> Result<()> {
    let size = usize::try_from(size).map_err(|_| ExrError::Base("allocation too large".into()))?;
    if size > buf.len() {
        buf.try_reserve(size - buf.len()).map_err(|_| ExrError::Base("out of memory".into()))?;
    }
    buf.resize(size, 0);
    Ok(())
}

/// The deep frame buffer: the count raster and one FLOAT sample slice per file channel, channel `k`'s
/// heap starting at `k * heap` bytes into the single sample buffer.
fn deep_fb(names: &[(Name, PixelType, i32, i32)], sc: &Slice, heap: u64) -> Result<DeepFrameBuffer> {
    let mut fb = DeepFrameBuffer::new();
    fb.insert_sample_count_slice(sc.clone())?;
    for (k, (name, ..)) in names.iter().enumerate() {
        let base = heap.checked_mul(k as u64).and_then(|n| i64::try_from(n).ok())
            .ok_or_else(|| ExrError::Base("heap too large".into()))?;
        fb.insert(name.clone(), DeepSlice::new(PixelType::Float, base, 8, 0, 4, 1, 1, 0.0, false, false)?)?;
    }
    Ok(fb)
}

/// One stable interleaved FLOAT layout, independent of a unit's sample count.
fn deep_unit_fb(names: &[(Name, PixelType, i32, i32)], sc: &Slice) -> Result<DeepFrameBuffer> {
    let stride = names
        .len()
        .checked_mul(4)
        .filter(|n| *n <= i32::MAX as usize)
        .ok_or_else(|| ExrError::Base("deep channel stride too large".into()))?;
    let mut fb = DeepFrameBuffer::new();
    fb.insert_sample_count_slice(sc.clone())?;
    for (k, (name, ..)) in names.iter().enumerate() {
        fb.insert(
            name.clone(),
            DeepSlice::new(
                PixelType::Float,
                (k * 4) as i64,
                8,
                0,
                stride,
                1,
                1,
                0.0,
                false,
                false,
            )?,
        )?;
    }
    Ok(fb)
}

/// Sum of the per-channel byte sizes (`realloc_deepdata`'s `user_bytes_per_element` sum).
fn elem_bytes(names: &[(Name, PixelType, i32, i32)]) -> u64 {
    names.iter().map(|(_, t, ..)| if *t == PixelType::Half { 2 } else { 4 }).sum()
}

/// Statistics of one line / tile of counts, as `readDeepScanLine` / `readDeepTile` compute them:
/// `file` = every sample of every channel the file holds, `buf` = the ones that would be buffered.
struct Unit {
    tot: u64,
    file: u64,
    buf: u64,
}

impl Unit {
    ///
    /// Deliberate difference: the products are 64-bit. The C++ tile check computes `localSampleCount *
    /// bytesPerSample` (`:722`, `:749`, `int bytesPerSample` at `:597`) and `channelCount *
    /// localSampleCount` (`:717`, `:727`) in 32-bit unsigned arithmetic, which wraps for a pixel with
    /// 2^32 / bytesPerSample or more samples (a corrupt count); there the C++ accepts a pixel it
    /// should skip. The scanline check (`:445`, `uint64_t bytesPerSample`) is 64-bit in the C++ too.
    /// `limit` is the per-pixel byte bound under which a pixel is buffered in reduceMemory mode
    /// (`gMaxBytesPerDeepPixel` for scanlines, `gMaxBytesPerDeepScanline` for tiles: `:521` vs `:722`).
    fn of(counts: impl Iterator<Item = u32>, nch: u64, bps: u64, reduce_memory: bool, strict: bool, limit: u64) -> Self {
        let mut u = Unit { tot: 0, file: 0, buf: 0 };
        for c in counts {
            let c = c as u64;
            u.tot += c;
            u.file += c * nch;
            let small = if strict { c * bps < limit } else { c * bps <= limit };
            if !reduce_memory || small {
                u.buf += c * nch;
            }
        }
        u
    }
}

/// Whether the core check decodes a deep line or tile of `bytes` sample bytes into memory:
/// `realloc_deepdata` (`ImfCheckFile.cpp:1313`) leaves the destination NULL (see [`check_payload`])
/// for no samples or `>= gMaxBytesPerDeepScanline` bytes.
fn core_decodes(bytes: u64) -> bool {
    bytes != 0 && bytes < MAX_BYTES_PER_DEEP_SCANLINE
}

/// `readDeepScanLine`'s test for reading a line when reducing memory (`:532`).
fn api_line_fits(file_samples: u64, buffered_samples: u64) -> bool {
    file_samples + buffered_samples < MAX_BYTES_PER_DEEP_SCANLINE
}

/// `readDeepScanLine<T>` (`ImfCheckFile.cpp:432`) and, with [`Guard::Core`], `readCoreScanlinePart`'s
/// deep branch.
fn read_deep_scanline<T: DeepScanIn>(inp: &mut T, opts: CheckOpts, guard: Guard) -> bool {
    (|| -> Result<bool> {
        let dw = inp.hdr().data_window()?;
        let w = extent(dw.min.x, dw.max.x);
        let bps = calculate_bytes_per_pixel(inp.hdr())? as u64;
        let names = chans(inp.hdr())?;
        let nch = names.len() as u64;
        // sample-count table size guard
        if matches!(guard, Guard::Api) && opts.reduce_memory && w * 4 > MAX_BYTES_PER_SCANLINE {
            return Ok(false);
        }
        if matches!(guard, Guard::Core) && opts.reduce_memory && w.wrapping_mul(elem_bytes(&names)) >= MAX_BYTES_PER_SCANLINE {
            // readCoreScanlinePart: `doread = false`: chunk info only.
            return raw_lines(inp, &dw, opts);
        }
        let mut sc = CountRaster::new(&Box2i {
            min: dw.min,
            max: imath_rs::V2i { x: dw.max.x, y: dw.min.y },
        })?;
        sc.slice.base_offset = -(dw.min.x as i64) * 4;
        sc.slice.y_stride = 0;
        let mut data = Vec::new();
        inp.set_fb(&deep_unit_fb(&names, &sc.slice)?)?;
        let mut threw = false;
        for y in ys(dw.min.y, dw.max.y, 1) {
            sc.bytes.fill(0);
            inp.counts(y, y, &mut sc.bytes)?;
            let read = match guard {
                Guard::Api => {
                    let u = Unit::of(sc.counts(), nch, bps, opts.reduce_memory, false, MAX_BYTES_PER_DEEP_PIXEL);
                    !opts.reduce_memory || api_line_fits(u.file, u.buf)
                }
                Guard::Core => core_decodes(sc.total().checked_mul(elem_bytes(&names))
                    .ok_or_else(|| ExrError::Base("deep unit byte size too large".into()))?),
            };
            if read {
                let heap = sc.total().checked_mul(4).ok_or_else(|| ExrError::Base("heap too large".into()))?;
                let bytes = heap.checked_mul(nch).ok_or_else(|| ExrError::Base("heap too large".into()))?;
                resize_scratch(&mut data, bytes)?;
                if inp.data(y, y, &sc.bytes, &mut data).is_err() {
                    threw = true;
                    if opts.reduce_time { return Ok(true); }
                }
            } else if matches!(guard, Guard::Core) {
                inp.raw(y)?;
                if inp.payload(y).is_err() {
                    threw = true;
                    if opts.reduce_time { return Ok(true); }
                }
            }
        }
        Ok(threw)
    })()
    .unwrap_or_else(failed)
}

/// Chunk-framing check of every scanline chunk (the `doread == false` branch of the core check).
fn raw_lines<T: DeepScanIn>(inp: &mut T, dw: &Box2i, opts: CheckOpts) -> Result<bool> {
    let mut bad = false;
    for y in ys(dw.min.y, dw.max.y, 1) {
        if inp.raw(y).is_err() {
            bad = true;
            if opts.reduce_time {
                break;
            }
        }
    }
    Ok(bad)
}

/// Rectangle of tile `(dx, dy)` inside level window `lw` (`dataWindowForTile`): clipped at the level's
/// edge.
fn tile_rect(lw: &Box2i, td: &exr_core::attr::TileDescription, dx: i32, dy: i32) -> Box2i {
    let x0 = lw.min.x as i64 + dx as i64 * td.x_size as i64;
    let y0 = lw.min.y as i64 + dy as i64 * td.y_size as i64;
    let x1 = (x0 + td.x_size as i64 - 1).min(lw.max.x as i64);
    let y1 = (y0 + td.y_size as i64 - 1).min(lw.max.y as i64);
    Box2i { min: imath_rs::V2i { x: x0 as i32, y: y0 as i32 }, max: imath_rs::V2i { x: x1 as i32, y: y1 as i32 } }
}

/// `readDeepTile<T>` (`ImfCheckFile.cpp:588`) and, with [`Guard::Core`], the deep branch of
/// `readCoreTiledPart`.
fn read_deep_tile<T: DeepTileIn>(inp: &mut T, opts: CheckOpts, guard: Guard) -> bool {
    (|| -> Result<bool> {
        let dw = inp.hdr().data_window()?;
        let bps = calculate_bytes_per_pixel(inp.hdr())? as u64;
        let names = chans(inp.hdr())?;
        let nch = names.len() as u64;
        let td = inp.hdr().tile_description()?;
        let (nxl, nyl) = inp.levels();
        let sc_bytes = td.x_size as u64 * td.y_size as u64 * 4;
        if matches!(guard, Guard::Api) && (opts.reduce_memory || opts.reduce_time) && sc_bytes > MAX_TILE_BYTES {
            return Ok(false);
        }
        let ripmap = td.mode == LevelMode::RipmapLevels;
        let mut sc = CountRaster::new(&tile_rect(&dw, &td, 0, 0))?;
        sc.slice.base_offset = 0;
        sc.slice.x_tile_coords = true;
        sc.slice.y_tile_coords = true;
        let mut data = Vec::new();
        inp.set_fb(&deep_unit_fb(&names, &sc.slice)?)?;
        let mut threw = false;
        // A failed step on level (lx, ly); `chunk` marks a chunk-info failure (the core check's `raw`).
        // API check (`readDeepTile`): every failure counts only on a level the file has
        // (`level_counts`), and reduceTime stops only when it counted.
        // Core check (`readCoreTiledPart`): a chunk-info failure counts only on a level the file has,
        // a decode failure always counts, and reduceTime stops the whole scan after *any* failure
        // (`keepgoing = false`), ignored ones included, with the verdict reached so far.
        macro_rules! flag {
            ($lx:expr, $ly:expr, $chunk:expr) => {
                match guard {
                    Guard::Api => {
                        if level_counts(ripmap, $lx, $ly) {
                            threw = true;
                            if opts.reduce_time {
                                return Ok(true);
                            }
                        }
                    }
                    Guard::Core => {
                        threw |= !$chunk || level_counts(ripmap, $lx, $ly);
                        if opts.reduce_time {
                            return Ok(threw);
                        }
                    }
                }
            };
        }

        for ly in 0..nyl {
            for lx in 0..nxl {
                let (nx, ny) = inp.ntiles(lx, ly)?;
                let lw = match inp.level_dw(lx, ly) {
                    Ok(b) => b,
                    Err(_) => {
                        // An invalid (non-diagonal MIPMAP) level: every tile read fails in the C++ too.
                        flag!(lx, ly, true);
                        continue;
                    }
                };
                // `readCoreTiledPart` decodes a level's tiles only if one tile's samples at the native
                // sizes (with the sample counts at 1 per pixel) are below `gMaxTileBytes` when
                // reducing memory (`:1561-1579`, evaluated on the level's first tile, `:1615`).
                let doread = !(matches!(guard, Guard::Core)
                    && opts.reduce_memory
                    && level_tile_bytes(&td, &lw, &names) >= MAX_TILE_BYTES);
                for y in 0..ny {
                    for x in 0..nx {
                        let rect = tile_rect(&lw, &td, x, y);
                        if matches!(guard, Guard::Core) && inp.raw(x, y, lx, ly).is_err() {
                            flag!(lx, ly, true);
                            continue;
                        }
                        if !doread {
                            // The decoder never runs (`:1615`): no count table is unpacked either.
                            continue;
                        }
                        // Better than the C++: the single tile `(x, y)` is asked for. `readDeepTile` passes
                        // `(x, y, x, y)` for `(dx1, dx2, dy1, dy2)` (`ImfCheckFile.cpp:698`), the tiles
                        // `[min (x, y), max (x, y)]` squared, which do not exist whenever `max (x, y)` reaches the
                        // number of tile columns or rows: `exrcheck` calls every deep tiled file with more tile
                        // columns than rows, and every deep RIPMAP file with a level with more rows than columns,
                        // bad. Upstream report: `.bughunt/upstream_openexr.md` section 4.
                        sc.set_active_window(&rect)?;
                        if inp.counts(x, y, lx, ly, &mut sc.bytes).is_err() {
                            flag!(lx, ly, false);
                            continue;
                        }
                        let mut empty_tile_bad = false;
                        let read = match guard {
                            Guard::Api => {
                                let u = Unit::of(
                                    sc.counts(),
                                    nch,
                                    bps,
                                    opts.reduce_memory,
                                    true,
                                    MAX_BYTES_PER_DEEP_SCANLINE,
                                );
                                // Found by a byte-flip sweep against exrcheck: a tile without samples whose
                                // chunk declares sample data that is not there is bad to the library path
                                // too (it is read back although `readTile` is never called for it).
                                empty_tile_bad = u.tot == 0 && inp.payload(x, y, lx, ly).is_err();
                                // `bufferSize > 0 && (!reduceMemory || (file + buf) * bps < 1000)` (:733)
                                u.buf > 0
                                    && (!opts.reduce_memory
                                        || (u.file + u.buf).wrapping_mul(bps) < MAX_BYTES_PER_DEEP_PIXEL)
                            }
                            Guard::Core => {
                                let bytes = sc.total().checked_mul(elem_bytes(&names))
                                    .ok_or_else(|| ExrError::Base("deep unit byte size too large".into()))?;
                                core_decodes(bytes)
                            }
                        };
                        if empty_tile_bad {
                            flag!(lx, ly, false);
                        }
                        if read {
                            let heap = sc.total().checked_mul(4).ok_or_else(|| ExrError::Base("heap too large".into()))?;
                            let bytes = heap.checked_mul(nch).ok_or_else(|| ExrError::Base("heap too large".into()))?;
                            resize_scratch(&mut data, bytes)?;
                            if inp.data(x, y, lx, ly, &sc.bytes, &mut data).is_err() {
                                flag!(lx, ly, false);
                            }
                        } else {
                            // Decoded into nowhere (`check_payload`), a failure is this tile's.
                            if matches!(guard, Guard::Core) && doread && inp.payload(x, y, lx, ly).is_err() {
                                flag!(lx, ly, false);
                            }
                        }
                    }
                }
            }
        }
        Ok(threw)
    })()
    .unwrap_or_else(failed)
}

// ---------------------------------------------------------------------------------------------
// runChecks
// ---------------------------------------------------------------------------------------------

/// A deep scanline part that the flat readers (`InputPart`, `InputFile`, `RgbaInputFile`) cannot flatten:
/// they composite it with `CompositeDeepScanLine`, which throws "missing a Z channel" / "missing an alpha
/// channel" unless the part has channels named exactly `Z` and `A` (`ImfCompositeDeepScanLine.cpp:103-123`).
/// Deep data without them is perfectly valid (ids, positions, ...); `exrcheck` calls it bad because it takes
/// any exception of the flat readers for damage (`ImfCheckFile.cpp:898`, `:1176`, `:1189`, where only
/// deep tiled parts are expected to throw). Better than the C++: such parts are not read through the flat
/// readers here (the deep readers still check them); a corrupt one is still bad. Reported upstream in
/// `openexr_bug.md`.
fn cannot_flatten(h: &Header) -> bool {
    Kind::of(h).is_ok_and(|k| k == Kind::DeepScan)
        && h.channels().is_ok_and(|c| c.find_channel("Z").is_none() || c.find_channel("A").is_none())
}

/// `enumsValid` (`ImfCheckFile.cpp:820`). exr-core's attribute readers already refuse out-of-range
/// `envmap` / `deepImageState` values (as C++ 3.4 core does), so a header that got this far can only
/// hold valid ones; the check is kept for a header built by other means.
fn enums_valid(h: &Header) -> bool {
    use exr_core::attr::{DeepImageState, Envmap};
    if h.has_envmap() && !matches!(h.envmap(), Ok(Envmap::LatLong | Envmap::Cube)) {
        return false;
    }
    if h.has_deep_image_state()
        && !matches!(
            h.deep_image_state(),
            Ok(DeepImageState::Messy | DeepImageState::Sorted | DeepImageState::NonOverlapping | DeepImageState::Tidy)
        )
    {
        return false;
    }
    true
}

/// `readMultiPart` (`ImfCheckFile.cpp:843`): every part through every part API. `Err` escapes to the
/// caller's `catch` (C++ propagates it).
fn read_multipart(multi: &mut MultiPartInputFile, src: Src, opts: CheckOpts) -> Result<bool> {
    let mut threw = false;
    for part in 0..multi.parts() {
        let h = multi.header(part)?;
        if !enums_valid(h) {
            threw = true;
        }
        let ty = Kind::of(h)?;
        let cannot_flatten = cannot_flatten(h);
        let dw = h.data_window()?;
        let bpp = calculate_bytes_per_pixel(h)? as u64;
        let width = extent(dw.min.x, dw.max.x);
        let lines = num_lines_in_buffer(h.compression()?)? as u64;
        // Very wide scanline parts take excessive memory; the scanline API also buffers a whole row of
        // tiles for a tiled part, so tall tiles count too.
        let mut wide = width.wrapping_mul(bpp).wrapping_mul(lines) > MAX_BYTES_PER_SCANLINE;
        let mut large_tiles = false;
        if ty.is_tiled() {
            let td = h.tile_description()?;
            let per_row = width.div_ceil(td.x_size as u64);
            let tile = td.x_size as u64 * td.y_size as u64;
            if tile.wrapping_mul(per_row).wrapping_mul(bpp) > MAX_TILE_BYTES_PER_SCANLINE {
                wide = true;
            }
            if tile_bytes(h, ty == Kind::DeepTile)? > MAX_TILE_BYTES {
                large_tiles = true;
            }
        }

        // `flushPartCache` is a no-op here: exr-core's per-part reader state keeps no position, every
        // read seeks through the offset table.
        if !opts.reduce_memory || !wide {
            // Reconstruction has already run. An incomplete flat table cannot be read, and its
            // reader would allocate metadata for every declared scanline before reporting that.
            if ty == Kind::Scan && !multi.part_complete(part)? {
                return Ok(true);
            }
            #[cfg(test)]
            FLAT_READER_ATTEMPTS.with(|n| n.set(n.get() + 1));
            let got = match multi.input_part(part) {
                Ok(mut pt) => read_scanline(&mut pt, opts, &|| deep_lines_fit(src, part)),
                Err(e) => failed(e),
            };
            // only 'DeepTiled' parts are expected to throw; all others are an error (here also a deep
            // part that cannot be flattened, see `cannot_flatten`)
            if got && ty != Kind::DeepTile && !cannot_flatten {
                threw = true;
            }
        }
        if !opts.reduce_memory || !large_tiles {
            let got = match multi.tiled_input_part(part) {
                Ok(mut pt) => read_tile(&mut pt, opts),
                Err(e) => failed(e),
            };
            if got && ty == Kind::Tiled {
                threw = true;
            }
        }
        if !opts.reduce_memory || !wide {
            let got = match multi.deep_scanline_input_part(part) {
                Ok(mut pt) => read_deep_scanline(&mut pt, opts, Guard::Api),
                Err(e) => failed(e),
            };
            if got && ty == Kind::DeepScan {
                threw = true;
            }
        }
        if !opts.reduce_memory || !large_tiles {
            let got = match multi.deep_tiled_input_part(part) {
                Ok(mut pt) => read_deep_tile(&mut pt, opts, Guard::Api),
                Err(e) => failed(e),
            };
            if got && ty == Kind::DeepTile {
                threw = true;
            }
        }
    }
    Ok(threw)
}

/// `runChecks<T>` (`ImfCheckFile.cpp:1082`): the multipart reader first, then each single-part API.
fn api_checks(src: Src, opts: CheckOpts) -> bool {
    let mut first_type = Kind::default();
    let mut first_cannot_flatten = false;
    // Assume the first part needs excessive memory until its header is checked; if the multipart
    // constructor throws, the first part stays "large".
    let mut large_tiles = true;
    let mut threw = false;
    let multi = (|| -> Result<bool> {
        let mut stream = src.open()?;
        let mut multi = MultiPartInputFile::new(&mut *stream)?;
        let h = multi.header(0)?;
        first_type = Kind::of(h)?;
        first_cannot_flatten = cannot_flatten(h);
        if first_type.is_tiled() {
            // the scanline API buffers a row of tiles for a tiled file: tall tiles cost memory
            if tile_bytes(h, first_type == Kind::DeepTile)? <= MAX_TILE_BYTES {
                large_tiles = false;
            }
        } else {
            // not tiled, so it cannot hold large tiles; the Tile and DeepTile API checks then run on
            // non-tiled files, where they are expected to throw
            large_tiles = false;
        }
        read_multipart(&mut multi, src, opts)
    })();
    threw |= multi.unwrap_or_else(failed);
    if threw {
        return true; // Later readers cannot change an already bad verdict.
    }

    // Every reader below opens its own stream (`resetInput`).
    macro_rules! attempt {
        ($open:expr, $read:expr) => {{
            (|| -> Result<bool> {
                let mut stream = src.open()?;
                let mut f = $open(&mut *stream)?;
                Ok($read(&mut f))
            })()
            .unwrap_or_else(failed)
        }};
    }

    if !opts.reduce_memory {
        let got = (|| -> Result<bool> {
            let mut stream = src.open()?;
            #[cfg(test)]
            FALLBACK_READER_ATTEMPTS.with(|n| n.set(n.get() + 1));
            let mut f = RgbaInputFile::new(&mut *stream)?;
            read_rgba(&mut f, opts, &|part| deep_lines_fit(src, part))
        })()
        .unwrap_or_else(failed);
        if got && first_type != Kind::DeepTile && !first_cannot_flatten {
            threw = true;
        }
        let got = attempt!(InputFile::new, |f: &mut InputFile| read_scanline(f, opts, &|| deep_lines_fit(src, 0)));
        if got && first_type != Kind::DeepTile && !first_cannot_flatten {
            threw = true;
        }
    }
    if !opts.reduce_memory || !large_tiles {
        let got = attempt!(TiledInputFile::new, |f: &mut TiledInputFile| read_tile(f, opts));
        if got && first_type == Kind::Tiled {
            threw = true;
        }
    }
    if !opts.reduce_memory {
        let got = attempt!(DeepScanLineInputFile::new, |f: &mut DeepScanLineInputFile| read_deep_scanline(
            f,
            opts,
            Guard::Api
        ));
        if got && first_type == Kind::DeepScan {
            threw = true;
        }
    }
    if !opts.reduce_memory || !large_tiles {
        let got = attempt!(DeepTiledInputFile::new, |f: &mut DeepTiledInputFile| read_deep_tile(f, opts, Guard::Api));
        if got && first_type == Kind::DeepTile {
            threw = true;
        }
    }
    threw
}

// ---------------------------------------------------------------------------------------------
// runCoreChecks
// ---------------------------------------------------------------------------------------------

/// The OpenEXRCore-style check (`runCoreChecks`, `checkCoreFile`, `readCoreScanlinePart`,
/// `readCoreTiledPart`, `ImfCheckFile.cpp:1346-1822`): every chunk of every part is located
/// (`exr_read_*_chunk_info`) and, unless the memory guards say otherwise, fully decoded into
/// per-channel native-type buffers.
///
/// exr-core has no separate chunk-info / decoding-pipeline layer, so:
/// * "chunk info" is the readers' raw-chunk call (`raw_pixel_data` / `raw_tile_data`), which locates
///   and frames the chunk without decoding it;
/// * "decode" is the reader's normal scanline / tile read into buffers of each channel's own pixel
///   type (`decode_to_ptr` per channel), one line / tile row wide like the C++ pipeline;
/// * a deep chunk is decoded only when `0 < bytes < 4096` (`realloc_deepdata`), through the same
///   count-then-data engine as the API-layer check; the others only get the chunk-info check (the C++
///   still runs the decoder with a NULL destination there, which also validates the compressed
///   payload — the one place this port checks less);
/// * the C++ ignores a failure of `exr_decoding_initialize` on a scanline part (`break` without
///   setting `frv`, :1386) — reproduced for a frame buffer that cannot be set.
fn core_checks(src: Src, opts: CheckOpts) -> bool {
    (|| -> Result<bool> {
        let mut stream = src.open()?;
        let mut multi = MultiPartInputFile::new(&mut *stream)?;
        for p in 0..multi.parts() {
            let h = multi.header(p)?;
            let (kind, unsupported_version) = (Kind::of(h)?, h.has_version() && h.version()? != 1);
            let bad = match kind {
                Kind::Scan if !multi.part_complete(p)? => true,
                Kind::Scan => core_scanline(&mut multi, p, opts)?,
                // `exr_decoding_initialize` refuses a deep part whose `version` is not 1
                // (`decoding.c:310-317`); `readCoreScanlinePart` answers that with a bare `break`, no `frv`
                // (`ImfCheckFile.cpp:1385-1386`), so a deep scanline part with another version is not
                // bad in the core check (a deep tiled one is: `:1554-1558` sets `frv`). Nothing of such a
                // part is decoded, and exr-core has already validated its chunk table at open.
                Kind::DeepScan if unsupported_version => false,
                Kind::DeepScan => match multi.deep_scanline_input_part(p) {
                    Ok(mut pt) => read_deep_scanline(&mut pt, opts, Guard::Core),
                    Err(e) => failed(e),
                },
                Kind::Tiled => core_tiled(&mut multi, p, opts)?,
                Kind::DeepTile => match multi.deep_tiled_input_part(p) {
                    Ok(mut pt) => read_deep_tile(&mut pt, opts, Guard::Core),
                    Err(e) => failed(e),
                },
                Kind::Other => false,
            };
            if bad {
                return Ok(true);
            }
        }
        Ok(false)
    })()
    .unwrap_or_else(failed)
}

/// Native-type frame buffer over one row of every channel, `y_stride == 0`: `(fb, bytes)` where
/// `width` is the number of pixels a row holds.
fn native_fb(names: &[(Name, PixelType, i32, i32)], width: u64, min_x: i32) -> Result<(FrameBuffer, Vec<u8>)> {
    let mut fb = FrameBuffer::new();
    let mut off = 0i64;
    for (name, ty, xs, ys) in names {
        let size = if *ty == PixelType::Half { 2usize } else { 4 };
        let base = off - (min_x / xs) as i64 * size as i64;
        fb.insert(name.clone(), Slice::new(*ty, base, size, 0, *xs, *ys, 0.0, false, false)?)?;
        off += width as i64 * size as i64;
    }
    Ok((fb, zeroed(off as u64, 0u8)?))
}

/// `readCoreScanlinePart` for a flat scanline part.
fn core_scanline(multi: &mut MultiPartInputFile, p: i32, opts: CheckOpts) -> Result<bool> {
    let h = multi.header(p)?;
    let dw = h.data_window()?;
    let lpc = num_lines_in_buffer(h.compression()?)?.max(1) as i64;
    let names = chans(h)?;
    let width = extent(dw.min.x, dw.max.x);
    let height = extent(dw.min.y, dw.max.y);
    #[cfg(test)]
    FLAT_READER_ATTEMPTS.with(|n| n.set(n.get() + 1));
    let mut rd = match multi.input_part(p) {
        Ok(r) => r,
        Err(_) => return Ok(true),
    };
    let mut init: Option<(bool, Vec<u8>)> = None;
    let mut chunk = 0u64;
    while chunk < height {
        #[cfg(test)]
        CORE_CHUNK_ATTEMPTS.with(|n| n.set(n.get() + 1));
        let y = (dw.min.y as i64 + chunk as i64) as i32;
        chunk += lpc as u64;
        let y2 = (y as i64 + lpc - 1).min(dw.max.y as i64) as i32;
        // The chunk-info step; when the decode below will run it validates the chunk itself.
        let decode = init.as_ref().is_some_and(|i| i.0);
        if !decode && rd.raw(y).is_err() {
            return Ok(true);
        }
        if init.is_none() {
            let bytes: u64 = names
                .iter()
                .map(|(_, t, ..)| width.wrapping_mul(if *t == PixelType::Half { 2 } else { 4 }).wrapping_mul(lpc as u64))
                .fold(0, u64::wrapping_add);
            let doread = !(opts.reduce_memory && bytes >= MAX_BYTES_PER_SCANLINE);
            let (fb, buf) = native_fb(&names, width, dw.min.x)?;
            if rd.set_fb(&fb).is_err() {
                break; // exr_decoding_initialize failure is not flagged (see the function docs)
            }
            init = Some((doread, buf));
        }
        if let Some((true, buf)) = init.as_mut()
            && rd.read(y, y2, buf).is_err()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `readCoreTiledPart` for a flat tiled part.
fn core_tiled(multi: &mut MultiPartInputFile, p: i32, opts: CheckOpts) -> Result<bool> {
    let h = multi.header(p)?;
    let td = h.tile_description()?;
    let names = chans(h)?;
    let mut rd = match multi.tiled_input_part(p) {
        Ok(r) => r,
        Err(_) => return Ok(true),
    };
    let ripmap = td.mode == LevelMode::RipmapLevels;
    let (nxl, nyl) = rd.levels();
    let mut bad = false;
    for ly in 0..nyl {
        for lx in 0..nxl {
            let lw = match rd.level_dw(lx, ly) {
                Ok(b) => b,
                Err(_) => {
                    bad |= level_counts(ripmap, lx, ly);
                    if opts.reduce_time {
                        return Ok(bad);
                    }
                    continue;
                }
            };
            let (levw, levh) = (extent(lw.min.x, lw.max.x), extent(lw.min.y, lw.max.y));
            let (curtw, curth) = ((td.x_size as u64).min(levw), (td.y_size as u64).min(levh));
            let mut init: Option<(bool, Vec<u8>)> = None;
            let (nx, ny) = rd.ntiles(lx, ly)?;
            for ty in 0..ny {
                for tx in 0..nx {
                    // chunk info; the decode validates the chunk itself once the decoder is set up
                    if !init.as_ref().is_some_and(|i| i.0) && rd.raw(tx, ty, lx, ly).is_err() {
                        bad |= level_counts(ripmap, lx, ly);
                        if opts.reduce_time {
                            return Ok(bad);
                        }
                        continue;
                    }
                    if init.is_none() {
                        let bytes: u64 = names
                            .iter()
                            .map(|(_, t, ..)| {
                                curtw.wrapping_mul(if *t == PixelType::Half { 2 } else { 4 }).wrapping_mul(curth)
                            })
                            .fold(0, u64::wrapping_add);
                        let doread = !(opts.reduce_memory && bytes >= MAX_TILE_BYTES);
                        let (fb, buf) = native_fb(&names, levw, lw.min.x)?;
                        if rd.set_fb(&fb).is_err() {
                            return Ok(true); // decoder initialisation failure is flagged for tiles (:1556)
                        }
                        init = Some((doread, buf));
                    }
                    if let Some((true, buf)) = init.as_mut()
                        && rd.read(tx, ty, lx, ly, buf).is_err()
                    {
                        bad = true;
                        if opts.reduce_time {
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }
    Ok(bad)
}

#[cfg(test)]
mod tests;
