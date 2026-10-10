//! `RgbaInputFile` — port of `Imf::RgbaInputFile` / `FromYca` (`ImfRgbaFile.{h,cpp}`).
//!
//! On disk an RGBA file is one of three shapes (`rgbaChannels()`,
//! `ImfRgbaFile.cpp:66-86`):
//!
//! * **luminance-chroma (YCA):** `RY` and/or `BY` present (chroma may be
//!   incomplete). `Y` (full res, filled `0.5` if absent) + `RY`/`BY` (2x2
//!   subsampled, filled `0.0` if either is absent) [+ `A`, filled `1.0`].
//!   Reconstruction = horizontal chroma reconstruct (per even scanline) ->
//!   vertical chroma reconstruct (odd scanlines) -> [`yca::yca_to_rgba`] ->
//!   [`yca::fix_saturation`], all with the 27-tap filters in [`super::yca`].
//! * **luminance only (grey):** `Y` present, no `RY`/`BY`. `Y` is read into
//!   red and duplicated into green/blue (`ImfRgbaFile.cpp:1354-1374`).
//! * **plain RGBA:** none of the above. `R`/`G`/`B`/`A` are read directly
//!   (each filled `0.0`/`1.0` if the file lacks that channel).
//!
//! A `layerName` (or `channelNamePrefix`) selects a named layer's channels
//! instead of the unprefixed ones (`prefixFromLayerName`, `:88-101`).
//!
//! The bulk one-shot decoder ([`decode_from_input`], backing
//! [`RgbaInputFile::read_all`]/[`read`](RgbaInputFile::read)/
//! [`read_stream`](RgbaInputFile::read_stream)) reads every YCA scanline ONCE
//! into a plain `Vec` (after horizontal reconstruction) and computes each
//! output line from that, rather than porting C++'s `_buf1`/`_buf2` ring
//! buffers — the per-scanline RESULT is independent of that caching, and a
//! single pass over the whole image needs no rotation bookkeeping. The
//! incremental engine backing `read_pixels` DOES port the ring buffers
//! faithfully (see below) since it must support partial/random-access reads
//! with bounded memory.
//!
//! [`RgbaInputFile`] DOES hold a live decoder across calls, like C++'s
//! `_inputPart`/`_fromYca`: it owns a [`MultiPartInputFile`], which parses the
//! file's headers + chunk offset tables exactly ONCE (at construction / at a
//! `set_part*` call) and is then reused — `read_pixels` never reopens or
//! reparses the file. [`MultiPartInputFile::input_part`] is cheap to call
//! per-request (it clones already-parsed `Header`/offset-table data, no
//! stream I/O), so a fresh, short-lived low-level [`scanline::InputFile`] is
//! built for each actual pixel read, exactly the way
//! [`MultiPartInputFile::input_part`] is documented to be used.
//!
//! [`RgbaInputFile<'a, S>`] is generic over the stream HOLDER `S`,
//! mirroring `MultiPartInputFile<'a, S>`'s
//! own split, which mirrors `scanline::OutputFile`/`OutputFileOwned` on the
//! write side: the default `S = &'a mut dyn IStream` borrows the caller's
//! stream (this is NOT a self-referential struct — the borrow is ordinary,
//! not internal); [`RgbaInputFile::open`] instead builds the
//! `S = Box<dyn IStream + 'a>` variant, OWNING a path-opened stream — C++'s
//! `RgbaInputFile(const char*, ...)` constructor, previously impossible to
//! port here because `MultiPartInputFile` had no owned-stream variant. The
//! one-shot convenience functions [`RgbaInputFile::read`]/
//! [`RgbaInputFile::read_stream`] remain for the "decode a path/stream once,
//! no persistent object" case.
//!
//! For the YCA path, [`read_pixels`](RgbaInputFile::read_pixels) does NOT
//! decode the whole image per call: [`YcaCache`] is a faithful, BOUNDED-MEMORY
//! port of C++ `FromYca`'s `_buf1[N+2]`/`_buf2[3]` rotating ring buffers
//! (`ImfRgbaFile.cpp:1000-1096`) — `Vec::rotate_left`/`rotate_right` move only
//! the row `Vec`s' headers, exactly like C++ rotating an array of row
//! POINTERS, so a sequential top-to-bottom scan costs O(1) amortized per new
//! line without ever holding more than `N + 2` (29) + 3 rows, regardless of
//! image height. Far random access (a jump `>= N + 2` rows) falls back to
//! refilling the whole (still bounded) ring, exactly as C++ does. See
//! [`yca_output_row`] for the line-by-line port of `FromYca::readPixels`.
//! [`RgbaInputFile::read_all`]/[`RgbaInputFile::read`]/[`RgbaInputFile::read_stream`]
//! additionally use a SEPARATE one-shot bulk decoder ([`decode_from_input`])
//! that reads the whole data window in a single low-level `read_pixels` call
//! (letting the low-level reader decode each on-disk chunk exactly once) —
//! faster than driving the incremental engine one line at a time for the
//! common "decode everything" case; both compute the identical formulas.
//!
//! `read_pixels`'s `base`/`x_stride`/`y_stride` parameters are the safe-Rust
//! translation of C++'s `Rgba* base, size_t xStride, size_t yStride`: C++
//! computes a pixel's address as `base + (x/xs)*xStride*sizeof(Rgba) +
//! (y/ys)*yStride*sizeof(Rgba)` where `base` is typically already offset by
//! `-dataWindow.min.x*xStride - dataWindow.min.y*yStride` so index 0 lands at
//! the data window's origin. `base` here IS that already-offset array (an
//! ordinary Rust slice — no raw pointers, no negative indexing): pixel `(x,
//! y)` lands at `base[(x - dw.min.x) * x_stride + (y - dw.min.y) * y_stride]`,
//! `x_stride`/`y_stride` in units of [`Rgba`] (not bytes; sampling is always 1
//! for RGBA data). `read_all` builds an owned, tightly packed [`RgbaImage`]
//! as the convenience path.

use imath_rs::{Box2i, Half};

use crate::attr::{ChannelList, Chromaticities, Compression, LineOrder, PixelType};
use crate::error::{ExrError, Result};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::Header;
use crate::io::{IStream, StdIfStream};
use crate::multipart::MultiPartInputFile;
use crate::rgba::pixel::RgbaChannels;
use crate::rgba::yca;
use crate::rgba::Rgba;
use crate::scanline;

use std::ops::DerefMut;
use std::path::Path;

/// A decoded luminance-chroma (or plain RGBA) image, reconstructed to
/// full-resolution RGBA.
pub struct RgbaImage {
    /// Data window (inclusive min/max), as in the header.
    pub data_window: Box2i,
    /// Full-resolution RGBA pixels, row-major over the data window (`w * h`).
    pub pixels: Vec<Rgba>,
}

impl RgbaImage {
    pub fn width(&self) -> usize {
        (self.data_window.max.x - self.data_window.min.x + 1) as usize
    }
    pub fn height(&self) -> usize {
        (self.data_window.max.y - self.data_window.min.y + 1) as usize
    }
}

/// Reader for luminance-chroma (and plain RGBA) scanline EXR files. See the
/// module doc for the shape it selects and for how it stays a persistent
/// decoder (like C++) without being a self-referential struct.
///
/// Generic over the stream HOLDER `S`,
/// mirroring [`MultiPartInputFile`]'s own `S` parameter: the default
/// `S = &'a mut dyn IStream` borrows the caller's stream (unchanged
/// behaviour); [`RgbaInputFile::open`] instead builds the
/// `S = Box<dyn IStream + 'a>` variant, so it can OWN a path-opened stream —
/// C++'s `RgbaInputFile(const char*, int)` constructor, which this crate
/// could not port before `MultiPartInputFile` gained the same split.
pub struct RgbaInputFile<'a, S = &'a mut dyn IStream> {
    /// Parses the file's headers + chunk offset tables ONCE; borrowed by
    /// `input_part` per pixel read (cheap — no stream I/O, see module doc).
    multipart: MultiPartInputFile<'a, S>,
    part: i32,
    channel_name_prefix: String,
    /// Which of the three shapes (plain RGBA, grey, YCA) this part/layer resolves to, plus the
    /// YCA incremental-decode memoization when applicable. Rebuilt whenever
    /// `part`/`channel_name_prefix` change.
    mode: DecodeMode,
    /// Persistent Plain/Grey staging buffer, reused across
    /// [`read_pixels`](Self::read_pixels) calls — see `decode_plain_range`/
    /// `decode_grey_range`'s doc for why this is the fix for the real
    /// "N chunk decodes for N requested rows" bug those replaced.
    range_staging: Vec<u8>,
}

/// Which decode path [`RgbaInputFile::read_pixels`]/[`RgbaInputFile::read_all`]
/// use for the current part/layer (mirrors the branch `decode_from_input`
/// takes), plus the YCA path's per-row memoization state.
enum DecodeMode {
    Plain,
    Grey,
    Yca(YcaCache),
}

/// Incremental YCA decode state — a faithful port of C++ `FromYca`'s
/// `_buf1[N+2]` / `_buf2[3]` ring buffers (`ImfRgbaFile.cpp:1000-1096`), NOT a
/// growing memo table: memory is bounded to `N+2` (29) + 3 rows regardless of
/// image height or access pattern. `buf1[i]` holds the YCA data (after
/// horizontal reconstruction if the row is even) for absolute scanline
/// `current_scan_line - N2 - 1 + i`; `buf2[k]` holds the pre-saturation-fix
/// RGBA for `current_scan_line - 1 + k`. A request for scanline `y` rotates
/// both rings by `dy = y - current_scan_line` (`Vec::rotate_left`/`rotate_right`
/// move only the row `Vec`s' headers, matching C++ rotating an array of row
/// POINTERS — no row data is copied) and refills only the newly-needed rows;
/// a jump larger than the ring (`|dy| >= N+2`) refills the whole ring, still
/// bounded. See [`yca_output_row`] for the exact port of `FromYca::readPixels`.
struct YcaCache {
    yw: [f32; 3],
    dw: Box2i,
    width: usize,
    /// C++ `_currentScanLine`; starts at the sentinel `dw.min.y - N - 2` so
    /// the FIRST request always takes the "jump" (full-refill) path.
    current_scan_line: i32,
    /// `N + 2` rows (C++ `_buf1`).
    buf1: Vec<Vec<Rgba>>,
    /// 3 rows (C++ `_buf2`).
    buf2: [Vec<Rgba>; 3],
    /// Persistent per-line scratch for `read_yca_scanline`:
    /// `line_buf` is the raw Xdr bytes for ONE scanline;
    /// `tmp_row` is the pre-reconstruction `Y`/`A`/(even-lx)`R`/`B` staging
    /// row, `width + N - 1` wide (the horizontal reconstruction filter's
    /// padding). Both are fully overwritten every call for the positions
    /// anything downstream actually reads — see `read_yca_scanline`'s doc
    /// and the poison test
    /// `poison_reused_yca_scanline_scratch_matches_fresh_instance` for why
    /// the odd-`lx` `r`/`b` "sentinel" positions (never explicitly written
    /// on an odd `lx`) don't need a reset despite starting non-zero: the
    /// horizontal reconstruction filter's taps only ever land on even `lx`.
    line_buf: Vec<u8>,
    tmp_row: Vec<Rgba>,
}

impl<'a> RgbaInputFile<'a, &'a mut dyn IStream> {
    // -- One-shot convenience API (part 0, no layer). exr-gpu depends on the
    // exact signatures of `read`/`read_stream`; keep them stable. -----------

    /// Read the file at `path` as full-resolution RGBA (part 0, no layer).
    pub fn read(path: impl AsRef<Path>) -> Result<RgbaImage> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
        let mut stream = StdIfStream::open(path_str)?;
        Self::read_stream(&mut stream)
    }

    /// Read a luminance-chroma (grey, or plain RGBA) scanline image from a
    /// stream (part 0, no layer).
    pub fn read_stream(stream: &mut dyn IStream) -> Result<RgbaImage> {
        let mut input = scanline::InputFile::new(stream)?;
        decode_from_input(&mut input, "")
    }

    // -- Object API: construct on a caller-owned stream, optionally select a
    // part/layer, then use the accessors or `read_pixels`/`read_all`. -------

    /// Open `stream`, part 0, no layer — port of `RgbaInputFile(IStream&, int)`.
    pub fn new(stream: &'a mut dyn IStream) -> Result<Self> {
        Self::with_part_and_layer(stream, 0, "")
    }

    /// Open `stream` at `part_number`, no layer — port of
    /// `RgbaInputFile(int, IStream&, int)`.
    pub fn with_part(stream: &'a mut dyn IStream, part_number: i32) -> Result<Self> {
        Self::with_part_and_layer(stream, part_number, "")
    }

    /// Open `stream`, part 0, selecting `layer_name` — port of
    /// `RgbaInputFile(IStream&, const std::string&, int)`.
    pub fn with_layer(stream: &'a mut dyn IStream, layer_name: &str) -> Result<Self> {
        Self::with_part_and_layer(stream, 0, layer_name)
    }

    /// Open `stream` at `part_number`, selecting `layer_name` — port of
    /// `RgbaInputFile(int, IStream&, const std::string&, int)`.
    pub fn with_part_and_layer(
        stream: &'a mut dyn IStream,
        part_number: i32,
        layer_name: &str,
    ) -> Result<Self> {
        let multipart = MultiPartInputFile::new(stream)?;
        Self::from_multipart(multipart, part_number, layer_name)
    }
}

impl<'a> RgbaInputFile<'a, Box<dyn IStream + 'a>> {
    /// Open `path`, OWNING the decoded stream, part 0, no layer — port of
    /// `RgbaInputFile(const char[], int)`, built on
    /// `MultiPartInputFile::new_owned`, which owns its stream holder.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
        let stream: Box<dyn IStream + 'a> = Box::new(StdIfStream::open(path_str)?);
        let multipart = MultiPartInputFile::new_owned(stream)?;
        Self::from_multipart(multipart, 0, "")
    }
}

// Shared read path, generic over the stream holder `S`. Covers BOTH the
// borrowed default (`S = &'a mut dyn IStream`) and the owned `Box<dyn
// IStream>` variant — every method below addresses the underlying stream
// through `MultiPartInputFile<'a, S>`'s own `DerefMut`-bounded methods, so
// the two paths are a single source of truth.
impl<'a, S> RgbaInputFile<'a, S>
where
    S: DerefMut<Target = dyn IStream + 'a>,
{
    /// Shared construction body for [`new`](RgbaInputFile::new) /
    /// [`with_part`](RgbaInputFile::with_part) /
    /// [`with_layer`](RgbaInputFile::with_layer) /
    /// [`with_part_and_layer`](RgbaInputFile::with_part_and_layer) (borrowed)
    /// and [`open`](RgbaInputFile::open) (owned): once the underlying
    /// [`MultiPartInputFile`] exists, resolving the layer prefix and building
    /// the initial [`DecodeMode`] is identical either way.
    fn from_multipart(
        multipart: MultiPartInputFile<'a, S>,
        part_number: i32,
        layer_name: &str,
    ) -> Result<Self> {
        let prefix = prefix_from_layer_name(layer_name, multipart.header(part_number)?)?;
        let mut file = Self {
            multipart,
            part: part_number,
            channel_name_prefix: prefix,
            mode: DecodeMode::Plain,
            range_staging: Vec::new(),
        };
        file.rebuild_mode()?;
        Ok(file)
    }

    /// Recompute [`DecodeMode`] for the current `part`/`channel_name_prefix`
    /// — called at construction and by every `set_*` mutator.
    fn rebuild_mode(&mut self) -> Result<()> {
        let header = self.multipart.header(self.part)?;
        let dw = header.data_window()?;
        let width = (dw.max.x - dw.min.x + 1) as usize;
        let rc = rgba_channels_of(header.channels()?, &self.channel_name_prefix);
        self.mode = if rc.intersects(RgbaChannels::WRITE_C) {
            let yw = yw_from_header(header)?;
            // C++ `FromYca::FromYca`: `_currentScanLine = dw.min.y - N - 2;`
            // — a sentinel before any valid scanline, so the first request's
            // `dy` is always `>= N + 2` and takes the full-ring-refill path.
            let sentinel = dw.min.y - yca::N as i32 - 2;
            DecodeMode::Yca(YcaCache {
                yw,
                dw,
                width,
                current_scan_line: sentinel,
                buf1: vec![vec![Rgba::default(); width]; yca::N + 2],
                buf2: [vec![Rgba::default(); width], vec![Rgba::default(); width], vec![Rgba::default(); width]],
                line_buf: Vec::new(),
                tmp_row: Vec::new(),
            })
        } else if rc.intersects(RgbaChannels::WRITE_Y) {
            DecodeMode::Grey
        } else {
            DecodeMode::Plain
        };
        Ok(())
    }

    /// Select a different layer of the CURRENT part — port of
    /// `RgbaInputFile::setLayerName`.
    pub fn set_layer_name(&mut self, layer_name: &str) -> Result<()> {
        self.channel_name_prefix = prefix_from_layer_name(layer_name, self.multipart.header(self.part)?)?;
        self.rebuild_mode()
    }

    /// Select a different part and layer — port of
    /// `RgbaInputFile::setPartAndLayer`.
    pub fn set_part_and_layer(&mut self, part_number: i32, layer_name: &str) -> Result<()> {
        let prefix = prefix_from_layer_name(layer_name, self.multipart.header(part_number)?)?;
        self.part = part_number;
        self.channel_name_prefix = prefix;
        self.rebuild_mode()
    }

    /// Select a different part, no layer — port of `RgbaInputFile::setPart`.
    pub fn set_part(&mut self, part_number: i32) -> Result<()> {
        self.set_part_and_layer(part_number, "")
    }

    // -- Accessors. All are cheap: `header`/chunk-offset parsing already
    // happened once, at construction. ----------------------------------

    /// This part's header — port of `RgbaInputFile::header`.
    pub fn header(&self) -> Result<&Header> {
        self.multipart.header(self.part)
    }

    /// The data window — port of `RgbaInputFile::dataWindow`.
    pub fn data_window(&self) -> Result<Box2i> {
        self.header()?.data_window()
    }

    /// The display window — port of `RgbaInputFile::displayWindow`.
    pub fn display_window(&self) -> Result<Box2i> {
        self.header()?.display_window()
    }

    /// The pixel aspect ratio — port of `RgbaInputFile::pixelAspectRatio`.
    pub fn pixel_aspect_ratio(&self) -> Result<f32> {
        self.header()?.pixel_aspect_ratio()
    }

    /// The scanline order — port of `RgbaInputFile::lineOrder`.
    pub fn line_order(&self) -> Result<LineOrder> {
        self.header()?.line_order()
    }

    /// The compression method — port of `RgbaInputFile::compression`.
    pub fn compression(&self) -> Result<Compression> {
        self.header()?.compression()
    }

    /// Which of R/G/B/A/Y/chroma this part (and selected layer) provides —
    /// port of `RgbaInputFile::channels` (`rgbaChannels(header.channels(),
    /// channelNamePrefix)`).
    pub fn channels(&self) -> Result<RgbaChannels> {
        Ok(rgba_channels_of(self.header()?.channels()?, &self.channel_name_prefix))
    }

    /// The raw file-format version word — port of `RgbaInputFile::version`
    /// (a single word for the whole file, same for every part).
    pub fn version(&self) -> i32 {
        self.multipart.version()
    }

    /// Number of parts in the file — port of `RgbaInputFile::parts`.
    pub fn parts(&self) -> i32 {
        self.multipart.parts()
    }

    /// `true` when every part's chunk offset table is complete — port of
    /// `RgbaInputFile::isComplete` (checks ALL parts, not just the selected
    /// one, matching C++).
    pub fn is_complete(&self) -> Result<bool> {
        for i in 0..self.multipart.parts() {
            if !self.multipart.part_complete(i)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // -- Pixel access. ---------------------------------------------------

    /// Decode the whole data window of the selected part/layer as
    /// full-resolution RGBA — the bulk one-shot path (see module doc).
    pub fn read_all(&mut self) -> Result<RgbaImage> {
        let mut input = self.multipart.input_part(self.part)?;
        decode_from_input(&mut input, &self.channel_name_prefix)
    }

    /// Read scanlines `[min(y1,y2), max(y1,y2)]` of the selected part/layer
    /// into `base` — port of `RgbaInputFile::readPixels(int, int)`. See the
    /// module doc for the `base`/`x_stride`/`y_stride` convention (the safe
    /// translation of C++'s caller pointer + byte strides) and for why this
    /// is O(1) amortized per new line on a sequential scan rather than
    /// O(height) per call.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if either scan line lies outside the data window, or
    /// an addressed pixel falls outside `base`.
    pub fn read_pixels(&mut self, y1: i32, y2: i32, base: &mut [Rgba], x_stride: isize, y_stride: isize) -> Result<()> {
        let dw = self.data_window()?;
        let (lo, hi) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        if lo < dw.min.y || hi > dw.max.y {
            return Err(ExrError::Arg(format!(
                "readPixels: scan line range [{lo}, {hi}] is outside the data window [{}, {}]",
                dw.min.y, dw.max.y
            )));
        }
        let width = (dw.max.x - dw.min.x + 1) as usize;

        let mut scatter_row = |y: i32, row: &[Rgba]| -> Result<()> {
            for (lx, &px) in row.iter().enumerate() {
                let idx = (lx as isize) * x_stride + (y - dw.min.y) as isize * y_stride;
                let idx = usize::try_from(idx)
                    .map_err(|_| ExrError::Arg("readPixels: destination index is negative.".into()))?;
                *base
                    .get_mut(idx)
                    .ok_or_else(|| ExrError::Arg("readPixels: destination index is out of range.".into()))? = px;
            }
            Ok(())
        };

        // Plain/Grey: ONE batched decode over [lo, hi] --
        // see `decode_plain_range`'s doc for why this replaces a per-row
        // loop. YCA keeps the per-row `yca_output_row` loop (C's own
        // `FromYca::readPixels` is line-by-line too, `ImfRgbaFile.cpp:
        // 886-900`) -- only its internal buffers become persistent.
        let mut mode = std::mem::replace(&mut self.mode, DecodeMode::Plain);
        let result = (|| -> Result<()> {
            match &mut mode {
                DecodeMode::Plain => {
                    let rows = decode_plain_range(
                        &mut self.multipart, self.part, &self.channel_name_prefix, lo, hi, &mut self.range_staging,
                    )?;
                    for (local_y, y) in (lo..=hi).enumerate() {
                        scatter_row(y, &rows[local_y * width..(local_y + 1) * width])?;
                    }
                    Ok(())
                }
                DecodeMode::Grey => {
                    let rows = decode_grey_range(
                        &mut self.multipart, self.part, &self.channel_name_prefix, lo, hi, &mut self.range_staging,
                    )?;
                    for (local_y, y) in (lo..=hi).enumerate() {
                        scatter_row(y, &rows[local_y * width..(local_y + 1) * width])?;
                    }
                    Ok(())
                }
                DecodeMode::Yca(cache) => {
                    for y in lo..=hi {
                        let row = yca_output_row(&mut self.multipart, self.part, &self.channel_name_prefix, cache, y)?;
                        scatter_row(y, &row)?;
                    }
                    Ok(())
                }
            }
        })();
        self.mode = mode;
        result
    }

    /// Read one scanline — port of `RgbaInputFile::readPixels(int)`. A
    /// single-row `base` (not one big enough to hold the whole image) needs
    /// `y_stride = 0` — the one absolute `y` touched must not push the index
    /// outside `base`.
    pub fn read_pixel_line(&mut self, scan_line: i32, base: &mut [Rgba], x_stride: isize, y_stride: isize) -> Result<()> {
        self.read_pixels(scan_line, scan_line, base, x_stride, y_stride)
    }

}

// ---------------------------------------------------------------------------
// Free helpers — ports of the file-local functions at the top of
// `ImfRgbaFile.cpp`.
// ---------------------------------------------------------------------------

/// `rgbaChannels(const ChannelList&, const string&)` — which of R/G/B/A/Y/
/// chroma `channelNamePrefix` + name resolves to in `ch`.
fn rgba_channels_of(ch: &ChannelList, channel_name_prefix: &str) -> RgbaChannels {
    let mut i = RgbaChannels::NONE;
    let has = |suffix: &str| ch.find_channel(&format!("{channel_name_prefix}{suffix}")).is_some();
    if has("R") {
        i |= RgbaChannels::WRITE_R;
    }
    if has("G") {
        i |= RgbaChannels::WRITE_G;
    }
    if has("B") {
        i |= RgbaChannels::WRITE_B;
    }
    if has("A") {
        i |= RgbaChannels::WRITE_A;
    }
    if has("Y") {
        i |= RgbaChannels::WRITE_Y;
    }
    if has("RY") || has("BY") {
        i |= RgbaChannels::WRITE_C;
    }
    i
}

/// `prefixFromLayerName` — `""` for the unnamed/default layer, else
/// `"<layerName>."`, EXCEPT a `multiView`ed file whose first (default) view is
/// named `layerName`, which also resolves to `""` (`ImfRgbaFile.cpp:88-101`).
///
/// v3.4.15 fixed a crash here vs v3.3.11: `hasMultiView(header) &&
/// multiView(header)[0] == layerName` indexed `[0]` unconditionally once
/// `hasMultiView` was true, even for an EMPTY `multiView` string vector (a
/// present-but-empty attribute); v3.4.15 guards with `if (!sv.empty())`
/// first. `sv.first()` below already returns `None` for an empty vector
/// rather than indexing, so this port already matched the fixed v3.4.15
/// behaviour before the diff was checked (verified via `git diff v3.3.11
/// v3.4.15 -- src/lib/OpenEXR/ImfRgbaFile.cpp`).
fn prefix_from_layer_name(layer_name: &str, header: &Header) -> Result<String> {
    if layer_name.is_empty() {
        return Ok(String::new());
    }
    if header.has_multi_view() {
        let sv = header.multi_view()?;
        if let Some(first) = sv.first()
            && first.as_str() == Some(layer_name)
        {
            return Ok(String::new());
        }
    }
    Ok(format!("{layer_name}."))
}

/// `ywFromHeader` — RGB->luminance weights from the header's chromaticities,
/// or Rec.709 default when absent.
pub(crate) fn yw_from_header(header: &Header) -> Result<[f32; 3]> {
    let chroma =
        if header.has_chromaticities() { *header.chromaticities()? } else { Chromaticities::default() };
    chroma.compute_yw()
}

// ---------------------------------------------------------------------------
// Decode dispatch — port of the body of `RgbaInputFile::setFrameBuffer` +
// `readPixels`, specialised to a one-shot whole-image decode (this crate does
// not keep a persistent frame buffer; see the module doc).
// ---------------------------------------------------------------------------

/// Decode the whole data window of `input`'s current header as full-resolution
/// RGBA, selecting R/G/B/A, Y, or Y+chroma exactly as `rgbaChannels` would.
///
/// Generic over [`scanline::GenericInputReader`] (not the concrete
/// `scanline::InputFile`) so it accepts EITHER the standalone reader
/// (`RgbaInputFile::new`'s single-part path) or the multipart-cached
/// `InputFilePartReader` view (`RgbaInputFile`'s `MultiPartInputFile` path)
/// — both expose the same three operations
/// (`header`, `set_frame_buffer`, `read_pixels`) but are genuinely
/// different concrete types.
fn decode_from_input<R: scanline::GenericInputReader>(
    input: &mut R,
    channel_name_prefix: &str,
) -> Result<RgbaImage> {
    let header = input.header().clone();
    let dw = header.data_window()?;
    let width = (dw.max.x - dw.min.x + 1) as usize;
    let height = (dw.max.y - dw.min.y + 1) as usize;

    let rc = rgba_channels_of(header.channels()?, channel_name_prefix);
    if rc.intersects(RgbaChannels::WRITE_C) {
        decode_yca(input, &header, dw, width, height, channel_name_prefix)
    } else if rc.intersects(RgbaChannels::WRITE_Y) {
        decode_grey(input, dw, width, height, channel_name_prefix)
    } else {
        decode_plain_rgba(input, dw, width, height, channel_name_prefix)
    }
}

/// Luminance-chroma (YCA) path — port of `FromYca`.
///
/// `Y`, `A` and BOTH chroma channels are always requested: a file
/// missing one of `RY`/`BY` (or `Y` itself, or `A`) gets that channel filled
/// (`0.0`/`0.5`/`1.0`) by the frame-buffer fill mechanism, exactly as C++'s
/// `FromYca::setFrameBuffer` inserts all four slices unconditionally (gated
/// only on `_readC`, which this function's caller has already established).
fn decode_yca<R: scanline::GenericInputReader>(
    input: &mut R,
    header: &Header,
    dw: Box2i,
    width: usize,
    height: usize,
    channel_name_prefix: &str,
) -> Result<RgbaImage> {
    let yw = yw_from_header(header)?;

    let cw = width / 2; // subsampled chroma width
    let ch = height / 2; // subsampled chroma height
    let yb = width * height;
    let ab = width * height;
    let cb = cw * ch;
    let y_off = 0usize;
    let a_off = y_off + yb * 2;
    let ry_off = a_off + ab * 2;
    let by_off = ry_off + cb * 2;
    let total = (yb + ab + 2 * cb) * 2;
    let mut buf = vec![0u8; total];

    let mut fb = FrameBuffer::new();
    fb.insert(format!("{channel_name_prefix}Y"), slice(y_off, dw, width, 1, 1, 0.5)?)?;
    fb.insert(format!("{channel_name_prefix}A"), slice(a_off, dw, width, 1, 1, 1.0)?)?;
    fb.insert(format!("{channel_name_prefix}RY"), slice(ry_off, dw, width, 2, 2, 0.0)?)?;
    fb.insert(format!("{channel_name_prefix}BY"), slice(by_off, dw, width, 2, 2, 0.0)?)?;
    input.set_frame_buffer(&fb)?;
    input.read_pixels(dw.min.y, dw.max.y, &mut buf)?;

    // Native-endian half reader for a plane.
    let half_at = |off: usize, idx: usize| -> Half {
        let b = off + idx * 2;
        Half::from_bits(u16::from_ne_bytes([buf[b], buf[b + 1]]))
    };

    // -- Per scanline: assemble YCA, reconstruct chroma horizontally. -------
    // `min.x`/`min.y` are multiples of the sampling factor (header rule), so a
    // local row/col index has the same even/odd parity as the absolute one.
    let n2 = yca::N2;
    let mut yca_rows: Vec<Vec<Rgba>> = Vec::with_capacity(height);
    for ly in 0..height {
        let mut tmp = vec![Rgba::default(); width + yca::N - 1];
        for lx in 0..width {
            let p = &mut tmp[n2 + lx];
            p.g = half_at(y_off, ly * width + lx);
            p.a = half_at(a_off, ly * width + lx);
        }
        // Chroma exists only on even scanlines, at even columns.
        if ly & 1 == 0 {
            let cy = ly / 2;
            let mut lx = 0;
            while lx < width {
                let cidx = cy * cw + lx / 2;
                tmp[n2 + lx].r = half_at(ry_off, cidx);
                tmp[n2 + lx].b = half_at(by_off, cidx);
                lx += 2;
            }
        }

        let mut row = vec![Rgba::default(); width];
        if ly & 1 == 0 {
            pad_tmp(&mut tmp, width);
            yca::reconstruct_chroma_horiz(&tmp, &mut row);
        } else {
            row.copy_from_slice(&tmp[n2..n2 + width]);
        }
        yca_rows.push(row);
    }

    // -- Per scanline: reconstruct chroma vertically + YCA->RGBA + fix sat. -
    let h1 = (height - 1) as i32;
    let clamp = |yy: i32| -> usize {
        if yy < 0 {
            0
        } else if yy > h1 {
            (h1 - 1).max(0) as usize
        } else {
            yy as usize
        } // C++ clamps y>yMax to yMax-1
    };

    let mut out = vec![Rgba::default(); width * height];
    for ly in 0..height {
        // buf2[k] = RGB of scanline (ly-1+k), k = 0,1,2.
        let mut buf2: [Vec<Rgba>; 3] = [
            vec![Rgba::default(); width],
            vec![Rgba::default(); width],
            vec![Rgba::default(); width],
        ];
        for (k, slot) in buf2.iter_mut().enumerate() {
            let s = ly as i32 - 1 + k as i32;
            if s & 1 == 0 {
                // Even scanline: chroma already valid → straight YCA->RGBA.
                yca::yca_to_rgba(&yw, &yca_rows[clamp(s)], slot);
            } else {
                // Odd scanline: reconstruct chroma from the surrounding N lines.
                let window: Vec<&[Rgba]> = (0..yca::N)
                    .map(|w| yca_rows[clamp(s - n2 as i32 + w as i32)].as_slice())
                    .collect();
                let mut tmp_yca = vec![Rgba::default(); width];
                yca::reconstruct_chroma_vert(&window, &mut tmp_yca);
                yca::yca_to_rgba(&yw, &tmp_yca, slot);
            }
        }
        let rows: [&[Rgba]; 3] = [&buf2[0], &buf2[1], &buf2[2]];
        yca::fix_saturation(&yw, &rows, &mut out[ly * width..(ly + 1) * width]);
    }

    Ok(RgbaImage { data_window: dw, pixels: out })
}

/// Luminance-only (grey) path — a `Y` channel with no chroma: read `Y` into
/// red and duplicate it into green/blue (`ImfRgbaFile.cpp:1220-1297,1354-1374`).
fn decode_grey<R: scanline::GenericInputReader>(
    input: &mut R,
    dw: Box2i,
    width: usize,
    height: usize,
    channel_name_prefix: &str,
) -> Result<RgbaImage> {
    let n = width * height;
    let y_off = 0usize;
    let a_off = y_off + n * 2;
    let total = (n + n) * 2;
    let mut buf = vec![0u8; total];

    let mut fb = FrameBuffer::new();
    fb.insert(format!("{channel_name_prefix}Y"), slice(y_off, dw, width, 1, 1, 0.0)?)?;
    fb.insert(format!("{channel_name_prefix}A"), slice(a_off, dw, width, 1, 1, 1.0)?)?;
    input.set_frame_buffer(&fb)?;
    input.read_pixels(dw.min.y, dw.max.y, &mut buf)?;

    let half_at = |off: usize, idx: usize| -> Half {
        let b = off + idx * 2;
        Half::from_bits(u16::from_ne_bytes([buf[b], buf[b + 1]]))
    };

    let mut out = vec![Rgba::default(); n];
    for (i, p) in out.iter_mut().enumerate() {
        let y = half_at(y_off, i);
        let a = half_at(a_off, i);
        *p = Rgba::new(y, y, y, a);
    }
    Ok(RgbaImage { data_window: dw, pixels: out })
}

/// Plain RGBA path — no `Y`, no chroma: `R`/`G`/`B`/`A` are read directly,
/// each filled if the file lacks that channel (`ImfRgbaFile.cpp:1220-1297`).
fn decode_plain_rgba<R: scanline::GenericInputReader>(
    input: &mut R,
    dw: Box2i,
    width: usize,
    height: usize,
    channel_name_prefix: &str,
) -> Result<RgbaImage> {
    let mut buf = vec![0u8; width * height * 8];
    input.set_frame_buffer(&plain_rgba_frame_buffer(channel_name_prefix, dw, width)?)?;
    input.read_pixels(dw.min.y, dw.max.y, &mut buf)?;
    Ok(RgbaImage { data_window: dw, pixels: rgba_from_interleaved(&buf) })
}

/// Build a `Slice` into the planar `buf` for a channel at byte `plane_off`.
pub(crate) fn slice(plane_off: usize, dw: Box2i, width: usize, xs: i32, ys: i32, fill: f64) -> Result<Slice> {
    let x_stride = 2usize; // HALF
    let y_stride = (width / xs as usize) * x_stride;
    Slice::make(
        PixelType::Half,
        plane_off,
        dw.min,
        width as i64,
        x_stride,
        y_stride,
        xs,
        ys,
        fill,
        false,
        false,
    )
}

/// Frame buffer reading `R`/`G`/`B`/`A` interleaved as half `[R, G, B, A]`
/// per pixel over `range` (x stride 8), the layout C++ `RgbaInputFile`
/// reads straight into its `Rgba` array. That shape is exactly what the
/// packed scatter fast path serves, where four separate planes plus a
/// second interleaving pass could not use it. A channel the file lacks is
/// filled (0 for R/G/B, 1 for A), which the generic scatter handles.
fn plain_rgba_frame_buffer(prefix: &str, range: Box2i, width: usize) -> Result<FrameBuffer> {
    let mut fb = FrameBuffer::new();
    for (i, (name, fill)) in [("R", 0.0), ("G", 0.0), ("B", 0.0), ("A", 1.0)].into_iter().enumerate() {
        let s = Slice::make(
            PixelType::Half,
            i * 2,
            range.min,
            width as i64,
            8,
            width * 8,
            1,
            1,
            fill,
            false,
            false,
        )?;
        fb.insert(format!("{prefix}{name}"), s)?;
    }
    Ok(fb)
}

/// Pixels from the interleaved half `[R, G, B, A]` bytes
/// [`plain_rgba_frame_buffer`] reads.
fn rgba_from_interleaved(bytes: &[u8]) -> Vec<Rgba> {
    let h = |b: &[u8]| Half::from_bits(u16::from_ne_bytes([b[0], b[1]]));
    bytes
        .as_chunks::<8>().0.iter()
        .map(|p| Rgba::new(h(&p[0..2]), h(&p[2..4]), h(&p[4..6]), h(&p[6..8])))
        .collect()
}

/// `padTmpBuf` — replicate the edge chroma into the N2 padding pixels on each
/// side before horizontal reconstruction (matches `ImfRgbaFile.cpp::padTmpBuf`).
pub(crate) fn pad_tmp(tmp: &mut [Rgba], width: usize) {
    let n2 = yca::N2;
    for i in 0..n2 {
        tmp[i] = tmp[n2];
        tmp[width + n2 + i] = tmp[width + n2 - 2];
    }
}

// ---------------------------------------------------------------------------
// Incremental (per-row) decode engine, used by `RgbaInputFile::read_pixels`.
// Each function reads exactly ONE on-disk scanline via a freshly-reopened
// (cheap — see module doc) `InputFile`; the YCA path additionally memoizes
// through `YcaCache` so a sequential scan stays O(1) amortized per new line.
// ---------------------------------------------------------------------------

/// Build a `Slice` for a SINGLE scanline `y`: like [`slice`], but the origin
/// is `(dw_min_x, y)` instead of the whole data window's minimum, so pixel
/// `(x, y)` lands at buffer offset `(x - dw_min_x) * x_stride` — offset 0 for
/// a buffer sized for exactly one row, regardless of where `y` falls in the
/// image (`Slice::make`'s `base_offset = ptr - offx - offy` cancels the `y`
/// term when `origin.y == y`).
pub(crate) fn slice_at_row(plane_off: usize, dw_min_x: i32, y: i32, width: usize, xs: i32, ys: i32, fill: f64) -> Result<Slice> {
    let x_stride = 2usize; // HALF
    let y_stride = (width / xs as usize) * x_stride;
    Slice::make(
        PixelType::Half,
        plane_off,
        imath_rs::V2i { x: dw_min_x, y },
        width as i64,
        x_stride,
        y_stride,
        xs,
        ys,
        fill,
        false,
        false,
    )
}

/// Reference implementation: decodes ONE scanline per call
/// via its own `input.read_pixels(y, y, ..)` -- kept ONLY as a `#[cfg(test)]`
/// byte-for-byte oracle for `decode_plain_range`'s output (see
/// `rgba::input::tests::plain_range_matches_row_by_row_reference` and
/// friends), since this is exactly the "N chunk decodes for N requested
/// rows" bug `decode_plain_range` fixes: called in a loop over a range, each
/// call independently re-locates and re-decodes whatever on-disk chunk
/// covers `y`, even when the PREVIOUS call's `y - 1` was in the SAME chunk.
#[cfg(test)]
fn decode_plain_row<'a, S: DerefMut<Target = dyn IStream + 'a>>(
    mp: &mut MultiPartInputFile<'a, S>,
    part: i32,
    prefix: &str,
    y: i32,
) -> Result<Vec<Rgba>> {
    let dw = mp.header(part)?.data_window()?;
    let width = (dw.max.x - dw.min.x + 1) as usize;
    let r_off = 0usize;
    let g_off = r_off + width * 2;
    let b_off = g_off + width * 2;
    let a_off = b_off + width * 2;
    let mut buf = vec![0u8; width * 4 * 2];

    let mut fb = FrameBuffer::new();
    fb.insert(format!("{prefix}R"), slice_at_row(r_off, dw.min.x, y, width, 1, 1, 0.0)?)?;
    fb.insert(format!("{prefix}G"), slice_at_row(g_off, dw.min.x, y, width, 1, 1, 0.0)?)?;
    fb.insert(format!("{prefix}B"), slice_at_row(b_off, dw.min.x, y, width, 1, 1, 0.0)?)?;
    fb.insert(format!("{prefix}A"), slice_at_row(a_off, dw.min.x, y, width, 1, 1, 1.0)?)?;
    let mut input = mp.input_part(part)?;
    input.set_frame_buffer(&fb)?;
    input.read_pixels(y, y, &mut buf)?;

    let half_at = |off: usize, idx: usize| -> Half {
        let b = off + idx * 2;
        Half::from_bits(u16::from_ne_bytes([buf[b], buf[b + 1]]))
    };
    let mut out = vec![Rgba::default(); width];
    for (i, p) in out.iter_mut().enumerate() {
        *p = Rgba::new(half_at(r_off, i), half_at(g_off, i), half_at(b_off, i), half_at(a_off, i));
    }
    Ok(out)
}

/// Reference implementation -- see `decode_plain_row`'s doc
/// (same "N chunk decodes for N requested rows" bug, same `#[cfg(test)]`
/// oracle role for `decode_grey_range`).
#[cfg(test)]
fn decode_grey_row<'a, S: DerefMut<Target = dyn IStream + 'a>>(
    mp: &mut MultiPartInputFile<'a, S>,
    part: i32,
    prefix: &str,
    y: i32,
) -> Result<Vec<Rgba>> {
    let dw = mp.header(part)?.data_window()?;
    let width = (dw.max.x - dw.min.x + 1) as usize;
    let y_off = 0usize;
    let a_off = y_off + width * 2;
    let mut buf = vec![0u8; width * 4];

    let mut fb = FrameBuffer::new();
    fb.insert(format!("{prefix}Y"), slice_at_row(y_off, dw.min.x, y, width, 1, 1, 0.0)?)?;
    fb.insert(format!("{prefix}A"), slice_at_row(a_off, dw.min.x, y, width, 1, 1, 1.0)?)?;
    let mut input = mp.input_part(part)?;
    input.set_frame_buffer(&fb)?;
    input.read_pixels(y, y, &mut buf)?;

    let half_at = |off: usize, idx: usize| -> Half {
        let b = off + idx * 2;
        Half::from_bits(u16::from_ne_bytes([buf[b], buf[b + 1]]))
    };
    let mut out = vec![Rgba::default(); width];
    for (i, p) in out.iter_mut().enumerate() {
        let yv = half_at(y_off, i);
        *p = Rgba::new(yv, yv, yv, half_at(a_off, i));
    }
    Ok(out)
}

/// Plain RGBA path, a whole `[y1, y2]` RANGE in ONE low-level `read_pixels`
/// call — batches exactly like the bulk `decode_plain_rgba`
/// does for the whole image, just scoped to the caller's range, with a
/// persistent `staging` buffer (grown/truncated via `ensure_len`, then fully
/// overwritten by the read) instead of a fresh allocation. This is the fix
/// for the real bug `decode_plain_row`/`#[cfg(test)]` reference: calling that
/// function once per row inside `[y1, y2]` re-locates and re-decodes
/// whatever on-disk chunk covers each row independently, so a chunk holding
/// N of the requested rows was decoded N times instead of once — matching C
/// `ImfRgbaFile.cpp:1343-1352`'s own single-range `readPixels(y1, y2)` call
/// per `RgbaInputFile::readPixels` invocation.
///
/// Returns `width * (y2 - y1 + 1)` pixels, row-major (row 0 == `y1`).
fn decode_plain_range<'a, S: DerefMut<Target = dyn IStream + 'a>>(
    mp: &mut MultiPartInputFile<'a, S>,
    part: i32,
    prefix: &str,
    y1: i32,
    y2: i32,
    staging: &mut Vec<u8>,
) -> Result<Vec<Rgba>> {
    let dw = mp.header(part)?.data_window()?;
    let width = (dw.max.x - dw.min.x + 1) as usize;
    let rows = (y2 - y1 + 1) as usize;
    crate::compressor::ensure_len(staging, width * rows * 8);

    let range = Box2i { min: imath_rs::V2i { x: dw.min.x, y: y1 }, max: imath_rs::V2i { x: dw.max.x, y: y2 } };
    let mut input = mp.input_part(part)?;
    input.set_frame_buffer(&plain_rgba_frame_buffer(prefix, range, width)?)?;
    input.read_pixels(y1, y2, staging)?;
    Ok(rgba_from_interleaved(staging))
}

/// Luminance-only (grey) path, a whole `[y1, y2]` range in ONE low-level
/// `read_pixels` call — see [`decode_plain_range`]'s doc for the full
/// rationale (identical shape, `Y`/`A` channels only).
fn decode_grey_range<'a, S: DerefMut<Target = dyn IStream + 'a>>(
    mp: &mut MultiPartInputFile<'a, S>,
    part: i32,
    prefix: &str,
    y1: i32,
    y2: i32,
    staging: &mut Vec<u8>,
) -> Result<Vec<Rgba>> {
    let dw = mp.header(part)?.data_window()?;
    let width = (dw.max.x - dw.min.x + 1) as usize;
    let rows = (y2 - y1 + 1) as usize;
    let n = width * rows;
    let y_off = 0usize;
    let a_off = y_off + n * 2;
    let total = n * 4;
    crate::compressor::ensure_len(staging, total);

    let range = Box2i { min: imath_rs::V2i { x: dw.min.x, y: y1 }, max: imath_rs::V2i { x: dw.max.x, y: y2 } };
    let mut fb = FrameBuffer::new();
    fb.insert(format!("{prefix}Y"), slice(y_off, range, width, 1, 1, 0.0)?)?;
    fb.insert(format!("{prefix}A"), slice(a_off, range, width, 1, 1, 1.0)?)?;
    let mut input = mp.input_part(part)?;
    input.set_frame_buffer(&fb)?;
    input.read_pixels(y1, y2, staging)?;

    let half_at = |off: usize, idx: usize| -> Half {
        let b = off + idx * 2;
        Half::from_bits(u16::from_ne_bytes([staging[b], staging[b + 1]]))
    };
    let mut out = vec![Rgba::default(); n];
    for (i, p) in out.iter_mut().enumerate() {
        let yv = half_at(y_off, i);
        *p = Rgba::new(yv, yv, yv, half_at(a_off, i));
    }
    Ok(out)
}

/// Euclidean mod (always non-negative for a positive `m`) — port of
/// `ImathFun.h`'s `modp`, used by `rotateBuf1`/`rotateBuf2`.
fn modp(d: i32, m: i32) -> i32 {
    ((d % m) + m) % m
}

/// Read ONE absolute scanline `y`'s luminance-chroma data (clamped to the
/// data window, `y > yMax` clamping to `yMax - 1`) — port of
/// `FromYca::readYCAScanLine`. No memoization here: the ring buffer in
/// [`yca_output_row`] IS the memoization, bounded to `N + 2` rows.
fn read_yca_scanline<'a, S: DerefMut<Target = dyn IStream + 'a>>(
    mp: &mut MultiPartInputFile<'a, S>,
    part: i32,
    prefix: &str,
    cache: &mut YcaCache,
    y: i32,
) -> Result<Vec<Rgba>> {
    let y = if y < cache.dw.min.y {
        cache.dw.min.y
    } else if y > cache.dw.max.y {
        cache.dw.max.y - 1
    } else {
        y
    };
    let width = cache.width;
    let cw = width / 2;
    // `dw.min.y` is a multiple of the chroma sampling factor (header rule),
    // so the ABSOLUTE parity used here matches C++'s `if (y & 1)` exactly.
    let even = y & 1 == 0;

    let yb = width;
    let ab = width;
    let cb = if even { cw } else { 0 };
    let y_off = 0usize;
    let a_off = y_off + yb * 2;
    let ry_off = a_off + ab * 2;
    let by_off = ry_off + cb * 2;
    // `line_buf` is a persistent, reused
    // buffer -- every byte within `[0, (yb+ab+2*cb)*2)` is written by
    // `read_pixels` below (a full-range decode into a freshly `ensure_len`'d
    // destination, same contract as every other `read_pixels`/`write_pixels`
    // call site).
    crate::compressor::ensure_len(&mut cache.line_buf, (yb + ab + 2 * cb) * 2);

    let mut fb = FrameBuffer::new();
    fb.insert(format!("{prefix}Y"), slice_at_row(y_off, cache.dw.min.x, y, width, 1, 1, 0.5)?)?;
    fb.insert(format!("{prefix}A"), slice_at_row(a_off, cache.dw.min.x, y, width, 1, 1, 1.0)?)?;
    if even {
        fb.insert(format!("{prefix}RY"), slice_at_row(ry_off, cache.dw.min.x, y, width, 2, 2, 0.0)?)?;
        fb.insert(format!("{prefix}BY"), slice_at_row(by_off, cache.dw.min.x, y, width, 2, 2, 0.0)?)?;
    }
    let mut input = mp.input_part(part)?;
    input.set_frame_buffer(&fb)?;
    input.read_pixels(y, y, &mut cache.line_buf)?;

    let half_at = |buf: &[u8], off: usize, idx: usize| -> Half {
        let b = off + idx * 2;
        Half::from_bits(u16::from_ne_bytes([buf[b], buf[b + 1]]))
    };

    let n2 = yca::N2;
    // `tmp_row` is a persistent, reused
    // buffer. `g`/`a` are written for EVERY `lx` (full coverage); `r`/`b` are
    // written only at EVEN `lx` when `even` — the ODD-`lx` positions keep
    // whatever they held from a PRIOR call (a "sentinel" that is NEVER
    // reset). This is sound (not a leak) because `reconstruct_chroma_horiz`'s
    // `RECONSTRUCT` filter taps only ever land on EVEN `lx` (odd output
    // positions interpolate from neighbouring EVEN samples; even output
    // positions read their OWN even sample directly) — verified empirically
    // by `poison_reused_yca_scanline_scratch_matches_fresh_instance`, not
    // just by tracing the taps.
    crate::compressor::ensure_len(&mut cache.tmp_row, width + yca::N - 1);
    for lx in 0..width {
        let p = &mut cache.tmp_row[n2 + lx];
        p.g = half_at(&cache.line_buf, y_off, lx);
        p.a = half_at(&cache.line_buf, a_off, lx);
    }
    if even {
        let mut lx = 0;
        while lx < width {
            cache.tmp_row[n2 + lx].r = half_at(&cache.line_buf, ry_off, lx / 2);
            cache.tmp_row[n2 + lx].b = half_at(&cache.line_buf, by_off, lx / 2);
            lx += 2;
        }
    }

    let mut row = vec![Rgba::default(); width];
    if even {
        pad_tmp(&mut cache.tmp_row, width);
        yca::reconstruct_chroma_horiz(&cache.tmp_row, &mut row);
    } else {
        // Odd rows carry no chroma sample at all; `r`/`b` stay `0.0`
        // (`Rgba::default()`), matching C++'s `_readC`-false zeroing branch —
        // `_readC` is always true when a `YcaCache` exists (only entered when
        // `WRITE_C` is set), so that branch never actually zeroes anything in
        // C++ either. These odd `r`/`b` values are never read downstream:
        // `reconstruct_chroma_vert`'s taps only touch EVEN absolute rows.
        row.copy_from_slice(&cache.tmp_row[n2..n2 + width]);
    }
    Ok(row)
}

/// The output row for absolute scanline `scan_line` — a faithful port of
/// `FromYca::readPixels(int scanLine)` (`ImfRgbaFile.cpp:904-1015`): rotate
/// `buf1`/`buf2` by `dy = scanLine - current_scan_line`, refill only the
/// newly-needed rows (or the whole ring, still bounded, on a jump `>= N+2`),
/// then `fixSaturation`. `Vec::rotate_left`/`rotate_right` moves only the row
/// `Vec`s' headers (24 bytes each), matching C++ rotating an array of row
/// POINTERS — no row data is copied on a sequential scan.
fn yca_output_row<'a, S: DerefMut<Target = dyn IStream + 'a>>(
    mp: &mut MultiPartInputFile<'a, S>,
    part: i32,
    prefix: &str,
    cache: &mut YcaCache,
    scan_line: i32,
) -> Result<Vec<Rgba>> {
    let n2 = yca::N2 as i32;
    let np2 = (yca::N + 2) as i32;
    let dy = scan_line - cache.current_scan_line;

    if dy.abs() < np2 {
        // `d = modp(dy, N+2); buf1[i] = old[(i+d) % (N+2)]` == `rotate_left(d)`.
        let d = modp(dy, np2) as usize;
        cache.buf1.rotate_left(d);
    }
    if dy.abs() < 3 {
        let d = modp(dy, 3) as usize;
        cache.buf2.rotate_left(d);
    }

    if dy < 0 {
        {
            let n = (-dy).min(np2);
            let y_min = scan_line - n2 - 1;
            for i in (0..n).rev() {
                cache.buf1[i as usize] = read_yca_scanline(mp, part, prefix, cache, y_min + i)?;
            }
        }
        {
            let n = (-dy).min(3);
            for i in 0..n {
                fill_buf2_slot(cache, scan_line, i, i as usize)?;
            }
        }
    } else {
        {
            let n = dy.min(np2);
            let y_max = scan_line + n2 + 1;
            for i in (0..n).rev() {
                cache.buf1[(yca::N as i32 + 1 - i) as usize] = read_yca_scanline(mp, part, prefix, cache, y_max - i)?;
            }
        }
        {
            let n = dy.min(3);
            // C++: `for (i = 2; i > 2 - n; --i)`.
            let mut i = 2;
            while i > 2 - n {
                fill_buf2_slot(cache, scan_line, i, i as usize)?;
                i -= 1;
            }
        }
    }

    let refs: [&[Rgba]; 3] = [&cache.buf2[0], &cache.buf2[1], &cache.buf2[2]];
    let mut out = vec![Rgba::default(); cache.width];
    yca::fix_saturation(&cache.yw, &refs, &mut out);
    cache.current_scan_line = scan_line;
    Ok(out)
}

/// Fill `cache.buf2[slot]` for absolute scanline `scan_line + i` — the body
/// shared by both directions' `buf2` refill loop in `FromYca::readPixels`
/// (`i` there ranges over `_buf1` positions `N2 + i`, i.e. offset `i` from the
/// window centre; `slot` is always `i` cast to `usize`, valid since `i` is
/// `0..3` or `2 - n + 1..=2`, both within `[0, 3)`).
fn fill_buf2_slot(cache: &mut YcaCache, scan_line: i32, i: i32, slot: usize) -> Result<()> {
    let n2 = yca::N2 as i32;
    let width = cache.width;
    let mut out = vec![Rgba::default(); width];
    if (scan_line + i) & 1 != 0 {
        yca::yca_to_rgba(&cache.yw, &cache.buf1[(n2 + i) as usize], &mut out);
    } else {
        let window: Vec<&[Rgba]> =
            cache.buf1[i as usize..i as usize + yca::N].iter().map(|v| v.as_slice()).collect();
        let mut tmp = vec![Rgba::default(); width];
        yca::reconstruct_chroma_vert(&window, &mut tmp);
        yca::yca_to_rgba(&cache.yw, &tmp, &mut out);
    }
    cache.buf2[slot] = out;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder};
    use crate::frame_buffer::Slice as WSlice;
    use crate::io::{MemIStream, MemOStream};
    use crate::scanline::OutputFile;
    use imath_rs::V2f;

    /// Build a tiny uncompressed EXR of the given HALF channels, each a
    /// constant value across every pixel, for round-trip tests below.
    fn write_exr(channels: &[(&str, f32)], width: usize, height: usize) -> Vec<u8> {
        let mut h = Header::with_dimensions(
            width as i32,
            height as i32,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for (name, _) in channels {
            cl.insert(*name, Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        }
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let n = width * height;
        let mut buf = vec![0u8; n * 2 * channels.len()];
        let mut fb = FrameBuffer::new();
        for (plane, (name, value)) in channels.iter().enumerate() {
            let base = plane * n * 2;
            for i in 0..n {
                let bits = Half::from_f32(*value).bits().to_ne_bytes();
                buf[base + i * 2..base + i * 2 + 2].copy_from_slice(&bits);
            }
            fb.insert(
                *name,
                WSlice {
                    type_: PixelType::Half,
                    base_offset: base as i64,
                    x_stride: 2,
                    y_stride: width * 2,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
        }

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&buf, height as i32).unwrap();
        }
        out.into_data()
    }

    /// Like [`write_exr`], but every pixel of every channel gets a distinct
    /// value (`base + (row*width + col + channel_idx) * 0.001`) instead of a
    /// constant, and the caller picks the compression — the batched-range
    /// matrix test needs both: non-constant content so a wrong row mapping
    /// is actually observable, and a real codec with a multi-line chunk so a
    /// non-chunk-aligned range genuinely straddles a chunk boundary.
    fn write_varying_exr(channels: &[&str], width: usize, height: usize, compression: Compression) -> Vec<u8> {
        let mut h = Header::with_dimensions(
            width as i32,
            height as i32,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            compression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for name in channels {
            cl.insert(*name, Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        }
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let n = width * height;
        let mut buf = vec![0u8; n * 2 * channels.len()];
        let mut fb = FrameBuffer::new();
        for (plane, name) in channels.iter().enumerate() {
            let base = plane * n * 2;
            for i in 0..n {
                let v = 0.01 + (i * channels.len() + plane) as f32 * 0.0003;
                let bits = Half::from_f32(v).bits().to_ne_bytes();
                buf[base + i * 2..base + i * 2 + 2].copy_from_slice(&bits);
            }
            fb.insert(
                *name,
                WSlice {
                    type_: PixelType::Half,
                    base_offset: base as i64,
                    x_stride: 2,
                    y_stride: width * 2,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
        }

        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&buf, height as i32).unwrap();
        }
        out.into_data()
    }

    /// `decode_plain_range`/`decode_grey_range` (the batched
    /// fix) must produce byte-identical output to the `#[cfg(test)]`
    /// `decode_plain_row`/`decode_grey_row` reference (one
    /// `input.read_pixels(y, y, ..)` call per row) across a MATRIX of
    /// compressions and a range that straddles a chunk boundary (NOT aligned
    /// to it), on non-constant content.
    #[test]
    fn plain_and_grey_range_match_row_by_row_reference_across_compressions() {
        // Height 20 with a 16-line-per-chunk codec (ZIP/PIZ/DWAA) gives two
        // chunks ([0,15], [16,19]); range [5, 18] straddles that boundary.
        let (w, h) = (6usize, 20usize);
        let (y1, y2) = (5i32, 18i32);
        for compression in [
            Compression::NoCompression,
            Compression::Zip,
            Compression::Piz,
            Compression::Dwaa,
        ] {
            // --- Plain RGBA ---
            let bytes = write_varying_exr(&["R", "G", "B", "A"], w, h, compression);
            let mut mis = MemIStream::new(bytes);
            let mut mp = MultiPartInputFile::new(&mut mis).unwrap();
            let batched = decode_plain_range(&mut mp, 0, "", y1, y2, &mut Vec::new()).unwrap();
            let mut row_by_row = Vec::new();
            for y in y1..=y2 {
                row_by_row.extend(decode_plain_row(&mut mp, 0, "", y).unwrap());
            }
            assert_eq!(
                batched, row_by_row,
                "plain RGBA: batched range decode diverges from the row-by-row \
                 reference at compression {compression:?}"
            );

            // --- Grey (Y-only) ---
            let bytes = write_varying_exr(&["Y", "A"], w, h, compression);
            let mut mis = MemIStream::new(bytes);
            let mut mp = MultiPartInputFile::new(&mut mis).unwrap();
            let batched = decode_grey_range(&mut mp, 0, "", y1, y2, &mut Vec::new()).unwrap();
            let mut row_by_row = Vec::new();
            for y in y1..=y2 {
                row_by_row.extend(decode_grey_row(&mut mp, 0, "", y).unwrap());
            }
            assert_eq!(
                batched, row_by_row,
                "grey: batched range decode diverges from the row-by-row \
                 reference at compression {compression:?}"
            );
        }
    }

    /// Explicit verification: `RgbaInputFile::
    /// read_pixels` must decode each on-disk CHUNK covering the requested
    /// range exactly ONCE, not once per requested SCANLINE -- the real bug
    /// the batched `decode_plain_range`/`decode_grey_range` fix replaces.
    /// A 10-row sub-range entirely inside ONE 16-line ZIP chunk must show
    /// exactly ONE `decode_chunk` call through the real, public
    /// `RgbaInputFile::read_pixels` entry point.
    #[test]
    fn read_pixels_decodes_each_chunk_once_not_once_per_row() {
        let (w, h) = (6usize, 32usize); // two 16-line ZIP chunks
        let bytes = write_varying_exr(&["R", "G", "B", "A"], w, h, Compression::Zip);
        let mut mis = MemIStream::new(bytes.clone());
        let mut file = RgbaInputFile::new(&mut mis).unwrap();

        // `base`/`x_stride`/`y_stride` address the FULL data window (index 0
        // == absolute y = dw.min.y = 0 here), per the module's own
        // convention -- sized for the whole image even though only a
        // sub-range is actually requested.
        crate::probe::reset(crate::probe::Probe::DecodeChunk);
        let mut base = vec![Rgba::default(); w * h];
        file.read_pixels(2, 11, &mut base, 1, w as isize).unwrap(); // 10 rows, one chunk
        assert_eq!(
            crate::probe::get(crate::probe::Probe::DecodeChunk),
            1,
            "a 10-row range entirely inside one 16-line ZIP chunk must decode \
             that chunk exactly once, not once per row"
        );

        // The first chunk remains cached; a straddling range only decodes the second.
        crate::probe::reset(crate::probe::Probe::DecodeChunk);
        let mut base2 = vec![Rgba::default(); w * h];
        file.read_pixels(10, 23, &mut base2, 1, w as isize).unwrap();
        assert_eq!(
            crate::probe::get(crate::probe::Probe::DecodeChunk),
            1,
            "the overlapping first chunk must be reused"
        );
        let mut fresh_stream = MemIStream::new(bytes);
        let mut fresh = RgbaInputFile::new(&mut fresh_stream).unwrap();
        crate::probe::reset(crate::probe::Probe::DecodeChunk);
        let mut fresh_base = vec![Rgba::default(); w * h];
        fresh.read_pixels(10, 23, &mut fresh_base, 1, w as isize).unwrap();
        assert_eq!(crate::probe::get(crate::probe::Probe::DecodeChunk), 2);
        assert_eq!(base2, fresh_base);
    }

    /// `YcaCache::line_buf`/`tmp_row` are now
    /// persistent, reused fields in `read_yca_scanline` instead of fresh
    /// allocations per scanline. Poison BOTH with implausible values
    /// (including `tmp_row`'s odd-`lx` `r`/`b` "sentinel" positions, which
    /// are never explicitly reset) BEFORE the
    /// first real `read_pixels` call, then decode the WHOLE image on that
    /// poisoned-then-reused instance and diff against a fresh instance —
    /// same methodology as B44's `poison_reused_scratch_matches_fresh_instance`.
    #[test]
    fn poison_reused_yca_scanline_scratch_matches_fresh_instance() {
        let (w, h) = (8usize, 6usize);
        let pixels: Vec<Rgba> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32;
                let y = (i / w) as f32;
                Rgba::new(
                    Half::from_f32(0.1 + x * 0.02),
                    Half::from_f32(0.2 + y * 0.03),
                    Half::from_f32(0.05 + (x + y) * 0.01),
                    Half::from_f32(0.9),
                )
            })
            .collect();
        let hdr = Header::with_dimensions(
            w as i32, h as i32, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::NoCompression,
        )
        .unwrap();

        let mut out = MemOStream::new();
        crate::rgba::output::RgbaOutputFile::write_stream(&mut out, &hdr, RgbaChannels::WRITE_YCA, &pixels).unwrap();
        let bytes = out.into_data();

        let mut poisoned_stream = MemIStream::new(bytes.clone());
        let mut poisoned = RgbaInputFile::new(&mut poisoned_stream).unwrap();
        match &mut poisoned.mode {
            DecodeMode::Yca(cache) => {
                cache.line_buf = vec![0xAAu8; 4096];
                cache.tmp_row = vec![
                    Rgba::new(
                        Half::from_f32(f32::NAN),
                        Half::from_f32(f32::NAN),
                        Half::from_f32(f32::NAN),
                        Half::from_f32(f32::NAN),
                    );
                    256
                ];
            }
            _ => panic!("expected YCA decode mode for a WRITE_YCA file"),
        }

        let mut fresh_stream = MemIStream::new(bytes);
        let mut fresh = RgbaInputFile::new(&mut fresh_stream).unwrap();

        let mut out_poisoned = vec![Rgba::default(); w * h];
        poisoned.read_pixels(0, (h - 1) as i32, &mut out_poisoned, 1, w as isize).unwrap();
        let mut out_fresh = vec![Rgba::default(); w * h];
        fresh.read_pixels(0, (h - 1) as i32, &mut out_fresh, 1, w as isize).unwrap();

        assert_eq!(
            out_poisoned, out_fresh,
            "a poisoned, reused YcaCache (line_buf/tmp_row) produced different decoded \
             pixels than a fresh instance"
        );
    }

    #[test]
    fn rgba_channels_of_detects_ry_without_by_as_chroma() {
        // The old code used `has("RY") && has("BY")`, so a file with
        // only ONE chroma channel was silently treated as having none.
        let mut cl = ChannelList::new();
        cl.insert("Y", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("RY", Channel::new(PixelType::Half, 2, 2, true)).unwrap();
        let rc = rgba_channels_of(&cl, "");
        assert!(rc.intersects(RgbaChannels::WRITE_C), "RY alone must set WRITE_C");

        let mut cl2 = ChannelList::new();
        cl2.insert("BY", Channel::new(PixelType::Half, 2, 2, true)).unwrap();
        let rc2 = rgba_channels_of(&cl2, "");
        assert!(rc2.intersects(RgbaChannels::WRITE_C), "BY alone must set WRITE_C");
    }

    #[test]
    fn rgba_channels_of_plain_and_grey() {
        let mut plain = ChannelList::new();
        for n in ["R", "G", "B", "A"] {
            plain.insert(n, Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        }
        let rc = rgba_channels_of(&plain, "");
        assert!(rc.contains(RgbaChannels::WRITE_RGBA));
        assert!(!rc.intersects(RgbaChannels::WRITE_Y | RgbaChannels::WRITE_C));

        let mut grey = ChannelList::new();
        grey.insert("Y", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        let rc = rgba_channels_of(&grey, "");
        assert!(rc.intersects(RgbaChannels::WRITE_Y));
        assert!(!rc.intersects(RgbaChannels::WRITE_C));
    }

    #[test]
    fn prefix_from_layer_name_rules() {
        let mut h = Header::with_dimensions(
            1,
            1,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        assert_eq!(prefix_from_layer_name("", &h).unwrap(), "");
        assert_eq!(prefix_from_layer_name("diffuse", &h).unwrap(), "diffuse.");

        h.add_multi_view(vec!["left".into(), "right".into()]).unwrap();
        assert_eq!(prefix_from_layer_name("left", &h).unwrap(), "", "default view has no prefix");
        assert_eq!(prefix_from_layer_name("right", &h).unwrap(), "right.");
    }

    #[test]
    fn plain_rgba_file_reads_without_a_y_channel() {
        // The old code hard-required a "Y" channel, so a plain
        // R/G/B/A file errored instead of reading.
        let bytes = write_exr(&[("R", 1.0), ("G", 2.0), ("B", 3.0), ("A", 0.5)], 2, 2);
        let mut stream = MemIStream::new(bytes);
        crate::probe::reset(crate::probe::Probe::ScatterChunkPacked);
        let image = RgbaInputFile::read_stream(&mut stream).expect("plain RGBA file must read");
        // An interleaved half RGBA frame buffer is exactly the packed
        // scatter's shape; the fast path must service the read.
        assert!(
            crate::probe::get(crate::probe::Probe::ScatterChunkPacked) > 0,
            "plain RGBA read did not take the packed scatter path"
        );
        assert_eq!(image.pixels.len(), 4);
        for p in &image.pixels {
            assert_eq!(p.r.to_f32(), 1.0);
            assert_eq!(p.g.to_f32(), 2.0);
            assert_eq!(p.b.to_f32(), 3.0);
            assert_eq!(p.a.to_f32(), 0.5);
        }
    }

    #[test]
    fn y_only_file_reads_as_grey() {
        // A Y-only file must read (grey = Y duplicated into R/G/B),
        // not error, and A defaults to 1.0 when absent.
        let bytes = write_exr(&[("Y", 0.25)], 2, 2);
        let mut stream = MemIStream::new(bytes);
        let image = RgbaInputFile::read_stream(&mut stream).expect("Y-only file must read");
        assert_eq!(image.pixels.len(), 4);
        for p in &image.pixels {
            assert_eq!(p.r.to_f32(), 0.25);
            assert_eq!(p.g.to_f32(), 0.25);
            assert_eq!(p.b.to_f32(), 0.25);
            assert_eq!(p.a.to_f32(), 1.0, "A defaults to 1.0 when absent");
        }
    }

    #[test]
    fn object_api_reads_the_same_pixels_as_the_free_function() {
        let bytes = write_exr(&[("R", 1.0), ("G", 2.0), ("B", 3.0), ("A", 0.5)], 2, 2);
        let via_free = RgbaInputFile::read_stream(&mut MemIStream::new(bytes.clone())).unwrap();

        let mut stream = MemIStream::new(bytes);
        let mut f = RgbaInputFile::new(&mut stream).unwrap();
        assert_eq!(f.parts(), 1);
        assert!(f.is_complete().unwrap());
        assert!(f.channels().unwrap().contains(RgbaChannels::WRITE_RGBA));
        let dw = f.data_window().unwrap();
        assert_eq!(dw, via_free.data_window);

        let via_object = f.read_all().unwrap();
        assert_eq!(via_object.pixels, via_free.pixels);

        // Single-row destination: y_stride = 0 (the one touched `y` must not
        // push the index outside `line`).
        let mut line = vec![Rgba::default(); 2];
        f.read_pixel_line(dw.min.y, &mut line, 1, 0).unwrap();
        assert_eq!(line, via_free.pixels[..2]);

        // read_pixels over the WHOLE image must match the bulk decode exactly
        // (cross-checks the incremental YCA-free plain-path engine too).
        let mut whole = vec![Rgba::default(); 4];
        f.read_pixels(dw.min.y, dw.max.y, &mut whole, 1, 2).unwrap();
        assert_eq!(whole, via_free.pixels);
    }

    /// `RgbaInputFile::open`
    /// OWNS a path-opened stream (no caller-side `&mut` to keep alive) --
    /// port of C++'s `RgbaInputFile(const char*, int)` constructor, which
    /// this crate could not provide before `MultiPartInputFile` gained the
    /// generic-over-stream-holder split (`MultiPartInputFile::new_owned`).
    /// Asserts it reads pixel-identical to the borrowed `new` path on the
    /// same bytes.
    #[test]
    fn open_owns_a_path_opened_stream_and_reads_identically_to_new() {
        let bytes = write_exr(&[("R", 1.0), ("G", 2.0), ("B", 3.0), ("A", 0.5)], 2, 2);
        let path = std::env::temp_dir()
            .join(format!("exr_core_rgba_open_{}.exr", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let mut owned = RgbaInputFile::open(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let mut stream = MemIStream::new(bytes);
        let mut borrowed = RgbaInputFile::new(&mut stream).unwrap();

        assert_eq!(owned.parts(), borrowed.parts());
        assert_eq!(owned.data_window().unwrap(), borrowed.data_window().unwrap());
        assert_eq!(
            owned.read_all().unwrap().pixels,
            borrowed.read_all().unwrap().pixels
        );
    }

    /// Incremental `read_pixels` on a YCA file must match the bulk decode
    /// used by `read_all`/`read_stream` line-for-line, whether read forward,
    /// backward, or as isolated single lines (exercising the memoization
    /// cache's edge-clamp / odd-vs-even-parity branch selection).
    #[test]
    fn incremental_read_pixels_matches_bulk_decode_on_a_yca_file() {
        // 4x4 Y + 2x2-subsampled RY/BY, non-constant so a wrong row/clamp
        // would very likely be caught (each channel varies per pixel/sample).
        let mut h = Header::with_dimensions(
            4,
            4,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Y", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("RY", Channel::new(PixelType::Half, 2, 2, true)).unwrap();
        cl.insert("BY", Channel::new(PixelType::Half, 2, 2, true)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let (w, hgt) = (4usize, 4usize);
        let (cw, ch) = (2usize, 2usize);
        let y_n = w * hgt;
        let c_n = cw * ch;
        let y_base = 0usize;
        let ry_base = y_n * 2;
        let by_base = ry_base + c_n * 2;
        let mut buf = vec![0u8; (y_n + 2 * c_n) * 2];
        let put = |buf: &mut [u8], off: usize, i: usize, v: f32| {
            let bits = Half::from_f32(v).bits().to_ne_bytes();
            buf[off + i * 2..off + i * 2 + 2].copy_from_slice(&bits);
        };
        for y in 0..hgt {
            for x in 0..w {
                put(&mut buf, y_base, y * w + x, 0.1 + (y * w + x) as f32 * 0.01);
            }
        }
        for cy in 0..ch {
            for cx in 0..cw {
                let idx = cy * cw + cx;
                put(&mut buf, ry_base, idx, 0.05 * (idx as f32 + 1.0));
                put(&mut buf, by_base, idx, -0.03 * (idx as f32 + 1.0));
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert(
            "Y",
            WSlice {
                type_: PixelType::Half,
                base_offset: y_base as i64,
                x_stride: 2,
                y_stride: w * 2,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        for (name, base) in [("RY", ry_base), ("BY", by_base)] {
            fb.insert(
                name,
                WSlice {
                    type_: PixelType::Half,
                    base_offset: base as i64,
                    x_stride: 2,
                    y_stride: cw * 2,
                    x_sampling: 2,
                    y_sampling: 2,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
        }
        let mut out = MemOStream::new();
        {
            let mut of = OutputFile::new(&mut out, &h).unwrap();
            of.set_frame_buffer(&fb).unwrap();
            of.write_pixels(&buf, hgt as i32).unwrap();
        }
        let bytes = out.into_data();

        let bulk = RgbaInputFile::read_stream(&mut MemIStream::new(bytes.clone())).unwrap();

        let mut stream = MemIStream::new(bytes);
        let mut f = RgbaInputFile::new(&mut stream).unwrap();
        assert!(f.channels().unwrap().intersects(RgbaChannels::WRITE_C));

        // Whole image, one call.
        let mut whole = vec![Rgba::default(); w * hgt];
        f.read_pixels(0, (hgt - 1) as i32, &mut whole, 1, w as isize).unwrap();
        assert_eq!(whole, bulk.pixels, "whole-image incremental read must match the bulk decode");

        // Re-request line-by-line, out of order, to exercise the cache from
        // a cold state again with a different access pattern.
        let mut stream2 = MemIStream::new(out_bytes_recreate(&h, &fb, &buf, hgt));
        let mut f2 = RgbaInputFile::new(&mut stream2).unwrap();
        for &y in &[3i32, 0, 2, 1] {
            // A single-row destination: y_stride = 0 (only one absolute `y`
            // is ever touched by a one-line call, so its stride term must
            // not push the index outside `line`).
            let mut line = vec![Rgba::default(); w];
            f2.read_pixel_line(y, &mut line, 1, 0).unwrap();
            let expected = &bulk.pixels[y as usize * w..(y as usize + 1) * w];
            assert_eq!(line, expected, "line {y} mismatch under out-of-order access");
        }
    }

    /// Re-run the same writer as `incremental_read_pixels_matches_bulk_decode_on_a_yca_file`
    /// to get a second, independent stream (the first was consumed by `MemIStream`).
    fn out_bytes_recreate(h: &Header, fb: &FrameBuffer, buf: &[u8], height: usize) -> Vec<u8> {
        let mut out = MemOStream::new();
        let mut of = OutputFile::new(&mut out, h).unwrap();
        of.set_frame_buffer(fb).unwrap();
        of.write_pixels(buf, height as i32).unwrap();
        out.into_data()
    }

    /// The YCA cache must be a BOUNDED ring (`N + 2` +
    /// 3 rows), not a memo table that grows with image height. Build a YCA
    /// image much taller than the ring, scan it top-to-bottom AND
    /// bottom-to-top, and check `buf1`/`buf2` never change length — plus
    /// cross-check every pixel against the bulk decoder, so a broken rotation
    /// (wrong values) would fail alongside an unbounded one (wrong sizes).
    #[test]
    fn yca_ring_buffer_stays_bounded_and_correct_past_its_own_size() {
        let (w, hgt) = (2usize, 64usize); // hgt > N + 2 (29)
        let mut h = Header::with_dimensions(
            w as i32,
            hgt as i32,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Y", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("RY", Channel::new(PixelType::Half, 2, 2, true)).unwrap();
        cl.insert("BY", Channel::new(PixelType::Half, 2, 2, true)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();

        let (cw, ch) = (w / 2, hgt / 2);
        let y_n = w * hgt;
        let c_n = cw * ch;
        let (y_base, ry_base, by_base) = (0usize, y_n * 2, y_n * 2 + c_n * 2);
        let mut buf = vec![0u8; (y_n + 2 * c_n) * 2];
        let put = |buf: &mut [u8], off: usize, i: usize, v: f32| {
            let bits = Half::from_f32(v).bits().to_ne_bytes();
            buf[off + i * 2..off + i * 2 + 2].copy_from_slice(&bits);
        };
        for y in 0..hgt {
            for x in 0..w {
                put(&mut buf, y_base, y * w + x, 0.05 + (y * w + x) as f32 * 0.001);
            }
        }
        for cy in 0..ch {
            for cx in 0..cw {
                let idx = cy * cw + cx;
                put(&mut buf, ry_base, idx, 0.02 * ((idx % 7) as f32 - 3.0));
                put(&mut buf, by_base, idx, -0.015 * ((idx % 5) as f32 - 2.0));
            }
        }
        let mut fb = FrameBuffer::new();
        fb.insert(
            "Y",
            WSlice {
                type_: PixelType::Half,
                base_offset: y_base as i64,
                x_stride: 2,
                y_stride: w * 2,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        for (name, base) in [("RY", ry_base), ("BY", by_base)] {
            fb.insert(
                name,
                WSlice {
                    type_: PixelType::Half,
                    base_offset: base as i64,
                    x_stride: 2,
                    y_stride: cw * 2,
                    x_sampling: 2,
                    y_sampling: 2,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
        }
        let bytes = out_bytes_recreate(&h, &fb, &buf, hgt);
        let bulk = RgbaInputFile::read_stream(&mut MemIStream::new(bytes.clone())).unwrap();

        // Forward scan.
        let mut stream = MemIStream::new(bytes.clone());
        let mut f = RgbaInputFile::new(&mut stream).unwrap();
        for y in 0..hgt as i32 {
            let mut line = vec![Rgba::default(); w];
            f.read_pixel_line(y, &mut line, 1, 0).unwrap();
            assert_eq!(line, bulk.pixels[y as usize * w..(y as usize + 1) * w], "forward scan, line {y}");
            match &f.mode {
                DecodeMode::Yca(cache) => {
                    assert_eq!(cache.buf1.len(), yca::N + 2, "buf1 must stay fixed-size");
                    assert_eq!(cache.buf2.len(), 3, "buf2 must stay fixed-size");
                }
                _ => panic!("expected YCA mode"),
            }
        }

        // Backward scan on a fresh object, to exercise the `dy < 0` branch.
        let mut stream2 = MemIStream::new(bytes);
        let mut f2 = RgbaInputFile::new(&mut stream2).unwrap();
        for y in (0..hgt as i32).rev() {
            let mut line = vec![Rgba::default(); w];
            f2.read_pixel_line(y, &mut line, 1, 0).unwrap();
            assert_eq!(line, bulk.pixels[y as usize * w..(y as usize + 1) * w], "backward scan, line {y}");
        }
    }
}
