//! `RgbaOutputFile` — port of `Imf::RgbaOutputFile` / `ToYca` (`ImfRgbaFile.{h,cpp}`).
//!
//! Writes one of the same three shapes [`super::input`] reads (see that
//! module's doc): luminance-chroma (YCA, when [`RgbaChannels::WRITE_C`] is
//! requested), luminance-only (grey, `WRITE_Y` without `WRITE_C`), or plain
//! R/G/B/A (`insertChannels`, `ImfRgbaFile.cpp:38-63`).
//!
//! # Streaming, bounded memory (the strip-writing use case at 16k-65k)
//! [`RgbaOutputFile`] is a faithful, STREAMING port of C++'s
//! `RgbaOutputFile`/`ToYca`, not a whole-image-at-once writer: `new` opens
//! the file and header, [`set_frame_buffer`](RgbaOutputFile::set_frame_buffer)
//! records how to read the CALLER's `&[Rgba]` buffer (`x_stride`/`y_stride`,
//! matching C++'s `Rgba* base, size_t xStride, size_t yStride`), and
//! [`write_pixels`](RgbaOutputFile::write_pixels) can be called repeatedly
//! with any number of scanlines per call (1, 7, 48, ... — the caller decides
//! the strip height; a 65536-row image is never materialised at once).
//!
//! For the YCA path this ports `ToYca`'s ACTUAL ring-buffer state machine
//! (`_buf[N]`, `_tmpBuf`, `rotateBuffers`/`duplicateLastBuffer`/
//! `duplicateSecondToLastBuffer`, `_currentScanLine`/`_linesConverted`) — see
//! [`ToYcaState`] — not a bulk/clamp-derived equivalent: memory is bounded by
//! `N` (27) rows regardless of image height or the caller's strip size,
//! exactly like C++. `Vec::rotate_left(1)` on the row-`Vec` array is the safe
//! Rust equivalent of C++ rotating an array of row POINTERS — no row data is
//! copied by a rotation, only `duplicate_last_buffer`/
//! `duplicate_second_to_last_buffer` (the edge-padding steps) actually clone
//! a row. Every low-level `scanline::OutputFile::write_pixels` call writes
//! exactly ONE converted scanline, from a freshly-built single-row
//! `FrameBuffer` (`slice_at_row`) — the same "reused one-row slice" pattern
//! the read side's incremental engine uses, verified there to be
//! byte-identical to C++.
//!
//! `write()`/`write_stream()` remain as thin one-shot convenience wrappers:
//! `new` + `set_frame_buffer` (tightly packed) + one `write_pixels` call for
//! the whole image.

use imath_rs::{Box2i, Half};

use crate::attr::{Channel, ChannelList, ChannelListAttribute, LineOrder, PixelType};
use crate::error::{ExrError, Result};
use crate::frame_buffer::FrameBuffer;
use crate::header::Header;
use crate::io::OStream;
use crate::rgba::input::{pad_tmp, slice_at_row, yw_from_header};
use crate::rgba::pixel::RgbaChannels;
use crate::rgba::yca;
use crate::rgba::Rgba;
use crate::scanline;

use std::path::Path;

/// The `roundYCA` defaults `ToYca` uses (`_roundY = 7; _roundC = 5;`,
/// `ImfRgbaFile.cpp:229-230`).
const DEFAULT_ROUND_Y: u32 = 7;
const DEFAULT_ROUND_C: u32 = 5;

/// Streaming writer for luminance-chroma (and plain RGBA) scanline EXR files.
/// See the module doc for the streaming contract and the ring-buffer port.
pub struct RgbaOutputFile<'a> {
    of: scanline::OutputFile<'a>,
    header: Header,
    dw: Box2i,
    width: usize,
    rc: RgbaChannels,
    line_order: LineOrder,
    /// The NEXT absolute scanline this writer will hand to the low-level
    /// `scanline::OutputFile` — C++ `RgbaOutputFile`/`ToYca`'s
    /// `_currentScanLine` (both paths track it identically: init to
    /// `dw.min.y`/`dw.max.y` by line order, stepped by one PER OUTPUT row).
    current_scan_line: i32,
    mode: WriteMode,
    /// Caller frame-buffer strides, in units of [`Rgba`] — set once via
    /// [`RgbaOutputFile::set_frame_buffer`] (C++ `Rgba* base, size_t
    /// xStride, size_t yStride`; `base` itself is supplied fresh to every
    /// [`write_pixels`](RgbaOutputFile::write_pixels) call instead of being
    /// stored, since C++ ToYca buffers the CONVERTED rows internally, not a
    /// borrow of the caller's memory).
    fb: Option<(isize, isize)>,
    /// Persistent Plain-mode batched-range scratch:
    /// `range_rgba` holds the gathered `Rgba`s for the WHOLE batch,
    /// row-major with row 0 == the batch's smallest absolute `y` (matching
    /// [`super::input::slice`]'s addressing convention regardless of line
    /// order); `range_staging` holds the resulting byte planes. See
    /// [`Self::write_plain_range`]'s doc for why this replaces a fresh
    /// `FrameBuffer` + fresh byte buffer rebuilt on EVERY single row.
    range_rgba: Vec<Rgba>,
    range_staging: Vec<u8>,
    /// Persistent per-INPUT-row scratch shared by Grey/YCA:
    /// unlike Plain, these two modes keep C++'s own per-row
    /// emission structure (`ToYca`'s ring buffer genuinely needs adjacent
    /// rows one at a time) -- only the allocations become persistent.
    row_buf: Vec<Rgba>,
    yca_row_staging: Vec<u8>,
}

/// Which decode path is active — mirrors [`super::input`]'s `DecodeMode`.
enum WriteMode {
    Plain,
    Grey { yw: [f32; 3] },
    Yca(ToYcaState),
}

/// `ToYca`'s ring-buffer state — a faithful, bounded-memory port of
/// `_buf[N]`/`_tmpBuf`/`_linesConverted` (`ImfRgbaFile.cpp:144-229`).
struct ToYcaState {
    yw: [f32; 3],
    write_y: bool,
    write_a: bool,
    width: usize,
    height: usize,
    /// C++ `_linesConverted` — count of INPUT rows converted so far.
    lines_converted: i32,
    round_y: u32,
    round_c: u32,
    /// `N` (27) horizontally-decimated rows — C++ `_buf[N]`.
    buf: Vec<Vec<Rgba>>,
    /// Persistent per-input-row scratch:
    /// `yca_row_scratch` holds `rgba_to_yca`'s output;
    /// `padded_scratch` holds the padded row `decimate_chroma_horiz` reads
    /// from. Both are fully overwritten every call (the middle region by a
    /// full-width copy, the padding by `pad_tmp`), so `ensure_len` is sound.
    yca_row_scratch: Vec<Rgba>,
    padded_scratch: Vec<Rgba>,
    /// Persistent per-OUTPUT-row scratch for `decimate_chroma_vert_and_write_scanline`
    /// -- fully overwritten either way (a full-width `copy_from_slice` or
    /// `decimate_chroma_vert`'s own full-width write).
    vert_tmp: Vec<Rgba>,
}

impl<'a> RgbaOutputFile<'a> {
    /// Open `stream` for `header`'s dimensions/compression/line order,
    /// writing the channels `rgba_channels` selects — port of
    /// `RgbaOutputFile(OStream&, const Header&, RgbaChannels, int)`.
    pub fn new(stream: &'a mut dyn OStream, header: &Header, rgba_channels: RgbaChannels) -> Result<Self> {
        let mut h = header.clone();
        insert_channels(&mut h, rgba_channels)?;
        let dw = h.data_window()?;
        let width = (dw.max.x - dw.min.x + 1) as usize;
        let height = (dw.max.y - dw.min.y + 1) as usize;
        let line_order = h.line_order()?;
        let of = scanline::OutputFile::new(stream, &h)?;

        let mode = if rgba_channels.intersects(RgbaChannels::WRITE_C) {
            WriteMode::Yca(ToYcaState {
                yw: yw_from_header(&h)?,
                write_y: rgba_channels.intersects(RgbaChannels::WRITE_Y),
                write_a: rgba_channels.intersects(RgbaChannels::WRITE_A),
                width,
                height,
                lines_converted: 0,
                round_y: DEFAULT_ROUND_Y,
                round_c: DEFAULT_ROUND_C,
                buf: vec![vec![Rgba::default(); width]; yca::N],
                yca_row_scratch: Vec::new(),
                padded_scratch: Vec::new(),
                vert_tmp: Vec::new(),
            })
        } else if rgba_channels.intersects(RgbaChannels::WRITE_Y) {
            WriteMode::Grey { yw: yw_from_header(&h)? }
        } else {
            WriteMode::Plain
        };

        let current_scan_line = if line_order == LineOrder::IncreasingY { dw.min.y } else { dw.max.y };
        Ok(Self {
            of,
            header: h,
            dw,
            width,
            rc: rgba_channels,
            line_order,
            current_scan_line,
            mode,
            fb: None,
            range_rgba: Vec::new(),
            range_staging: Vec::new(),
            row_buf: Vec::new(),
            yca_row_staging: Vec::new(),
        })
    }

    /// Record how to read the caller's buffer in [`write_pixels`](Self::write_pixels)
    /// — port of `RgbaOutputFile::setFrameBuffer(Rgba*, size_t xStride, size_t
    /// yStride)`, minus the pointer (supplied per call instead — see the
    /// module doc). `x_stride`/`y_stride` are in units of [`Rgba`].
    pub fn set_frame_buffer(&mut self, x_stride: isize, y_stride: isize) -> Result<()> {
        self.fb = Some((x_stride, y_stride));
        Ok(())
    }

    /// Change the `Y`/chroma rounding (`ToYca::setYCRounding`); only has an
    /// effect on the YCA path. Must be called before the relevant
    /// [`write_pixels`](Self::write_pixels) calls to affect them (C++ applies
    /// the CURRENT rounding at write time, not retroactively).
    pub fn set_yc_rounding(&mut self, round_y: u32, round_c: u32) {
        if let WriteMode::Yca(state) = &mut self.mode {
            state.round_y = round_y;
            state.round_c = round_c;
        }
    }

    /// Write `num_scan_lines` scanlines, reading them from `pixels` via the
    /// strides [`set_frame_buffer`](Self::set_frame_buffer) recorded — port
    /// of `RgbaOutputFile::writePixels(int)`. May be called repeatedly with
    /// any strip height; memory stays bounded (see the module doc).
    ///
    /// # Errors
    /// [`ExrError::Arg`] if [`set_frame_buffer`](Self::set_frame_buffer) was
    /// never called, or `pixels` is too short for the strides given.
    pub fn write_pixels(&mut self, pixels: &[Rgba], num_scan_lines: i32) -> Result<()> {
        let (x_stride, y_stride) = self.fb.ok_or_else(|| {
            ExrError::Arg("No frame buffer was specified as the pixel data source.".into())
        })?;

        // Plain: ONE batched range write --
        // see `write_plain_range`'s doc for why this replaces a per-row
        // loop (matches the read-side `decode_plain_range` fix exactly).
        // Grey/YCA keep C++'s own per-INPUT-row emission structure (`ToYca`
        // genuinely needs adjacent rows one at a time for chroma
        // reconstruction, and Grey reuses that same machinery) -- only the
        // per-row `Rgba` gather buffer becomes persistent.
        if matches!(self.mode, WriteMode::Plain) {
            return self.write_plain_range(pixels, num_scan_lines, x_stride, y_stride);
        }

        for i in 0..num_scan_lines as isize {
            let mut row = std::mem::take(&mut self.row_buf);
            crate::compressor::ensure_len(&mut row, self.width);
            for (x, slot) in row.iter_mut().enumerate() {
                let idx = i * y_stride + x as isize * x_stride;
                let idx = usize::try_from(idx)
                    .map_err(|_| ExrError::Arg("writePixels: source index is negative.".into()))?;
                *slot = *pixels
                    .get(idx)
                    .ok_or_else(|| ExrError::Arg("writePixels: source index is out of range.".into()))?;
            }
            let result = self.convert_and_write_row(&row);
            self.row_buf = row;
            result?;
        }
        Ok(())
    }

    /// Write a WHOLE `num_scan_lines`-row batch of Plain-mode pixels in ONE
    /// low-level `write_pixels` call — mirrors
    /// `decode_plain_range`'s read-side fix. Builds ONE `FrameBuffer`
    /// covering the batch's absolute y-range plus ONE persistent byte
    /// staging buffer, instead of a fresh `FrameBuffer` and a fresh byte
    /// buffer rebuilt on EVERY input row — the real bug: besides the
    /// allocation churn, forcing every low-level `write_pixels` call down to
    /// exactly 1 line also prevented that writer's own PARALLEL block-write
    /// path from ever seeing more than one line per call. Matches C
    /// `RgbaOutputFile::writePixels`'s single `_outputFile->writePixels(n)`
    /// call over a frame buffer set ONCE.
    ///
    /// `pixels`/`x_stride`/`y_stride` address INPUT row `i`
    /// (`0..num_scan_lines`) the same way the old per-row loop did;
    /// `range_rgba`/`range_staging` are laid out with row 0 == the batch's
    /// SMALLEST absolute `y` regardless of line order (matching
    /// [`super::input::slice`]'s own addressing convention), since the
    /// low-level writer reads the frame buffer by ABSOLUTE `y`, not by
    /// input order.
    fn write_plain_range(&mut self, pixels: &[Rgba], num_scan_lines: i32, x_stride: isize, y_stride: isize) -> Result<()> {
        let n = num_scan_lines as usize;
        let width = self.width;
        if n == 0 {
            return Ok(());
        }
        let step: i32 = if self.line_order == LineOrder::IncreasingY { 1 } else { -1 };
        let y_start = self.current_scan_line;
        let range_min = if step == 1 { y_start } else { y_start - (n as i32 - 1) };
        let range_max = if step == 1 { y_start + (n as i32 - 1) } else { y_start };

        crate::compressor::ensure_len(&mut self.range_rgba, n * width);
        for i in 0..n as i32 {
            let y = y_start + i * step;
            let local_row = (y - range_min) as usize;
            for x in 0..width {
                let idx = (i as isize) * y_stride + (x as isize) * x_stride;
                let idx = usize::try_from(idx)
                    .map_err(|_| ExrError::Arg("writePixels: source index is negative.".into()))?;
                let px = *pixels
                    .get(idx)
                    .ok_or_else(|| ExrError::Arg("writePixels: source index is out of range.".into()))?;
                self.range_rgba[local_row * width + x] = px;
            }
        }

        let mut next = 0usize;
        let mut plane = |want: bool| -> Option<usize> {
            if !want {
                return None;
            }
            let off = next;
            next += n * width * 2;
            Some(off)
        };
        let r_off = plane(self.rc.intersects(RgbaChannels::WRITE_R));
        let g_off = plane(self.rc.intersects(RgbaChannels::WRITE_G));
        let b_off = plane(self.rc.intersects(RgbaChannels::WRITE_B));
        let a_off = plane(self.rc.intersects(RgbaChannels::WRITE_A));
        crate::compressor::ensure_len(&mut self.range_staging, next);

        for (i, p) in self.range_rgba[..n * width].iter().enumerate() {
            if let Some(off) = r_off {
                put_half(&mut self.range_staging, off, i, p.r);
            }
            if let Some(off) = g_off {
                put_half(&mut self.range_staging, off, i, p.g);
            }
            if let Some(off) = b_off {
                put_half(&mut self.range_staging, off, i, p.b);
            }
            if let Some(off) = a_off {
                put_half(&mut self.range_staging, off, i, p.a);
            }
        }

        let range_box = Box2i {
            min: imath_rs::V2i { x: self.dw.min.x, y: range_min },
            max: imath_rs::V2i { x: self.dw.max.x, y: range_max },
        };
        let mut fb = FrameBuffer::new();
        for (off, name) in [(r_off, "R"), (g_off, "G"), (b_off, "B"), (a_off, "A")] {
            if let Some(off) = off {
                fb.insert(name, crate::rgba::input::slice(off, range_box, width, 1, 1, 0.0)?)?;
            }
        }
        self.of.set_frame_buffer(&fb)?;
        self.of.write_pixels(&self.range_staging[..next], num_scan_lines)?;

        self.current_scan_line = y_start + step * n as i32;
        Ok(())
    }

    /// One INPUT row: dispatch to the active [`WriteMode`], writing zero
    /// (never, for Plain/Grey) or more (Yca, during the end-of-image flush)
    /// OUTPUT scanlines to the low-level writer.
    fn convert_and_write_row(&mut self, row: &[Rgba]) -> Result<()> {
        // `mode` is taken out for the duration so `write_output_row` (which
        // needs `&mut self.of`/`self.current_scan_line`) can be called from
        // within the `Yca` arm without an aliased double-borrow of `self`.
        let mut mode = std::mem::replace(&mut self.mode, WriteMode::Plain);
        let result = match &mut mode {
            WriteMode::Plain => self.write_output_row(row),
            WriteMode::Grey { yw } => {
                let write_a = self.rc.intersects(RgbaChannels::WRITE_A);
                let mut out = vec![Rgba::default(); self.width];
                yca::rgba_to_yca(yw, write_a, row, &mut out);
                self.write_grey_output_row(write_a, &out)
            }
            WriteMode::Yca(state) => yca_convert_and_write_row(self, state, row),
        };
        self.mode = mode;
        result
    }

    /// Write ONE already-final row (Plain: as-is; Grey/Yca: already `Y`/A`/
    /// chroma in `Rgba` layout) to the low-level writer at
    /// `self.current_scan_line`, then step it — shared tail of every mode.
    fn write_output_row(&mut self, row: &[Rgba]) -> Result<()> {
        write_plain_row(&mut self.of, self.dw, self.width, self.rc, self.current_scan_line, row)?;
        self.step_scan_line();
        Ok(())
    }

    /// Luminance-only row (`Y` and optionally `A`, no chroma): C++ `ToYca::writePixels`'s
    /// `_writeY && !_writeC` branch.
    fn write_grey_output_row(&mut self, write_a: bool, row: &[Rgba]) -> Result<()> {
        write_yca_row_bytes(
            &mut self.of, self.dw, self.width, true, false, write_a, self.current_scan_line, row,
            &mut self.yca_row_staging,
        )?;
        self.step_scan_line();
        Ok(())
    }

    fn step_scan_line(&mut self) {
        if self.line_order == LineOrder::IncreasingY {
            self.current_scan_line += 1;
        } else {
            self.current_scan_line -= 1;
        }
    }

    // -- Accessors — port of the matching `RgbaOutputFile` methods, all
    // simple reads of the (post-`insertChannels`) header. -------------------

    pub fn header(&self) -> &Header {
        &self.header
    }
    /// The `(x_stride, y_stride)` [`set_frame_buffer`](Self::set_frame_buffer)
    /// last installed, in units of [`Rgba`] — port of
    /// `RgbaOutputFile::frameBuffer()`. C++ returns the registered
    /// `FrameBuffer` object; the base-relative stride pair IS its entire
    /// caller-visible content (this port has no persistent low-level
    /// `FrameBuffer` — see the module doc), so it is what's returned here.
    /// `None` before the first [`set_frame_buffer`](Self::set_frame_buffer) call.
    pub fn frame_buffer(&self) -> Option<(isize, isize)> {
        self.fb
    }
    pub fn data_window(&self) -> Box2i {
        self.dw
    }
    pub fn display_window(&self) -> Result<Box2i> {
        self.header.display_window()
    }
    pub fn pixel_aspect_ratio(&self) -> Result<f32> {
        self.header.pixel_aspect_ratio()
    }
    pub fn screen_window_center(&self) -> Result<imath_rs::V2f> {
        self.header.screen_window_center()
    }
    pub fn screen_window_width(&self) -> Result<f32> {
        self.header.screen_window_width()
    }
    pub fn line_order(&self) -> LineOrder {
        self.line_order
    }
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.header.compression()
    }
    /// The channels this file writes — port of `RgbaOutputFile::channels`.
    pub fn channels(&self) -> RgbaChannels {
        self.rc
    }
    /// C++ `RgbaOutputFile::currentScanLine` (`ToYca::currentScanLine` for
    /// the YCA path, the low-level writer's own position otherwise — both
    /// equal `self.current_scan_line` here since every mode steps it
    /// identically, once per OUTPUT row).
    pub fn current_scan_line(&self) -> i32 {
        self.current_scan_line
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk — port of `RgbaOutputFile::updatePreviewImage`
    /// (`ImfRgbaFile.cpp:716-719`, a one-line forward to
    /// `OutputFile::updatePreviewImage`).
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.of.update_preview_image(new_pixels)
    }

    /// Overwrite `length` bytes of an already-written scanline's chunk,
    /// starting `offset` bytes into it, with the byte `c` — port of
    /// `RgbaOutputFile::breakScanLine` (`ImfRgbaFile.cpp:732-735`, a one-line
    /// forward to `OutputFile::breakScanLine`; test-only corrupt-chunk
    /// helper).
    pub fn break_scan_line(&mut self, y: i32, offset: u64, length: usize, c: u8) -> Result<()> {
        self.of.break_scan_line(y, offset, length, c)
    }

    // -- One-shot convenience API. -------------------------------------

    /// Write `pixels` (row-major, tightly packed over `header`'s data
    /// window) to `path` in ONE call — port of `RgbaOutputFile(const char*,
    /// const Header&, RgbaChannels, int)` + one `setFrameBuffer` +
    /// `writePixels(height)`.
    pub fn write(path: impl AsRef<Path>, header: &Header, rgba_channels: RgbaChannels, pixels: &[Rgba]) -> Result<()> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
        crate::io::write_file(path_str, |stream| Self::write_stream(stream, header, rgba_channels, pixels))
    }

    /// Stream variant of [`write`](Self::write).
    pub fn write_stream<'s>(
        stream: &'s mut dyn OStream,
        header: &Header,
        rgba_channels: RgbaChannels,
        pixels: &[Rgba],
    ) -> Result<()> {
        let dw = header.data_window()?;
        let width = (dw.max.x - dw.min.x + 1) as usize;
        let height = (dw.max.y - dw.min.y + 1) as usize;
        if pixels.len() != width * height {
            return Err(ExrError::Arg(format!(
                "RgbaOutputFile::write: pixel buffer has {} pixels, expected {} ({width}x{height})",
                pixels.len(),
                width * height
            )));
        }
        let mut f = RgbaOutputFile::<'s>::new(stream, header, rgba_channels)?;
        f.set_frame_buffer(1, width as isize)?;
        f.write_pixels(pixels, height as i32)
    }
}

/// `RGBAtoYCA`-then-filter-then-write for ONE input row of the YCA path —
/// the per-row body of `ToYca::writePixels`'s filtering branch
/// (`ImfRgbaFile.cpp:363-437`), including the exact start/end-of-image
/// padding sequence.
fn yca_convert_and_write_row(out: &mut RgbaOutputFile<'_>, state: &mut ToYcaState, row: &[Rgba]) -> Result<()> {
    let n2 = yca::N2;
    let n = yca::N;

    // RGBAtoYCA, pad, horizontal decimate into the newly-rotated last slot.
    // `yca_row_scratch`/
    // `padded_scratch` are persistent, reused buffers -- `rgba_to_yca`
    // writes every element of the former, and the latter's middle region
    // (a full-width copy) plus its padding (`pad_tmp`) cover the whole of
    // the latter every call.
    crate::compressor::ensure_len(&mut state.yca_row_scratch, state.width);
    yca::rgba_to_yca(&state.yw, state.write_a, row, &mut state.yca_row_scratch);
    crate::compressor::ensure_len(&mut state.padded_scratch, state.width + n - 1);
    state.padded_scratch[n2..n2 + state.width].copy_from_slice(&state.yca_row_scratch);
    pad_tmp(&mut state.padded_scratch, state.width);

    state.buf.rotate_left(1);
    let last = n - 1;
    yca::decimate_chroma_horiz(&state.padded_scratch, &mut state.buf[last]);

    // `if (_linesConverted == 0) { N2x duplicateLastBuffer(); }`
    if state.lines_converted == 0 {
        for _ in 0..n2 {
            duplicate_last_buffer(state);
        }
    }
    state.lines_converted += 1;

    // `if (_linesConverted > N2) decimateChromaVertAndWriteScanLine();`
    if state.lines_converted > n2 as i32 {
        decimate_chroma_vert_and_write_scanline(out, state)?;
    }

    // End-of-image flush: `if (_linesConverted >= _height) { ... }`
    if state.lines_converted >= state.height as i32 {
        for _ in 0..(n2 as i32 - state.height as i32).max(0) {
            duplicate_last_buffer(state);
        }
        duplicate_second_to_last_buffer(state);
        state.lines_converted += 1;
        decimate_chroma_vert_and_write_scanline(out, state)?;

        for _ in 1..(state.height as i32).min(n2 as i32) {
            duplicate_last_buffer(state);
            state.lines_converted += 1;
            decimate_chroma_vert_and_write_scanline(out, state)?;
        }
    }
    Ok(())
}

/// `rotateBuffers` + copy — `duplicateLastBuffer`/`duplicateSecondToLastBuffer`
/// (`ImfRgbaFile.cpp:486-497`). `Vec::rotate_left(1)` is the safe-Rust
/// equivalent of C++ rotating the `_buf[N]` POINTER array; only the
/// subsequent `clone_from_slice`-style copy actually duplicates row data,
/// exactly as C++'s `memcpy` does.
fn duplicate_last_buffer(state: &mut ToYcaState) {
    state.buf.rotate_left(1);
    let n = state.buf.len();
    state.buf[n - 1] = state.buf[n - 2].clone();
}

fn duplicate_second_to_last_buffer(state: &mut ToYcaState) {
    state.buf.rotate_left(1);
    let n = state.buf.len();
    state.buf[n - 1] = state.buf[n - 3].clone();
}

/// `decimateChromaVertAndWriteScanLine` (`ImfRgbaFile.cpp:499-510`): vertical
/// decimation (even `_linesConverted`) or a straight copy of the centre row
/// (odd), `roundYCA` when writing both `Y` and chroma, then ONE low-level
/// scanline write at `out.current_scan_line`.
fn decimate_chroma_vert_and_write_scanline(out: &mut RgbaOutputFile<'_>, state: &mut ToYcaState) -> Result<()> {
    let n2 = yca::N2;
    // `vert_tmp` is a persistent,
    // reused buffer -- both branches below fully overwrite it.
    crate::compressor::ensure_len(&mut state.vert_tmp, state.width);
    if state.lines_converted & 1 != 0 {
        state.vert_tmp.copy_from_slice(&state.buf[n2]);
    } else {
        let refs: Vec<&[Rgba]> = state.buf.iter().map(|v| v.as_slice()).collect();
        yca::decimate_chroma_vert(&refs, &mut state.vert_tmp);
    }
    // C++: `if (_writeY && _writeC) roundYCA(...)` — `_writeC` is always true
    // in a `WriteMode::Yca` state.
    if state.write_y {
        let src = state.vert_tmp.clone();
        yca::round_yca(state.round_y, state.round_c, &src, &mut state.vert_tmp);
    }
    write_yca_row_bytes(
        &mut out.of, out.dw, state.width, state.write_y, true, state.write_a, out.current_scan_line, &state.vert_tmp,
        &mut out.yca_row_staging,
    )?;
    out.step_scan_line();
    Ok(())
}

/// `insertChannels` — build the channel list `rgbaChannels` requests
/// (`ImfRgbaFile.cpp:38-63`): `Y`/chroma if either is requested, else any
/// subset of `R`/`G`/`B`; `A` independently either way.
fn insert_channels(h: &mut Header, rc: RgbaChannels) -> Result<()> {
    let mut cl = ChannelList::new();
    if rc.intersects(RgbaChannels::WRITE_Y | RgbaChannels::WRITE_C) {
        if rc.intersects(RgbaChannels::WRITE_Y) {
            cl.insert("Y", Channel::new(PixelType::Half, 1, 1, false))?;
        }
        if rc.intersects(RgbaChannels::WRITE_C) {
            cl.insert("RY", Channel::new(PixelType::Half, 2, 2, true))?;
            cl.insert("BY", Channel::new(PixelType::Half, 2, 2, true))?;
        }
    } else {
        if rc.intersects(RgbaChannels::WRITE_R) {
            cl.insert("R", Channel::new(PixelType::Half, 1, 1, false))?;
        }
        if rc.intersects(RgbaChannels::WRITE_G) {
            cl.insert("G", Channel::new(PixelType::Half, 1, 1, false))?;
        }
        if rc.intersects(RgbaChannels::WRITE_B) {
            cl.insert("B", Channel::new(PixelType::Half, 1, 1, false))?;
        }
    }
    if rc.intersects(RgbaChannels::WRITE_A) {
        cl.insert("A", Channel::new(PixelType::Half, 1, 1, false))?;
    }
    h.insert("channels", ChannelListAttribute::new(cl))?;
    Ok(())
}

/// Write a half's bits, native-endian, at `buf[off + idx*2 .. +2]`.
fn put_half(buf: &mut [u8], off: usize, idx: usize, v: Half) {
    buf[off + idx * 2..off + idx * 2 + 2].copy_from_slice(&v.bits().to_ne_bytes());
}

/// Write ONE YCA output scanline (`row` already has final `Y`/chroma/`A`
/// values) at absolute row `y`, reusing a fresh single-row `FrameBuffer`
/// (`slice_at_row`) exactly like the read side's incremental engine.
///
/// `write_c` mirrors `ToYca::setFrameBuffer` (`ImfRgbaFile.cpp`, v3.4.15):
/// `RY`/`BY` slices exist only when chroma is written. The luminance-only
/// (Grey) path has no chroma channels, and its data window may have an odd
/// width -- the header's "width is a multiple of x sampling" rule only binds
/// files that carry the 2x2 chroma channels.
#[allow(clippy::too_many_arguments)] // 1:1 with the C++ call site's own parameter list + the new persistent `staging`
fn write_yca_row_bytes(
    of: &mut scanline::OutputFile<'_>,
    dw: Box2i,
    width: usize,
    write_y: bool,
    write_c: bool,
    write_a: bool,
    y: i32,
    row: &[Rgba],
    staging: &mut Vec<u8>,
) -> Result<()> {
    // One chroma sample per even `x` (the loop below): exactly `div_ceil`,
    // which equals `width / 2` for the even widths a chroma file must have.
    let cw = if write_c { width.div_ceil(2) } else { 0 };
    let y_off = 0usize;
    let a_off = y_off + width * 2;
    let ry_off = a_off + width * 2;
    let by_off = ry_off + cw * 2;
    // `staging` is a persistent, reused buffer
    // (see `RgbaOutputFile::yca_row_staging`'s doc) instead of a fresh
    // allocation per row: every byte below is either written this call (Y/A
    // for all `x`, RY/BY for every even `x` -- `cw` entries via `x/2`, an
    // exact, gap-free count) or belongs to a plane the FrameBuffer never
    // references (`write_y`/`write_a` false), so a stale byte there is never
    // observed either way.
    crate::compressor::ensure_len(staging, (width + width + cw + cw) * 2);

    for (x, p) in row.iter().enumerate() {
        if write_y {
            put_half(staging, y_off, x, p.g);
        }
        if write_a {
            put_half(staging, a_off, x, p.a);
        }
    }
    // Only meaningful for an EVEN `y` (the frame buffer's `y_sampling = 2`
    // slices skip odd rows entirely, so writing here for an odd `y` is
    // harmless — never read).
    if write_c {
        for (x, p) in row.iter().enumerate().step_by(2) {
            put_half(staging, ry_off, x / 2, p.r);
            put_half(staging, by_off, x / 2, p.b);
        }
    }

    let mut fb = FrameBuffer::new();
    if write_y {
        fb.insert("Y", slice_at_row(y_off, dw.min.x, y, width, 1, 1, 0.0)?)?;
    }
    if write_c {
        fb.insert("RY", slice_at_row(ry_off, dw.min.x, y, width, 2, 2, 0.0)?)?;
        fb.insert("BY", slice_at_row(by_off, dw.min.x, y, width, 2, 2, 0.0)?)?;
    }
    if write_a {
        fb.insert("A", slice_at_row(a_off, dw.min.x, y, width, 1, 1, 0.0)?)?;
    }
    of.set_frame_buffer(&fb)?;
    of.write_pixels(staging, 1)
}

/// Write ONE plain (`R`/`G`/`B`/`A`, no conversion) output scanline at
/// absolute row `y` — the streaming form of the old bulk `write_plain`.
fn write_plain_row(
    of: &mut scanline::OutputFile<'_>,
    dw: Box2i,
    width: usize,
    rc: RgbaChannels,
    y: i32,
    row: &[Rgba],
) -> Result<()> {
    let mut next = 0usize;
    let mut plane = |want: bool| -> Option<usize> {
        if !want {
            return None;
        }
        let off = next;
        next += width * 2;
        Some(off)
    };
    let r_off = plane(rc.intersects(RgbaChannels::WRITE_R));
    let g_off = plane(rc.intersects(RgbaChannels::WRITE_G));
    let b_off = plane(rc.intersects(RgbaChannels::WRITE_B));
    let a_off = plane(rc.intersects(RgbaChannels::WRITE_A));
    let mut buf = vec![0u8; next];

    let mut fb = FrameBuffer::new();
    for (off, name) in [(r_off, "R"), (g_off, "G"), (b_off, "B"), (a_off, "A")] {
        if let Some(off) = off {
            fb.insert(name, slice_at_row(off, dw.min.x, y, width, 1, 1, 0.0)?)?;
        }
    }
    for (x, p) in row.iter().enumerate() {
        if let Some(off) = r_off {
            put_half(&mut buf, off, x, p.r);
        }
        if let Some(off) = g_off {
            put_half(&mut buf, off, x, p.g);
        }
        if let Some(off) = b_off {
            put_half(&mut buf, off, x, p.b);
        }
        if let Some(off) = a_off {
            put_half(&mut buf, off, x, p.a);
        }
    }
    of.set_frame_buffer(&fb)?;
    of.write_pixels(&buf, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Compression, LineOrder, PreviewImage, PreviewRgba};
    use crate::io::{MemIStream, MemOStream};
    use crate::rgba::RgbaInputFile;
    use imath_rs::{Half, V2f};

    fn test_header(w: i32, h: i32) -> Header {
        Header::with_dimensions(
            w,
            h,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap()
    }

    fn gradient(w: usize, h: usize) -> Vec<Rgba> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    Rgba::new(
                        Half::from_f32(x as f32 * 0.01),
                        Half::from_f32(y as f32 * 0.02),
                        Half::from_f32((x + y) as f32 * 0.005),
                        Half::from_f32(0.75),
                    )
                })
            })
            .collect()
    }

    /// `write_plain_range` (the batched fix) must
    /// produce a byte-IDENTICAL file to the OLD per-row reference
    /// (`write_output_row` called once per row, exactly the pre-fix shape),
    /// across both line orders and a couple of real codecs with multi-line
    /// chunks -- a non-chunk-aligned height (9, vs ZIP's 16-line chunk)
    /// exercises a genuinely partial final chunk too.
    #[test]
    fn plain_range_write_matches_row_by_row_reference_both_line_orders() {
        let (w, h) = (6usize, 9usize);
        for line_order in [LineOrder::IncreasingY, LineOrder::DecreasingY] {
            for compression in [Compression::NoCompression, Compression::Zip, Compression::Piz] {
                let hdr = Header::with_dimensions(
                    w as i32, h as i32, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, line_order, compression,
                )
                .unwrap();
                let pixels = gradient(w, h);

                // NEW batched path: one public write_pixels(pixels, h) call.
                let mut out_new = MemOStream::new();
                {
                    let mut f = RgbaOutputFile::new(&mut out_new, &hdr, RgbaChannels::WRITE_RGBA).unwrap();
                    f.set_frame_buffer(1, w as isize).unwrap();
                    f.write_pixels(&pixels, h as i32).unwrap();
                }
                let bytes_new = out_new.into_data();

                // OLD reference: call the private per-row `write_output_row`
                // directly, once per row -- the exact pre-fix code path,
                // still present (not deleted) for exactly this purpose.
                let mut out_old = MemOStream::new();
                {
                    let mut f = RgbaOutputFile::new(&mut out_old, &hdr, RgbaChannels::WRITE_RGBA).unwrap();
                    for y in 0..h {
                        f.write_output_row(&pixels[y * w..(y + 1) * w]).unwrap();
                    }
                }
                let bytes_old = out_old.into_data();

                assert_eq!(
                    bytes_new, bytes_old,
                    "batched Plain write diverges from the row-by-row reference \
                     at line_order {line_order:?}, compression {compression:?}"
                );

                // And the batched output must itself still read back correctly.
                // `write_pixels`'s row 0 always corresponds to whatever
                // `current_scan_line` starts at (dw.min.y for IncreasingY,
                // dw.max.y for DecreasingY -- matching C++'s own contract),
                // so the expected ABSOLUTE-row-major image is `pixels`
                // unchanged for IncreasingY, or row-reversed for DecreasingY.
                let mut mis = MemIStream::new(bytes_new);
                let img = RgbaInputFile::read_stream(&mut mis).unwrap();
                let expected: Vec<Rgba> = if line_order == LineOrder::IncreasingY {
                    pixels.clone()
                } else {
                    (0..h).rev().flat_map(|y| pixels[y * w..(y + 1) * w].iter().copied()).collect()
                };
                assert_eq!(img.pixels, expected, "batched write must round-trip correctly");
            }
        }
    }

    /// Explicit verification (write side): a single
    /// `RgbaOutputFile::write_pixels(pixels, n)` call must make exactly ONE
    /// low-level `write_pixels` call, not one per requested scanline -- the
    /// real bug the batched fix replaces (besides the allocation churn,
    /// forcing every low-level call to exactly 1 line also defeats that
    /// writer's own parallel block-write path).
    #[test]
    fn write_pixels_makes_one_low_level_call_not_one_per_row() {
        let (w, h) = (6usize, 20usize);
        let hdr = Header::with_dimensions(
            w as i32, h as i32, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Zip,
        )
        .unwrap();
        let pixels = gradient(w, h);

        crate::probe::reset(crate::probe::Probe::WritePixels);
        let mut out = MemOStream::new();
        {
            let mut f = RgbaOutputFile::new(&mut out, &hdr, RgbaChannels::WRITE_RGBA).unwrap();
            f.set_frame_buffer(1, w as isize).unwrap();
            f.write_pixels(&pixels, h as i32).unwrap();
        }
        assert_eq!(
            crate::probe::get(crate::probe::Probe::WritePixels),
            1,
            "one RgbaOutputFile::write_pixels(.., h) call must make exactly one \
             low-level write_pixels call, not one per row"
        );
    }

    /// Luminance-only writes carry no chroma channels, so an odd data-window width is legal
    /// (C++ `exr2aces` converts `golden_tiled_rgba_grey.exr`, 67 px wide). The writer used to
    /// stage `RY`/`BY` for it anyway and indexed one chroma sample past its buffer.
    #[test]
    fn grey_write_with_odd_width_round_trips() {
        let (w, h) = (67usize, 5usize);
        let pixels = gradient(w, h);
        let hdr = test_header(w as i32, h as i32);
        let mut out = MemOStream::new();
        RgbaOutputFile::write_stream(&mut out, &hdr, RgbaChannels::WRITE_YA, &pixels).unwrap();

        let mut mis = MemIStream::new(out.into_data());
        let img = RgbaInputFile::read_stream(&mut mis).unwrap();
        let yw = yw_from_header(&hdr).unwrap();
        let mut want = vec![Rgba::default(); w];
        for y in 0..h {
            yca::rgba_to_yca(&yw, true, &pixels[y * w..(y + 1) * w], &mut want);
            for (x, (got, want)) in img.pixels[y * w..(y + 1) * w].iter().zip(&want).enumerate() {
                assert_eq!((got.g.bits(), got.a.bits()), (want.g.bits(), want.a.bits()), "pixel ({x}, {y})");
            }
        }
    }

    /// `ToYcaState`'s
    /// `yca_row_scratch`/`padded_scratch`/`vert_tmp` are now persistent,
    /// reused fields instead of fresh allocations per input row. Poison all
    /// three with implausible values BEFORE the first real `write_pixels`
    /// call, write a multi-row image on that poisoned-then-reused instance,
    /// and diff the resulting FILE BYTES against a fresh instance -- same
    /// methodology as every other poison test this round.
    #[test]
    fn poison_reused_yca_write_scratch_matches_fresh_instance() {
        let (w, h) = (8usize, 10usize);
        let pixels = gradient(w, h);
        let hdr = test_header(w as i32, h as i32);

        let mut out_poisoned = MemOStream::new();
        {
            let mut f = RgbaOutputFile::new(&mut out_poisoned, &hdr, RgbaChannels::WRITE_YCA).unwrap();
            if let WriteMode::Yca(state) = &mut f.mode {
                state.yca_row_scratch = vec![Rgba::new(Half::from_f32(f32::NAN), Half::from_f32(f32::NAN), Half::from_f32(f32::NAN), Half::from_f32(f32::NAN)); 512];
                state.padded_scratch = state.yca_row_scratch.clone();
                state.vert_tmp = state.yca_row_scratch.clone();
            } else {
                panic!("expected YCA write mode for a WRITE_YCA request");
            }
            f.set_frame_buffer(1, w as isize).unwrap();
            f.write_pixels(&pixels, h as i32).unwrap();
        }
        let bytes_poisoned = out_poisoned.into_data();

        let mut out_fresh = MemOStream::new();
        RgbaOutputFile::write_stream(&mut out_fresh, &hdr, RgbaChannels::WRITE_YCA, &pixels).unwrap();
        let bytes_fresh = out_fresh.into_data();

        assert_eq!(
            bytes_poisoned, bytes_fresh,
            "a poisoned, reused ToYcaState produced different file bytes than a fresh instance"
        );
    }

    /// Port of `RgbaOutputFile::updatePreviewImage` -- a one-line forward to
    /// the low-level `OutputFile::updatePreviewImage`, wired now that the
    /// scanline writer supports it. Writes a plain-RGBA file whose header carries a 2x2 preview,
    /// rewrites the preview after the pixel data is written, and asserts the
    /// scan line pixels survive untouched while the re-parsed header shows
    /// the new preview pixels.
    #[test]
    fn update_preview_image_rewrites_pixels_and_preserves_scan_lines() {
        let (w, h) = (4usize, 3usize);
        let pixels = gradient(w, h);
        let mut hdr = test_header(w as i32, h as i32);
        hdr.set_preview_image(
            PreviewImage::new(2, 2, Some(vec![PreviewRgba::new(1, 2, 3, 4); 4])).unwrap(),
        )
        .unwrap();

        let mut out = MemOStream::new();
        {
            let mut f = RgbaOutputFile::new(&mut out, &hdr, RgbaChannels::WRITE_RGBA).unwrap();
            f.set_frame_buffer(1, w as isize).unwrap();
            f.write_pixels(&pixels, h as i32).unwrap();
            let new_preview = vec![
                PreviewRgba::new(10, 20, 30, 40),
                PreviewRgba::new(50, 60, 70, 80),
                PreviewRgba::new(90, 100, 110, 120),
                PreviewRgba::new(130, 140, 150, 160),
            ];
            f.update_preview_image(&new_preview).unwrap();
        }
        let bytes = out.into_data();

        let mut stream = MemIStream::new(bytes.clone());
        let mut inf = RgbaInputFile::new(&mut stream).unwrap();
        let mut got = vec![Rgba::default(); w * h];
        inf.read_pixels(0, h as i32 - 1, &mut got, 1, w as isize).unwrap();
        assert_eq!(got, pixels, "scan line data must survive a preview rewrite");

        let mut stream2 = MemIStream::new(bytes);
        let (reread, _version) = Header::read_checked(&mut stream2).unwrap();
        assert_eq!(
            reread.preview_image().unwrap().pixels(),
            &[
                PreviewRgba::new(10, 20, 30, 40),
                PreviewRgba::new(50, 60, 70, 80),
                PreviewRgba::new(90, 100, 110, 120),
                PreviewRgba::new(130, 140, 150, 160),
            ]
        );
    }

    /// C++ throws `LogicExc` when the file has no preview image.
    #[test]
    fn update_preview_image_without_a_preview_is_a_logic_error() {
        let (w, h) = (2usize, 2usize);
        let hdr = test_header(w as i32, h as i32);
        let mut out = MemOStream::new();
        let mut f = RgbaOutputFile::new(&mut out, &hdr, RgbaChannels::WRITE_RGBA).unwrap();
        f.set_frame_buffer(1, w as isize).unwrap();
        f.write_pixels(&gradient(w, h), h as i32).unwrap();
        let err = f
            .update_preview_image(&[PreviewRgba::new(0, 0, 0, 0)])
            .unwrap_err();
        assert!(matches!(err, ExrError::Logic(_)), "{err:?}");
    }

    /// Port of `RgbaOutputFile::breakScanLine` -- a one-line forward, wired
    /// now that the scanline writer supports it.
    #[test]
    fn break_scan_line_overwrites_bytes_of_an_already_written_chunk() {
        let (w, h) = (4usize, 3usize);
        let pixels = gradient(w, h);
        let hdr = test_header(w as i32, h as i32);

        let mut out = MemOStream::new();
        {
            let mut f = RgbaOutputFile::new(&mut out, &hdr, RgbaChannels::WRITE_RGBA).unwrap();
            f.set_frame_buffer(1, w as isize).unwrap();
            f.write_pixels(&pixels, h as i32).unwrap();
        }
        let original = out.into_data();

        let mut out = MemOStream::new();
        {
            let mut f = RgbaOutputFile::new(&mut out, &hdr, RgbaChannels::WRITE_RGBA).unwrap();
            f.set_frame_buffer(1, w as isize).unwrap();
            f.write_pixels(&pixels, h as i32).unwrap();
            f.break_scan_line(1, 0, 2, 0xAA).unwrap();
        }
        let broken = out.into_data();

        assert_eq!(original.len(), broken.len(), "breakScanLine only overwrites, never resizes");
        let diff = original.iter().zip(broken.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 2, "exactly the 2 broken bytes must differ");
    }
}
