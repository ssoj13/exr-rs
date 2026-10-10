//! `TiledRgbaOutputFile` — port of `Imf::TiledRgbaOutputFile` / `ToYa`
//! (`ImfTiledRgbaFile.{h,cpp}`).
//!
//! Tiled EXR does not support subsampled (chroma) channels at all — C++
//! `insertChannels` THROWS if `WRITE_C` is requested (`ImfTiledRgbaFile.cpp:47-52`)
//! — so this is simpler than the scanline [`super::output`]: only two shapes,
//! luminance-only (`WRITE_Y`, ported as `ToYa`: a per-pixel `RGBAtoYCA`, NO
//! filtering/subsampling/ring-buffer, unlike the scanline `ToYca`) or plain
//! R/G/B/A. `insert_channels` returns [`ExrError::Arg`] for `WRITE_C`
//! (matching the C++ throw) instead of silently doing something else.
//!
//! Same streaming, per-call-pixels shape as [`super::output::RgbaOutputFile`]:
//! `new` -> `set_frame_buffer(x_stride, y_stride)` -> repeated
//! `write_tile(pixels, dx, dy, lx, ly)`. All level/tile-address bookkeeping
//! (mip/rip level sizes, tile counts, `dataWindowForTile`) is delegated
//! directly to [`tiled::TiledOutputFile`], which already implements it — this
//! module only adds the RGBA<->channel conversion per tile. Each tile's pixels
//! are supplied in TILE-LOCAL coordinates (`pixels[ly_local * y_stride +
//! lx_local * x_stride]`, `(0, 0)` = the tile's own top-left), matching the
//! low-level `Slice`'s `x_tile_coords`/`y_tile_coords` addressing — no
//! per-row "shifted origin" trick is needed here (unlike the scanline path),
//! since the low-level tiled writer already offers tile-relative addressing
//! natively.

use imath_rs::Box2i;

use crate::attr::{Channel, ChannelListAttribute, LevelMode, LevelRoundingMode, PixelType};
use crate::attr::{ChannelList, Compression, LineOrder};
use crate::error::{ExrError, Result};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::Header;
use crate::attr::TileDescription;
use crate::io::OStream;
use crate::rgba::input::yw_from_header;
use crate::rgba::pixel::RgbaChannels;
use crate::rgba::yca;
use crate::rgba::Rgba;
use crate::tiled;

use std::path::Path;

/// Streaming writer for tiled R/G/B(/A) or luminance(/A) EXR files. See the
/// module doc for the streaming contract and why chroma is rejected.
pub struct TiledRgbaOutputFile<'a> {
    of: tiled::TiledOutputFile<'a>,
    header: Header,
    dw: Box2i,
    rc: RgbaChannels,
    mode: TiledWriteMode,
    fb: Option<(isize, isize)>,
    /// Persistent per-tile scratch: a tile IS the chunk
    /// here (same reasoning as `TiledRgbaInputFile`'s own fields), so this
    /// is pure allocation-elimination, not a redundant-decode fix.
    /// `tile_rows` holds the gathered `Rgba`s; `tile_staging` the resulting
    /// byte planes; `yca_pixels` the Grey path's intermediate `RGBAtoYCA`
    /// conversion.
    tile_rows: Vec<Rgba>,
    tile_staging: Vec<u8>,
    yca_pixels: Vec<Rgba>,
}

enum TiledWriteMode {
    Plain,
    Grey { yw: [f32; 3] },
}

impl<'a> TiledRgbaOutputFile<'a> {
    /// Open `stream`, overriding `header`'s channel list and tile description
    /// — port of `TiledRgbaOutputFile(OStream&, const Header&, RgbaChannels,
    /// int, int, LevelMode, LevelRoundingMode, int)`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if `rgba_channels` requests [`RgbaChannels::WRITE_C`]
    /// — tiled EXR does not support subsampled channels (C++ throws the
    /// equivalent `ArgExc` from `insertChannels`).
    pub fn new(
        stream: &'a mut dyn OStream,
        header: &Header,
        rgba_channels: RgbaChannels,
        tile_x_size: u32,
        tile_y_size: u32,
        level_mode: LevelMode,
        rounding_mode: LevelRoundingMode,
    ) -> Result<Self> {
        let mut h = header.clone();
        insert_channels(&mut h, rgba_channels)?;
        h.set_tile_description(TileDescription {
            x_size: tile_x_size,
            y_size: tile_y_size,
            mode: level_mode,
            rounding_mode,
        })?;
        let dw = h.data_window()?;
        let of = tiled::TiledOutputFile::new(stream, &h)?;

        let mode = if rgba_channels.intersects(RgbaChannels::WRITE_Y) {
            TiledWriteMode::Grey { yw: yw_from_header(&h)? }
        } else {
            TiledWriteMode::Plain
        };

        Ok(Self {
            of,
            header: h,
            dw,
            rc: rgba_channels,
            mode,
            fb: None,
            tile_rows: Vec::new(),
            tile_staging: Vec::new(),
            yca_pixels: Vec::new(),
        })
    }

    /// Record how to read the caller's buffer in [`write_tile`](Self::write_tile)
    /// — port of `TiledRgbaOutputFile::setFrameBuffer` minus the pointer
    /// (supplied per call instead — see the module doc). Units of [`Rgba`].
    pub fn set_frame_buffer(&mut self, x_stride: isize, y_stride: isize) -> Result<()> {
        self.fb = Some((x_stride, y_stride));
        Ok(())
    }

    /// Write tile `(dx, dy, lx, ly)`, reading its pixels from `pixels` via the
    /// strides [`set_frame_buffer`](Self::set_frame_buffer) recorded — port of
    /// `TiledRgbaOutputFile::writeTile(int, int, int, int)`.
    pub fn write_tile(&mut self, pixels: &[Rgba], dx: i32, dy: i32, lx: i32, ly: i32) -> Result<()> {
        let (x_stride, y_stride) = self.fb.ok_or_else(|| {
            ExrError::Arg("No frame buffer was specified as the pixel data source.".into())
        })?;
        let tile_dw = self.of.data_window_for_tile(dx, dy, lx, ly)?;
        let tw = (tile_dw.max.x - tile_dw.min.x + 1) as usize;
        let th = (tile_dw.max.y - tile_dw.min.y + 1) as usize;

        // `tile_rows` is a persistent, reused buffer --
        // every element is written by the gather loop below, so
        // `ensure_len` is sound.
        crate::compressor::ensure_len(&mut self.tile_rows, tw * th);
        for ly_local in 0..th {
            for lx_local in 0..tw {
                let idx = ly_local as isize * y_stride + lx_local as isize * x_stride;
                let idx = usize::try_from(idx)
                    .map_err(|_| ExrError::Arg("writeTile: source index is negative.".into()))?;
                self.tile_rows[ly_local * tw + lx_local] = *pixels
                    .get(idx)
                    .ok_or_else(|| ExrError::Arg("writeTile: source index is out of range.".into()))?;
            }
        }

        match &self.mode {
            TiledWriteMode::Plain => write_tile_plain(
                &mut self.of, self.rc, tw, th, dx, dy, lx, ly, &self.tile_rows, &mut self.tile_staging,
            ),
            TiledWriteMode::Grey { yw } => write_tile_grey(
                &mut self.of, self.rc, yw, tw, th, dx, dy, lx, ly, &self.tile_rows,
                &mut self.tile_staging, &mut self.yca_pixels,
            ),
        }
    }

    /// Convenience for a `ONE_LEVEL`/`MIPMAP_LEVELS` file — port of
    /// `writeTile(int dx, int dy, int l)`.
    pub fn write_tile_l(&mut self, pixels: &[Rgba], dx: i32, dy: i32, l: i32) -> Result<()> {
        self.write_tile(pixels, dx, dy, l, l)
    }

    // -- Accessors — delegate directly to the low-level writer, which already
    // implements all level/tile-address bookkeeping. -----------------------

    pub fn header(&self) -> &Header {
        &self.header
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
    pub fn line_order(&self) -> Result<LineOrder> {
        self.header.line_order()
    }
    pub fn compression(&self) -> Result<Compression> {
        self.header.compression()
    }
    pub fn channels(&self) -> RgbaChannels {
        self.rc
    }
    /// The `(x_stride, y_stride)` last installed — see
    /// [`super::output::RgbaOutputFile::frame_buffer`] for why this is the
    /// faithful equivalent of C++'s `frameBuffer()` here.
    pub fn frame_buffer(&self) -> Option<(isize, isize)> {
        self.fb
    }
    pub fn tile_x_size(&self) -> u32 {
        self.of.tile_x_size()
    }
    pub fn tile_y_size(&self) -> u32 {
        self.of.tile_y_size()
    }
    pub fn level_mode(&self) -> LevelMode {
        self.of.level_mode()
    }
    pub fn level_rounding_mode(&self) -> LevelRoundingMode {
        self.of.level_rounding_mode()
    }
    pub fn num_levels(&self) -> Result<i32> {
        self.of.num_levels()
    }
    pub fn num_x_levels(&self) -> i32 {
        self.of.num_x_levels()
    }
    pub fn num_y_levels(&self) -> i32 {
        self.of.num_y_levels()
    }
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        self.of.is_valid_level(lx, ly)
    }
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        self.of.level_width(lx)
    }
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        self.of.level_height(ly)
    }
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.of.num_x_tiles(lx)
    }
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.of.num_y_tiles(ly)
    }
    pub fn data_window_for_level(&self, lx: i32, ly: i32) -> Result<Box2i> {
        self.of.data_window_for_level(lx, ly)
    }
    pub fn data_window_for_level_l(&self, l: i32) -> Result<Box2i> {
        self.of.data_window_for_level_l(l)
    }
    pub fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
        self.of.data_window_for_tile(dx, dy, lx, ly)
    }

    // -- One-shot convenience API. ------------------------------------------

    /// Write the whole level-0 image in ONE call (loops `write_tile` over
    /// every tile in the file's own order) — NOT a C++ method; a convenience
    /// analogous to [`super::output::RgbaOutputFile::write`].
    pub fn write(
        path: impl AsRef<Path>,
        header: &Header,
        rgba_channels: RgbaChannels,
        tile_x_size: u32,
        tile_y_size: u32,
        pixels: &[Rgba],
    ) -> Result<()> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
        crate::io::write_file(path_str, |stream| {
            Self::write_stream(stream, header, rgba_channels, tile_x_size, tile_y_size, pixels)
        })
    }

    /// Stream variant of [`write`](Self::write). ONE_LEVEL only (the common
    /// case); use the streaming API directly for MIPMAP/RIPMAP.
    pub fn write_stream<'s>(
        stream: &'s mut dyn OStream,
        header: &Header,
        rgba_channels: RgbaChannels,
        tile_x_size: u32,
        tile_y_size: u32,
        pixels: &[Rgba],
    ) -> Result<()> {
        let dw = header.data_window()?;
        let width = (dw.max.x - dw.min.x + 1) as usize;
        let height = (dw.max.y - dw.min.y + 1) as usize;
        if pixels.len() != width * height {
            return Err(ExrError::Arg(format!(
                "TiledRgbaOutputFile::write: pixel buffer has {} pixels, expected {} ({width}x{height})",
                pixels.len(),
                width * height
            )));
        }
        let mut f = TiledRgbaOutputFile::<'s>::new(
            stream,
            header,
            rgba_channels,
            tile_x_size,
            tile_y_size,
            LevelMode::OneLevel,
            LevelRoundingMode::RoundDown,
        )?;
        let nx = f.num_x_tiles(0)?;
        let ny = f.num_y_tiles(0)?;
        for dy in 0..ny {
            for dx in 0..nx {
                let tile_dw = f.data_window_for_tile(dx, dy, 0, 0)?;
                let base = (tile_dw.min.y - dw.min.y) as isize * width as isize
                    + (tile_dw.min.x - dw.min.x) as isize;
                let tw = (tile_dw.max.x - tile_dw.min.x + 1) as usize;
                let th = (tile_dw.max.y - tile_dw.min.y + 1) as usize;
                let mut tile_pixels = vec![Rgba::default(); tw * th];
                for row in 0..th {
                    let src_off = (base + row as isize * width as isize) as usize;
                    tile_pixels[row * tw..(row + 1) * tw].copy_from_slice(&pixels[src_off..src_off + tw]);
                }
                // Gathered as a tightly-packed `tw`-wide buffer; set the
                // frame buffer to match before each tile (edge tiles have a
                // narrower `tw` than the file's nominal tile width).
                f.set_frame_buffer(1, tw as isize)?;
                f.write_tile(&tile_pixels, dx, dy, 0, 0)?;
            }
        }
        Ok(())
    }
}

/// `insertChannels` — build the channel list `rgbaChannels` requests
/// (`ImfTiledRgbaFile.cpp:38-71`): `Y` alone if requested (no chroma — tiled
/// EXR rejects it), else any subset of `R`/`G`/`B`; `A` independently either
/// way.
fn insert_channels(h: &mut Header, rc: RgbaChannels) -> Result<()> {
    if rc.intersects(RgbaChannels::WRITE_C) {
        return Err(ExrError::Arg(
            "Tiled image files do not support subsampled chroma channels.".into(),
        ));
    }
    let mut cl = ChannelList::new();
    if rc.intersects(RgbaChannels::WRITE_Y) {
        cl.insert("Y", Channel::new(PixelType::Half, 1, 1, false))?;
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
fn put_half(buf: &mut [u8], off: usize, idx: usize, v: imath_rs::Half) {
    buf[off + idx * 2..off + idx * 2 + 2].copy_from_slice(&v.bits().to_ne_bytes());
}

/// A tile-local `Slice` (`x_tile_coords`/`y_tile_coords = true`, so pixel
/// `(x, y)` WITHIN the tile lands at `base_offset + y*y_stride + x*x_stride`
/// — no per-tile origin shift needed, unlike the scanline path's
/// `slice_at_row`).
fn tile_slice(plane_off: usize, tw: usize, xs: i32, ys: i32) -> Slice {
    let x_stride = 2usize; // HALF
    let y_stride = (tw / xs as usize) * x_stride;
    Slice {
        type_: PixelType::Half,
        base_offset: plane_off as i64,
        x_stride,
        y_stride,
        x_sampling: xs,
        y_sampling: ys,
        fill_value: 0.0,
        x_tile_coords: true,
        y_tile_coords: true,
    }
}

/// Luminance-only tile — port of `ToYa::writeTile`: per-pixel `RGBAtoYCA`
/// (NO filtering — tiled EXR has no chroma), `Y`/`A` written straight through.
#[allow(clippy::too_many_arguments)]
fn write_tile_grey(
    of: &mut tiled::TiledOutputFile<'_>,
    rc: RgbaChannels,
    yw: &[f32; 3],
    tw: usize,
    th: usize,
    dx: i32,
    dy: i32,
    lx: i32,
    ly: i32,
    rows: &[Rgba],
    staging: &mut Vec<u8>,
    yca_pixels: &mut Vec<Rgba>,
) -> Result<()> {
    let write_a = rc.intersects(RgbaChannels::WRITE_A);
    let n = tw * th;
    // `yca_pixels`/`staging` are persistent, reused
    // buffers -- every element/byte is fully overwritten below.
    crate::compressor::ensure_len(yca_pixels, n);
    yca::rgba_to_yca(yw, write_a, rows, yca_pixels);

    let y_off = 0usize;
    let a_off = y_off + n * 2;
    crate::compressor::ensure_len(staging, (n + n) * 2);
    for (i, p) in yca_pixels.iter().enumerate() {
        put_half(staging, y_off, i, p.g);
        if write_a {
            put_half(staging, a_off, i, p.a);
        }
    }

    let mut fb = FrameBuffer::new();
    fb.insert("Y", tile_slice(y_off, tw, 1, 1))?;
    if write_a {
        fb.insert("A", tile_slice(a_off, tw, 1, 1))?;
    }
    of.set_frame_buffer(&fb)?;
    of.write_tile(staging, dx, dy, lx, ly)
}

/// Plain R/G/B(/A) tile — no conversion, whichever channels `rc` requests.
#[allow(clippy::too_many_arguments)]
fn write_tile_plain(
    of: &mut tiled::TiledOutputFile<'_>,
    rc: RgbaChannels,
    tw: usize,
    th: usize,
    dx: i32,
    dy: i32,
    lx: i32,
    ly: i32,
    rows: &[Rgba],
    staging: &mut Vec<u8>,
) -> Result<()> {
    let n = tw * th;
    let mut next = 0usize;
    let mut plane = |want: bool| -> Option<usize> {
        if !want {
            return None;
        }
        let off = next;
        next += n * 2;
        Some(off)
    };
    let r_off = plane(rc.intersects(RgbaChannels::WRITE_R));
    let g_off = plane(rc.intersects(RgbaChannels::WRITE_G));
    let b_off = plane(rc.intersects(RgbaChannels::WRITE_B));
    let a_off = plane(rc.intersects(RgbaChannels::WRITE_A));
    // `staging` is a persistent, reused buffer -- every
    // byte within `[0, next)` is written by the loop below.
    crate::compressor::ensure_len(staging, next);

    let mut fb = FrameBuffer::new();
    for (off, name) in [(r_off, "R"), (g_off, "G"), (b_off, "B"), (a_off, "A")] {
        if let Some(off) = off {
            fb.insert(name, tile_slice(off, tw, 1, 1))?;
        }
    }
    for (i, p) in rows.iter().enumerate() {
        if let Some(off) = r_off {
            put_half(staging, off, i, p.r);
        }
        if let Some(off) = g_off {
            put_half(staging, off, i, p.g);
        }
        if let Some(off) = b_off {
            put_half(staging, off, i, p.b);
        }
        if let Some(off) = a_off {
            put_half(staging, off, i, p.a);
        }
    }
    of.set_frame_buffer(&fb)?;
    of.write_tile(&staging[..next], dx, dy, lx, ly)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{MemIStream, MemOStream};
    use crate::rgba::TiledRgbaInputFile;
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

    /// A plain-RGBA tiled file must round-trip byte-for-value through
    /// `TiledRgbaOutputFile` -> `TiledRgbaInputFile`, including a partial
    /// (edge) tile — the file is 20x13 with 8x8 tiles, so the last column and
    /// row of tiles are narrower/shorter than the nominal tile size.
    #[test]
    fn plain_rgba_round_trips_including_edge_tiles() {
        let (w, h) = (20usize, 13usize);
        let pixels = gradient(w, h);
        let hdr = test_header(w as i32, h as i32);

        let mut out = MemOStream::new();
        TiledRgbaOutputFile::write_stream(&mut out, &hdr, RgbaChannels::WRITE_RGBA, 8, 8, &pixels).unwrap();
        let bytes = out.into_data();

        let mut stream = MemIStream::new(bytes);
        let mut f = TiledRgbaInputFile::new(&mut stream).unwrap();
        assert!(f.channels().unwrap().contains(RgbaChannels::WRITE_RGBA));
        let nx = f.num_x_tiles(0).unwrap();
        let ny = f.num_y_tiles(0).unwrap();
        for dy in 0..ny {
            for dx in 0..nx {
                let tile_dw = f.data_window_for_tile(dx, dy, 0, 0).unwrap();
                let tw = (tile_dw.max.x - tile_dw.min.x + 1) as usize;
                let th = (tile_dw.max.y - tile_dw.min.y + 1) as usize;
                let mut tile = vec![Rgba::default(); tw * th];
                f.read_tile(dx, dy, 0, 0, &mut tile, 1, tw as isize).unwrap();
                for ly in 0..th {
                    for lx in 0..tw {
                        let x = (tile_dw.min.x + lx as i32) as usize;
                        let y = (tile_dw.min.y + ly as i32) as usize;
                        assert_eq!(tile[ly * tw + lx], pixels[y * w + x], "pixel ({x},{y})");
                    }
                }
            }
        }
    }

    /// A `Y`-only tiled file must read back grey (`R = G = B = Y`), matching
    /// `FromYa`'s zero-chroma special case.
    #[test]
    fn y_only_round_trips_as_grey() {
        let (w, h) = (16usize, 16usize);
        let pixels = gradient(w, h);
        let hdr = test_header(w as i32, h as i32);

        let mut out = MemOStream::new();
        TiledRgbaOutputFile::write_stream(&mut out, &hdr, RgbaChannels::WRITE_YA, 8, 8, &pixels).unwrap();
        let bytes = out.into_data();

        let mut stream = MemIStream::new(bytes);
        let mut f = TiledRgbaInputFile::new(&mut stream).unwrap();
        assert!(f.channels().unwrap().intersects(RgbaChannels::WRITE_Y));
        assert!(!f.channels().unwrap().intersects(RgbaChannels::WRITE_C));

        let mut tile = vec![Rgba::default(); 8 * 8];
        f.read_tile(0, 0, 0, 0, &mut tile, 1, 8).unwrap();
        for p in &tile {
            assert_eq!(p.r, p.g);
            assert_eq!(p.g, p.b);
        }
    }

    /// Writing chroma to a tiled file must error (tiled EXR has no
    /// subsampled-channel support) — the C++ `ArgExc` equivalent.
    #[test]
    fn write_c_is_rejected() {
        let h = test_header(8, 8);
        let mut out = MemOStream::new();
        let result = TiledRgbaOutputFile::new(
            &mut out,
            &h,
            RgbaChannels::WRITE_YC,
            8,
            8,
            LevelMode::OneLevel,
            LevelRoundingMode::RoundDown,
        );
        assert!(matches!(result, Err(ExrError::Arg(_))));
    }
}
