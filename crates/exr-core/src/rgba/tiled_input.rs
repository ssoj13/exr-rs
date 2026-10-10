//! `TiledRgbaInputFile` — port of `Imf::TiledRgbaInputFile` / `FromYa`
//! (`ImfTiledRgbaFile.{h,cpp}`).
//!
//! Tiled EXR has no chroma channels (see [`super::tiled_output`]'s module
//! doc), so there are only two shapes: luminance-only (`Y` present — ported
//! as `FromYa`: read `Y`/`A` per tile, chroma is always `0` so
//! `YCAtoRGBA`'s special case makes every pixel grey, `R=G=B=Y`) or plain
//! R/G/B/A (any subset, each filled `0.0`/`1.0` if absent — matches C++
//! `TiledRgbaInputFile::setFrameBuffer`'s unconditional R/G/B/A insert).
//!
//! Same streaming, per-call-pixels shape as [`super::input::RgbaInputFile`]:
//! `new(stream)` opens the file (parses headers once); `read_tile(dx, dy, lx,
//! ly, out, x_stride, y_stride)` reads ONE tile at a time, no whole-image
//! buffer. Level/tile-address bookkeeping is delegated to
//! [`tiled::TiledInputFile`]. A tile's pixels are TILE-LOCAL in `out`
//! (`out[ly_local * y_stride + lx_local * x_stride]`, `(0, 0)` = the tile's
//! own top-left) — matching the low-level `Slice`'s native
//! `x_tile_coords`/`y_tile_coords` addressing, so no per-row "shifted
//! origin" trick is needed (unlike the scanline path).

use imath_rs::{Box2i, Half};

use crate::attr::{ChannelList, LevelMode, PixelType};
use crate::error::{ExrError, Result};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::Header;
use crate::io::IStream;
use crate::rgba::input::yw_from_header;
use crate::rgba::pixel::RgbaChannels;
use crate::rgba::Rgba;
use crate::tiled;

/// Reader for tiled luminance(/A) or plain R/G/B(/A) EXR files. See the
/// module doc for the two shapes and the streaming contract.
pub struct TiledRgbaInputFile<'a> {
    inp: tiled::TiledInputFile<'a>,
    channel_name_prefix: String,
    mode: TiledReadMode,
    /// Persistent per-tile scratch: a tile IS the chunk
    /// here (unlike scanline's multi-line chunks), so there is no
    /// "N-times redecode" bug to fix -- `buf`/`out` were simply fresh
    /// allocations on EVERY `read_tile` call. `tile_staging` holds the raw
    /// Xdr bytes `read_tile_plain`/`read_tile_grey` decode into;
    /// `tile_rgba` holds the resulting `Rgba`s.
    tile_staging: Vec<u8>,
    tile_rgba: Vec<Rgba>,
}

enum TiledReadMode {
    Plain,
    Grey { yw: [f32; 3] },
}

impl<'a> TiledRgbaInputFile<'a> {
    /// Open `stream`, no layer — port of `TiledRgbaInputFile(IStream&, int)`.
    pub fn new(stream: &'a mut dyn IStream) -> Result<Self> {
        Self::with_layer(stream, "")
    }

    /// Open `stream`, selecting `layer_name` — port of
    /// `TiledRgbaInputFile(IStream&, const std::string&, int)`.
    pub fn with_layer(stream: &'a mut dyn IStream, layer_name: &str) -> Result<Self> {
        let inp = tiled::TiledInputFile::new(stream)?;
        let prefix = prefix_from_layer_name(layer_name, inp.header())?;
        let mode = rebuild_mode(inp.header(), &prefix)?;
        Ok(Self {
            inp,
            channel_name_prefix: prefix,
            mode,
            tile_staging: Vec::new(),
            tile_rgba: Vec::new(),
        })
    }

    /// Select a different layer — port of `TiledRgbaInputFile::setLayerName`.
    pub fn set_layer_name(&mut self, layer_name: &str) -> Result<()> {
        self.channel_name_prefix = prefix_from_layer_name(layer_name, self.inp.header())?;
        self.mode = rebuild_mode(self.inp.header(), &self.channel_name_prefix)?;
        Ok(())
    }

    /// Read tile `(dx, dy, lx, ly)` into `out`, tile-local (`(0, 0)` = the
    /// tile's own top-left; `x_stride`/`y_stride` in units of [`Rgba`]) —
    /// port of `TiledRgbaInputFile::readTile(int, int, int, int)`.
    ///
    /// The `(dx, dy, lx, ly)` tile-address quadruplet is this crate's
    /// established tile-addressing convention (`tiled::TiledInputFile`'s own
    /// `data_window_for_tile`/`is_valid_tile`, `TileOffsets::{get,
    /// slot_index}`, and this file's own `read_tile_plain`/`read_tile_grey`
    /// helpers below all take it the same way) — bundling it into a
    /// one-off param struct here would diverge from every neighbouring
    /// function's signature rather than reduce the real complexity, so this
    /// follows the same `#[allow(clippy::too_many_arguments)]` precedent
    /// used throughout `tiled/`/`deep/` for tile/pixel-address functions.
    #[allow(clippy::too_many_arguments)]
    pub fn read_tile(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        out: &mut [Rgba],
        x_stride: isize,
        y_stride: isize,
    ) -> Result<()> {
        let tile_dw = self.inp.data_window_for_tile(dx, dy, lx, ly)?;
        let tw = (tile_dw.max.x - tile_dw.min.x + 1) as usize;
        let th = (tile_dw.max.y - tile_dw.min.y + 1) as usize;

        match &self.mode {
            TiledReadMode::Plain => read_tile_plain(
                &mut self.inp, &self.channel_name_prefix, tw, th, dx, dy, lx, ly,
                &mut self.tile_staging, &mut self.tile_rgba,
            )?,
            TiledReadMode::Grey { yw } => read_tile_grey(
                &mut self.inp, &self.channel_name_prefix, yw, tw, th, dx, dy, lx, ly,
                &mut self.tile_staging, &mut self.tile_rgba,
            )?,
        };

        for ly_local in 0..th {
            for lx_local in 0..tw {
                let idx = ly_local as isize * y_stride + lx_local as isize * x_stride;
                let idx = usize::try_from(idx)
                    .map_err(|_| ExrError::Arg("readTile: destination index is negative.".into()))?;
                *out
                    .get_mut(idx)
                    .ok_or_else(|| ExrError::Arg("readTile: destination index is out of range.".into()))? =
                    self.tile_rgba[ly_local * tw + lx_local];
            }
        }
        Ok(())
    }

    /// Convenience for a `ONE_LEVEL`/`MIPMAP_LEVELS` file — port of
    /// `readTile(int dx, int dy, int l)`.
    pub fn read_tile_l(
        &mut self,
        dx: i32,
        dy: i32,
        l: i32,
        out: &mut [Rgba],
        x_stride: isize,
        y_stride: isize,
    ) -> Result<()> {
        self.read_tile(dx, dy, l, l, out, x_stride, y_stride)
    }

    // -- Accessors — delegate directly to the low-level reader. -------------

    pub fn header(&self) -> &Header {
        self.inp.header()
    }
    pub fn data_window(&self) -> Result<Box2i> {
        self.header().data_window()
    }
    pub fn display_window(&self) -> Result<Box2i> {
        self.header().display_window()
    }
    pub fn pixel_aspect_ratio(&self) -> Result<f32> {
        self.header().pixel_aspect_ratio()
    }
    pub fn line_order(&self) -> Result<crate::attr::LineOrder> {
        self.header().line_order()
    }
    pub fn compression(&self) -> Result<crate::attr::Compression> {
        self.header().compression()
    }
    /// Which of R/G/B/A/Y this part (and selected layer) provides — port of
    /// `TiledRgbaInputFile::channels` (no chroma bit: tiled files never have it).
    pub fn channels(&self) -> Result<RgbaChannels> {
        rgba_channels_of(self.header().channels()?, &self.channel_name_prefix)
    }
    pub fn is_complete(&self) -> bool {
        self.inp.is_complete()
    }
    pub fn tile_x_size(&self) -> u32 {
        self.inp.tile_x_size()
    }
    pub fn tile_y_size(&self) -> u32 {
        self.inp.tile_y_size()
    }
    pub fn level_mode(&self) -> LevelMode {
        self.inp.level_mode()
    }
    pub fn num_levels(&self) -> Result<i32> {
        self.inp.num_levels()
    }
    pub fn is_valid_level(&self, lx: i32, ly: i32) -> bool {
        self.inp.is_valid_level(lx, ly)
    }
    pub fn level_width(&self, lx: i32) -> Result<i32> {
        self.inp.level_width(lx)
    }
    pub fn level_height(&self, ly: i32) -> Result<i32> {
        self.inp.level_height(ly)
    }
    pub fn num_x_tiles(&self, lx: i32) -> Result<i32> {
        self.inp.num_x_tiles(lx)
    }
    pub fn num_y_tiles(&self, ly: i32) -> Result<i32> {
        self.inp.num_y_tiles(ly)
    }
    pub fn data_window_for_tile(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Result<Box2i> {
        self.inp.data_window_for_tile(dx, dy, lx, ly)
    }
}

/// `rgbaChannels` (tiled variant — no chroma bit) — `ImfTiledRgbaFile.cpp:73-89`.
fn rgba_channels_of(ch: &ChannelList, prefix: &str) -> Result<RgbaChannels> {
    let mut i = RgbaChannels::NONE;
    let has = |suffix: &str| ch.find_channel(&format!("{prefix}{suffix}")).is_some();
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
    Ok(i)
}

/// `prefixFromLayerName` — identical rule to the scanline path
/// (`ImfTiledRgbaFile.cpp:91-104`). Same v3.4.15 empty-`multiView`-vector fix
/// as `input.rs`'s `prefix_from_layer_name` — `sv.first()` already returns
/// `None` for an empty vector instead of indexing `[0]` unconditionally, so
/// this port already matched (verified via `git diff v3.3.11 v3.4.15 --
/// src/lib/OpenEXR/ImfTiledRgbaFile.cpp`).
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

fn rebuild_mode(header: &Header, prefix: &str) -> Result<TiledReadMode> {
    let rc = rgba_channels_of(header.channels()?, prefix)?;
    Ok(if rc.intersects(RgbaChannels::WRITE_Y) {
        TiledReadMode::Grey { yw: yw_from_header(header)? }
    } else {
        TiledReadMode::Plain
    })
}

/// Read a half's bits, native-endian, from `buf[off + idx*2 .. +2]`.
fn half_at(buf: &[u8], off: usize, idx: usize) -> Half {
    let b = off + idx * 2;
    Half::from_bits(u16::from_ne_bytes([buf[b], buf[b + 1]]))
}

/// A tile-local `Slice` (`x_tile_coords`/`y_tile_coords = true`) — the
/// read-side counterpart of [`super::tiled_output`]'s `tile_slice`.
fn tile_slice(plane_off: usize, tw: usize, fill: f64) -> Slice {
    Slice {
        type_: PixelType::Half,
        base_offset: plane_off as i64,
        x_stride: 2,
        y_stride: tw * 2,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: fill,
        x_tile_coords: true,
        y_tile_coords: true,
    }
}

/// Luminance-only tile — port of `FromYa::readTile`: read `Y`/`A`, chroma is
/// always `0` so every pixel comes out grey (`R = G = B = Y`), matching
/// `YCAtoRGBA`'s zero-chroma special case.
#[allow(clippy::too_many_arguments)]
fn read_tile_grey(
    inp: &mut tiled::TiledInputFile<'_>,
    prefix: &str,
    _yw: &[f32; 3],
    tw: usize,
    th: usize,
    dx: i32,
    dy: i32,
    lx: i32,
    ly: i32,
    staging: &mut Vec<u8>,
    out: &mut Vec<Rgba>,
) -> Result<()> {
    let n = tw * th;
    let y_off = 0usize;
    let a_off = y_off + n * 2;
    // `staging`/`out` are persistent, reused buffers (see
    // `TiledRgbaInputFile`'s field doc) -- every byte/element is fully
    // overwritten below, so `ensure_len` (grow/truncate, no unconditional
    // zero-fill) is sound.
    crate::compressor::ensure_len(staging, (n + n) * 2);

    let mut fb = FrameBuffer::new();
    // C++ `FromYa::setFrameBuffer`: Y fill 0.0, A fill 1.0.
    fb.insert(format!("{prefix}Y"), tile_slice(y_off, tw, 0.0))?;
    fb.insert(format!("{prefix}A"), tile_slice(a_off, tw, 1.0))?;
    inp.set_frame_buffer(&fb)?;
    inp.read_tile(dx, dy, lx, ly, staging)?;

    crate::compressor::ensure_len(out, n);
    for (i, p) in out.iter_mut().enumerate() {
        let y = half_at(staging, y_off, i);
        let a = half_at(staging, a_off, i);
        *p = Rgba::new(y, y, y, a);
    }
    Ok(())
}

/// Plain R/G/B(/A) tile — port of `TiledRgbaInputFile::setFrameBuffer`'s
/// else-branch: R/G/B/A always requested, each filled if absent.
#[allow(clippy::too_many_arguments)]
fn read_tile_plain(
    inp: &mut tiled::TiledInputFile<'_>,
    prefix: &str,
    tw: usize,
    th: usize,
    dx: i32,
    dy: i32,
    lx: i32,
    ly: i32,
    staging: &mut Vec<u8>,
    out: &mut Vec<Rgba>,
) -> Result<()> {
    let n = tw * th;
    let r_off = 0usize;
    let g_off = r_off + n * 2;
    let b_off = g_off + n * 2;
    let a_off = b_off + n * 2;
    // See `read_tile_grey`'s doc for why `ensure_len` is
    // sound here too (full overwrite every call).
    crate::compressor::ensure_len(staging, n * 4 * 2);

    let mut fb = FrameBuffer::new();
    fb.insert(format!("{prefix}R"), tile_slice(r_off, tw, 0.0))?;
    fb.insert(format!("{prefix}G"), tile_slice(g_off, tw, 0.0))?;
    fb.insert(format!("{prefix}B"), tile_slice(b_off, tw, 0.0))?;
    fb.insert(format!("{prefix}A"), tile_slice(a_off, tw, 1.0))?;
    inp.set_frame_buffer(&fb)?;
    inp.read_tile(dx, dy, lx, ly, staging)?;

    crate::compressor::ensure_len(out, n);
    for (i, p) in out.iter_mut().enumerate() {
        *p = Rgba::new(
            half_at(staging, r_off, i),
            half_at(staging, g_off, i),
            half_at(staging, b_off, i),
            half_at(staging, a_off, i),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Compression, LineOrder};
    use crate::io::{MemIStream, MemOStream};
    use crate::rgba::tiled_output::TiledRgbaOutputFile;
    use crate::rgba::pixel::RgbaChannels;
    use imath_rs::{Half, V2f};

    fn test_header(w: i32, h: i32) -> Header {
        Header::with_dimensions(
            w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::NoCompression,
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

    /// `TiledRgbaInputFile`'s `tile_staging`/`tile_rgba`
    /// scratch is now reused across `read_tile` calls instead of a fresh
    /// `Vec` per call. Poison BOTH fields with implausible values before the
    /// FIRST real call on one instance, then read a sequence of tiles of
    /// GENUINELY DIFFERENT sizes (a full 8x8 tile, then edge tiles narrower
    /// AND shorter than nominal) on that same poisoned-then-reused instance,
    /// diffing every tile against a FRESH instance -- same methodology as
    /// B44's `poison_reused_scratch_matches_fresh_instance`.
    #[test]
    fn poison_reused_tile_scratch_matches_fresh_instance_across_tile_sizes() {
        let (w, h) = (20usize, 13usize); // 8x8 tiles -> edge tiles at the right/bottom
        let pixels = gradient(w, h);
        let hdr = test_header(w as i32, h as i32);
        let mut out = MemOStream::new();
        TiledRgbaOutputFile::write_stream(&mut out, &hdr, RgbaChannels::WRITE_RGBA, 8, 8, &pixels).unwrap();
        let bytes = out.into_data();

        let mut poisoned_stream = MemIStream::new(bytes.clone());
        let mut poisoned = TiledRgbaInputFile::new(&mut poisoned_stream).unwrap();
        // Poison BEFORE the first real call, oversized relative to any real
        // tile here (8*8=64 Rgba / 64*4*2=512 bytes max), so both growth
        // (edge tile -> smaller) and any later regrowth start from genuinely
        // stale bytes, not zeros.
        poisoned.tile_staging = vec![0xAAu8; 4096];
        poisoned.tile_rgba = vec![Rgba::new(Half::from_f32(f32::NAN), Half::from_f32(f32::NAN), Half::from_f32(f32::NAN), Half::from_f32(f32::NAN)); 256];

        let mut fresh_stream = MemIStream::new(bytes);
        let mut fresh = TiledRgbaInputFile::new(&mut fresh_stream).unwrap();

        let nx = poisoned.num_x_tiles(0).unwrap();
        let ny = poisoned.num_y_tiles(0).unwrap();
        for dy in 0..ny {
            for dx in 0..nx {
                let tile_dw = poisoned.data_window_for_tile(dx, dy, 0, 0).unwrap();
                let tw = (tile_dw.max.x - tile_dw.min.x + 1) as usize;
                let th = (tile_dw.max.y - tile_dw.min.y + 1) as usize;

                let mut tile_poisoned = vec![Rgba::default(); tw * th];
                poisoned.read_tile(dx, dy, 0, 0, &mut tile_poisoned, 1, tw as isize).unwrap();
                let mut tile_fresh = vec![Rgba::default(); tw * th];
                fresh.read_tile(dx, dy, 0, 0, &mut tile_fresh, 1, tw as isize).unwrap();

                assert_eq!(
                    tile_poisoned, tile_fresh,
                    "poisoned, reused tile scratch produced a different tile ({dx},{dy}) \
                     than a fresh instance"
                );
            }
        }
    }
}
