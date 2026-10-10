//! Channel storage and the ONE layout code path of the [`super::Image`] facade.
//!
//! A level of an image (level 0, or any MIP/RIP level) is a set of channel planes
//! ([`Planes`]) *without* a window: the window is always derived from the image's
//! data window and tiling ([`level_window`]), so a plane set can never disagree with
//! its geometry. Reading ([`ReadPlan`]) and writing ([`Planes::write_buf`]) both
//! build their frame buffers through [`plane_slice`], so a plane is addressed the
//! same way in both directions and for every level.

use std::collections::BTreeMap;

use imath_rs::Box2i;

use super::{ChannelData, Unsupported};
use crate::attr::{Channel, ChannelList, LevelMode, PixelType, TileDescription};
use crate::error::{ExrError, Result};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::Header;
use crate::misc::{num_samples, pixel_type_size};
use crate::tiled::misc as level_math;

/// One channel: its samples and the sampling it was stored at. Sampling lives in
/// the record (not beside it) so insert / replace / remove touch one entry.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Chan {
    pub data: ChannelData,
    pub xs: i32,
    pub ys: i32,
}

impl Chan {
    /// A full-resolution channel (`xSampling == ySampling == 1`).
    pub fn full(data: ChannelData) -> Self {
        Self { data, xs: 1, ys: 1 }
    }
}

/// The channel planes of one level, alphabetical (= header) order.
#[derive(Clone, Debug, PartialEq, Default)]
pub(super) struct Planes {
    pub chans: BTreeMap<String, Chan>,
}

/// `(columns, rows)` of samples a channel of sampling `(xs, ys)` holds over `window`
/// — the single checked layout boundary for every plane (full-resolution, subsampled,
/// any level). Rejects an inverted/empty window, a sampling below 1, and a count that
/// does not fit `usize`.
pub(super) fn sample_dims(window: Box2i, xs: i32, ys: i32) -> Result<(usize, usize)> {
    let width = i64::from(window.max.x) - i64::from(window.min.x) + 1;
    let height = i64::from(window.max.y) - i64::from(window.min.y) + 1;
    if width <= 0 || height <= 0 {
        return Err(ExrError::Arg(format!(
            "Data window has non-positive extent ({width}x{height}); the Image \
             facade needs a positive width and height."
        )));
    }
    if xs < 1 || ys < 1 {
        return Err(ExrError::Arg(format!("Channel sampling {xs}x{ys} must be at least 1x1.")));
    }
    let n = (
        usize::try_from(num_samples(xs, window.min.x, window.max.x)),
        usize::try_from(num_samples(ys, window.min.y, window.max.y)),
    );
    match n {
        (Ok(w), Ok(h)) if w.checked_mul(h).is_some() => Ok((w, h)),
        _ => Err(ExrError::Arg(format!(
            "Data window {width}x{height} overflows the addressable pixel count."
        ))),
    }
}

/// Data window of level `(lx, ly)` of an image whose level 0 is `dw`.
pub(super) fn level_window(td: &TileDescription, dw: Box2i, lx: i32, ly: i32) -> Result<Box2i> {
    level_math::data_window_for_level(td, dw.min.x, dw.max.x, dw.min.y, dw.max.y, lx, ly)
}

/// Whether `(lx, ly)` is a level of `td` over `dw`: the diagonal for MIPMAP, the full
/// grid otherwise (ONE_LEVEL is a 1x1 grid). Same set as [`level_grid`], without
/// building it.
pub(super) fn has_level(td: &TileDescription, dw: Box2i, lx: i32, ly: i32) -> Result<bool> {
    let nx = level_math::num_x_levels(td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?;
    let ny = level_math::num_y_levels(td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?;
    Ok((0..nx).contains(&lx) && (0..ny).contains(&ly) && (td.mode != LevelMode::MipmapLevels || lx == ly))
}

/// Every level `(lx, ly)` of `td` over `dw`, in FILE order (`ly` outer, `lx` inner;
/// MIP levels are the diagonal). `ONE_LEVEL` is just `(0, 0)`.
pub(super) fn level_grid(td: &TileDescription, dw: Box2i) -> Result<Vec<(i32, i32)>> {
    let nx = level_math::num_x_levels(td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?;
    let ny = level_math::num_y_levels(td, dw.min.x, dw.max.x, dw.min.y, dw.max.y)?;
    Ok(match td.mode {
        LevelMode::MipmapLevels => (0..nx).map(|l| (l, l)).collect(),
        _ => (0..ny).flat_map(|ly| (0..nx).map(move |lx| (lx, ly))).collect(),
    })
}

/// The one level validator: level `(lx, ly)` must exist in `td`'s grid and `planes`
/// must be exactly level 0's channels (`base`: names, types, sampling) at that
/// level's derived window. Used by `Image::insert_level` and again before any tiled
/// write emits a byte.
pub(super) fn check_level(
    td: &TileDescription,
    dw: Box2i,
    base: &Planes,
    lx: i32,
    ly: i32,
    planes: &Planes,
) -> Result<()> {
    if !has_level(td, dw, lx, ly)? {
        return Err(ExrError::Arg(format!("Level ({lx}, {ly}) does not exist in the {:?} tiling.", td.mode)));
    }
    let same = planes.chans.len() == base.chans.len()
        && planes.chans.iter().zip(&base.chans).all(|((n, c), (bn, b))| {
            n == bn && c.data.pixel_type() == b.data.pixel_type() && (c.xs, c.ys) == (b.xs, b.ys)
        });
    if !same {
        return Err(ExrError::Arg(format!(
            "Level ({lx}, {ly}) must have exactly level 0's channels (names, types, sampling)."
        )));
    }
    planes.check(level_window(td, dw, lx, ly)?)
}

/// The frame-buffer slice of a plane at byte `offset` holding `window`'s samples at
/// sampling `(xs, ys)`, `cols` samples per row. `Slice::make` bakes the window origin
/// into `base_offset`, so absolute pixel coordinates index a zero-origin plane.
fn plane_slice(type_: PixelType, offset: usize, window: Box2i, (xs, ys): (i32, i32), cols: usize) -> Result<Slice> {
    let size = pixel_type_size(type_)?;
    let width = i64::from(window.max.x) - i64::from(window.min.x) + 1;
    Slice::make(type_, offset, window.min, width, size, size * cols, xs, ys, 0.0, false, false)
}

impl Planes {
    /// The header channel list for these planes.
    pub fn channel_list(&self) -> Result<ChannelList> {
        let mut list = ChannelList::new();
        for (name, c) in &self.chans {
            list.insert(name, Channel::new(c.data.pixel_type(), c.xs, c.ys, false))?;
        }
        Ok(list)
    }

    /// Check every plane's length against `window` at its own sampling.
    pub fn check(&self, window: Box2i) -> Result<()> {
        for (name, c) in &self.chans {
            let (w, h) = sample_dims(window, c.xs, c.ys)?;
            if c.data.len() != w * h {
                return Err(ExrError::Arg(format!(
                    "Channel \"{name}\" has {} samples but the data window needs {} \
                     ({w}x{h} at sampling {}x{}).",
                    c.data.len(),
                    w * h,
                    c.xs,
                    c.ys
                )));
            }
        }
        Ok(())
    }

    /// Channel-planar native-endian byte buffer plus the [`FrameBuffer`] addressing
    /// it, for the low-level writers. Planes follow each other in header order.
    pub fn write_buf(&self, window: Box2i) -> Result<(Vec<u8>, FrameBuffer)> {
        self.check(window)?;
        let mut total = 0usize;
        for c in self.chans.values() {
            total = total
                .checked_add(c.data.len() * c.data.sample_size()?)
                .ok_or_else(|| ExrError::Arg("Channel plane layout overflows usize.".into()))?;
        }
        let mut buffer: Vec<u8> = Vec::with_capacity(total);
        let mut fb = FrameBuffer::new();
        for (name, c) in &self.chans {
            let (cols, _) = sample_dims(window, c.xs, c.ys)?;
            fb.insert(name.as_str(), plane_slice(c.data.pixel_type(), buffer.len(), window, (c.xs, c.ys), cols)?)?;
            // Native bytes straight into `buffer`: the writer's gather
            // re-encodes to XDR.
            match &c.data {
                ChannelData::Half(v) => buffer.extend(v.iter().flat_map(|h| h.bits().to_ne_bytes())),
                ChannelData::Float(v) => buffer.extend(v.iter().flat_map(|f| f.to_ne_bytes())),
                ChannelData::Uint(v) => buffer.extend(v.iter().flat_map(|u| u.to_ne_bytes())),
            }
        }
        Ok((buffer, fb))
    }
}

/// One channel's place in the channel-planar read buffer.
struct PlanChan {
    name: String,
    type_: PixelType,
    sampling: (i32, i32),
    cols: usize,
    rows: usize,
    /// Byte offset of the plane within the buffer.
    offset: usize,
    /// Plane size in bytes.
    len: usize,
}

/// How to read every channel of a part over `window` into one channel-planar byte
/// buffer, and how to split that buffer back into [`Planes`]. Shared by the scanline
/// and tiled readers and by every level, so they cannot diverge.
pub(super) struct ReadPlan {
    window: Box2i,
    chans: Vec<PlanChan>,
    pub total: usize,
}

impl ReadPlan {
    /// Layout for `header`'s channels over `window` (the data window, or a level's).
    /// With `native == false` a subsampled channel is rejected with
    /// [`Unsupported::Subsampled`] (the strict `Image::read` contract); with `true`
    /// it is laid out at its own sampling.
    pub fn new(header: &Header, window: Box2i, native: bool) -> Result<Self> {
        let mut chans = Vec::new();
        let mut total = 0usize;
        for (name, ch) in header.channels()?.iter() {
            if !native && (ch.x_sampling != 1 || ch.y_sampling != 1) {
                // Structured kind (not a string): consumers route on it.
                return Err(ExrError::Unsupported(Unsupported::Subsampled));
            }
            // `api::Image` stays `String`-keyed — a hard
            // error, never a lossy conversion, on a non-UTF-8 channel name.
            let name = name
                .to_str()
                .ok_or_else(|| ExrError::Input(format!("non-UTF-8 channel name \"{name}\" (api facade)")))?;
            let (cols, rows) = sample_dims(window, ch.x_sampling, ch.y_sampling)?;
            let size = pixel_type_size(ch.type_)?;
            let len = cols
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(size))
                .ok_or_else(|| ExrError::Arg(format!("Channel plane {cols}x{rows} overflows usize.")))?;
            chans.push(PlanChan {
                name: name.to_string(),
                type_: ch.type_,
                sampling: (ch.x_sampling, ch.y_sampling),
                cols,
                rows,
                offset: total,
                len,
            });
            total = total
                .checked_add(len)
                .ok_or_else(|| ExrError::Arg("Channel plane layout overflows usize.".into()))?;
        }
        Ok(Self { window, chans, total })
    }

    /// The [`FrameBuffer`] whose slices address the channel-planar buffer.
    pub fn frame_buffer(&self) -> Result<FrameBuffer> {
        let mut fb = FrameBuffer::new();
        for c in &self.chans {
            fb.insert(c.name.as_str(), plane_slice(c.type_, c.offset, self.window, c.sampling, c.cols)?)?;
        }
        Ok(fb)
    }

    /// Split the filled buffer (native-endian, as the low-level readers scatter it)
    /// into typed planes.
    pub fn into_planes(self, buffer: &[u8]) -> Result<Planes> {
        let mut chans = BTreeMap::new();
        for c in self.chans {
            let plane = buffer
                .get(c.offset..c.offset + c.len)
                .ok_or_else(|| ExrError::Logic(format!("Read buffer too small for channel \"{}\" plane.", c.name)))?;
            let data = ChannelData::from_ne_bytes(c.type_, plane, c.cols * c.rows)?;
            chans.insert(c.name, Chan { data, xs: c.sampling.0, ys: c.sampling.1 });
        }
        Ok(Planes { chans })
    }
}
