//! `SourceImage` — the source-agnostic decoded image the GPU upload consumes.
//!
//! WHY this exists: the layer-grouping, format-selection and packing logic must
//! NOT care where the channels came from. Phase A fills a `SourceImage` from the
//! single-part `api::Image` facade; Phase B will fill it from `MultiPartInputFile`
//! (per part), and [`SourceImage::from_reader`] from the low-level `FrameBuffer`
//! (subsampled channels at their native sampling).
//! Unifying on "name → (pixel_type, native-endian planar bytes) + dims" lets the
//! upload path be written ONCE and reused by every source adapter.
//!
//! Bytes are NATIVE-endian (host order): the decoder produced host-typed values,
//! and `queue.write_texture` wants native bytes for the matching `f16/f32/u32`
//! texture format. One channel buffer is `width * height * sample_size` bytes,
//! raster order over the data window — exactly `exr-core`'s planar `ChannelData`.

use exr_core::attr::PixelType;
use exr_core::rgba::RgbaImage;
use exr_core::scanline::GenericInputReader;
use exr_core::{ChannelData, Image};

use crate::error::{GpuError, Result};

/// One decoded channel: its full (dot-qualified) name, EXR pixel type, and its
/// pixels as a flat native-endian byte buffer (`width * height * sample_size`).
#[derive(Clone, Debug)]
pub(crate) struct SourceChannel {
    /// Full channel name as it appears in the EXR header (e.g. `"diffuse.R"`).
    pub name: String,
    /// The sample type (`Half` = 2 B, `Float`/`Uint` = 4 B).
    pub pixel_type: PixelType,
    /// Native-endian planar pixels, `width * height * pixel_type.size()` bytes.
    pub bytes: Vec<u8>,
    /// Samples per row: the data-window width for a full-resolution channel,
    /// `num_samples(x_sampling, min.x, max.x)` for a subsampled one.
    pub width: u32,
    /// Sample rows.
    pub height: u32,
    /// `(x_sampling, y_sampling)`; `(1, 1)` = full resolution.
    pub sampling: (i32, i32),
}

/// A decoded image reduced to the minimum the GPU path needs: dimensions plus a
/// flat list of typed channel byte-planes. Source-agnostic (see module docs).
#[derive(Clone, Debug)]
pub(crate) struct SourceImage {
    /// Data-window width in pixels.
    pub width: u32,
    /// Data-window height in pixels.
    pub height: u32,
    /// All channels, in the EXR's (alphabetical) header order.
    pub channels: Vec<SourceChannel>,
}

impl SourceImage {
    /// Build from a single-part `api::Image`.
    ///
    /// Converts each typed `ChannelData` plane to native-endian bytes once. The
    /// facade has already rejected deep / multipart / subsampled, so every channel
    /// here is guaranteed full-resolution (`width * height` samples).
    pub fn from_image(img: &Image) -> Result<Self> {
        // `Image::width/height` are data-window extents and are always >= 1 for a
        // valid file (the header sanity check guarantees min <= max).
        let width = u32::try_from(img.width())
            .map_err(|_| GpuError::Gpu(format!("invalid image width {}", img.width())))?;
        let height = u32::try_from(img.height())
            .map_err(|_| GpuError::Gpu(format!("invalid image height {}", img.height())))?;
        let expected = (width as usize) * (height as usize);

        let mut channels = Vec::new();
        for (name, data) in img.channels() {
            // The facade guarantees full-res planes; assert defensively rather
            // than silently upload a short buffer.
            if data.len() != expected {
                return Err(GpuError::Gpu(format!(
                    "channel \"{name}\" has {} samples, expected {expected} ({width}x{height})",
                    data.len()
                )));
            }
            channels.push(SourceChannel {
                name: name.to_string(),
                pixel_type: data.pixel_type(),
                bytes: channel_to_bytes(data),
                width,
                height,
                sampling: (1, 1),
            });
        }
        Ok(Self { width, height, channels })
    }

    /// Build from a reconstructed luminance-chroma image: de-interleave the RGBA
    /// `Rgba` pixels into planar R, G, B, A half channels. Used by the subsampled
    /// path (the facade rejects subsampled, so `exr-gpu` reconstructs full-res RGBA
    /// via `RgbaInputFile`, then treats it as a normal 4-channel image).
    pub fn from_rgba(img: &RgbaImage) -> Result<Self> {
        let width = u32::try_from(img.width())
            .map_err(|_| GpuError::Gpu(format!("invalid rgba width {}", img.width())))?;
        let height = u32::try_from(img.height())
            .map_err(|_| GpuError::Gpu(format!("invalid rgba height {}", img.height())))?;
        let n = img.pixels.len();
        let (mut r, mut g, mut b, mut a) = (
            Vec::with_capacity(n * 2),
            Vec::with_capacity(n * 2),
            Vec::with_capacity(n * 2),
            Vec::with_capacity(n * 2),
        );
        for p in &img.pixels {
            r.extend_from_slice(&p.r.bits().to_ne_bytes());
            g.extend_from_slice(&p.g.bits().to_ne_bytes());
            b.extend_from_slice(&p.b.bits().to_ne_bytes());
            a.extend_from_slice(&p.a.bits().to_ne_bytes());
        }
        let mk = |name: &str, bytes: Vec<u8>| SourceChannel {
            name: name.to_string(),
            pixel_type: PixelType::Half,
            bytes,
            width,
            height,
            sampling: (1, 1),
        };
        let channels = vec![mk("R", r), mk("G", g), mk("B", b), mk("A", a)];
        Ok(Self { width, height, channels })
    }
}

impl SourceImage {
    /// Build from a scanline reader with every channel at its NATIVE sampling:
    /// an `xs x ys` channel becomes a `num_samples(xs, min.x, max.x) x
    /// num_samples(ys, min.y, max.y)` plane read straight from the file, with no
    /// reconstruction — the low-level `FrameBuffer` path (C++
    /// `Slice(type, base, xStride, yStride, xSampling, ySampling)`), which is
    /// what reads a subsampled file exactly as stored. Works for a single-part
    /// `InputFile` and a multipart scanline part alike.
    pub fn from_reader<R: GenericInputReader>(reader: &mut R) -> Result<Self> {
        let dw = reader.header().data_window().map_err(GpuError::from)?;
        let extent = |lo: i32, hi: i32| {
            u32::try_from(i64::from(hi) - i64::from(lo) + 1)
                .map_err(|_| GpuError::Gpu(format!("invalid data window [{lo}, {hi}]")))
        };
        let (width, height) = (extent(dw.min.x, dw.max.x)?, extent(dw.min.y, dw.max.y)?);
        // The native-sampling read is exr-core's one reader of it (`api::planes`).
        let channels = exr_core::api::planes::planes(reader)
            .map_err(GpuError::from)?
            .into_iter()
            .map(|p| SourceChannel {
                bytes: channel_to_bytes(&p.data),
                pixel_type: p.data.pixel_type(),
                name: p.name,
                width: p.width,
                height: p.height,
                sampling: p.sampling,
            })
            .collect();
        Ok(Self { width, height, channels })
    }
}

/// Flatten a typed `ChannelData` plane into native-endian bytes.
fn channel_to_bytes(data: &ChannelData) -> Vec<u8> {
    match data {
        // `Half::bits()` is the raw IEEE-754 binary16 word; native-endian bytes
        // match the `*16Float` texture formats.
        ChannelData::Half(v) => v.iter().flat_map(|h| h.bits().to_ne_bytes()).collect(),
        ChannelData::Float(v) => v.iter().flat_map(|f| f.to_ne_bytes()).collect(),
        ChannelData::Uint(v) => v.iter().flat_map(|u| u.to_ne_bytes()).collect(),
    }
}
