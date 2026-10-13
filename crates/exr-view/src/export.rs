//! Saving what the viewer shows, in the viewer (File > Save view / Save view as HDR)
//! and headless (`exr-rs view <file> --screenshot <out>` [`--hdr`]): an SDR 8-bit sRGB
//! PNG, or HDR: the display light as an EXR / an HDR10 PQ or HLG PNG (`save_hdr`, below).
//!
//! The image is the display transform of `transfer.rs` (exposure, gamma, channel
//! isolate, sRGB OETF: the CPU twin the GPU test holds the tile shader to), or the
//! OCIO transform's CPU twin (`ocio::Transform::display`) when OCIO is shown, applied
//! to the shown content, not a grab of the window: egui-wgpu's frame capture reads
//! only `Rgba8Unorm`/`Bgra8Unorm` surfaces and this viewer draws to an sRGB one.
//! The content is level 0 of an in-RAM frame (the effect's output when one runs),
//! or the overview level of a file-backed frame ([`ImageBackend::export_level`]).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::image::{DecodedFrame, DisplayTarget, ImageBackend, ImageStage};
use crate::transfer;

/// One saved image: display-encoded 8-bit RGB.
pub struct Shot {
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
    /// `w * h * 3` bytes.
    rgb: Vec<u8>,
    #[cfg(feature = "ofx")]
    _permit: Option<crate::tiles::region::Permit>,
}

impl Shot {
    /// Encoded RGB pixels remain attached to their physical admission owner.
    pub fn rgb(&self) -> &[u8] { &self.rgb }
}

/// The look a shot is encoded with (the viewer's display controls).
#[derive(Clone)]
pub struct Look {
    /// Exposure in stops.
    pub exposure: f32,
    /// Gamma dial.
    pub gamma: f32,
    /// `transfer::mode::*`.
    pub mode: u32,
    /// `transfer::tonemap::*`.
    pub tonemap: u32,
    /// The OCIO transform of the shown frame's colour image in place of the built-in decode,
    /// the tone operator and the sRGB encode (`None` = the built-in transform).
    pub ocio: Option<Arc<crate::ocio::Transform>>,
    /// How the frame's stored values are read (`ocio::State::frame`): the built-in decode, the
    /// alpha kind and the display gain for the output the file is saved for.
    pub src: transfer::Src,
    /// SDR white in nits: the nits of canvas light 1.0 in an HDR file.
    pub white: f32,
    /// The HLG display's peak in nits (the viewer's `Target::peak`: Settings > Display, else
    /// the OS's, else 1000): an HLG PNG is encoded for it, so the file is what an HLG window
    /// shows. HLG is relative to its display; PQ and the EXR ignore this.
    pub peak: f32,
}

impl Default for Look {
    /// The neutral look: 0 EV, gamma 1, RGBA, no tone operator, no OCIO.
    fn default() -> Self {
        Self {
            exposure: 0.0,
            gamma: 1.0,
            mode: transfer::mode::RGBA,
            tonemap: transfer::tonemap::NONE,
            ocio: None,
            src: transfer::Src::LINEAR,
            white: crate::present::SCRGB_NITS,
            peak: egui_display::export::HLG_REFERENCE_PEAK_NITS,
        }
    }
}

/// Pixels per parallel work unit of the export loops (the tail unit is ragged).
/// Every conversion runs over these, so no full-size f32 intermediate exists.
const STRIP: usize = 1 << 16;

/// Level 0 (or the overview level) of what the viewer shows, linear RGBA f32.
enum RawSource {
    Original(Arc<Vec<f32>>),
    #[cfg(feature = "ofx")]
    Processed(Box<std::sync::Mutex<crate::consumers::PixelRows>>),
}

/// Frozen linear content; processed rows retain accepted owners and admitted
/// encoder scratch before this value can escape to a delayed job.
pub struct Raw {
    pub w: u32,
    pub h: u32,
    pub level: u32,
    source: RawSource,
}
impl Raw {
    fn validate_original(w: u32, h: u32, px: &[f32]) -> Result<(), crate::tiles::region::RegionError> {
        let expected = crate::tiles::region::pixel_count(w, h)?.checked_mul(4)
            .ok_or("export sample count overflows")?;
        if px.len() != expected { return Err("export metadata does not match frozen original storage".into()); }
        Ok(())
    }
    fn original(w: u32, h: u32, level: u32, px: Arc<Vec<f32>>) -> Result<Self, crate::tiles::region::RegionError> {
        Self::validate_original(w, h, &px)?;
        Ok(Self { w, h, level, source: RawSource::Original(px) })
    }
}

struct OriginalRows<'a> {
    pixels: &'a [f32],
    width: usize,
    rgba: Vec<f32>,
    alpha: Vec<f32>,
    encoded: Vec<u8>,
}
enum RowSession<'a> {
    Original(OriginalRows<'a>),
    #[cfg(feature = "ofx")]
    Processed(std::sync::MutexGuard<'a, crate::consumers::PixelRows>),
}
fn scratch_vec<T: Clone>(len: usize, fill: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len)?;
    values.resize(len, fill);
    Ok(values)
}
impl<'a> RowSession<'a> {
    fn new(raw: &'a Raw) -> Result<Self> {
        match &raw.source {
            RawSource::Original(pixels) => {
                Raw::validate_original(raw.w, raw.h, pixels)?;
                let width = usize::try_from(raw.w)?;
                Ok(Self::Original(OriginalRows { pixels, width,
                    rgba: scratch_vec(width.checked_mul(4).context("export row overflows")?, 0.0)?,
                    alpha: scratch_vec(width, 0.0)?,
                    encoded: scratch_vec(width.checked_mul(6).context("export row overflows")?, 0)? }))
            }
            #[cfg(feature = "ofx")]
            RawSource::Processed(rows) => {
                let rows = rows.lock().map_err(|_| anyhow::anyhow!("export scratch lock was poisoned"))?;
                if (raw.w, raw.h) != rows.size() || raw.level != rows.level() {
                    return Err(anyhow::Error::new(crate::tiles::region::RegionError::Failed(
                        "export metadata does not match frozen processed rows".into())));
                }
                Ok(Self::Processed(rows))
            },
        }
    }
    fn load(&mut self, y: u32) -> Result<()> {
        match self {
            Self::Original(rows) => {
                let len = rows.width.checked_mul(4).context("export row overflows")?;
                let start = usize::try_from(y)?.checked_mul(len).context("export row offset overflows")?;
                let end = start.checked_add(len).context("export row offset overflows")?;
                rows.rgba.copy_from_slice(rows.pixels.get(start..end).context("export row is outside frozen source")?);
            }
            #[cfg(feature = "ofx")]
            Self::Processed(rows) => { rows.row(y).map_err(anyhow::Error::new)?; }
        }
        Ok(())
    }
    fn buffers(&mut self) -> (&mut [f32], &mut [f32], &mut [u8]) {
        match self {
            Self::Original(rows) => (&mut rows.rgba, &mut rows.alpha, &mut rows.encoded),
            #[cfg(feature = "ofx")]
            Self::Processed(rows) => {
                let rows = &mut **rows;
                (&mut rows.rgba, &mut rows.alpha, &mut rows.encoded)
            }
        }
    }
    fn canvas(&mut self, y: u32, look: &Look) -> Result<()> {
        self.load(y)?;
        let (rgba, alpha, _) = self.buffers();
        let pixels: &mut [[f32; 4]] = bytemuck::try_cast_slice_mut(rgba)
            .map_err(|_| anyhow::anyhow!("export RGBA row is not aligned"))?;
        shade(pixels, alpha, look)
    }
    fn light(&mut self, y: u32, look: &Look) -> Result<()> {
        self.canvas(y, look)?;
        let (rgba, _, _) = self.buffers();
        for pixel in rgba.as_chunks_mut::<4>().0 {
            for channel in &mut pixel[..3] { *channel = transfer::eotf(*channel); }
        }
        Ok(())
    }
    fn sdr(&mut self, y: u32, look: &Look) -> Result<&[u8]> {
        self.canvas(y, look)?;
        let (rgba, _, encoded) = self.buffers();
        let count = rgba.len() / 4 * 3;
        for (pixel, out) in rgba.as_chunks::<4>().0.iter().zip(encoded[..count].as_chunks_mut::<3>().0.iter_mut()) {
            for (value, code) in pixel[..3].iter().zip(out) { *code = transfer::code8(*value); }
        }
        Ok(&encoded[..count])
    }
    /// Row `y` as big-endian 16-bit BT.2020 RGB of the HDR PNG signal `png`.
    fn hdr_png(&mut self, y: u32, look: &Look, png: HdrPng) -> Result<&[u8]> {
        self.light(y, look)?;
        let (rgba, _, encoded) = self.buffers();
        for (pixel, out) in rgba.as_chunks::<4>().0.iter().zip(encoded.as_chunks_mut::<6>().0.iter_mut()) {
            let nits = crate::present::rec2020_nits([pixel[0], pixel[1], pixel[2]], look.white);
            for (code, bytes) in png.codes(nits, look.peak).into_iter().zip(out.as_chunks_mut::<2>().0.iter_mut()) {
                bytes.copy_from_slice(&((code.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes());
            }
        }
        Ok(encoded)
    }
    fn half(&mut self, y: u32, look: &Look) -> Result<&[u8]> {
        self.light(y, look)?;
        let (rgba, _, encoded) = self.buffers();
        for (pixel, out) in rgba.as_chunks::<4>().0.iter().zip(encoded.as_chunks_mut::<6>().0.iter_mut()) {
            for (value, bytes) in pixel[..3].iter().zip(out.as_chunks_mut::<2>().0.iter_mut()) {
                bytes.copy_from_slice(&imath_rs::Half::new(*value).bits().to_ne_bytes());
            }
        }
        Ok(encoded)
    }
}

impl Look {
    /// A digest of everything the display values depend on (the scopes' upload key): equal
    /// looks give equal ids, a change of any dial, the transform or the frame's source reading
    /// gives another. Never 0 (0 is the scopes' "stored values").
    pub(crate) fn id(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let floats = [self.exposure, self.gamma, self.src.gain, self.white, self.peak];
        floats.map(f32::to_bits).hash(&mut h);
        (self.mode, self.tonemap, self.src.straight).hash(&mut h);
        format!("{:?}", self.src.builtin).hash(&mut h);
        self.ocio.as_ref().map(|t| t.key.as_str()).hash(&mut h);
        h.finish().max(1)
    }
}

/// [`shade`] over flat stored RGBA samples (`4 * n` floats), in place.
pub(crate) fn shade_flat(rgba: &mut [f32], look: &Look) -> Result<()> {
    let px: &mut [[f32; 4]] = bytemuck::try_cast_slice_mut(rgba).map_err(|_| anyhow::anyhow!("RGBA samples are not whole pixels"))?;
    let mut alpha = vec![0.0; px.len()];
    shade(px, &mut alpha, look)
}

/// The one CPU display transform of every export path: stored RGBA `px` to canvas values in
/// place, through the shown frame's OCIO transform (colour mode) or the built-in twin of the tile
/// shader; `alpha` is the OCIO twin's per-pixel scratch (`px.len()` long).
fn shade(px: &mut [[f32; 4]], alpha: &mut [f32], look: &Look) -> Result<()> {
    if let Some(t) = look.ocio.as_ref().filter(|_| look.mode == transfer::mode::RGBA) {
        return t.display_scratch(px, alpha, &look.src, look.exposure, look.gamma).map_err(anyhow::Error::msg);
    }
    for p in px {
        *p = transfer::display_pixel(*p, &look.src, look.exposure, look.gamma, look.mode, look.tonemap);
    }
    Ok(())
}

/// Stored RGBA (whole pixels) through the display transform: the canvas value per pixel
/// (display-encoded RGB, premultiplied in light). Alpha is dropped: EXR alpha is premultiplied and
/// PNG's is straight, so the image is saved as the viewer paints it, without the checkerboard
/// behind it.
fn canvas_px(stored: &[f32], look: &Look) -> Result<Vec<[f32; 3]>> {
    let mut px: Vec<[f32; 4]> = stored.as_chunks::<4>().0.to_vec();
    let mut alpha = vec![0.0; px.len()];
    shade(&mut px, &mut alpha, look)?;
    Ok(px.iter().map(|d| [d[0], d[1], d[2]]).collect())
}

/// Stored `w * h * 4` RGBA through the display transform, as SDR 8-bit RGB.
pub fn encode(stored: &[f32], look: &Look) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    let strips: Vec<Vec<u8>> = stored
        .par_chunks(STRIP * 4)
        .map(|c| Ok(canvas_px(c, look)?.into_iter().flat_map(|d| d.map(transfer::code8)).collect()))
        .collect::<Result<_>>()?;
    Ok(strips.concat())
}

/// The display light of `linear` RGBA pixels: the canvas value decoded back to linear
/// Rec.709, 1.0 = SDR white (`look.white` nits). This is the stage the present pass
/// starts from (`eotf`, then x white nits), so the HDR files, the PQ signal and an
/// scRGB screen agree. Out-of-gamut and negative values are kept (`eotf` is the
/// extended sRGB curve).
#[cfg(test)]
fn light_px(linear: &[f32], look: &Look) -> Vec<[f32; 3]> {
    let mut d = canvas_px(linear, look).expect("the display transform");
    for p in &mut d {
        *p = p.map(transfer::eotf);
    }
    d
}

/// The grab of `img` as shown, `None` while a file-backed overview is still
/// streaming in (the level has been requested).
pub fn grab(
    img: &mut ImageBackend,
    target: &mut DisplayTarget,
    max_dim: u32,
    stage: Option<&mut (dyn ImageStage + '_)>,
) -> Result<Option<Raw>, crate::tiles::region::RegionError> {
    #[cfg(feature = "ofx")]
    if let Some(pixels) = target.accepted_pixels() {
        let level = pixels.level(u64::MAX, max_dim);
        let rows = crate::consumers::PixelRows::for_export(pixels, level)?;
        let (w, h) = rows.size();
        return Ok(Some(Raw { w, h, level, source: RawSource::Processed(Box::new(std::sync::Mutex::new(rows))) }));
    }
    let Some((level, w, h, px)) = img.export_level(target, max_dim, stage) else { return Ok(None); };
    Raw::original(w, h, level, px).map(Some)
}

impl Raw {
    /// Consume the frozen source. A failed RGB admission releases its pins,
    /// so a delayed Shot cannot wait for memory while blocking promotion.
    pub fn sdr(self, look: &Look) -> Result<Shot> {
        match &self.source {
            RawSource::Original(px) => {
                Self::validate_original(self.w, self.h, px)?;
                Ok(Shot { w: self.w, h: self.h, rgb: encode(px, look)?,
                    #[cfg(feature = "ofx")] _permit: None })
            },
            #[cfg(feature = "ofx")]
            RawSource::Processed(_) => self.processed_sdr(look),
        }
    }
    #[cfg(feature = "ofx")]
    fn processed_sdr(&self, look: &Look) -> Result<Shot> {
        let mut session = RowSession::new(self)?;
        let rows = match &session {
            RowSession::Processed(rows) => rows,
            RowSession::Original(_) => bail!("processed shot received an original row source"),
        };
        let budget = rows.snapshot().budget().map_err(anyhow::Error::new)?;
        let bytes = crate::tiles::region::pixel_count(self.w, self.h).map_err(anyhow::Error::msg)?
            .checked_mul(3).context("shot byte count overflows")?;
        let own = rows.bytes().checked_add(bytes).context("shot storage overflows")?;
        if own > budget.candidate_limit() {
            return Err(anyhow::Error::new(crate::tiles::region::RegionError::Budget {
                needed: own, limit: budget.candidate_limit() }));
        }
        let mut permit = budget.reserve(bytes).map_err(anyhow::Error::new)?;
        let mut rgb = scratch_vec(bytes, 0)?;
        let actual = rows.bytes().checked_add(rgb.capacity()).context("shot capacity overflows")?;
        if actual > budget.candidate_limit() {
            return Err(anyhow::Error::new(crate::tiles::region::RegionError::Budget {
                needed: actual, limit: budget.candidate_limit() }));
        }
        permit.resize(rgb.capacity()).map_err(anyhow::Error::new)?;
        let row_bytes = usize::try_from(self.w)?.checked_mul(3).context("shot row overflows")?;
        for y in 0..self.h {
            let offset = usize::try_from(y)?.checked_mul(row_bytes).context("shot row offset overflows")?;
            rgb[offset..offset + row_bytes].copy_from_slice(session.sdr(y, look)?);
        }
        Ok(Shot { w: self.w, h: self.h, rgb, _permit: Some(permit) })
    }
}

// Both PNG formats must explicitly finish the outer writer: its Drop ignores
// IEND write errors, while stream.finish() only completes image data.
fn write_png_rows<W: std::io::Write>(
    mut writer: png::Writer<W>,
    write_rows: impl FnOnce(&mut png::StreamWriter<'_, W>) -> Result<()>,
) -> Result<()> {
    let mut stream = writer.stream_writer().context("PNG data")?;
    write_rows(&mut stream)?;
    stream.finish().context("PNG data end")?;
    writer.finish().context("PNG end")
}

/// Stream display RGB8 from one frozen row session. Encoder-private compression
/// storage is outside pixel admission; our row/alpha/format buffers are admitted.
pub fn write_sdr_png(path: &Path, raw: &Raw, look: &Look) -> Result<()> {
    use std::io::Write;
    let mut rows = RowSession::new(raw)?;
    exr_core::io::write_file(path, |out| -> Result<()> {
        let mut encoder = png::Encoder::new(out, raw.w, raw.h);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let writer = encoder.write_header().context("PNG header")?;
        write_png_rows(writer, |stream| {
            for y in 0..raw.h { stream.write_all(rows.sdr(y, look)?).context("PNG data")?; }
            Ok(())
        })
    }).with_context(|| format!("write {}", path.display()))
}

/// Write `shot` as an sRGB PNG.
pub fn write_png(path: &Path, shot: &Shot) -> Result<()> {
    exr_core::io::write_file(path, |out| -> Result<()> {
        let mut enc = png::Encoder::new(out, shot.w, shot.h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut writer = enc.write_header().context("PNG header")?;
        writer.write_image_data(shot.rgb()).context("PNG data")?;
        writer.finish().context("PNG end")?;
        Ok(())
    })
    .with_context(|| format!("write {}", path.display()))
}

/// The signal of an HDR PNG: the HDR half of the shared `egui_display` PNG encodings
/// (`PngEncoding`, Playa's), without the SDR case the HDR writer has no use for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HdrPng {
    /// HDR10: BT.2100 PQ, absolute nits.
    #[default]
    Pq,
    /// BT.2100 HLG for the display peak of `Look::peak`.
    Hlg,
}

impl HdrPng {
    /// Both, in menu order.
    pub const ALL: [Self; 2] = [Self::Pq, Self::Hlg];

    /// The shared encoding, whose labels the UI shows.
    pub fn encoding(self) -> egui_display::export::PngEncoding {
        match self {
            Self::Pq => egui_display::export::PngEncoding::Hdr10,
            Self::Hlg => egui_display::export::PngEncoding::Hlg,
        }
    }

    /// PNG `cICP` code points (ITU-T H.273, PNG 3rd ed. §cICP): BT.2020 primaries (9),
    /// the transfer (PQ 16, HLG 18), RGB matrix (0), full range (1).
    fn cicp(self) -> [u8; 4] {
        match self {
            Self::Pq => [9, 16, 0, 1],
            Self::Hlg => [9, 18, 0, 1],
        }
    }

    /// BT.2020 light in nits to the 0..1 codes, through the shared `egui_display` curves the
    /// present pass also uses. HLG is encoded for a display of `peak` nits, the one the HLG
    /// window encodes for (`Look::peak`), so the file is what that window shows (at 1000 nits
    /// the BT.2408 SDR white lands at 75 %). Out-of-gamut (negative) BT.2020 components go in
    /// as they are, as on screen: `hlg` takes the luminance of the whole pixel and clips the
    /// signal, which is what OCIO's HLG display does
    /// (`present::tests::hlg_matches_ocio_at_1000_nits`).
    fn codes(self, nits: [f32; 3], peak: f32) -> [f32; 3] {
        match self {
            Self::Pq => nits.map(crate::present::pq),
            Self::Hlg => egui_display::hlg(nits, peak),
        }
    }
}
/// PNG 3rd ed. `cLLI` stores MaxCLL / MaxFALL in units of 0.0001 cd/m².
const CLLI_UNIT: f32 = 10_000.0;
/// The light of one pixel as the PQ signal carries it, in nits: the largest of R, G, B
/// of the BT.2020 light, held to PQ's 0..10000 cd/m² (what the encoded signal can say;
/// negatives and NaN count as 0, inf as the peak).
fn peak_nits(l: [f32; 3], white: f32) -> f32 {
    let n = crate::present::rec2020_nits(l, white);
    let m = n[0].max(n[1]).max(n[2]);
    if m.is_nan() { 0.0 } else { m.clamp(0.0, crate::present::PQ_PEAK) }
}

/// The content light levels of `linear` (CTA-861.3, nits): MaxCLL = the largest
/// per-pixel [`peak_nits`], MaxFALL = their frame average.
#[cfg(test)]
fn light_levels(linear: &[f32], look: &Look) -> (f32, f32) {
    use rayon::prelude::*;
    let (max, sum) = linear
        .par_chunks(STRIP * 4)
        .map(|lin| {
            light_px(lin, look).into_iter().fold((0.0f32, 0.0f64), |(max, sum), l| {
                let peak = peak_nits(l, look.white);
                (max.max(peak), sum + f64::from(peak))
            })
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0.max(b.0), a.1 + b.1));
    (max, (sum / (linear.len() / 4).max(1) as f64) as f32)
}

/// `linear` as big-endian 16-bit PQ RGB, the PNG's sample bytes.
#[cfg(test)]
fn pq_be(linear: &[f32], look: &Look) -> Vec<u8> {
    use rayon::prelude::*;
    linear
        .par_chunks(STRIP * 4)
        .flat_map_iter(|lin| {
            light_px(lin, look).into_iter().flat_map(|l| {
                crate::present::rec2020_nits(l, look.white)
                    .into_iter()
                    .flat_map(|v| ((crate::present::pq(v) * 65535.0).round() as u16).to_be_bytes())
            })
        })
        .collect()
}

#[cfg(all(test, feature = "ofx"))]
thread_local! {
    // Writer-thread-local synchronization only, never a pixel/owner registry.
    static STATISTICS_CHECKPOINT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}
#[cfg(all(test, feature = "ofx"))]
pub(crate) fn set_statistics_checkpoint(checkpoint: Box<dyn FnOnce()>) {
    STATISTICS_CHECKPOINT.with(|slot| *slot.borrow_mut() = Some(checkpoint));
}

/// The content light levels of a PQ PNG (`cLLI`, CTA-861.3): MaxCLL / MaxFALL in nits,
/// a first pass over `rows` (`cLLI` precedes the pixel data).
fn clli(rows: &mut RowSession<'_>, raw: &Raw, look: &Look) -> Result<[u8; 8]> {
    let mut max_cll = 0.0f32;
    let mut sum = 0.0f64;
    for y in 0..raw.h {
        rows.light(y, look)?;
        let (rgba, _, _) = rows.buffers();
        for pixel in rgba.as_chunks::<4>().0 {
            let peak = peak_nits([pixel[0], pixel[1], pixel[2]], look.white);
            max_cll = max_cll.max(peak);
            sum += f64::from(peak);
        }
    }
    let count = crate::tiles::region::pixel_count(raw.w, raw.h).map_err(anyhow::Error::msg)?;
    let max_fall = (sum / count as f64) as f32;
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&((max_cll * CLLI_UNIT).round() as u32).to_be_bytes());
    out[4..].copy_from_slice(&((max_fall * CLLI_UNIT).round() as u32).to_be_bytes());
    Ok(out)
}

/// Write `raw` as an HDR PNG: 16-bit RGB of the `png` signal in BT.2020, tagged with
/// `cICP` (9/16/0/1 for PQ, 9/18/0/1 for HLG). A PQ file also carries `cLLI`; an HLG one
/// does not (its light is relative to the display, as the shared writer has it). No
/// `sRGB`/`gAMA` chunk: they would contradict `cICP`. The display light is `look`'s, in
/// nits from `look.white`. The samples are converted one row at a time: no whole-image
/// signal buffer.
pub fn write_hdr_png(path: &Path, raw: &Raw, look: &Look, png: HdrPng) -> Result<()> {
    use std::io::Write;
    // One guard and the same accepted owners span both passes.
    let mut rows = RowSession::new(raw)?;
    let clli = match png {
        HdrPng::Pq => Some(clli(&mut rows, raw, look)?),
        HdrPng::Hlg => None,
    };
    #[cfg(all(test, feature = "ofx"))]
    STATISTICS_CHECKPOINT.with(|slot| {
        let checkpoint = slot.borrow_mut().take();
        if let Some(checkpoint) = checkpoint { checkpoint(); }
    });
    exr_core::io::write_file(path, |out| -> Result<()> {
        let mut enc = png::Encoder::new(out, raw.w, raw.h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Sixteen);
        let mut writer = enc.write_header().context("PNG header")?;
        writer.write_chunk(png::chunk::cICP, &png.cicp()).context("PNG cICP")?;
        if let Some(clli) = clli {
            writer.write_chunk(png::chunk::cLLI, &clli).context("PNG cLLI")?;
        }
        write_png_rows(writer, |stream| {
            for y in 0..raw.h { stream.write_all(rows.hdr_png(y, look, png)?).context("PNG data")?; }
            Ok(())
        })
    }).with_context(|| format!("write {}", path.display()))
}

/// Write `raw` as an EXR of the display light: half RGB, linear Rec.709,
/// `chromaticities` stated explicitly and `whiteLuminance` = `look.white`, the
/// luminance in cd/m² of RGB (1,1,1) (OpenEXR `ImfStandardAttributes.h`), so
/// pixel x whiteLuminance is nits. Native RGBHalf rows stream into the existing
/// scanline writer; no whole-image channel planes are assembled.
pub fn write_exr(path: &Path, raw: &Raw, look: &Look) -> Result<()> {
    use exr_core::{Channel, Chromaticities, ChromaticitiesAttribute, Compression,
        FloatAttribute, LineOrder, PixelType};
    use exr_core::frame_buffer::{FrameBuffer, Slice};
    use exr_core::header::Header;
    let mut rows = RowSession::new(raw)?;
    let mut header = Header::with_dimensions(i32::try_from(raw.w)?, i32::try_from(raw.h)?,
        1.0, imath_rs::V2f::new(0.0, 0.0), 1.0, LineOrder::IncreasingY, Compression::Zip)?;
    for name in ["R", "G", "B"] {
        header.channels_mut()?.insert(name, Channel::new(PixelType::Half, 1, 1, false))?;
    }
    header.insert("chromaticities", ChromaticitiesAttribute::new(Chromaticities::default()))?;
    header.insert("whiteLuminance", FloatAttribute::new(look.white))?;
    #[cfg(feature = "ofx")]
    if let RowSession::Processed(processed) = &rows {
        let lineage = processed.snapshot().lineage();
        *header.pixel_aspect_ratio_mut()? = lineage.pixel_aspect as f32;
        header.insert("exrViewPrimaryDataWindow", exr_core::Box2iAttribute::new(lineage.data_window))?;
        header.insert("exrViewPrimaryDisplayWindow", exr_core::Box2iAttribute::new(lineage.display_window))?;
        header.insert("exrViewPrimaryIndex", exr_core::IntAttribute::new(lineage.primary_index))?;
        header.insert("exrViewTime", exr_core::DoubleAttribute::new(lineage.time))?;
        header.insert("exrViewLevel", exr_core::IntAttribute::new(i32::try_from(raw.level)?))?;
    }
    // The caller supplies a different native RGBHalf row on each call; zero
    // y stride is intentional. The existing writer owns the ZIP16 chunk buffer.
    let mut layout = FrameBuffer::new();
    for (name, offset) in [("R", 0), ("G", 2), ("B", 4)] {
        layout.insert(name, Slice::new(PixelType::Half, offset, 6, 0, 1, 1, 0.0, false, false)?)?;
    }
    exr_core::io::write_file(path, |out| -> Result<()> {
        let mut writer = exr_core::scanline::OutputFile::new(out, &header)?;
        writer.set_frame_buffer(&layout)?;
        for y in 0..raw.h { writer.write_pixels(rows.half(y, look)?, 1)?; }
        writer.finish()?;
        Ok(())
    }).with_context(|| format!("write {}", path.display()))
}

/// Save `raw` as the HDR format the extension of `path` names: `.exr` = display
/// light, `.png` = the `png` signal (HDR10 PQ or HLG). Anything else is an error, not a guess.
pub fn save_hdr(path: &Path, raw: &Raw, look: &Look, png: HdrPng) -> Result<()> {
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("exr") => write_exr(path, raw, look),
        Some("png") => write_hdr_png(path, raw, look, png),
        _ => bail!("{}: HDR view export writes .exr (display light) or .png (HDR10 PQ / HLG)", path.display()),
    }
}

/// How long the headless shot waits for a file-backed overview.
const HEADLESS_WAIT: Duration = Duration::from_secs(600);

/// Open `input` as the viewer does (file-backed above `backing` bytes) and wait for its
/// level 0 / overview; `ram_budget` sizes the tile cache. No window, no GPU.
fn grab_wait(input: &Path, backing: u64, ram_budget: usize, upload: usize) -> Result<(Raw, Arc<DecodedFrame>)> {
    let frame = Arc::new(DecodedFrame::open(input, backing).map_err(|e| anyhow::anyhow!("{}: {e}", input.display()))?);
    let mut img = ImageBackend::from_frame(frame.clone());
    let mut target = DisplayTarget::new(ram_budget, upload);
    let deadline = Instant::now() + HEADLESS_WAIT;
    loop {
        if let Some(raw) = grab(&mut img, &mut target, u32::MAX, None)? {
            return Ok((raw, frame));
        }
        if Instant::now() > deadline {
            bail!("{}: the overview did not arrive within {HEADLESS_WAIT:?}", input.display());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The neutral look (0 EV, gamma 1, RGBA, no tone operator) of `frame` for `target`, through
/// the OCIO choice `sel` as the viewer would show it (`ocio::State::frame`: the frame's colour
/// space, its view, the same processor). OCIO on but not shown (a config that does not load, a
/// colour space the config lacks) is an error, not a silent built-in shot.
fn headless_look(frame: &DecodedFrame, sel: &crate::ocio::Sel, target: &crate::present::Target) -> Result<Look> {
    let mut colour = crate::ocio::State::new(sel.clone());
    colour.set_hdr(target.hdr);
    let src = colour.frame(frame, target).map_err(anyhow::Error::msg)?;
    if let Some(e) = &colour.err {
        bail!("OCIO: {e}");
    }
    Ok(Look { ocio: colour.active().cloned(), src, white: target.white, ..Look::default() })
}

/// Open `input`, take the shot with the neutral look through `sel` and write it to `output` as
/// an SDR PNG.
pub fn headless(
    input: &Path,
    output: &Path,
    sel: &crate::ocio::Sel,
    backing: u64,
    ram_budget: usize,
    upload: usize,
) -> Result<Shot> {
    let (raw, frame) = grab_wait(input, backing, ram_budget, upload)?;
    let target = crate::present::Target { hdr: false, white: HEADLESS_WHITE, ..crate::present::Target::default() };
    let shot = raw.sdr(&headless_look(&frame, sel, &target)?)?;
    write_png(output, &shot)?;
    Ok(shot)
}

/// SDR white of a headless HDR save: the BT.2408 HDR reference white, what an HDR10
/// display shows for the neutral look.
pub const HEADLESS_WHITE: f32 = crate::present::HDR_REF_WHITE;

/// Like [`headless`], saved as HDR (`save_hdr`: `.exr`, or `.png` as `png`) for an HDR output
/// with SDR white at [`HEADLESS_WHITE`]. Returns the size.
pub fn headless_hdr(
    input: &Path,
    output: &Path,
    png: HdrPng,
    sel: &crate::ocio::Sel,
    backing: u64,
    ram_budget: usize,
    upload: usize,
) -> Result<(u32, u32)> {
    let (raw, frame) = grab_wait(input, backing, ram_budget, upload)?;
    let target = crate::present::Target { hdr: true, white: HEADLESS_WHITE, ..crate::present::Target::default() };
    save_hdr(output, &raw, &headless_look(&frame, sel, &target)?, png)?;
    Ok((raw.w, raw.h))
}

#[cfg(test)]
mod tests {
    #[test]
    fn streamed_png_propagates_iend_write_failure() {
        use std::io::Write;
        struct FailIend { attempts: usize }
        impl Write for FailIend {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes == b"IEND" {
                    self.attempts += 1;
                    return Err(std::io::Error::other("injected IEND failure"));
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        for depth in [png::BitDepth::Eight, png::BitDepth::Sixteen] {
            let mut output = FailIend { attempts: 0 };
            let mut encoder = png::Encoder::new(&mut output, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(depth);
            let writer = encoder.write_header().unwrap();
            let pixels = vec![0; if depth == png::BitDepth::Eight { 3 } else { 6 }];
            let error = write_png_rows(writer, |stream| {
                stream.write_all(&pixels)?;
                Ok(())
            }).expect_err("a failed IEND must fail the save, rather than disappear in Drop");
            assert!(format!("{error:#}").contains("injected IEND failure"), "{error:#}");
            assert_eq!(output.attempts, 1, "the fault must happen at final IEND writing");
        }
    }

    use super::*;

    /// The encode is the display twin per pixel, alpha dropped.
    #[test]
    fn encode_is_the_display_transform() {
        let px = [0.18f32, 0.5, 1.0, 0.25, 4.0, 0.0, -1.0, 1.0];
        let look = Look { exposure: 1.0, gamma: 1.2, mode: transfer::mode::RGBA, tonemap: transfer::tonemap::AGX, ..Look::default() };
        let want: Vec<u8> = px
            .as_chunks::<4>().0.iter()
            .flat_map(|p| {
                let d = transfer::display_pixel([p[0], p[1], p[2], p[3]], &transfer::Src::LINEAR, 1.0, 1.2, transfer::mode::RGBA, transfer::tonemap::AGX);
                [transfer::code8(d[0]), transfer::code8(d[1]), transfer::code8(d[2])]
            })
            .collect();
        assert_eq!(encode(&px, &look).unwrap(), want);
        assert_eq!(encode(&px, &Look::default()).unwrap()[..3], [118, 188, 255], "18% grey is sRGB code 118");
    }

    /// With OCIO the colour image is its CPU twin (ACES 2.0 SDR: 18 % grey at 0.349,
    /// code 89); an isolated channel stays on the built-in path.
    #[test]
    fn encode_uses_ocio_for_colour_only() {
        let o = crate::ocio::Ocio::load("ocio://default").unwrap();
        let t = Arc::new(o.transform(&o.resolve(&crate::ocio::Sel::default(), &o.scene(), false).unwrap(), None, false).unwrap());
        let px = [0.18f32, 0.18, 0.18, 1.0];
        let look = Look { ocio: Some(t), ..Look::default() };
        assert_eq!(encode(&px, &look).unwrap(), [89, 89, 89]);
        let red = Look { mode: transfer::mode::R, ..look };
        assert_eq!(encode(&px, &red).unwrap(), encode(&px, &Look { mode: transfer::mode::R, ..Look::default() }).unwrap());
    }

    /// Headless: a resident file is saved at full resolution and reads back as the
    /// encode of its level 0; a file-backed one is saved at its overview level.
    #[test]
    fn headless_writes_the_shown_image() {
        let tmp = tempfile::Builder::new().prefix("exr-view-shot-").tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let src = dir.join("shot.exr");
        let (w, h) = (300usize, 200usize);
        let plane = |f: &dyn Fn(usize, usize) -> f32| (0..w * h).map(|i| f(i % w, i / w)).collect::<Vec<f32>>();
        let win = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(0, 0), imath_rs::V2i::new(w as i32 - 1, h as i32 - 1));
        exr_core::api::Image::new(win)
            .with_channel("R", exr_core::api::ChannelData::Float(plane(&|x, _| x as f32 / 300.0)))
            .with_channel("G", exr_core::api::ChannelData::Float(plane(&|_, y| y as f32 / 200.0)))
            .with_channel("B", exr_core::api::ChannelData::Float(plane(&|_, _| 0.18)))
            .write(&src, exr_core::Compression::Zip)
            .unwrap();

        let out = dir.join("shot.png");
        let shot = headless(&src, &out, &crate::ocio::Sel::default(), u64::MAX, 1 << 30, 64 << 20).unwrap();
        assert_eq!((shot.w, shot.h), (300, 200));
        let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&out).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height, info.color_type), (300, 200, png::ColorType::Rgb));
        let at = |x: usize, y: usize| &buf[(y * w + x) * 3..(y * w + x) * 3 + 3];
        let want = |x: usize, y: usize| encode(&[x as f32 / 300.0, y as f32 / 200.0, 0.18, 1.0], &Look::default()).unwrap();
        for (x, y) in [(0, 0), (150, 100), (299, 199)] {
            assert_eq!(at(x, y), &want(x, y)[..], "({x}, {y})");
        }

        // Backing budget 1 byte: file-backed, saved at the overview level (the
        // whole image here is within it, so level 0 is the overview).
        let backed = headless(&src, &dir.join("backed.png"), &crate::ocio::Sel::default(), 1, 1 << 30, 64 << 20).unwrap();
        assert_eq!((backed.w, backed.h), (300, 200));
        // A file-backed level is assembled from the pyramid's f16 tiles, a resident
        // frame's level 0 is its f32 pixels: f16 rounding moves a value across an
        // 8-bit code boundary at most by one code.
        let off = backed.rgb.iter().zip(&shot.rgb).map(|(a, b)| a.abs_diff(*b)).max();
        assert_eq!(backed.rgb.len(), shot.rgb.len());
        assert!(off.is_some_and(|d| d <= 1), "file-backed vs resident: max code difference {off:?}");
    }

    // ---- HDR export --------------------------------------------------------------

    /// A unique temp dir per test, removed when the guard drops.
    fn tmp(name: &str) -> tempfile::TempDir {
        tempfile::Builder::new().prefix(&format!("exr-view-hdr-{name}-")).tempdir().unwrap()
    }

    /// `px` as a `w x h` grab.
    fn raw(w: u32, h: u32, px: Vec<f32>) -> Raw {
        assert_eq!(px.len(), (w * h * 4) as usize);
        Raw::original(w, h, 0, Arc::new(px)).unwrap()
    }

    /// Deterministic pseudo-random sample in `0..range` / 1000.
    fn noise(i: u32, range: u32) -> f32 {
        (i.wrapping_mul(2_654_435_761) >> 8) as f32 % range as f32 / 1000.0
    }

    /// The chunks of a PNG file, read off the bytes (not through the encoder's structs).
    fn chunks(bytes: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let (mut at, mut out) = (8, Vec::new());
        while at < bytes.len() {
            let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            let name: [u8; 4] = bytes[at + 4..at + 8].try_into().unwrap();
            out.push((name, bytes[at + 8..at + 8 + len].to_vec()));
            at += 12 + len;
        }
        out
    }

    /// Four pixels of known PQ: black, white, half grey and pure red, at 203 nits SDR white.
    /// Expected codes are ST 2084 hand-computed in f64 (m1 2610/16384, m2 2523/4096*128,
    /// c1 3424/4096, c2 2413/4096*32, c3 2392/4096*32) on the BT.2020 nits of the
    /// Rec.709 -> 2020 matrix derived from the primaries (`transfer::SRGB_TO_REC2020`).
    #[test]
    fn pq_png_holds_the_known_values_and_tags() {
        let guard = tmp("pq");
        let dir = guard.path().to_path_buf();
        let path = dir.join("pq.png");
        let px = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0];
        write_hdr_png(&path, &raw(2, 2, px), &Look { white: 203.0, ..Look::default() }, HdrPng::Pq).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let cs = chunks(&bytes);
        let get = |n: &[u8; 4]| cs.iter().find(|(c, _)| c == n).map(|(_, d)| d.clone());
        assert_eq!(get(b"cICP").unwrap(), [9, 16, 0, 1], "BT.2020, PQ, RGB, full range");
        // MaxCLL 203 nits, MaxFALL (0 + 203 + 101.5 + 127.3622) / 4 nits, in 0.0001 cd/m2.
        let clli = get(b"cLLI").unwrap();
        let (cll, fall) = (u32::from_be_bytes(clli[..4].try_into().unwrap()), u32::from_be_bytes(clli[4..].try_into().unwrap()));
        assert_eq!(cll, 2_030_000);
        assert!(fall.abs_diff(1_079_657) <= 2, "MaxFALL {fall}");
        for bad in [b"sRGB", b"gAMA", b"iCCP", b"cHRM"] {
            assert!(get(bad).is_none(), "{} contradicts cICP", String::from_utf8_lossy(bad));
        }

        let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height, info.color_type, info.bit_depth), (2, 2, png::ColorType::Rgb, png::BitDepth::Sixteen));
        let code = |i: usize| u16::from_be_bytes([buf[i * 2], buf[i * 2 + 1]]);
        let want: [[u16; 3]; 4] = [[0, 0, 0], [38055; 3], [33395; 3], [34900, 21431, 14422]];
        for (p, w) in want.iter().enumerate() {
            for (c, w) in w.iter().enumerate() {
                assert!(code(p * 3 + c).abs_diff(*w) <= 1, "pixel {p} channel {c}: {} vs {w}", code(p * 3 + c));
            }
        }
    }

    /// cLLI describes the encoded signal: per pixel the largest of R, G, B (here G and B
    /// dominate, not only R), held to PQ's 10000 cd/m2 (a 20300-nit pixel and an inf
    /// pixel count as 10000), a NaN pixel as 0. Hand values: the exact Rec.709 -> 2020
    /// matrix at 203 nits gives a G-only 0.5 pixel max(R,G,B) = 93.33335 nits and a
    /// B-only 0.25 pixel 45.45146 nits.
    #[test]
    fn clli_is_the_clamped_max_channel() {
        let guard = tmp("clli");
        let dir = guard.path().to_path_buf();
        let path = dir.join("c.png");
        let nan = f32::NAN;
        let px = vec![
            0.0, 0.5, 0.0, 1.0, // G
            0.0, 0.0, 0.25, 1.0, // B
            100.0, 100.0, 100.0, 1.0, // 20300 nits, clamped
            f32::INFINITY, 0.0, 0.0, 1.0, // inf, clamped
            nan, nan, nan, 1.0, // 0
        ];
        write_hdr_png(&path, &raw(5, 1, px), &Look { white: 203.0, ..Look::default() }, HdrPng::Pq).unwrap();
        let cs = chunks(&std::fs::read(&path).unwrap());
        let clli = &cs.iter().find(|(c, _)| c == b"cLLI").unwrap().1;
        let (cll, fall) = (u32::from_be_bytes(clli[..4].try_into().unwrap()), u32::from_be_bytes(clli[4..].try_into().unwrap()));
        assert_eq!(cll, 100_000_000, "MaxCLL is PQ's 10000 cd/m2, not u32::MAX");
        // (93.33335 + 45.45146 + 10000 + 10000 + 0) / 5 nits.
        assert!(fall.abs_diff(40_277_570) <= 3, "MaxFALL {fall}");
    }

    /// MaxCLL of a frame whose brightest pixel is G-dominant: the max must look at all
    /// three channels (R-only would read 14 nits).
    #[test]
    fn clli_reads_every_channel() {
        let (cll, fall) = light_levels(&[0.0, 1.0, 0.0, 1.0], &Look { white: 203.0, ..Look::default() });
        assert!((cll - 186.6667).abs() < 1e-2 && (fall - cll).abs() < 1e-3, "{cll} {fall}");
    }

    /// PQ at the anchor luminances, through the one `present::pq` the PNG uses.
    #[test]
    fn pq_anchors() {
        for (nits, code) in [(0.0, 0u32), (100.0, 33297), (203.0, 38055), (1000.0, 49271), (10000.0, 65535)] {
            // f64 hand values; the f32 twin may land one code away (203 nits sits at x.0).
            let got = (crate::present::pq(nits) * 65535.0).round() as u32;
            assert!(got.abs_diff(code) <= 1, "{nits} nits: {got} vs {code}");
        }
    }

    /// The file is the present pass's signal: every pixel of a ragged multi-strip image
    /// (more than one `STRIP`, tail unit partial) equals `present::encode`'s PQ of the
    /// canvas, within one 16-bit code.
    #[test]
    fn pq_png_is_the_present_signal() {
        use crate::present::{Encoding, Target, encode as present};
        let (w, h) = (311u32, 241u32);
        assert!(w as usize * h as usize > STRIP && !((w * h) as usize).is_multiple_of(STRIP));
        let px: Vec<f32> = (0..w * h * 4).map(|i| noise(i, 4000) - 0.1).collect();
        let look = Look { exposure: 0.5, gamma: 1.1, white: 300.0, ..Look::default() };
        let bytes = pq_be(&px, &look);
        let samples: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|b| u16::from_be_bytes(*b)).collect();
        let target = Target { hdr: true, white: 300.0, peak: 1000.0 };
        for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
            let d = transfer::display_pixel([p[0], p[1], p[2], p[3]], &transfer::Src::LINEAR, 0.5, 1.1, transfer::mode::RGBA, transfer::tonemap::NONE);
            let want = present([d[0], d[1], d[2]], Encoding::Pq, target);
            for c in 0..3 {
                let got = f32::from(samples[i * 3 + c]) / 65535.0;
                assert!((got - want[c]).abs() <= 0.6 / 65535.0, "pixel {i} ch {c}: {got} vs {}", want[c]);
            }
        }
    }

    /// The 16-bit RGB samples of a PNG file, decoded by the `png` crate.
    fn png_samples(path: &Path) -> (png::OutputInfo, Vec<u16>) {
        let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap())).read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        (info, buf.as_chunks::<2>().0.iter().map(|b| u16::from_be_bytes(*b)).collect())
    }

    /// An HLG PNG is tagged BT.2020 / HLG (cICP 9/18/0/1), carries no cLLI (HLG light is
    /// relative to its display) and nothing contradicting cICP, and holds BT.2408's
    /// anchor: SDR white (203 nits) on the 1000-nit reference display is 75 % signal.
    #[test]
    fn hlg_png_holds_the_reference_white_and_tags() {
        let guard = tmp("hlg");
        let dir = guard.path().to_path_buf();
        let path = dir.join("hlg.png");
        let px = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        write_hdr_png(&path, &raw(2, 1, px), &Look { white: 203.0, ..Look::default() }, HdrPng::Hlg).unwrap();
        let cs = chunks(&std::fs::read(&path).unwrap());
        let get = |n: &[u8; 4]| cs.iter().find(|(c, _)| c == n).map(|(_, d)| d.clone());
        assert_eq!(get(b"cICP").unwrap(), [9, 18, 0, 1], "BT.2020, HLG, RGB, full range");
        for bad in [b"cLLI", b"sRGB", b"gAMA", b"iCCP", b"cHRM"] {
            assert!(get(bad).is_none(), "{} in an HLG PNG", String::from_utf8_lossy(bad));
        }
        let (info, s) = png_samples(&path);
        assert_eq!((info.width, info.height, info.color_type, info.bit_depth), (2, 1, png::ColorType::Rgb, png::BitDepth::Sixteen));
        assert_eq!(&s[..3], &[0, 0, 0]);
        for c in &s[3..] {
            let v = f32::from(*c) / 65535.0;
            assert!((v - 0.75).abs() < 0.005, "SDR white at {v}, BT.2408 says 0.75");
        }
    }

    /// The HLG file is the present pass's HLG signal for the viewer's display peak
    /// (`Look::peak`, here a 600-nit display): every pixel of the file (written row by row)
    /// equals `present::encode`'s HLG of the canvas at that peak within one 16-bit code,
    /// negatives and out-of-gamut values included; the same image for a 1000-nit display is
    /// a different file.
    #[test]
    fn hlg_png_is_the_present_signal() {
        use crate::present::{Encoding, Target, encode as present};
        let (w, h) = (37u32, 23u32);
        let px: Vec<f32> = (0..w * h * 4).map(|i| noise(i, 6000) - 0.2).collect();
        let look = Look { exposure: 0.5, gamma: 1.1, white: 300.0, peak: 600.0, ..Look::default() };
        let guard = tmp("hlg-present");
        let dir = guard.path().to_path_buf();
        let path = dir.join("h.png");
        write_hdr_png(&path, &raw(w, h, px.clone()), &look, HdrPng::Hlg).unwrap();
        let (_, samples) = png_samples(&path);
        let at_1000 = dir.join("h1000.png");
        write_hdr_png(&at_1000, &raw(w, h, px.clone()), &Look { peak: 1000.0, ..look.clone() }, HdrPng::Hlg).unwrap();
        assert_ne!(png_samples(&at_1000).1, samples, "the display peak must change the HLG signal");
        let target = Target { hdr: true, white: 300.0, peak: 600.0 };
        for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
            let d = transfer::display_pixel([p[0], p[1], p[2], p[3]], &transfer::Src::LINEAR, 0.5, 1.1, transfer::mode::RGBA, transfer::tonemap::NONE);
            let want = present([d[0], d[1], d[2]], Encoding::Hlg, target);
            for c in 0..3 {
                let got = f32::from(samples[i * 3 + c]) / 65535.0;
                assert!((got - want[c]).abs() <= 0.6 / 65535.0, "pixel {i} ch {c}: {got} vs {}", want[c]);
            }
        }
    }

    /// The EXR holds the display light exactly (half), out-of-gamut and negative values
    /// included, with explicit Rec.709 chromaticities and `whiteLuminance` = SDR white,
    /// and no alpha. Neutral look: light = the raw values; +1 EV doubles them.
    #[test]
    fn exr_holds_the_display_light() {
        use exr_core::api::ChannelData;
        let guard = tmp("exr");
        let dir = guard.path().to_path_buf();
        let (w, h) = (4u32, 1u32);
        // Values exactly representable in half, so f32 round trips land on them.
        let vals = [0.25f32, 1.0, 4.0, -0.5, 8.0, 0.5, 2.0, 0.0, 16.0, -2.0, 0.125, 1.0];
        let px: Vec<f32> = vals.chunks(3).flat_map(|c| [c[0], c[1], c[2], 0.3]).collect();
        for (ev, name) in [(0.0f32, "a.exr"), (1.0, "b.exr")] {
            let path = dir.join(name);
            let look = Look { exposure: ev, white: 203.0, ..Look::default() };
            write_exr(&path, &raw(w, h, px.clone()), &look).unwrap();
            let img = exr_core::api::read(&path).unwrap();
            assert!(img.channel("A").is_none(), "alpha is dropped");
            for (c, n) in ["R", "G", "B"].iter().enumerate() {
                let ChannelData::Half(d) = img.channel(n).unwrap() else { panic!("{n} is not half") };
                for p in 0..4 {
                    assert_eq!(d[p].to_f32(), vals[p * 3 + c] * 2f32.powf(ev), "{n} pixel {p} at {ev} EV");
                }
            }
            let wl = img.attributes().get("whiteLuminance").and_then(|a| a.as_any().downcast_ref::<exr_core::FloatAttribute>());
            assert_eq!(wl.map(|a| a.value), Some(203.0));
            let ch = img.attributes().get("chromaticities").and_then(|a| a.as_any().downcast_ref::<exr_core::ChromaticitiesAttribute>());
            let c = ch.map(|a| a.value).expect("chromaticities");
            assert_eq!((c.red, c.green, c.blue, c.white), ([0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.3127, 0.3290]));
        }
    }

    /// A display-light EXR reopens as absolute light: its `whiteLuminance` makes it absolute CIE
    /// XYZ (`exr_import::TagOrigin::Luminance`), so the built-in decode holds the saved nits (in
    /// units of 100) and an HDR output shows them whatever its SDR white; an SDR output shows
    /// 100 nits at white. An isolated channel is data and is not scaled.
    #[test]
    fn absolute_light_round_trips_and_scales_only_colour() {
        let guard = tmp("absolute");
        let path = guard.path().join("a.exr");
        let px = vec![1.0f32, 0.5, 0.25, 1.0];
        write_exr(&path, &raw(1, 1, px.clone()), &Look { white: 300.0, ..Look::default() }).unwrap();
        let frame = DecodedFrame::load(&path).unwrap();
        assert!(matches!(frame.tag.origin, exr_import::TagOrigin::Luminance { .. }), "{:?}", frame.tag);
        let builtin = exr_import::builtin(&frame.tag).unwrap();
        assert!(builtin.absolute);
        let target = |hdr, white| crate::present::Target { hdr, white, peak: 1000.0 };
        let mut colour = crate::ocio::State::new(crate::ocio::Sel::default());
        let src = colour.frame(&frame, &target(true, 150.0)).unwrap();
        // 1.0 at 300 nits is 2.0 x a 150-nit SDR white.
        let lit = canvas_px(&px, &Look { src, ..Look::default() }).unwrap();
        let want = canvas_px(&[2.0, 1.0, 0.5, 1.0], &Look::default()).unwrap();
        for (a, b) in lit[0].iter().zip(want[0]) {
            assert!((a - b).abs() < 1e-4, "{lit:?} vs {want:?}");
        }
        assert_eq!(colour.frame(&frame, &target(false, 80.0)).unwrap().gain, 1.0, "SDR shows 100 nits at white");
        let red = |src| Look { src, mode: transfer::mode::R, ..Look::default() };
        assert_eq!(canvas_px(&px, &red(src)).unwrap(), canvas_px(&px, &red(transfer::Src::LINEAR)).unwrap(), "an isolated channel is data");
    }


    #[test]
    fn streamed_exr_zip_rows_keep_orientation_across_chunks_and_partial_tail() {
        use exr_core::api::ChannelData;
        let (w, h) = (3u32, 35u32);
        let mut pixels = Vec::new();
        for y in 0..h { for x in 0..w {
            pixels.extend([y as f32 / 8.0 + x as f32 / 32.0,
                1.0 + y as f32 / 16.0, -0.25 - x as f32 / 8.0, 0.3]);
        }}
        let guard = tmp("zip-row-orientation");
        let path = guard.path().join("rows.exr");
        write_exr(&path, &raw(w, h, pixels.clone()), &Look::default()).unwrap();
        let image = exr_core::api::read(&path).unwrap();
        for (channel, name) in ["R", "G", "B"].iter().enumerate() {
            let ChannelData::Half(samples) = image.channel(name).unwrap() else { panic!("expected half") };
            assert_eq!(samples.len(), (w * h) as usize);
            for y in 0..h { for x in 0..w {
                let i = (y * w + x) as usize;
                let wanted = imath_rs::Half::new(pixels[i * 4 + channel]).bits();
                assert_eq!(samples[i].bits(), wanted, "{name} ({x},{y}), including ZIP boundaries 15/16/31/32");
            }}
        }
    }

    /// The EXR light, re-encoded, is the canvas the SDR path would show: the two formats
    /// are one chain seen at two stages. Ragged multi-strip image again.
    #[test]
    fn exr_light_reencodes_to_the_canvas() {
        use exr_core::api::ChannelData;
        let (w, h) = (311u32, 241u32);
        let px: Vec<f32> = (0..w * h * 4).map(|i| noise(i, 3000)).collect();
        let look = Look { exposure: -0.5, gamma: 1.3, tonemap: transfer::tonemap::REINHARD, ..Look::default() };
        let guard = tmp("exr2");
        let path = guard.path().join("l.exr");
        write_exr(&path, &raw(w, h, px.clone()), &look).unwrap();
        let img = exr_core::api::read(&path).unwrap();
        let chan = |n: &str| match img.channel(n).unwrap() {
            ChannelData::Half(d) => d.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            _ => panic!("half"),
        };
        let (r, g, b) = (chan("R"), chan("G"), chan("B"));
        for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
            let d = transfer::display_pixel([p[0], p[1], p[2], p[3]], &transfer::Src::LINEAR, -0.5, 1.3, transfer::mode::RGBA, transfer::tonemap::REINHARD);
            for (got, want) in [(r[i], d[0]), (g[i], d[1]), (b[i], d[2])] {
                // Half keeps 11 bits: the light is within 2^-10 relative, the encoded
                // value (below 1) moves less than that.
                assert!((transfer::oetf(got) - want).abs() <= 2e-3, "pixel {i}: {} vs {want}", transfer::oetf(got));
            }
        }
    }

    /// With OCIO the light is the decoded OCIO canvas (ACES 2.0 SDR: 18 % grey shows at
    /// 0.349 encoded, the same 89/255 as the SDR test).
    #[test]
    fn light_follows_ocio() {
        let o = crate::ocio::Ocio::load("ocio://default").unwrap();
        let t = Arc::new(o.transform(&o.resolve(&crate::ocio::Sel::default(), &o.scene(), false).unwrap(), None, false).unwrap());
        let look = Look { ocio: Some(t), ..Look::default() };
        let l = light_px(&[0.18, 0.18, 0.18, 1.0], &look)[0];
        for c in l {
            assert!((transfer::oetf(c) - 0.349).abs() < 2e-3, "{c}");
        }
    }

    /// The SDR encode over a ragged multi-strip image is the per-pixel display transform
    /// (the strips change nothing).
    #[test]
    fn sdr_encode_is_strip_independent() {
        let n = STRIP + STRIP / 2 + 7;
        let px: Vec<f32> = (0..(n * 4) as u32).map(|i| noise(i, 2500)).collect();
        let got = encode(&px, &Look::default()).unwrap();
        assert_eq!(got.len(), n * 3);
        for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
            let d = transfer::display_pixel([p[0], p[1], p[2], p[3]], &transfer::Src::LINEAR, 0.0, 1.0, transfer::mode::RGBA, transfer::tonemap::NONE);
            assert_eq!(got[i * 3..i * 3 + 3], [d[0], d[1], d[2]].map(transfer::code8), "pixel {i}");
        }
    }


    #[cfg(feature = "ofx")]
    fn processed_fixture(budget: &crate::tiles::region::RegionBudget) -> crate::consumers::AcceptedPixels {
        use crate::consumers::{AcceptedCoverage, AcceptedPixels};
        use crate::tiles::grid::{Grid, PixRect, TileId};
        use crate::tiles::processed::{ChargedPixelStorage, ProcessedLineage, ProcessedWork, output_frame};
        use crate::tiles::view::{Pick, SourceDesc, TileKey};
        use ofx_runtime::ofx_host::{OfxRectI, PixelStorage};
        let grid = Grid::on_lattice(3, 35, (0, -35)).unwrap();
        let bounds = OfxRectI { x1: 0, y1: 0, x2: 3, y2: 35 };
        let window = imath_rs::Box2i::from_min_max(imath_rs::V2i::new(10, -3), imath_rs::V2i::new(12, 31));
        let lineage = Arc::new(ProcessedLineage { primary_index: -7, frame_id: 42,
            data_window: window, display_window: window, pixel_aspect: 1.5,
            selection: crate::image::Selection::default(), time: -6.75 });
        let attempt = budget.begin_attempt().unwrap();
        let mut storage = ChargedPixelStorage::allocate(bounds, &attempt).unwrap();
        let pixels: &mut [f32] = bytemuck::cast_slice_mut(storage.bytes_mut().unwrap());
        for (i, pixel) in pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            pixel.copy_from_slice(&[(i / 3) as f32 / 8.0, (i % 3) as f32 / 4.0, -0.25, 1.0]);
        }
        let frame = output_frame(storage, bounds, bounds, 1.5).unwrap();
        let work = ProcessedWork::from_rendered(frame, 0, lineage.time, attempt.clone(), lineage.clone()).unwrap();
        drop(attempt);
        let id = TileId { level: 0, tx: 0, ty: 0 };
        let rect = PixRect { x: 0, y: 0, w: 3, h: 35 };
        let pick = Pick { id, from: id, texels: grid.texture_rect(id, rect).unwrap() };
        let entry = AcceptedCoverage { pick, key: TileKey { source: 7, run: work.run, id },
            view: work.view(rect).unwrap() };
        drop(work);
        AcceptedPixels::new(19, SourceDesc { frame: 42, view: lineage.selection.clone(), pixels_gen: 3, stage: 8 },
            grid, lineage, vec![], entry).unwrap()
    }

    #[cfg(feature = "ofx")]
    #[test]
    fn processed_shot_pending_consumes_pins_and_success_keeps_rgb_charge() {
        use crate::tiles::region::{RegionBudget, RegionError};
        let budget = RegionBudget::new(8192);
        let snapshot = processed_fixture(&budget);
        let output_charge = budget.used();
        let make = || {
            let rows = crate::consumers::PixelRows::for_export(snapshot.clone(), 0).unwrap();
            Raw { w: 3, h: 35, level: 0, source: RawSource::Processed(Box::new(std::sync::Mutex::new(rows))) }
        };
        let raw = make();
        assert_eq!(snapshot.retirement_credit(), 0);
        let pressure_bytes = 8192 - budget.used();
        let pressure = budget.reserve(pressure_bytes).unwrap();
        let error = match raw.sdr(&Look::default()) { Ok(_) => panic!("expected physical pressure"), Err(error) => error };
        assert_eq!(error.downcast_ref::<RegionError>(), Some(&RegionError::Pending));
        assert_eq!(snapshot.retirement_credit(), output_charge, "consuming Pending releases delayed Raw pins");
        assert_eq!(budget.used(), output_charge + pressure_bytes, "row scratch was physically released");
        drop(pressure);
        let shot = make().sdr(&Look::default()).unwrap();
        assert_eq!(shot.rgb().len(), 3 * 35 * 3);
        assert_eq!(budget.used(), output_charge + shot.rgb().len(), "Shot retains only its actual RGB owner");
        drop(shot);
        assert_eq!(budget.used(), output_charge);
        drop(snapshot);
        assert_eq!(budget.used(), 0);
    }

    #[cfg(feature = "ofx")]
    #[test]
    fn processed_streamed_exr_freezes_primary_provenance_and_top_first_pixels() {
        let budget = crate::tiles::region::RegionBudget::new(8192);
        let snapshot = processed_fixture(&budget);
        let rows = crate::consumers::PixelRows::for_export(snapshot, 0).unwrap();
        let raw = Raw { w: 3, h: 35, level: 0, source: RawSource::Processed(Box::new(std::sync::Mutex::new(rows))) };
        let guard = tmp("processed-provenance");
        let path = guard.path().join("processed.exr");
        write_exr(&path, &raw, &Look::default()).unwrap();
        let image = exr_core::api::read(&path).unwrap();
        let exr_core::api::ChannelData::Half(red) = image.channel("R").unwrap() else { panic!("half") };
        for y in 0..35 { for x in 0..3 { assert_eq!(red[y * 3 + x].to_f32(), y as f32 / 8.0); }}
        let attrs = image.attributes();
        assert_eq!(attrs.get("exrViewPrimaryIndex").unwrap().as_any()
            .downcast_ref::<exr_core::IntAttribute>().unwrap().value, -7);
        assert_eq!(attrs.get("exrViewTime").unwrap().as_any()
            .downcast_ref::<exr_core::DoubleAttribute>().unwrap().value.to_bits(), (-6.75f64).to_bits());
        drop(raw);
        assert_eq!(budget.used(), 0);
    }


    #[test]
    fn mutated_original_dimensions_fail_before_creating_output() {
        let mut source = raw(1, 1, vec![0.0, 0.0, 0.0, 1.0]);
        source.w = 2;
        let guard = tmp("corrupt-original-metadata");
        let path = guard.path().join("never.png");
        let error = write_sdr_png(&path, &source, &Look::default()).unwrap_err();
        assert!(matches!(error.downcast_ref::<crate::tiles::region::RegionError>(),
            Some(crate::tiles::region::RegionError::Failed(_))));
        assert!(!path.exists());
        assert!(source.sdr(&Look::default()).is_err());
    }

    #[cfg(feature = "ofx")]
    #[test]
    fn mutated_processed_metadata_fails_before_files_or_new_pixel_charges() {
        use crate::tiles::region::{RegionBudget, RegionError};
        let budget = RegionBudget::new(8192);
        let snapshot = processed_fixture(&budget);
        let output_charge = budget.used();
        for change_level in [false, true] {
            let rows = crate::consumers::PixelRows::for_export(snapshot.clone(), 0).unwrap();
            let mut source = Raw { w: 3, h: 35, level: 0,
                source: RawSource::Processed(Box::new(std::sync::Mutex::new(rows))) };
            if change_level { source.level = 1; } else { source.w = 4; }
            let before = budget.used();
            let guard = tmp("corrupt-processed-metadata");
            let path = guard.path().join("never.exr");
            let error = write_exr(&path, &source, &Look::default()).unwrap_err();
            assert!(matches!(error.downcast_ref::<RegionError>(), Some(RegionError::Failed(_))));
            assert!(!path.exists());
            assert_eq!(budget.used(), before);
            let error = match source.sdr(&Look::default()) { Ok(_) => panic!("expected corrupted geometry"), Err(error) => error };
            assert!(matches!(error.downcast_ref::<RegionError>(), Some(RegionError::Failed(_))));
            assert_eq!(budget.used(), output_charge);
            assert_eq!(snapshot.retirement_credit(), output_charge);
        }
        drop(snapshot);
        assert_eq!(budget.used(), 0);
    }

    /// An extension that names no HDR format is an error.
    #[test]
    fn hdr_save_refuses_other_extensions() {
        let r = raw(1, 1, vec![0.0, 0.0, 0.0, 1.0]);
        let guard = tmp("bad");
        let dir = guard.path().to_path_buf();
        for name in ["x.jpg", "x", "x.tif"] {
            let e = save_hdr(&dir.join(name), &r, &Look::default(), HdrPng::Pq).unwrap_err();
            assert!(e.to_string().contains(".exr"), "{e}");
        }
    }

    /// A PNG write that fails half way leaves the old file intact and no temporary
    /// behind (the all-or-nothing `write_file`); a good one replaces it.
    #[test]
    fn png_writes_are_all_or_nothing() {
        let guard = tmp("atomic");
        let dir = guard.path().to_path_buf();
        let path = dir.join("keep.png");
        std::fs::write(&path, b"old").unwrap();
        // Too few bytes for 2x2 RGB: the encoder fails after the header.
        let bad = Shot { w: 2, h: 2, rgb: vec![0; 5], #[cfg(feature = "ofx")] _permit: None };
        assert!(write_png(&path, &bad).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names, [std::ffi::OsString::from("keep.png")], "no temporary left");
        write_png(&path, &Shot { w: 1, h: 1, rgb: vec![1, 2, 3], #[cfg(feature = "ofx")] _permit: None }).unwrap();
        assert_eq!(&std::fs::read(&path).unwrap()[..8], b"\x89PNG\r\n\x1a\n");
    }
}
