//! `dispatch.rs` — codec routing / orchestration over the `exv-gpu-kernels` decoders.
//!
//! Holds the high-level entry points: [`decode_still_gpu`] (route a whole 1-frame `.exv` still to the right
//! GPU/CPU path), [`GpuImage`]/[`GpuFrame`] (a decoded still resident in VRAM + display conversion),
//! [`GpuDecoder`], [`RgbaFormat`], [`GpuDecodeError`], [`decode_wave_batched`], [`read_back_image`],
//! and the CPU export gather (per-sample decode = `exv_codec::image::sample_f32`, the ONE copy).

use crate::prelude::*;
// Pure decode kernels + GpuContext + the low-level plumbing (`storage_u32`, `e`, `map_read_blocking`,
// `Pipelines`, `decode_exvrt_into`) now come from the extracted `exv-gpu-kernels` crate.
use bufpool::Cancel;
use exv_codec::GpuKind;
use exv_codec::image::sample_f32;
use exv_format::half::f32_to_half;
use exv_gpu_kernels::*;

/// A whole decoded still resident in VRAM (GPU-first, NO readback): one composite buffer per channel,
/// each in its [`GpuTile`] word representation (`enc`). Produced by [`decode_still_gpu`]; the
/// viewer/processing consumes the buffers on the GPU. [`read_back_image`] packs them to native plane
/// bytes (test/export only) — bit-for-bit == `exv_codec::read_still`.
pub struct GpuImage {
    pub width: u32,
    pub height: u32,
    /// One composite plane per channel, in header channel order.
    pub planes: Vec<GpuTile>,
}

/// **Decode a 1-frame `.exv` still on the GPU into VRAM — NO readback on the hot path.**
///
/// Mirrors the container geometry of `exv_video::read_still`, but routes every `(tile, channel)` blob
/// by its tag ([`classify_blob`]): a rANS tile decodes on the GPU in a whole-frame batch (LosslessWave /
/// LosslessFast, see `decode_rans_batched`; EXV-RT blocks via [`RtBatch`]); a raw/deflate tile — or a rANS tile the GPU cannot take (e.g.
/// wavelet `esz>2`) — falls back to CPU `exv_codec::decode_channel` and is uploaded into the SAME VRAM
/// composite (intra-tile mixed residency). GPU-decoded tiles never leave VRAM.
///
/// Whole-image path: each per-channel composite must fit VRAM (`max_buffer_size`); huge images use
/// tiled streaming (`exv-tile`, future). Lossless profiles only (LOSSLESS_FAST / WAVELET53); a lossy
/// file returns a clean `Err` (use CPU `read_still`). Composite is in word rep (4× the packed plane);
/// packing to bytes happens once, on CPU, in [`read_back_image`] (avoids GPU sub-word byte packing).
///
/// `limits` = the caller's [`DecodeLimits`]: its [`exv_format::Verify`] policy gates the per-blob CRC check and its
/// budget bounds the host planes of CPU-fallback tiles ([`GpuDecodeError::Budget`] over it).
pub fn decode_still_gpu(ctx: &GpuContext, data: &[u8], limits: &DecodeLimits) -> Result<GpuImage, GpuDecodeError> {
    // `data` is a 1-frame `.exv` CONTAINER, not a bare header — pull the still's header via the unified
    // `still_frame_body` seam (the same one the tile decoders below use). The downstream decoders
    // re-extract the frame body from `data` themselves, so only the header is needed here.
    let (header, _body) = exv_video::still_frame_body(data)
        .ok_or_else(|| GpuDecodeError::Corrupt("bad exv container".to_string()))?;
    let codec = CodecId(header.codec_id);
    // Typed open-time codec validation: unknown / vendor / not built / bad params => `GpuDecodeError::Codec`.
    let profile = exv_codec::registry::open(codec, &header.codec_params)?;
    if header.width == 0 || header.height == 0 {
        return Err(GpuDecodeError::Corrupt("empty image".to_string()));
    }
    // Single routing decision (transform → GPU path | NoGpuDecoder). The service NEVER silently
    // CPU-mimics an unsupported codec: it returns `NoGpuDecoder` and the CALLER decides (viewer → CPU
    // read_still; gate/bench → hard error). GPU-path decoders' internal errors wrap as `Corrupt`.
    match profile.gpu_kind() {
        // LosslessFast: whole-image batched GPU decode (rANS -> reorder+delta undelta, one thread per tile-plane)
        // into per-channel `BytePerWord` VRAM planes; raw/deflate tiles composite from the CPU. (The serial per-plane
        // prefix sum is slow for ONE tile, but across a frame's thousands of tiles it fills the GPU.)
        Some(GpuKind::RansFast) => Ok(decode_rans_batched(ctx, data, &header, &profile, Stage::Fast, limits, &Cancel::new())?),
        // EXV-RT (B-RT.3): whole-image batched GPU decode. Per-channel VRAM composite; F16 block tiles
        // decode on the GPU (bit-exact vs CPU at q=0), non-F16/raw tiles composite from the CPU.
        Some(GpuKind::ExvRt) => Ok(decode_blockrt_batched(ctx, data, &header, &profile, limits, &Cancel::new())?),
        // BC6H (hardware BC6H_SF16 Mode 11 + optional lossless rANS residual): whole-image GPU decode.
        Some(GpuKind::Bc6h) => Ok(decode_bc6h_batched(ctx, data, &header, &profile, limits, &Cancel::new())?),
        // LosslessWave: whole-image batched GPU decode (rANS -> repack -> inverse 5/3). The old "bug #16"
        // (wrong pixels for the default 128-lane encoder) was the batched rANS kernel decoding only lanes
        // 0..63; it now strides lanes over its 64 threads and is parity-tested for lanes 1..255.
        Some(GpuKind::RansWave) => Ok(decode_wave_batched(ctx, data, &header, &profile, limits)?),
        // HTJ2K tiles are decoded by jph-rs (CPU) / jph-wgpu (GPU, the resident atlas), not this whole-still
        // rANS path.
        Some(GpuKind::Htj2k) => Err(GpuDecodeError::NoGpuDecoder {
            codec,
            why: "htj2k decodes via jph-rs (CPU) / jph-wgpu (GPU atlas), not the exv rANS GPU path",
        }),
        // Lossy wavelet / lossy HTJ2K: the registry reports no GPU route.
        None => Err(GpuDecodeError::NoGpuDecoder { codec, why: "lossy codec is not GPU-decodable; use CPU read_still" }),
    }
}

/// Typed decode-service error (replaces the ad-hoc `String` of the free decoders).
///
/// The GPU decode service NEVER silently falls back to a CPU mimic of an unsupported codec: it returns
/// [`NoGpuDecoder`](Self::NoGpuDecoder) and the CALLER decides (a viewer → CPU `read_still`; a
/// gate/bench → hard error). The other variants mirror [`GpuError`] 1:1 (`From<GpuError>`), so the caller
/// can tell a bad file (`Corrupt`) from a device that cannot do it (`DeviceLimit`/`Device` → CPU fallback)
/// from a routing outcome (`Unsupported`).
#[derive(Debug)]
pub enum GpuDecodeError {
    /// This codec has no GPU decoder here — not an error in the stream, a routing outcome. `why` is a
    /// static human reason; `codec` is the on-disk [`CodecId`]. The caller falls back to CPU.
    NoGpuDecoder { codec: CodecId, why: &'static str },
    /// The header's codec cannot be opened at all (unknown / vendor / not built / bad params): the CPU cannot
    /// decode it either, so this is NOT a CPU-fallback signal.
    Codec(exv_codec::OpenError),
    /// Malformed/short stream, bad tag, or inconsistent geometry (message preserved).
    Corrupt(String),
    /// A device limit blocks the operation (e.g. image exceeds `max_texture_dimension_2d`, a buffer
    /// exceeds the binding size, a dispatch exceeds the per-dimension cap). The CPU decoder can serve it.
    DeviceLimit(String),
    /// A GPU path exists but does not take this input (dtype/tag/quant it cannot decode) — use the CPU.
    Unsupported(String),
    /// wgpu/driver failure (device lost, validation error, readback failure).
    Device(String),
    /// The caller's [`DecodeLimits`] budget refuses the host bytes this decode would hold (NOT a CPU-fallback signal:
    /// the CPU decoder obeys the same budget).
    Budget(String),
}

impl From<exv_codec::OpenError> for GpuDecodeError {
    fn from(e: exv_codec::OpenError) -> Self {
        GpuDecodeError::Codec(e)
    }
}

impl From<GpuError> for GpuDecodeError {
    fn from(e: GpuError) -> Self {
        match e {
            GpuError::Limit(m) => GpuDecodeError::DeviceLimit(m),
            GpuError::Corrupt(m) => GpuDecodeError::Corrupt(m),
            GpuError::Unsupported(m) => GpuDecodeError::Unsupported(m),
            GpuError::Device(m) => GpuDecodeError::Device(m),
            GpuError::Budget(m) => GpuDecodeError::Budget(m),
        }
    }
}

impl std::fmt::Display for GpuDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuDecodeError::NoGpuDecoder { codec, why } => {
                write!(f, "no GPU decoder for codec {}: {why}", codec.0)
            }
            GpuDecodeError::Codec(e) => write!(f, "cannot open codec: {e}"),
            GpuDecodeError::Corrupt(m) => write!(f, "corrupt/unsupported .exv: {m}"),
            GpuDecodeError::DeviceLimit(m) => write!(f, "device limit: {m}"),
            GpuDecodeError::Unsupported(m) => write!(f, "unsupported on GPU: {m}"),
            GpuDecodeError::Device(m) => write!(f, "GPU device: {m}"),
            GpuDecodeError::Budget(m) => write!(f, "decode budget: {m}"),
        }
    }
}

impl std::error::Error for GpuDecodeError {}

/// Interleaved-RGBA CPU export pixel format for [`GpuFrame::read_back_rgba`] (matches the capi's
/// `exv_session_decode` output formats).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RgbaFormat {
    /// 4×4-byte little-endian f32 per pixel.
    F32,
    /// 4×2-byte little-endian f16 per pixel.
    F16,
}

/// A decoded still frame + the shared decode context — the first-class, device-unified output of
/// [`GpuDecoder::decode_still`].
///
/// # Why
/// Wraps the VRAM [`GpuImage`] together with the `Arc<GpuContext>` that owns it, so the display
/// conversion ([`to_rgba_texture`](Self::to_rgba_texture)) and readback paths need no extra `ctx`
/// argument and cannot outlive the device. Owns the `GpuImage` (planes are not `Clone`); the context is
/// shared (`Arc`).
pub struct GpuFrame {
    pub width: u32,
    pub height: u32,
    inner: GpuImage,
    ctx: Arc<GpuContext>,
}

impl GpuFrame {
    /// Wrap a decoded [`GpuImage`] with its owning context (used by [`GpuDecoder::decode_still`]).
    fn new(inner: GpuImage, ctx: Arc<GpuContext>) -> Self {
        Self { width: inner.width, height: inner.height, inner, ctx }
    }

    /// The raw per-channel VRAM plane `i` (header channel order), or `None` if out of range.
    pub fn channel_buffer(&self, i: usize) -> Option<&GpuTile> {
        self.inner.planes.get(i)
    }

    /// Number of decoded channel planes.
    pub fn channel_count(&self) -> usize {
        self.inner.planes.len()
    }

    /// **Gather up to four decoded channel planes into a fresh `Rgba32Float` STORAGE texture on the
    /// GPU — zero CPU readback.**
    ///
    /// `rgb`/`a` are indices into the channel planes (header order); a `None` rgb slot writes 0.0, a
    /// `None` alpha writes 1.0. Each plane is word-unpacked per its [`GpuTile`] `enc`/`esz` and
    /// converted dtype→f32 **byte-identically to the CPU viewer/plugin** ([`RGBA_GATHER_SHADER`]). The
    /// texture is created `STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC`, so the caller binds it
    /// straight into a render pass. This is the load-bearing display path that replaces the CPU
    /// gather+convert the viewer/capi do today.
    pub fn to_rgba_texture(
        &self,
        rgb: [Option<usize>; 3],
        a: Option<usize>,
    ) -> Result<wgpu::Texture, GpuDecodeError> {
        let ctx = &self.ctx;
        let (w, h) = (self.width, self.height);
        let lim = ctx.device.limits();
        if w > lim.max_texture_dimension_2d || h > lim.max_texture_dimension_2d {
            return Err(GpuDecodeError::DeviceLimit(format!(
                "rgba texture {w}x{h} exceeds max_texture_dimension_2d {}",
                lim.max_texture_dimension_2d
            )));
        }
        let planes = &self.inner.planes;
        // A bound buffer is required for all four slots even when a channel is absent; reuse plane 0 as
        // the harmless dummy (its words are never read — `present==0` short-circuits in the shader).
        let dummy = planes
            .first()
            .ok_or_else(|| GpuDecodeError::Corrupt("frame has no planes".to_string()))?;

        // params: [width, height] then 5 u32 per slot (present, off_words, enc_kind, esz, dtype_code).
        let mut params: Vec<u32> = vec![w, h];
        let slots = [rgb[0], rgb[1], rgb[2], a];
        let mut bufs: [&wgpu::Buffer; 4] = [&dummy.buf, &dummy.buf, &dummy.buf, &dummy.buf];
        for (s, sel) in slots.iter().enumerate() {
            match sel {
                Some(ci) => {
                    let t = planes.get(*ci).ok_or_else(|| {
                        GpuDecodeError::Corrupt(format!(
                            "channel index {ci} out of range ({} planes)",
                            planes.len()
                        ))
                    })?;
                    let (enc_kind, esz) = match t.enc {
                        TileEnc::BytePerWord => (0u32, t.dtype.size() as u32),
                        TileEnc::ElemPerWord(esz) => (1u32, esz as u32),
                    };
                    params.extend_from_slice(&[1, t.off_u32()?, enc_kind, esz, t.dtype.code() as u32]);
                    bufs[s] = &t.buf;
                }
                None => params.extend_from_slice(&[0, 0, 0, 0, 0]),
            }
        }

        Ok(ctx.scoped(|| {
        let tex = tex2d(
            &ctx.device,
            "exv.rgba32f",
            w,
            h,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let pbuf =
            storage_u32(&ctx.device, &ctx.queue, "exv.rgba.params", &params, wgpu::BufferUsages::empty());

        let pipe = &ctx.pipelines.rgba_gather;
        let layout = pipe.get_bind_group_layout(0);
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exv.rgba.bg"),
            layout: &layout,
            entries: &[
                e(0, bufs[0]),
                e(1, bufs[1]),
                e(2, bufs[2]),
                e(3, bufs[3]),
                e(4, &pbuf),
                et(5, &view),
            ],
        });
        let mut enc = encoder(&ctx.device, "exv.rgba.enc");
        pass(ctx, &mut enc, "exv.rgba.pass", pipe, &bg, [w.div_ceil(WG2), h.div_ceil(WG2), 1])?;
        ctx.queue.submit([enc.finish()]);
        Ok(tex)
        })?)
    }

    /// Copy every channel plane out of VRAM to native-endian packed bytes (test/export only). Reuses
    /// [`read_back_image`]; the result is bit-for-bit == `exv_codec::read_still`'s planes.
    pub fn read_back(&self) -> Result<Vec<Vec<u8>>, GpuDecodeError> {
        Ok(read_back_image(&self.ctx, &self.inner)?)
    }

    /// **CPU** interleaved-RGBA gather (test/export & the capi path): the same channel-pick, dtype→f32
    /// conversion and interleave the plugin's `exv_session_decode` does, matching its byte layout. `rgb`/`a` index
    /// the channel planes (`None` rgb → 0.0, `None` alpha → 1.0); `stride` is the destination row
    /// stride in bytes (0 = tightly packed `width * bytes_per_pixel`). Does a full readback then gathers
    /// on the CPU — the GPU-resident equivalent is [`to_rgba_texture`](Self::to_rgba_texture).
    pub fn read_back_rgba(
        &self,
        rgb: [Option<usize>; 3],
        a: Option<usize>,
        fmt: RgbaFormat,
        stride: usize,
    ) -> Result<Vec<u8>, GpuDecodeError> {
        let planes = self.read_back()?; // native per-channel bytes
        let (w, h) = (self.width as usize, self.height as usize);
        let bpp = match fmt {
            RgbaFormat::F32 => 16,
            RgbaFormat::F16 => 8,
        };
        let row_bytes = w * bpp;
        let row_stride = if stride == 0 { row_bytes } else { stride };
        if row_stride < row_bytes {
            return Err(GpuDecodeError::Corrupt(format!("stride {row_stride} < row {row_bytes}")));
        }
        // Resolve each slot to (plane bytes, dtype). An out-of-range index is a hard error.
        let resolve = |ci: Option<usize>| -> Result<Option<(&[u8], Dtype)>, GpuDecodeError> {
            match ci {
                None => Ok(None),
                Some(i) => {
                    let p = planes.get(i).ok_or_else(|| {
                        GpuDecodeError::Corrupt(format!("channel index {i} out of range"))
                    })?;
                    let d = self.inner.planes.get(i).map(|t| t.dtype).ok_or_else(|| {
                        GpuDecodeError::Corrupt(format!("channel index {i} out of range"))
                    })?;
                    Ok(Some((p.as_slice(), d)))
                }
            }
        };
        let rp = resolve(rgb[0])?;
        let gp = resolve(rgb[1])?;
        let bp = resolve(rgb[2])?;
        let ap = resolve(a)?;

        let mut out = vec![0u8; row_stride * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let r = rp.map(|(p, d)| sample_f32(p, d, i)).unwrap_or(0.0);
                let g = gp.map(|(p, d)| sample_f32(p, d, i)).unwrap_or(0.0);
                let b = bp.map(|(p, d)| sample_f32(p, d, i)).unwrap_or(0.0);
                let al = ap.map(|(p, d)| sample_f32(p, d, i)).unwrap_or(1.0);
                let o = y * row_stride + x * bpp;
                match fmt {
                    RgbaFormat::F32 => {
                        for (k, v) in [r, g, b, al].iter().enumerate() {
                            out[o + k * 4..o + k * 4 + 4].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                    RgbaFormat::F16 => {
                        for (k, v) in [r, g, b, al].iter().enumerate() {
                            out[o + k * 2..o + k * 2 + 2]
                                .copy_from_slice(&f32_to_half(*v).to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// The first-class GPU decode facade: one handle, `Arc<GpuContext>` inside, that a consumer holds and
/// calls [`decode_still`](Self::decode_still) on. Built either standalone ([`new`](Self::new)) or on an
/// app's existing device ([`from_wgpu`](Self::from_wgpu), the device-unified path).
pub struct GpuDecoder {
    ctx: Arc<GpuContext>,
}

impl GpuDecoder {
    /// Standalone decoder on its OWN wgpu device (CLI/tests/benches).
    pub fn new() -> GpuResult<Self> {
        Ok(Self { ctx: Arc::new(GpuContext::new()?) })
    }

    /// Decoder that SHARES the caller's wgpu device/queue/adapter (device unification) — decoded VRAM
    /// and [`GpuFrame::to_rgba_texture`] outputs are directly usable by the caller's render pass.
    pub fn from_wgpu(device: wgpu::Device, queue: wgpu::Queue, adapter: wgpu::Adapter) -> GpuResult<Self> {
        Ok(Self { ctx: Arc::new(GpuContext::from_wgpu(device, queue, adapter)?) })
    }

    /// The shared decode context (for `vram_budget`/`drain`/etc.).
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Decode a whole single-part 1-frame `.exv` still into a VRAM [`GpuFrame`] under the caller's `limits` (as
    /// [`decode_still_gpu`]). Returns [`GpuDecodeError::NoGpuDecoder`] for a codec with no GPU path (the caller falls
    /// back to CPU).
    pub fn decode_still(&self, bytes: &[u8], limits: &DecodeLimits) -> Result<GpuFrame, GpuDecodeError> {
        let img = decode_still_gpu(&self.ctx, bytes, limits)?;
        Ok(GpuFrame::new(img, Arc::clone(&self.ctx)))
    }
}

/// One `(tile, channel)` work item handed out by [`items`]: the still-encoded blob plus its geometry.
#[derive(Clone, Copy)]
pub(crate) struct Item<'a> {
    t: u32,
    ci: usize,
    dtype: Dtype,
    esz: usize,
    blob: &'a [u8],
    /// Tile origin and size in the image.
    x: u32,
    y: u32,
    tw: u32,
    th: u32,
}

/// Every `(tile, channel)` blob of a 1-frame still in container order — the ONE tile loop shared by the batched
/// wave, EXV-RT and fast decoders (they used to carry three copies, which is how a bug fixed in one stayed in the
/// others). The canonical `FrameReader` parses the on-disk framing and hands out each still-encoded sub-blob (no
/// decode); callers only classify/route. The items borrow `data` (not the local reader), so the batched hosts
/// keep the blobs for their parallel parse. Degenerate geometry (zero dims, a tile rect outside the image) is
/// `Corrupt` here, once, so the callers' plane indexing cannot go out of bounds. The CRC checks follow `limits.verify`.
fn items<'d>(data: &'d [u8], h: &exv_format::Header, limits: &DecodeLimits) -> GpuResult<Vec<Item<'d>>> {
    if h.width == 0 || h.height == 0 || h.tiles_x() == 0 || h.tile_h == 0 {
        return Err(corrupt("empty image / zero tile grid"));
    }
    let (header, body) = exv_video::still_frame_body(data).ok_or_else(|| corrupt("bad exv container"))?;
    let reader = exv_codec::frame::FrameReader::open(body, header).map_err(|e| corrupt(&e.to_string()))?.with_limits(*limits);
    if reader.tile_count() as u64 != h.tile_count() {
        return Err(corrupt("tile count mismatch"));
    }
    use rayon::prelude::*;
    let tx_n = h.tiles_x();
    // Tiles in parallel: each one's CRC check (the CPU-side integrity gate before any upload, once per tile, not per
    // channel) is the only real work here; the indexed collect keeps tile order.
    let per_tile = (0..reader.tile_count())
        .into_par_iter()
        .map(|t| {
            let (x, y, tw, th) = h.tile_rect(t % tx_n, t / tx_n);
            if x as u64 + tw as u64 > h.width as u64 || y as u64 + th as u64 > h.height as u64 {
                return Err(corrupt("tile rect outside the image"));
            }
            let subs = reader.tile_chans(t).map_err(|e| GpuError::from_decode(e, &format!("tile {t}")))?;
            Ok(h.channels
                .iter()
                .enumerate()
                .zip(subs)
                .map(|((ci, ch), blob)| Item { t, ci, dtype: ch.dtype, esz: ch.dtype.size(), blob, x, y, tw, th })
                .collect::<Vec<_>>())
        })
        .collect::<GpuResult<Vec<Vec<Item<'d>>>>>()?;
    Ok(per_tile.into_iter().flatten().collect())
}

/// CPU-decode one tile (`exv_codec::decode_channel`, the reference) and check its packed length.
fn cpu_tile(it: &Item, profile: &exv_codec::Profile) -> GpuResult<Vec<u8>> {
    let packed = exv_codec::decode_channel(it.blob, it.dtype, it.tw, it.th, profile)
        .map_err(|e| GpuError::from_decode(e, &format!("CPU decode (tile {}, ch {})", it.t, it.ci)))?;
    if packed.len() != it.tw as usize * it.th as usize * it.esz {
        return Err(GpuError::Corrupt(format!("CPU tile size mismatch (tile {}, ch {})", it.t, it.ci)));
    }
    Ok(packed)
}

/// Scatter a CPU-decoded packed tile into a plane BAND (the rows from `band_y0` on) in word representation `rep`
/// (`ElemPerWord`: one element per word; `BytePerWord`: one byte per word). `w` = the plane width in pixels;
/// indices are in range because [`items`] checked the tile rect and the callers size the band.
fn scatter(band: &mut [u32], band_y0: u32, packed: &[u8], rep: TileEnc, w: u32, it: &Item) {
    let (tw, esz) = (it.tw as usize, it.esz);
    for r in 0..it.th as usize {
        for col in 0..tw {
            let px = ((it.y - band_y0) as usize + r) * w as usize + it.x as usize + col;
            let src = (r * tw + col) * esz;
            match rep {
                TileEnc::ElemPerWord(_) => {
                    band[px] = packed[src..src + esz].iter().enumerate().fold(0u32, |v, (i, &b)| v | (b as u32) << (8 * i));
                }
                TileEnc::BytePerWord => {
                    for (i, &b) in packed[src..src + esz].iter().enumerate() {
                        band[px * esz + i] = b as u32;
                    }
                }
            }
        }
    }
}

/// A VRAM composite buffer of `words` u32 (limit-checked): `STORAGE` = kernel target, `COPY_SRC` =
/// readback, `COPY_DST` = the CPU-tile upload.
fn composite(ctx: &GpuContext, label: &str, words: usize) -> GpuResult<wgpu::Buffer> {
    ctx.check_buf(label, words as u64 * 4)?;
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    Ok(new_buf(&ctx.device, label, (words.max(1) * 4) as u64, usage))
}

/// `usize` → `u32` for a descriptor field; beyond 2^32 the shaders cannot address it (`Limit`, not a wrap).
fn u(v: usize) -> GpuResult<u32> {
    u32::try_from(v).map_err(|_| GpuError::Limit(format!("{v} exceeds the u32 shader index range")))
}

/// Second stage of a rANS-family whole-image decode, after the shared batched rANS pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// LosslessWave: repack -> inverse 5/3 columns -> rows (`ElemPerWord` planes in ONE combined composite).
    Wave,
    /// LosslessFast: reorder+delta inverse (`BytePerWord`, one composite per channel).
    Fast,
}

/// Test probe: chunks dispatched by [`RansBatch::run`] (the chunking gate asserts a small injected cap splits a frame).
#[cfg(test)]
pub(crate) static BATCH_CHUNKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A parsed rANS blob: the model + borrowed lanes and the lane streams as the ONE contiguous slice they are stored in
/// (`rans::parse_span`), so a unit's streams upload with a single memcpy.
type Parsed<'a> = (rans::Prepared<&'a [u8]>, &'a [u8]);

/// One GPU-decodable rANS unit of a whole-image wave / fast decode: the parsed stream, where its samples land, and its
/// frame-wide table / stream / lane bases (assigned by [`RansBatch::add`]).
struct RansUnit<'a> {
    prep: rans::Prepared<&'a [u8]>,
    /// The lane streams as the one contiguous slice they are stored in (`rans::parse_span`): ONE memcpy per unit.
    span: &'a [u8],
    /// Tile geometry: element size, tile w/h, tile origin.
    esz: u32,
    tw: u32,
    th: u32,
    x: u32,
    y: u32,
    /// Plane width in pixels (the descriptor's row stride).
    full_w: u32,
    /// Word base of the unit's channel plane inside its composite buffer.
    chan_base: u32,
    table: usize,
    stream_base: u32,
    lane_base: u32,
}

/// Host-side accumulator of every GPU-decodable rANS unit of a whole-image wave / fast decode. [`add`](Self::add) is O(1)
/// bookkeeping (dedupe a table, assign stream / lane bases); [`run`](Self::run) lays every frame-wide buffer out in
/// parallel straight into the upload staging memory and builds the `Desc` array (std430: [`DESC_WORDS`] u32 per unit,
/// WGSL `Desc` order). `lane_off` holds ABSOLUTE byte offsets into the shared `streams` buffer (the one big
/// correctness hazard of this layout).
#[derive(Default)]
struct RansBatch<'a> {
    units: Vec<RansUnit<'a>>,
    /// Deduplicated models: index of the first unit that carries each (its `Prepared` holds the arrays) and its
    /// `(freq_base, cum_base, slot_base)` word bases. `cum` and `slot` derive from `freq`, so identical models
    /// (common across tiles of one channel) share ONE copy instead of 16 KiB of `slot` per unit.
    tables: Vec<(usize, [u32; 3])>,
    table_ix: std::collections::HashMap<(u8, [u16; 256]), usize>,
    freq_words: usize,
    slot_words: usize,
    stream_bytes: usize,
    lane_words: usize,
}

impl<'a> RansBatch<'a> {
    /// Append one rANS unit (`prep` = a `rans::parse`d tile) that decodes into plane word base `chan_base`.
    fn add(&mut self, (prep, span): (rans::Prepared<&'a [u8]>, &'a [u8]), it: &Item, chan_base: u32, full_w: u32) -> GpuResult<()> {
        let stream_base = u(self.stream_bytes)?;
        let lane_base = u(self.lane_words / 2)?;
        let key = (prep.cfg.table_log, prep.freq);
        let table = match self.table_ix.get(&key) {
            Some(&i) => i,
            None => {
                let bases = [u(self.freq_words)?, u(self.freq_words)?, u(self.slot_words)?];
                self.freq_words += 256;
                self.slot_words += prep.slot_to_sym.len();
                self.tables.push((self.units.len(), bases));
                self.table_ix.insert(key, self.tables.len() - 1);
                self.tables.len() - 1
            }
        };
        self.stream_bytes += span.len();
        u(self.stream_bytes)?;
        self.lane_words += 2 * prep.lane_streams.len();
        self.units.push(RansUnit {
            prep,
            span,
            esz: it.esz as u32,
            tw: it.tw,
            th: it.th,
            x: it.x,
            y: it.y,
            full_w,
            chan_base,
            table,
            stream_base,
            lane_base,
        });
        Ok(())
    }

    /// Upload the frame-wide buffers once, then run the four batched stages (rANS -> repack -> cols -> rows) over
    /// `comp` for every chunk of units, ALL in ONE encoder / ONE submit. A chunk is a run of units whose symbol
    /// scratch fits the chunk cap ([`GpuContext::chunk_words`], from the device's real buffer limit): a frame
    /// whose whole symbol buffer would exceed it (8k LosslessWave with 4-byte samples: 2.7 GB > 2 GiB) decodes in
    /// several chunks that REUSE one pooled scratch (wgpu orders the passes). Only the descriptor array is per
    /// chunk; tables / streams / lane spans are shared. Each stage is ONE workgroup per unit on a 2-D grid (the
    /// kernels rebuild the unit id from `wid.y * num_workgroups.x + wid.x`), so a 16K image of 128² tiles x 4
    /// channels (65536 units) works.
    fn run(&self, ctx: &GpuContext, comp: &wgpu::Buffer, stage: Stage, cancel: &Cancel) -> GpuResult<()> {
        cancelled_gpu(cancel)?;
        let device = &ctx.device;
        let syms: Vec<usize> = self.units.iter().map(|un| un.prep.n).collect();
        let chunks = plan_chunks(syms.iter().map(|&n| (0, n)), ctx.chunk_words(syms.iter().copied().max().unwrap_or(0)))?;
        for (what, bytes) in [
            ("batch freq", self.freq_words * 4),
            ("batch slot", self.slot_words * 4),
            ("batch streams", self.stream_bytes.next_multiple_of(4)),
            ("batch lane_off", self.lane_words * 4),
        ] {
            ctx.check_buf(what, bytes as u64)?;
        }
        // Frame-wide buffers: every part (a table / a unit's streams / a unit's lane spans) is its own rayon task.
        let prep_of = |i: usize| &self.units[self.tables[i].0].prep;
        let tab_parts = vec![256 * 4; self.tables.len()];
        let b_freq = storage_parts(ctx, "b.freq", &tab_parts, |i, mut d| {
            let v: Vec<u32> = prep_of(i).freq.iter().map(|&f| f as u32).collect();
            d.copy_from_slice(bytemuck::cast_slice(&v));
        })?;
        let b_cum = storage_parts(ctx, "b.cum", &tab_parts, |i, mut d| {
            let v: Vec<u32> = prep_of(i).cum.iter().map(|&c| c as u32).collect();
            d.copy_from_slice(bytemuck::cast_slice(&v));
        })?;
        let slot_parts: Vec<usize> = (0..self.tables.len()).map(|i| prep_of(i).slot_to_sym.len() * 4).collect();
        let b_slot = storage_parts(ctx, "b.slot", &slot_parts, |i, mut d| {
            let v: Vec<u32> = prep_of(i).slot_to_sym.iter().map(|&s| s as u32).collect();
            d.copy_from_slice(bytemuck::cast_slice(&v));
        })?;
        let stream_parts: Vec<usize> = self.units.iter().map(|un| un.span.len()).collect();
        let b_streams = storage_parts(ctx, "b.streams", &stream_parts, |i, mut d| d.copy_from_slice(self.units[i].span))?;
        let lane_parts: Vec<usize> = self.units.iter().map(|un| un.prep.lane_streams.len() * 8).collect();
        let b_off = storage_parts(ctx, "b.off", &lane_parts, |i, mut d| {
            let un = &self.units[i];
            let mut v: Vec<u32> = Vec::with_capacity(un.prep.lane_streams.len() * 2);
            let mut at = un.stream_base as usize;
            for s in &un.prep.lane_streams {
                // `add` checked the running stream total against u32, so every span fits.
                v.extend_from_slice(&[at as u32, (at + s.len()) as u32]);
                at += s.len();
            }
            d.copy_from_slice(bytemuck::cast_slice(&v));
        })?;
        // The symbol scratch is the same size for every frame of a clip: pooled (see `Scratch`).
        let sym_words = chunks.iter().map(|c| c.words).max().unwrap_or(0);
        let b_sym = ctx.scratch("b.sym", (sym_words.max(1) * 4) as u64);
        let pl = &ctx.pipelines;
        let mut enc = encoder(device, "exv.wave.batched");
        #[cfg(test)]
        BATCH_CHUNKS.fetch_add(chunks.len(), std::sync::atomic::Ordering::Relaxed);
        for c in &chunks {
            cancelled_gpu(cancel)?;
            // Chunk descriptors (one row per unit, parallel parts) with their symbol offsets rebased onto the scratch.
            let mut base = 0u32;
            let mut sym_of = Vec::with_capacity(c.units.len());
            for &n in &syms[c.units.clone()] {
                sym_of.push(base);
                base += u(n)?; // <= the chunk cap (u32-addressable), so no wrap
            }
            let parts = vec![DESC_WORDS * 4; c.units.len()];
            let b_desc = storage_parts(ctx, "b.desc", &parts, |k, mut d| {
                let un = &self.units[c.units.start + k];
                let t = &self.tables[un.table];
                let (cfg, n) = (un.prep.cfg, un.prep.n);
                // Typed to `DESC_WORDS`: a field added to / dropped from the WGSL `Desc` fails to compile here.
                let row: [u32; DESC_WORDS] = [
                    cfg.lanes as u32, cfg.table_log as u32, n as u32, un.lane_base,
                    t.1[0], t.1[1], t.1[2], sym_of[k],
                    un.esz, un.tw * un.th, un.tw, un.th,
                    un.full_w, un.x, un.y, un.chan_base,
                ];
                d.copy_from_slice(bytemuck::cast_slice(&row));
            })?;
            let bg_rans = bind_bufs(device, &pl.rans_b, "b.rans", &[&b_desc, &b_freq, &b_cum, &b_slot, &b_streams, &b_off, &b_sym]);
            let grid = ctx.split_1d("rans units", c.units.len() as u64)?;
            pass(ctx, &mut enc, "rans", &pl.rans_b, &bg_rans, grid)?;
            match stage {
                Stage::Wave => {
                    let bg_rep = bind_bufs(device, &pl.repack_b, "b.repack", &[&b_desc, &b_sym, comp]);
                    let bg_cols = bind_bufs(device, &pl.wav_cols_b, "b.cols", &[&b_desc, comp]);
                    let bg_rows = bind_bufs(device, &pl.wav_rows_b, "b.rows", &[&b_desc, comp]);
                    pass(ctx, &mut enc, "repack", &pl.repack_b, &bg_rep, grid)?;
                    pass(ctx, &mut enc, "cols", &pl.wav_cols_b, &bg_cols, grid)?;
                    pass(ctx, &mut enc, "rows", &pl.wav_rows_b, &bg_rows, grid)?;
                }
                Stage::Fast => {
                    // One workgroup (parallel prefix scan) per (unit, plane): `MAX_ESZ` workgroups per unit.
                    let bg_un = bind_bufs(device, &pl.undelta_b, "b.undelta", &[&b_desc, &b_sym, comp]);
                    let g = ctx.split_1d("undelta", c.units.len() as u64 * MAX_ESZ as u64)?;
                    pass(ctx, &mut enc, "undelta", &pl.undelta_b, &bg_un, g)?;
                }
            }
        }
        ctx.queue.submit([enc.finish()]);
        Ok(())
    }
}

/// Host composite planes for the CPU-fallback tiles of a whole-image decode: one `Vec<u32>` per channel in
/// word representation, uploaded ONCE per touched channel (a single `write_buffer` instead of one per tile row —
/// that per-tile upload was the wave bottleneck). A plane is allocated (zeroed) only when a tile actually lands in
/// it: an all-GPU frame no longer pays for hundreds of MB of untouched host planes. The upload must precede the
/// GPU dispatches that overwrite the GPU-tile regions of mixed channels (queue writes apply at the next submit).
struct CpuPlanes {
    /// Word length of channel `ci`'s plane.
    words: Vec<usize>,
    /// `planes[ci]` is empty until the channel is first touched.
    planes: Vec<Vec<u32>>,
    /// The caller's host budget: the one live host plane (the pipelined hosts free each after its upload) must fit it.
    limits: DecodeLimits,
}

impl CpuPlanes {
    /// `words[ci]` = the word length of channel `ci`'s plane.
    fn new(words: impl IntoIterator<Item = usize>, limits: &DecodeLimits) -> Self {
        let words: Vec<usize> = words.into_iter().collect();
        let planes = vec![Vec::new(); words.len()];
        Self { words, planes, limits: *limits }
    }

    /// Allocate channel `ci`'s zeroed host plane: budget-checked first, then fallibly (a forged size is a typed error).
    fn alloc(&mut self, ci: usize) -> GpuResult<()> {
        let words = self.words.get(ci).copied().ok_or_else(|| corrupt("CPU plane channel index"))?;
        let bytes = (words as u64).saturating_mul(4);
        self.limits.check(bytes).map_err(|e| GpuError::from_decode(e, &format!("CPU-fallback plane of channel {ci}")))?;
        let mut plane = Vec::new();
        plane.try_reserve_exact(words).map_err(|_| GpuError::Limit(format!("cannot allocate a {bytes} B host plane")))?;
        plane.resize(words, 0u32);
        self.planes[ci] = plane;
        Ok(())
    }

    /// CPU-decode every item and scatter it into its channel plane, in PARALLEL: tiles of one tile-row of one
    /// channel own a disjoint band of rows of that plane, so each rayon task decodes and scatters its band's tiles
    /// with no shared mutable state and at most one packed tile alive per thread. `rep(ci)` = the plane's word
    /// representation; `tile_h` = the nominal tile height (band = tile row).
    fn put_all(&mut self, items: &[Item], profile: &exv_codec::Profile, rep: impl Fn(usize) -> TileEnc + Sync, w: u32, tile_h: u32) -> GpuResult<()> {
        use rayon::prelude::*;
        for ci in 0..self.planes.len() {
            let mut bands: std::collections::BTreeMap<u32, Vec<&Item>> = std::collections::BTreeMap::new();
            for it in items.iter().filter(|it| it.ci == ci) {
                bands.entry(it.y / tile_h).or_default().push(it);
            }
            if bands.is_empty() {
                continue;
            }
            if self.planes[ci].is_empty() {
                self.alloc(ci)?;
            }
            // Words per pixel of this plane: one per element, or one per byte (`BytePerWord`).
            let enc = rep(ci);
            let wpp = match enc {
                TileEnc::ElemPerWord(_) => 1,
                TileEnc::BytePerWord => bands.values().next().and_then(|v| v.first()).map_or(1, |it| it.esz),
            };
            let band_words = tile_h as usize * w as usize * wpp;
            self.planes[ci].par_chunks_mut(band_words).enumerate().try_for_each(|(bi, band)| -> GpuResult<()> {
                for it in bands.get(&(bi as u32)).into_iter().flatten() {
                    let packed = cpu_tile(it, profile)?;
                    scatter(band, bi as u32 * tile_h, &packed, enc, w, it);
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Write every touched channel plane to `buf(ci)` at word offset `off(ci)` and FREE it (a plane is uploaded
    /// once; the pipelined hosts call this per channel, so only one host plane is alive at a time). A big plane goes
    /// through the device's pooled upload staging ([`write_pooled`]), submitted at once: the ordering the dispatches
    /// that follow rely on is the same as for a queued write.
    fn upload(&mut self, ctx: &GpuContext, buf: impl Fn(usize) -> (wgpu::Buffer, usize)) -> GpuResult<()> {
        for (ci, plane) in self.planes.iter_mut().enumerate() {
            if !plane.is_empty() {
                let (b, off) = buf(ci);
                write_pooled(ctx, "exv.cpu_plane", &b, (off * 4) as u64, bytemuck::cast_slice(plane))?;
                *plane = Vec::new();
            }
        }
        Ok(())
    }
}

/// Whole-image **LosslessWave** (Wavelet53) GPU decode (`decode_rans_batched`, wave stage): rANS -> repack -> inverse
/// 5/3 columns -> rows over a descriptor grid, one submit per channel, chunked below the device buffer limit. `limits` as
/// [`decode_still_gpu`].
pub fn decode_wave_batched(
    ctx: &GpuContext,
    data: &[u8],
    h: &exv_format::Header,
    profile: &exv_codec::Profile,
    limits: &DecodeLimits,
) -> GpuResult<GpuImage> {
    decode_rans_batched(ctx, data, h, profile, Stage::Wave, limits, &Cancel::new())
}

/// Whole-image GPU decode of the rANS-family profiles into a per-channel VRAM [`GpuImage`] via a descriptor
/// grid: ONE workgroup per unit, ONE dispatch per stage.
/// * [`Stage::Wave`] (LosslessWave, Wavelet53): rANS -> repack -> inverse 5/3 columns -> rows; all channels share ONE
///   combined composite (`ElemPerWord`, `w*h` words each); esz>2 channels decode on the CPU.
/// * [`Stage::Fast`] (LosslessFast, reorder+delta): rANS -> batched undelta (one thread per tile-plane); one
///   `BytePerWord` composite per channel; every dtype size decodes on the GPU.
///
/// Every unit's rANS data is concatenated into a shared buffer set + a `Desc` array — this removes the per-unit
/// allocation that the measurement showed dominates. raw/deflate tiles (or `table_log > MAX_GPU_TABLE_LOG`) fall
/// back to the CPU (tile rows in parallel) into the same composite; the final stage writes straight into the
/// composite at the tile origin (no scatter).
///
/// Pipelined per CHANNEL: blobs are parsed in parallel, the channel's batch ([`RansBatch`]) is laid out and
/// submitted, and only then does the host start the next channel — the GPU decodes channel `c` while the CPU
/// prepares `c+1` (a single end-of-frame submit left the GPU idle for the whole host phase). Within a channel the
/// frame is cut into chunks below the device buffer limit ([`RansBatch::run`]).
fn decode_rans_batched(
    ctx: &GpuContext,
    data: &[u8],
    h: &exv_format::Header,
    profile: &exv_codec::Profile,
    stage: Stage,
    limits: &DecodeLimits,
    cancel: &Cancel,
) -> GpuResult<GpuImage> {
    use rayon::prelude::*;
    let (w, hgt) = (h.width, h.height);
    let nch = h.channels.len();
    let overflow = || GpuError::Limit("image size overflows usize".to_string());
    let plane_words = (w as usize).checked_mul(hgt as usize).ok_or_else(overflow)?;
    // Per channel: (word offset inside its buffer, words, representation). Wave shares ONE composite across channels;
    // Fast gives each channel its own (`BytePerWord` planes differ in size per dtype).
    let layout: Vec<(usize, usize, TileEnc)> = match stage {
        Stage::Wave => (0..nch).map(|ci| (ci * plane_words, plane_words, TileEnc::ElemPerWord(h.channels[ci].dtype.size()))).collect(),
        Stage::Fast => h
            .channels
            .iter()
            .map(|ch| plane_words.checked_mul(ch.dtype.size()).map(|n| (0, n, TileEnc::BytePerWord)).ok_or_else(overflow))
            .collect::<GpuResult<_>>()?,
    };
    ctx.scoped(|| {
        let bufs: Vec<wgpu::Buffer> = match stage {
            Stage::Wave => {
                let comp = composite(ctx, "exv.composite", plane_words.checked_mul(nch).ok_or_else(overflow)?)?;
                vec![comp; nch]
            }
            Stage::Fast => layout.iter().map(|&(_, n, _)| composite(ctx, "exv.fastplane", n)).collect::<GpuResult<_>>()?,
        };
        let all = items(data, h, limits)?;
        let mut cpu = CpuPlanes::new(layout.iter().map(|&(_, n, _)| n), limits);
        for ci in 0..nch {
            cancelled_gpu(cancel)?;
            let (off, _, enc) = layout[ci];
            let chan: Vec<Item> = all.iter().filter(|it| it.ci == ci).copied().collect();
            // Route + parse every blob in parallel (pure): `Some` = a GPU rANS unit, `None` = a CPU-decoded tile.
            let routed: Vec<Option<Parsed<'_>>> = chan
                .par_iter()
                .map(|it| {
                    if (stage == Stage::Fast || it.esz <= 2)
                        && let Some(BlobKind::Rans(p)) = classify_blob(it.blob)
                        && let Some((prep, span)) = rans::parse_span(p)
                        // table_log > MAX_GPU_TABLE_LOG would overflow the kernel's workgroup table -> CPU fallback
                        // below, not silent GPU corruption.
                        && prep.n == it.tw as usize * it.th as usize * it.esz
                        && prep.cfg.table_log <= MAX_GPU_TABLE_LOG
                    {
                        Some((prep, span))
                    } else {
                        None
                    }
                })
                .collect();
            let mut batch = RansBatch::default();
            let mut cpu_items: Vec<Item> = Vec::new();
            for (it, prep) in chan.iter().zip(routed) {
                match prep {
                    Some(prep) => batch.add(prep, it, u(off)?, w)?,
                    None => cpu_items.push(*it),
                }
            }
            // CPU-decoded tiles first, then the GPU dispatch overwrites the rANS-tile regions of a mixed channel.
            cpu.put_all(&cpu_items, profile, |_| enc, w, h.tile_h)?;
            cpu.upload(ctx, |c| (bufs[c].clone(), layout[c].0))?;
            if !batch.units.is_empty() {
                batch.run(ctx, &bufs[ci], stage, cancel)?;
            }
        }
        let planes = (0..nch)
            .map(|ci| GpuTile { buf: bufs[ci].clone(), off_words: layout[ci].0, words: layout[ci].1, w, h: hgt, dtype: h.channels[ci].dtype, enc: layout[ci].2 })
            .collect();
        Ok(GpuImage { width: w, height: hgt, planes })
    })
}

/// Whole-image **EXV-RT** (`blockrt_dwt`, `Transform::BlockRt`) GPU decode: ONE VRAM composite PER
/// channel (word rep, `w*h` words each), every `(tile, channel)` blob routed by its tag. F16 tiles coded by the
/// block codec (`BlobKind::BlockRt`) are parsed in parallel ([`RtTile::parse`], borrowing the container bytes) and
/// decoded as one [`RtBatch`] per channel: one set of storage buffers, one encoder, one submit PER CHANNEL, the
/// host preparing channel `c+1` while the GPU decodes `c` (the old per-tile path was 89-99% host overhead). A
/// `BlobKind::Raw` tile (a non-F16 channel, or an F16 tile that did not shrink below raw) is CPU-decoded
/// (`exv_codec::decode_channel`) into a host plane and uploaded ONCE per dirty channel BEFORE that channel's GPU
/// dispatch overwrites its regions (the wave path's mixed-residency ordering). Any other tag (Rans/Deflate/Htj2k)
/// inside an EXV-RT image is a mismatched/corrupt stream -> clean `Err` (NO silent mimic).
///
/// Bit-for-bit == `exv_codec::read_still` at q=0 (the pure-integer inverse path). Each channel composite must
/// fit VRAM (`max_buffer_size`); the symbol scratch is chunked below the device limit ([`RtBatch::build`]).
fn decode_blockrt_batched(
    ctx: &GpuContext,
    data: &[u8],
    h: &exv_format::Header,
    profile: &exv_codec::Profile,
    limits: &DecodeLimits,
    cancel: &Cancel,
) -> GpuResult<GpuImage> {
    use rayon::prelude::*;
    let (w, hgt) = (h.width, h.height);
    let nch = h.channels.len();
    let plane_words = (w as usize).checked_mul(hgt as usize).ok_or_else(|| GpuError::Limit("image size overflows usize".to_string()))?;
    ctx.scoped(|| {
        // One composite buffer PER channel (word rep).
        let bufs = (0..nch).map(|_| composite(ctx, "exv.rt.composite", plane_words)).collect::<GpuResult<Vec<_>>>()?;
        // Classify EVERY blob first (a corrupt tag must fail before any GPU work): BlockRt tiles become GPU jobs
        // (borrowing `data`), Raw tiles CPU work.
        let mut jobs: Vec<Vec<(Item, &[u8])>> = vec![Vec::new(); nch];
        let mut raw: Vec<Item> = Vec::new();
        for it in items(data, h, limits)? {
            match classify_blob(it.blob) {
                // F16 block-coded tile -> GPU. The block codec only ever emits F16 planes; a BlockRt tag on any
                // other dtype is a corrupt stream (hard Err, never a silent decode).
                Some(BlobKind::BlockRt(payload)) => {
                    if it.esz != 2 {
                        return Err(GpuError::Corrupt(format!("exv-rt GPU: BlockRt tile on non-F16 channel {} (esz={})", it.ci, it.esz)));
                    }
                    jobs[it.ci].push((it, payload));
                }
                // Raw tile (a non-F16 channel, or an F16 tile that did not shrink): CPU decode into the host plane.
                Some(BlobKind::Raw(_)) => raw.push(it),
                // An EXV-RT image carries only BlockRt (F16) + Raw (everything else) blobs. Rans/Deflate/Htj2k/None
                // is a mismatched/corrupt stream -> hard Err (no silent mimic of another codec).
                Some(BlobKind::Rans(_)) => return Err(GpuError::Corrupt(format!("exv-rt GPU: unexpected rANS blob in EXV-RT image (tile {}, ch {})", it.t, it.ci))),
                Some(BlobKind::Deflate(_)) => return Err(GpuError::Corrupt(format!("exv-rt GPU: unexpected deflate blob in EXV-RT image (tile {}, ch {})", it.t, it.ci))),
                Some(BlobKind::Htj2k(_)) => return Err(GpuError::Corrupt(format!("exv-rt GPU: unexpected htj2k blob in EXV-RT image (tile {}, ch {})", it.t, it.ci))),
                Some(BlobKind::Bc6h(_)) => return Err(GpuError::Corrupt(format!("exv-rt GPU: unexpected bc6h blob in EXV-RT image (tile {}, ch {})", it.t, it.ci))),
                None => return Err(GpuError::Corrupt(format!("exv-rt GPU: bad blob tag (tile {}, ch {})", it.t, it.ci))),
            }
        }
        let mut cpu = CpuPlanes::new((0..nch).map(|_| plane_words), limits);
        for (ci, chan) in jobs.iter().enumerate() {
            cancelled_gpu(cancel)?;
            let raw_ci: Vec<Item> = raw.iter().filter(|it| it.ci == ci).copied().collect();
            cpu.put_all(&raw_ci, profile, |c| TileEnc::ElemPerWord(h.channels[c].dtype.size()), w, h.tile_h)?;
            cpu.upload(ctx, |c| (bufs[c].clone(), 0))?;
            if chan.is_empty() {
                continue;
            }
            // Parse + validate every blob in parallel (pure, borrowed); the merge into the batch is O(1) per tile.
            let tiles = chan.par_iter().map(|(it, payload)| RtTile::parse(payload, it.tw, it.th)).collect::<GpuResult<Vec<_>>>()?;
            let mut batch = RtBatch::new();
            for (tile, (it, _)) in tiles.into_iter().zip(chan) {
                // full_w/full_h = the channel dims, so the shader's fused crop drops padded pixels beyond the edge.
                batch.push(tile, RtDest::plane(0, it.x, it.y, w, hgt))?;
            }
            batch.build(ctx, &[&bufs[ci]], cancel)?.dispatch(ctx)?;
        }
        let planes = bufs
            .into_iter()
            .enumerate()
            .map(|(ci, buf)| GpuTile {
                buf,
                off_words: 0,
                words: plane_words,
                w,
                h: hgt,
                dtype: h.channels[ci].dtype,
                enc: TileEnc::ElemPerWord(h.channels[ci].dtype.size()),
            })
            .collect();
        Ok(GpuImage { width: w, height: hgt, planes })
    })
}

/// Whole-image **BC6H** (`Transform::Bc6h`) GPU decode: one VRAM composite per channel (`ElemPerWord`, `w*h` words each).
/// F16 tiles coded as `BlobKind::Bc6h` decode directly on the GPU into their `(x, y)` tile origin inside the channel
/// composite via [`decode_bc6h_into`]; `BlobKind::Raw` tiles (non-F16 channels) are CPU-decoded and uploaded once per
/// channel. Bit-for-bit identical to `exv_codec::read_still` in both `Lossy` and `LosslessResidual` modes.
fn decode_bc6h_batched(
    ctx: &GpuContext,
    data: &[u8],
    h: &exv_format::Header,
    profile: &exv_codec::Profile,
    limits: &DecodeLimits,
    cancel: &Cancel,
) -> GpuResult<GpuImage> {
    let (w, hgt) = (h.width, h.height);
    let nch = h.channels.len();
    let plane_words = (w as usize).checked_mul(hgt as usize).ok_or_else(|| GpuError::Limit("image size overflows usize".to_string()))?;
    ctx.scoped(|| {
        let bufs = (0..nch).map(|_| composite(ctx, "exv.bc6h.composite", plane_words)).collect::<GpuResult<Vec<_>>>()?;
        let mut jobs: Vec<Vec<(Item, &[u8])>> = vec![Vec::new(); nch];
        let mut raw: Vec<Item> = Vec::new();
        for it in items(data, h, limits)? {
            match classify_blob(it.blob) {
                Some(BlobKind::Bc6h(payload)) => {
                    if it.dtype != Dtype::F16 {
                        return Err(GpuError::Corrupt(format!("bc6h GPU: Bc6h tile on non-F16 channel {}", it.ci)));
                    }
                    jobs[it.ci].push((it, payload));
                }
                Some(BlobKind::Raw(_)) => raw.push(it),
                Some(other) => return Err(GpuError::Corrupt(format!("bc6h GPU: unexpected blob {other:?} in BC6H image (tile {}, ch {})", it.t, it.ci))),
                None => return Err(GpuError::Corrupt(format!("bc6h GPU: bad blob tag (tile {}, ch {})", it.t, it.ci))),
            }
        }
        let mut cpu = CpuPlanes::new((0..nch).map(|_| plane_words), limits);
        for (ci, chan) in jobs.iter().enumerate() {
            cancelled_gpu(cancel)?;
            let raw_ci: Vec<Item> = raw.iter().filter(|it| it.ci == ci).copied().collect();
            cpu.put_all(&raw_ci, profile, |c| TileEnc::ElemPerWord(h.channels[c].dtype.size()), w, h.tile_h)?;
            cpu.upload(ctx, |c| (bufs[c].clone(), 0))?;
            for &(it, payload) in chan {
                cancelled_gpu(cancel)?;
                decode_bc6h_into(ctx, payload, it.dtype, it.tw, it.th, w, it.x, it.y, &bufs[ci])?;
            }
        }
        let planes = bufs
            .into_iter()
            .enumerate()
            .map(|(ci, buf)| GpuTile {
                buf,
                off_words: 0,
                words: plane_words,
                w,
                h: hgt,
                dtype: h.channels[ci].dtype,
                enc: TileEnc::ElemPerWord(h.channels[ci].dtype.size()),
            })
            .collect();
        Ok(GpuImage { width: w, height: hgt, planes })
    })
}

/// One resident-atlas tile's GPU-decodable channels: `(channel index, tag-stripped payload)` + the tile's pixel size.
/// The atlas collects ALL tiles a request misses (the INTRA ancestors of SKIP tiles included) as `TileJob`s and decodes
/// them in ONE batch ([`decode_tiles_wave_gpu`] / [`decode_tiles_exvrt_gpu`]).
#[cfg(feature = "htj2k-backend")]
pub(crate) struct TileJob<'a> {
    pub chans: Vec<(usize, &'a [u8])>,
    pub tw: u32,
    pub th: u32,
}

/// Cut `jobs` into runs whose total output (`words(job)` u32 words) fits ONE device buffer, so a request of thousands of
/// tiles stays under the binding limit (typed `Limit` for a single tile above it). Each run = one batch / one submit.
#[cfg(feature = "htj2k-backend")]
fn plan_jobs(ctx: &GpuContext, jobs: &[TileJob]) -> GpuResult<Vec<std::ops::Range<usize>>> {
    let cap = (ctx.sym_cap / 4) as usize;
    let words = |j: &TileJob| j.tw as usize * j.th as usize * j.chans.len();
    Ok(plan_chunks(jobs.iter().map(|j| (0, words(j))), cap)?.into_iter().map(|c| c.units).collect())
}

/// Decode the GPU-eligible channels of MANY tiles — `jobs[i].chans` = `(channel index, tag-stripped rANS wave payload)` —
/// in ONE [`RansBatch`] per run of tiles (LosslessWave stages): payloads are parsed in PARALLEL, every frame-wide buffer is
/// laid out straight into the upload staging, and the whole run is ONE encoder / ONE submit into one composite (a
/// tile-channel's words start at its own `chan_base`; tile sizes may differ: edge tiles). The resident atlas used to run the
/// stages once per tile. Returns, per job, the [`GpuTile`] views (word rep, `tw*th` words each) of the channels it decoded.
/// A payload that does not parse or does not match the tile geometry is `Corrupt`; a channel with
/// `table_log > MAX_GPU_TABLE_LOG` (the batched kernel's workgroup table cannot hold it) is simply NOT returned, so the
/// caller decodes it on the CPU.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn decode_tiles_wave_gpu(ctx: &GpuContext, dtypes: &[Dtype], jobs: &[TileJob], cancel: &Cancel) -> GpuResult<Vec<Vec<(usize, GpuTile)>>> {
    use rayon::prelude::*;
    cancelled_gpu(cancel)?;
    // Pure parse + validation of every (tile, channel) blob in parallel; `None` = a table the GPU kernel cannot hold.
    let parsed: Vec<Vec<Option<Parsed<'_>>>> = jobs
        .par_iter()
        .map(|j| {
            j.chans
                .iter()
                .map(|&(ci, payload)| {
                    let dtype = *dtypes.get(ci).ok_or_else(|| corrupt("tile channel index out of range"))?;
                    let (prep, span) = rans::parse_span(payload).ok_or_else(|| corrupt("bad rANS container"))?;
                    let words = j.tw as usize * j.th as usize;
                    if prep.n != words * dtype.size() {
                        return Err(GpuError::Corrupt(format!("size mismatch: n={} != {words}*{}", prep.n, dtype.size())));
                    }
                    Ok((prep.cfg.table_log <= MAX_GPU_TABLE_LOG).then_some((prep, span)))
                })
                .collect::<GpuResult<Vec<_>>>()
        })
        .collect::<GpuResult<_>>()?;
    let mut parsed: Vec<std::vec::IntoIter<Option<Parsed<'_>>>> = parsed.into_iter().map(Vec::into_iter).collect();
    let mut out: Vec<Vec<(usize, GpuTile)>> = jobs.iter().map(|_| Vec::new()).collect();
    ctx.scoped(|| {
        for run in plan_jobs(ctx, jobs)? {
            cancelled_gpu(cancel)?;
            let mut batch = RansBatch::default();
            let mut placed: Vec<(usize, usize, usize)> = Vec::new(); // (job, channel, word base)
            let mut words = 0usize;
            for ji in run {
                let j = &jobs[ji];
                for &(ci, _) in &j.chans {
                    let Some(prep) = parsed[ji].next().ok_or_else(|| corrupt("parsed channel count"))? else { continue };
                    let dtype = dtypes[ci];
                    let it = Item { t: 0, ci, dtype, esz: dtype.size(), blob: &[], x: 0, y: 0, tw: j.tw, th: j.th };
                    batch.add(prep, &it, u(words)?, j.tw)?;
                    placed.push((ji, ci, words));
                    words += j.tw as usize * j.th as usize;
                }
            }
            if placed.is_empty() {
                continue;
            }
            let comp = composite(ctx, "exv.tiles.composite", words)?;
            batch.run(ctx, &comp, Stage::Wave, cancel)?;
            for (ji, ci, base) in placed {
                let (tw, th, dtype) = (jobs[ji].tw, jobs[ji].th, dtypes[ci]);
                let tile = GpuTile { buf: comp.clone(), off_words: base, words: tw as usize * th as usize, w: tw, h: th, dtype, enc: TileEnc::ElemPerWord(dtype.size()) };
                out[ji].push((ci, tile));
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// Decode the F16 `BlockRt` channels of MANY tiles — `jobs[i].chans` = `(channel index, tag-stripped payload)` — as one
/// [`RtBatch`] per run of tiles: blobs parsed in parallel, ONE output buffer per channel holding every tile's plane back to
/// back ([`RtDest::base`]), ONE encoder / ONE submit. Returns, per job, the [`GpuTile`]s (word rep, `tw*th` words) in
/// `chans` order. The resident atlas's EXV-RT route (it used to decode, and submit, tile by tile).
#[cfg(feature = "htj2k-backend")]
pub(crate) fn decode_tiles_exvrt_gpu(ctx: &GpuContext, dtypes: &[Dtype], jobs: &[TileJob], cancel: &Cancel) -> GpuResult<Vec<Vec<(usize, GpuTile)>>> {
    use rayon::prelude::*;
    cancelled_gpu(cancel)?;
    for j in jobs {
        if let Some(&(ci, _)) = j.chans.iter().find(|&&(ci, _)| dtypes.get(ci).map(|d| d.size()) != Some(2)) {
            return Err(GpuError::Unsupported(format!("exv-rt GPU: channel {ci} is not an f16 plane")));
        }
    }
    let parsed: Vec<Vec<RtTile>> = jobs
        .par_iter()
        .map(|j| j.chans.iter().map(|&(_, payload)| RtTile::parse(payload, j.tw, j.th)).collect::<GpuResult<Vec<_>>>())
        .collect::<GpuResult<_>>()?;
    let mut parsed: Vec<std::vec::IntoIter<RtTile>> = parsed.into_iter().map(Vec::into_iter).collect();
    let mut out: Vec<Vec<(usize, GpuTile)>> = jobs.iter().map(|_| Vec::new()).collect();
    ctx.scoped(|| {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        for run in plan_jobs(ctx, jobs)? {
            cancelled_gpu(cancel)?;
            // One output buffer per distinct channel of the run; a tile-channel's plane starts at its channel's running base.
            let mut chans: Vec<usize> = run.clone().flat_map(|ji| jobs[ji].chans.iter().map(|&(ci, _)| ci)).collect();
            chans.sort_unstable();
            chans.dedup();
            let mut base = vec![0usize; chans.len()];
            let mut batch = RtBatch::new();
            let mut placed: Vec<(usize, usize, usize)> = Vec::new(); // (job, channel slot, word base)
            for ji in run {
                let j = &jobs[ji];
                for &(ci, _) in &j.chans {
                    let slot = chans.binary_search(&ci).map_err(|_| corrupt("rt channel slot"))?;
                    let tile = parsed[ji].next().ok_or_else(|| corrupt("parsed channel count"))?;
                    batch.push(tile, RtDest { ch: slot, base: u(base[slot])?, x0: 0, y0: 0, full_w: j.tw, full_h: j.th })?;
                    placed.push((ji, slot, base[slot]));
                    base[slot] += j.tw as usize * j.th as usize;
                }
            }
            if batch.is_empty() {
                continue;
            }
            let outs: Vec<wgpu::Buffer> = base
                .iter()
                .map(|&w| {
                    ctx.check_buf("exv-rt out", w as u64 * 4)?;
                    Ok(new_buf(&ctx.device, "rt.tiles.out", (w.max(1) * 4) as u64, usage))
                })
                .collect::<GpuResult<_>>()?;
            let refs: Vec<&wgpu::Buffer> = outs.iter().collect();
            // `build` checks `cancel` before its frame-wide uploads and before every chunk's staging + submit.
            batch.build(ctx, &refs, cancel)?.dispatch(ctx)?;
            for (ji, slot, base) in placed {
                let (tw, th, ci) = (jobs[ji].tw, jobs[ji].th, chans[slot]);
                let tile = GpuTile { buf: outs[slot].clone(), off_words: base, words: tw as usize * th as usize, w: tw, h: th, dtype: dtypes[ci], enc: TileEnc::ElemPerWord(2) };
                out[ji].push((ci, tile));
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// Decode the F16 `Bc6h` channels of MANY tiles into [`GpuTile`]s for the resident atlas.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn decode_tiles_bc6h_gpu(ctx: &GpuContext, dtypes: &[Dtype], jobs: &[TileJob], cancel: &Cancel) -> GpuResult<Vec<Vec<(usize, GpuTile)>>> {
    cancelled_gpu(cancel)?;
    let mut out: Vec<Vec<(usize, GpuTile)>> = Vec::with_capacity(jobs.len());
    for j in jobs {
        cancelled_gpu(cancel)?;
        let mut tiles = Vec::with_capacity(j.chans.len());
        for &(ci, payload) in &j.chans {
            cancelled_gpu(cancel)?;
            let dt = *dtypes.get(ci).ok_or_else(|| corrupt("bc6h channel index out of range"))?;
            let gt = decode_tile_bc6h_vram(ctx, payload, dt, j.tw, j.th)?;
            tiles.push((ci, gt));
        }
        out.push(tiles);
    }
    Ok(out)
}

/// Pack a whole [`GpuImage`] out of VRAM to native per-channel plane bytes (test/export only).
pub fn read_back_image(ctx: &GpuContext, img: &GpuImage) -> GpuResult<Vec<Vec<u8>>> {
    img.planes.iter().map(|p| read_back(ctx, p)).collect()
}
