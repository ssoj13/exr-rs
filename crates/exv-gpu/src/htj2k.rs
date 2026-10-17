//! `htj2k.rs` — HTJ2K-on-GPU integration and resident texture adapters.
//!
//! Gated behind `htj2k-backend` (module decl in the crate root). Holds
//! [`decode_htj2k_still_planes_gpu`] (lossless still -> per-channel byte planes),
//! [`decode_htj2k_resident`], [`Htj2kGpuDecoder`], and the resident atlas re-exports
//! ([`ResidentTexture`], [`ContentKey`], [`DisplayKey`]).

use crate::prelude::*;
use exv_gpu_kernels::*;

/// Per-decode entropy-pool VRAM bound handed to every HTJ2K decode (peak VRAM of one tile/frame decode).
/// ONE value for the one-shot, persistent and atlas paths (the viewer's resident session passes it too).
pub const DECODE_BUDGET: u64 = 256 << 20;

/// Self-contained compatibility adapter replacing the external `jph-wgpu-rs` crate over `exv_codec::htj2k` +
/// `crate::wgpu_tile_atlas` + `GpuContext`.
#[cfg(feature = "htj2k-backend")]
#[allow(dead_code)]
pub mod jph_compat {
    use crate::prelude::*;
    pub use crate::wgpu_tile_atlas::{
        ContentKey, DisplayKey, RegionKey, ResidentTexture, TileAtlas, TileKeys,
    };
    use exv_format::half::half_to_f32;
    use exv_gpu_kernels::{encoder, map_read_blocking, new_buf, tex2d};

    /// Shared GPU device/queue/adapter handle.
    #[derive(Clone)]
    pub struct Gpu {
        pub device: wgpu::Device,
        pub queue: wgpu::Queue,
        pub adapter: wgpu::Adapter,
    }

    impl Gpu {
        /// Adopt an existing `wgpu` device, queue, and adapter.
        pub fn from_parts(device: wgpu::Device, queue: wgpu::Queue, adapter: wgpu::Adapter) -> Self {
            Self { device, queue, adapter }
        }
    }

    /// Parsed J2K codestream configuration.
    #[derive(Clone, Copy, Debug)]
    pub struct DecoderConfig {
        pub color_transform: bool,
    }

    /// Parse SIZ and COD markers from a single-tile J2K codestream.
    fn parse_j2k_header(data: &[u8]) -> Result<(u32, u32, Vec<Dtype>, bool), String> {
        let be16 = |d: &[u8], p: usize| -> Option<u16> {
            Some(u16::from_be_bytes(d.get(p..p.checked_add(2)?)?.try_into().ok()?))
        };
        let be32 = |d: &[u8], p: usize| -> Option<u32> {
            Some(u32::from_be_bytes(d.get(p..p.checked_add(4)?)?.try_into().ok()?))
        };
        if be16(data, 0) != Some(0xFF4F) {
            return Err("HTJ2K: missing SOC marker".to_string());
        }
        let mut p = 2usize;
        let mut siz_parsed: Option<(u32, u32, Vec<Dtype>)> = None;
        let mut color_transform = false;
        let mut cod_seen = false;
        while let Some(m) = be16(data, p) {
            if m == 0xFF90 {
                break;
            }
            let len = be16(data, p + 2).ok_or_else(|| "HTJ2K: truncated marker length".to_string())? as usize;
            if len < 2 {
                return Err("HTJ2K: invalid marker segment length".to_string());
            }
            let body = data.get(p + 4..p + 2 + len).ok_or_else(|| "HTJ2K: truncated marker body".to_string())?;
            match m {
                0xFF51 if siz_parsed.is_none() => {
                    let nch = be16(body, 34).ok_or_else(|| "HTJ2K SIZ: truncated".to_string())? as usize;
                    if body.len() != 36 + 3 * nch {
                        return Err("HTJ2K SIZ: unexpected segment length".to_string());
                    }
                    let w = be32(body, 2).ok_or_else(|| "HTJ2K SIZ: width".to_string())?;
                    let h = be32(body, 6).ok_or_else(|| "HTJ2K SIZ: height".to_string())?;
                    let mut dtypes = Vec::with_capacity(nch);
                    for c in body[36..].as_chunks::<3>().0 {
                        let bits = u32::from(c[0] & 0x7F) + 1;
                        let signed = (c[0] & 0x80) != 0;
                        let dt = match (bits, signed) {
                            (16, true) => Dtype::F16,
                            (32, true) => Dtype::F32,
                            (32, false) => Dtype::U32,
                            (16, false) => Dtype::U16,
                            (8, false) => Dtype::U8,
                            _ => return Err(format!("HTJ2K SIZ: unsupported component ({bits} bits, signed={signed})")),
                        };
                        dtypes.push(dt);
                    }
                    siz_parsed = Some((w, h, dtypes));
                }
                0xFF52 if !cod_seen => {
                    if body.len() < 10 {
                        return Err("HTJ2K COD: truncated".to_string());
                    }
                    color_transform = body[4] != 0;
                    cod_seen = true;
                }
                _ => {}
            }
            p += 2 + len;
        }
        let (w, h, dtypes) = siz_parsed.ok_or_else(|| "HTJ2K: missing SIZ marker".to_string())?;
        if !cod_seen {
            return Err("HTJ2K: missing COD marker".to_string());
        }
        Ok((w, h, dtypes, color_transform))
    }

    /// Single-tile HTJ2K decoder handle.
    pub struct Decoder {
        payload: Vec<u8>,
        w: u32,
        h: u32,
        dtypes: Vec<Dtype>,
        cfg: DecoderConfig,
    }

    impl Decoder {
        /// Open a single-tile HTJ2K codestream.
        pub fn open(_gpu: &Gpu, payload: Vec<u8>, _budget: u64) -> Result<Self, String> {
            let (w, h, dtypes, color_transform) = parse_j2k_header(&payload)?;
            Ok(Self {
                payload,
                w,
                h,
                dtypes,
                cfg: DecoderConfig { color_transform },
            })
        }

        /// Codestream configuration.
        pub fn config(&self) -> &DecoderConfig {
            &self.cfg
        }

        /// Full-resolution level index (`0` = full resolution in `decode_level_planes`).
        pub fn num_levels(&self) -> u32 {
            0
        }

        /// Decode all 16-bit components into `Vec<i32>` samples. 32-bit components defer to the CPU path.
        pub fn read_full(&mut self) -> Result<Vec<Vec<i32>>, String> {
            if self.dtypes.iter().any(|&d| d != Dtype::F16) {
                return Err("64-bit magnitude path unsupported on GPU HTJ2K still kernel".to_string());
            }
            let (planes, rw, rh) = exv_codec::htj2k::decode_htj2k_multi_level(
                &self.payload,
                &self.dtypes,
                self.w,
                self.h,
                0,
                exv_codec::htj2k::Wavelet::Rev53,
            )
            .map_err(|e| e.to_string())?;
            if (rw, rh) != (self.w, self.h) {
                return Err("HTJ2K full-res dimension mismatch".to_string());
            }
            Ok(planes
                .into_iter()
                .map(|p| {
                    p.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&c| i32::from(i16::from_le_bytes(c)))
                        .collect()
                })
                .collect())
        }
    }

    /// Pack 1..=4 `F16` planes (`rw * rh * 2` bytes each) into an interleaved `Rgba16Float` buffer (`rw * rh * 8` bytes).
    fn pack_rgba16f(planes: &[Vec<u8>], rw: u32, rh: u32, dst: &mut Vec<u8>) {
        let npx = (rw as usize) * (rh as usize);
        dst.resize(npx * 8, 0);
        let nch = planes.len();
        for i in 0..npx {
            let c0 = &planes[0][i * 2..i * 2 + 2];
            let c1 = if nch >= 2 { &planes[1][i * 2..i * 2 + 2] } else { c0 };
            let c2 = if nch >= 3 {
                &planes[2][i * 2..i * 2 + 2]
            } else if nch == 1 {
                c0
            } else {
                &[0, 0]
            };
            let c3 = if nch >= 4 {
                &planes[3][i * 2..i * 2 + 2]
            } else {
                &0x3C00u16.to_le_bytes()
            };
            let o = i * 8;
            dst[o..o + 2].copy_from_slice(c0);
            dst[o + 2..o + 4].copy_from_slice(c1);
            dst[o + 4..o + 6].copy_from_slice(c2);
            dst[o + 6..o + 8].copy_from_slice(c3);
        }
    }

    /// Upload an `Rgba16Float` byte buffer (`rw * rh * 8` bytes) into a `wgpu::Texture`.
    fn upload_rgba16f(gpu: &Gpu, rw: u32, rh: u32, tex: &wgpu::Texture, rgba_bytes: &[u8]) {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(rw * 8),
                rows_per_image: Some(rh),
            },
            wgpu::Extent3d {
                width: rw,
                height: rh,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Persistent HTJ2K resident decode context.
    pub struct ResidentCtx {
        pack_buf: Vec<u8>,
        cached_tex: Option<(u32, u32, u32, Vec<u8>, ResidentTexture)>,
    }

    impl ResidentCtx {
        /// Create a resident decode context.
        pub fn new(_gpu: &Gpu) -> Result<Self, String> {
            Ok(Self {
                pack_buf: Vec::new(),
                cached_tex: None,
            })
        }

        /// Decode into a fresh owned `Rgba16Float` `ResidentTexture` at full resolution (`level = 0`).
        pub fn decode_to_texture_fresh(&mut self, gpu: &Gpu, payload: &[u8], budget: u64) -> Result<ResidentTexture, String> {
            self.decode_level_fresh(gpu, payload, 0, budget)
        }

        /// Decode into a fresh owned `Rgba16Float` `ResidentTexture` at resolution `level`.
        pub fn decode_level_fresh(&mut self, gpu: &Gpu, payload: &[u8], level: u32, _budget: u64) -> Result<ResidentTexture, String> {
            let (w, h, dtypes, color_transform) = parse_j2k_header(payload)?;
            if color_transform {
                return Err("HTJ2K: RCT color transform unsupported".to_string());
            }
            if dtypes.is_empty() || dtypes.len() > 4 || dtypes.iter().any(|&d| d != Dtype::F16) {
                return Err("HTJ2K packed RGBA16F requires 1..=4 F16 components".to_string());
            }
            let (planes, rw, rh) = exv_codec::htj2k::decode_htj2k_multi_level(
                payload,
                &dtypes,
                w,
                h,
                level,
                exv_codec::htj2k::Wavelet::Rev53,
            )
            .map_err(|e| e.to_string())?;
            pack_rgba16f(&planes, rw, rh, &mut self.pack_buf);
            let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
            let tex = tex2d(&gpu.device, "exv.htj2k.rgba16f", rw, rh, wgpu::TextureFormat::Rgba16Float, usage);
            upload_rgba16f(gpu, rw, rh, &tex, &self.pack_buf);
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Ok(ResidentTexture::new(tex, view, rw, rh))
        }

        /// Decode into the persistent cached `Rgba16Float` `ResidentTexture` at full resolution (`level = 0`).
        pub fn decode_to_texture(&mut self, gpu: &Gpu, payload: &[u8], budget: u64) -> Result<ResidentTexture, String> {
            self.decode_level(gpu, payload, 0, budget)
        }

        /// Decode into the persistent cached `Rgba16Float` `ResidentTexture` at resolution `level`.
        pub fn decode_level(&mut self, gpu: &Gpu, payload: &[u8], level: u32, _budget: u64) -> Result<ResidentTexture, String> {
            if let Some((cw, ch, cl, cpay, ctex)) = &self.cached_tex
                && *cl == level
                && cpay.as_slice() == payload
            {
                upload_rgba16f(gpu, *cw, *ch, &ctex.tex, &self.pack_buf);
                return Ok(ctex.clone());
            }
            let (w, h, dtypes, color_transform) = parse_j2k_header(payload)?;
            if color_transform {
                return Err("HTJ2K: RCT color transform unsupported".to_string());
            }
            if dtypes.is_empty() || dtypes.len() > 4 || dtypes.iter().any(|&d| d != Dtype::F16) {
                return Err("HTJ2K packed RGBA16F requires 1..=4 F16 components".to_string());
            }
            let (planes, rw, rh) = exv_codec::htj2k::decode_htj2k_multi_level(
                payload,
                &dtypes,
                w,
                h,
                level,
                exv_codec::htj2k::Wavelet::Rev53,
            )
            .map_err(|e| e.to_string())?;
            pack_rgba16f(&planes, rw, rh, &mut self.pack_buf);
            let reuse = match &self.cached_tex {
                Some((cw, ch, _, _, ctex)) if *cw == rw && *ch == rh => ctex.clone(),
                _ => {
                    let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
                    let tex = tex2d(&gpu.device, "exv.htj2k.warm", rw, rh, wgpu::TextureFormat::Rgba16Float, usage);
                    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                    ResidentTexture::new(tex, view, rw, rh)
                }
            };
            upload_rgba16f(gpu, rw, rh, &reuse.tex, &self.pack_buf);
            self.cached_tex = Some((rw, rh, level, payload.to_vec(), reuse.clone()));
            Ok(reuse)
        }

        /// Decode all components at resolution `level` (`0` = full resolution) into per-channel `ResidentTexture` planes.
        pub fn decode_level_planes(
            &mut self,
            gpu: &Gpu,
            payload: &[u8],
            level: u32,
            _budget: u64,
        ) -> Result<Vec<ResidentTexture>, String> {
            let (w, h, dtypes, color_transform) = parse_j2k_header(payload)?;
            if color_transform {
                return Err("HTJ2K: RCT color transform unsupported".to_string());
            }
            if dtypes.iter().any(|&d| !matches!(d, Dtype::F16 | Dtype::F32)) {
                return Err("HTJ2K GPU planar decode requires F16/F32 components".to_string());
            }
            let (planes, rw, rh) = exv_codec::htj2k::decode_htj2k_multi_level(
                payload,
                &dtypes,
                w,
                h,
                level,
                exv_codec::htj2k::Wavelet::Rev53,
            )
            .map_err(|e| e.to_string())?;
            let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
            let mut out = Vec::with_capacity(planes.len());
            for (plane_bytes, &dt) in planes.iter().zip(&dtypes) {
                let (format, bpp) = match dt {
                    Dtype::F16 => (wgpu::TextureFormat::R16Float, 2u32),
                    Dtype::F32 => (wgpu::TextureFormat::R32Float, 4u32),
                    _ => return Err(format!("unsupported planar dtype {dt:?}")),
                };
                let tex = tex2d(&gpu.device, "exv.htj2k.plane", rw, rh, format, usage);
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    plane_bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(rw * bpp),
                        rows_per_image: Some(rh),
                    },
                    wgpu::Extent3d {
                        width: rw,
                        height: rh,
                        depth_or_array_layers: 1,
                    },
                );
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                out.push(ResidentTexture::new_plane(tex, view, rw, rh, format));
            }
            Ok(out)
        }
    }

    /// One-shot full-resolution decode to a fresh `Rgba16Float` `ResidentTexture`.
    pub fn decode_to_texture(gpu: &Gpu, payload: &[u8], budget: u64) -> Result<ResidentTexture, String> {
        decode_level(gpu, payload, 0, budget)
    }

    /// One-shot level decode to a fresh `Rgba16Float` `ResidentTexture`.
    pub fn decode_level(gpu: &Gpu, payload: &[u8], level: u32, budget: u64) -> Result<ResidentTexture, String> {
        let mut rctx = ResidentCtx::new(gpu)?;
        let tex = rctx.decode_level_fresh(gpu, payload, level, budget)?;
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        Ok(tex)
    }

    /// Texture readback helpers for parity tests.
    pub mod texture {
        use super::*;

        /// Read back a single-channel `ResidentTexture` (`R16Float`, `R32Float`, or `R32Uint`) as `Vec<f32>`.
        pub fn read_plane(
            gpu: &Gpu,
            plane: &ResidentTexture,
            w: u32,
            h: u32,
            format: wgpu::TextureFormat,
        ) -> Result<Vec<f32>, String> {
            let bpp = match format {
                wgpu::TextureFormat::R16Float => 2usize,
                wgpu::TextureFormat::R32Float | wgpu::TextureFormat::R32Uint => 4usize,
                other => return Err(format!("read_plane: unsupported format {other:?}")),
            };
            let row_bytes = (w as usize) * bpp;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
            let pitch = row_bytes.next_multiple_of(align);
            let total = (pitch * (h as usize)).max(4) as u64;
            let staging = new_buf(&gpu.device, "exv.read_plane", total, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let mut enc = encoder(&gpu.device, "exv.read_plane.enc");
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &plane.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(pitch as u32),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit([enc.finish()]);
            let raw = map_read_blocking(&gpu.device, &staging).map_err(|e| e.to_string())?;
            let mut out = Vec::with_capacity((w as usize) * (h as usize));
            for y in 0..h as usize {
                let row = &raw[y * pitch..y * pitch + row_bytes];
                match bpp {
                    2 => {
                        for &c in row.as_chunks::<2>().0 {
                            out.push(half_to_f32(u16::from_le_bytes(c)));
                        }
                    }
                    _ => {
                        for &c in row.as_chunks::<4>().0 {
                            out.push(f32::from_le_bytes(c));
                        }
                    }
                }
            }
            Ok(out)
        }

        /// Read back an `Rgba16Float` `ResidentTexture` as `Vec<f32>` (`w * h * 4` floats).
        pub fn read_texture(gpu: &Gpu, tex: &ResidentTexture, w: u32, h: u32) -> Result<Vec<f32>, String> {
            let row_bytes = (w as usize) * 8;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
            let pitch = row_bytes.next_multiple_of(align);
            let total = (pitch * (h as usize)).max(4) as u64;
            let staging = new_buf(&gpu.device, "exv.read_tex", total, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let mut enc = encoder(&gpu.device, "exv.read_tex.enc");
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(pitch as u32),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit([enc.finish()]);
            let raw = map_read_blocking(&gpu.device, &staging).map_err(|e| e.to_string())?;
            let mut out = Vec::with_capacity((w as usize) * (h as usize) * 4);
            for y in 0..h as usize {
                let row = &raw[y * pitch..y * pitch + row_bytes];
                for &c in row.as_chunks::<2>().0 {
                    out.push(half_to_f32(u16::from_le_bytes(c)));
                }
            }
            Ok(out)
        }
    }
}

/// Decode a **lossless HTJ2K** 1-frame `.exv` still on the GPU via the jph-wgpu backend (the third backend),
/// returning full-image per-channel byte planes identical to `exv_codec::image::read_still`'s
/// `Image.planes`.
#[cfg(feature = "htj2k-backend")]
pub fn decode_htj2k_still_planes_gpu(ctx: &GpuContext, data: &[u8], limits: &exv_format::DecodeLimits) -> Result<Vec<Vec<u8>>, String> {
    use exv_codec::{BlobKind, GpuKind, classify_blob};
    use jph_wgpu_rs::{Decoder, Gpu};

    let (header, body) = exv_video::still_frame_body(data).ok_or_else(|| "bad exv container".to_string())?;
    let reader = exv_codec::frame::FrameReader::open(body, header).map_err(|e| format!("bad exv frame: {e}"))?.with_limits(*limits);
    let profile = reader.profile().map_err(|e| e.to_string())?;
    if profile.gpu_kind() != Some(GpuKind::Htj2k) {
        return Err("not lossless HTJ2K; use CPU read_still".to_string());
    }
    let h = reader.header().clone();
    let (iw, ih) = (h.width as usize, h.height as usize);
    let nch = h.channels.len();
    if let Some(c) = h.channels.iter().find(|c| !exv_codec::htj2k_gpu_decodable(c.dtype) || !matches!(c.dtype.size(), 2 | 4)) {
        return Err(format!("HTJ2K GPU decode: {:?} components are CPU-decode only", c.dtype));
    }
    let total = h.frame_bytes().ok_or_else(|| "HTJ2K GPU decode: image size overflows".to_string())?;
    limits.check(total).map_err(|e| format!("HTJ2K GPU decode: {e}"))?;
    let mut planes: Vec<Vec<u8>> = h
        .channels
        .iter()
        .map(|c| {
            iw.checked_mul(ih)
                .and_then(|n| n.checked_mul(c.dtype.size()))
                .filter(|&n| n <= isize::MAX as usize / 2)
                .map(|n| vec![0u8; n])
                .ok_or_else(|| "HTJ2K GPU decode: image size overflows".to_string())
        })
        .collect::<Result<_, _>>()?;

    let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
    let budget = DECODE_BUDGET;
    let txn = h.tiles_x();
    for t in 0..reader.tile_count() {
        let (tx, ty) = (t % txn, t / txn);
        let (x, y, tw, th) = h.tile_rect(tx, ty);
        let blob = reader.tile_blob(t).map_err(|e| format!("tile {t}: {e}"))?;
        let payload = match classify_blob(blob).ok_or_else(|| "bad blob tag".to_string())? {
            BlobKind::Htj2k(p) => p,
            _ => return Err("non-HTJ2K tile in an HTJ2K image".to_string()),
        };
        let mut dec = Decoder::open(&gpu, payload.to_vec(), budget).map_err(|e| format!("open: {e}"))?;
        let comps = dec.read_full().map_err(|e| format!("read_full: {e}"))?;
        if comps.len() != nch {
            return Err(format!("tile {t}: {} components, expected {nch}", comps.len()));
        }
        let (twu, thu) = (tw as usize, th as usize);
        for (ci, samp) in comps.iter().enumerate() {
            if samp.len() != twu * thu {
                return Err(format!("tile {t} ch {ci}: {} samples, expected {}", samp.len(), twu * thu));
            }
            let esz = h.channels[ci].dtype.size();
            let plane = &mut planes[ci];
            for r in 0..thu {
                let dst_row = ((y as usize + r) * iw + x as usize) * esz;
                let src_row = r * twu;
                for c in 0..twu {
                    let v = samp[src_row + c];
                    let dst = dst_row + c * esz;
                    if esz == 2 {
                        plane[dst..dst + 2].copy_from_slice(&(v as i16).to_le_bytes());
                    } else {
                        plane[dst..dst + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(planes)
}

/// The resident, sampleable RGBA16F decode target: the zero-readback display handle the viewer binds directly.
#[cfg(feature = "htj2k-backend")]
pub use jph_wgpu_rs::ResidentTexture;

/// Atlas invalidation keys so a host crate can pass them to [`crate::ExvResidentAtlas::request`].
#[cfg(feature = "htj2k-backend")]
pub use jph_wgpu_rs::{ContentKey, DisplayKey};

/// Decode a **single-tile lossless HTJ2K** 1-frame `.exv` still to a **resident, sampleable RGBA16F texture**
/// on the shared device.
#[cfg(feature = "htj2k-backend")]
pub fn decode_htj2k_resident(ctx: &GpuContext, data: &[u8], level: u32, limits: &exv_format::DecodeLimits) -> Result<ResidentTexture, String> {
    use jph_wgpu_rs::{Gpu, decode_level, decode_to_texture};
    let payload = htj2k_frame_codestream(data, limits)?;
    let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
    let budget = DECODE_BUDGET;
    if level == 0 {
        decode_to_texture(&gpu, payload, budget).map_err(|e| format!("decode_to_texture: {e}"))
    } else {
        decode_level(&gpu, payload, level, budget).map_err(|e| format!("decode_level: {e}"))
    }
}

/// Extract a single-tile lossless-HTJ2K frame's multi-component codestream payload from a `.exv`.
#[cfg(feature = "htj2k-backend")]
fn htj2k_frame_codestream<'a>(data: &'a [u8], limits: &exv_format::DecodeLimits) -> Result<&'a [u8], String> {
    use exv_codec::{BlobKind, GpuKind, classify_blob};
    let (header, body) = exv_video::still_frame_body(data).ok_or_else(|| "bad exv container".to_string())?;
    let reader = exv_codec::frame::FrameReader::open(body, header).map_err(|e| format!("bad exv frame: {e}"))?.with_limits(*limits);
    let profile = reader.profile().map_err(|e| e.to_string())?;
    if profile.gpu_kind() != Some(GpuKind::Htj2k) {
        return Err("not lossless HTJ2K; resident decode unavailable".to_string());
    }
    if reader.tile_count() != 1 {
        return Err(format!(
            "resident decode needs a single-tile frame (tile == image); got {} tiles",
            reader.tile_count()
        ));
    }
    let blob = reader.tile_blob(0).map_err(|e| format!("tile 0: {e}"))?;
    match classify_blob(blob).ok_or_else(|| "bad blob tag".to_string())? {
        BlobKind::Htj2k(p) => Ok(p),
        _ => Err("non-HTJ2K tile in an HTJ2K image".to_string()),
    }
}

/// Persistent HTJ2K GPU decoder for the viewer / realtime path.
#[cfg(feature = "htj2k-backend")]
pub struct Htj2kGpuDecoder {
    gpu: jph_wgpu_rs::Gpu,
    ctx: jph_wgpu_rs::ResidentCtx,
}

#[cfg(feature = "htj2k-backend")]
impl Htj2kGpuDecoder {
    /// Build over exv-gpu's shared device.
    pub fn new(ctx: &GpuContext) -> Result<Self, String> {
        let gpu = jph_wgpu_rs::Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let rctx = jph_wgpu_rs::ResidentCtx::new(&gpu).map_err(|e| format!("resident ctx: {e}"))?;
        Ok(Self { gpu, ctx: rctx })
    }

    /// Warm resident decode of a single-tile lossless HTJ2K `.exv` frame at resolution `level` ->
    /// resident RGBA16F texture, reusing the cached GPU objects.
    pub fn decode(&mut self, data: &[u8], level: u32, limits: &exv_format::DecodeLimits) -> Result<ResidentTexture, String> {
        let payload = htj2k_frame_codestream(data, limits)?;
        let budget = DECODE_BUDGET;
        if level == 0 {
            self.ctx.decode_to_texture(&self.gpu, payload, budget).map_err(|e| format!("decode_to_texture: {e}"))
        } else {
            self.ctx.decode_level(&self.gpu, payload, level, budget).map_err(|e| format!("decode_level: {e}"))
        }
    }
}
