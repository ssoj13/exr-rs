//! `testkit.rs` — shared fixtures of the exv-gpu in-crate GPU gates (cfg(test) only).
//!
//! **Why:** the gates used to carry five hand-rolled f32->f16 converters, six near-identical synthetic image
//! builders, two RGBA readbacks and four copies of the compose max-abs loop. One builder per concept here keeps
//! the fixtures byte-identical across gates and gives a single place to change them.

use crate::prelude::*;
#[cfg(feature = "htj2k-backend")]
use exv_codec::image::Image;
use exv_format::half::f32_to_half;
use exv_format::{Channel, Header};
use exv_gpu_kernels::{GpuContext, map_read_blocking, staging};

/// f32 -> f16 little-endian bytes (round-to-nearest-even): the codec's own conversion, so a ramp fed to an
/// encoder round-trips exactly at q=0.
pub(crate) fn f16_le(v: f32) -> [u8; 2] {
    f32_to_half(v).to_le_bytes()
}

/// Deterministic xorshift64 (fixtures must be reproducible; no `rand` dependency).
pub(crate) fn xorshift(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

/// A single-frame [`Header`] over `chans` (`(name, dtype)` in channel order); `params` are the codec params.
pub(crate) fn header(w: u32, h: u32, tile_w: u32, tile_h: u32, codec_id: u32, params: Vec<u8>, chans: &[(&str, Dtype)]) -> Header {
    Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w,
        tile_h,
        codec_id,
        codec_params: params,
        channels: chans.iter().map(|&(n, dtype)| Channel { name: n.into(), dtype }).collect(),
    }
}

/// Walk every `(tile, channel)` sub-blob of a 1-frame `.exv` and hand its classification to `f` (a bad
/// container yields no calls). One walker for every "which tags does this fixture really contain" gate.
pub(crate) fn for_blobs(data: &[u8], mut f: impl FnMut(Option<BlobKind<'_>>)) {
    let Some((header, body)) = exv_video::still_frame_body(data) else { return };
    let Ok(reader) = exv_codec::frame::FrameReader::open(body, header) else { return };
    for t in 0..reader.tile_count() {
        for ci in 0..reader.header().channels.len() {
            if let Ok(blob) = reader.tile_channel_blob(t, ci) {
                f(classify_blob(blob));
            }
        }
    }
}

/// Read an `Rgba32Float` texture back to a flat `w*h*4` f32 vec (handles the 256-byte `bytes_per_row`
/// alignment `copy_texture_to_buffer` requires). Waits through the shared non-blocking map path.
pub(crate) fn readback_rgba32f(ctx: &GpuContext, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<f32> {
    let unpadded = w * 16; // RGBA * f32 bytes/pixel
    let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let rb = staging(&ctx.device, "exv.rgba.rb", (padded as u64 * h as u64).max(4));
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("exv.rgba.rb.enc") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let bytes = map_read_blocking(&ctx.device, &rb).expect("map rgba readback");
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let row = &bytes[(y * padded) as usize..(y * padded + unpadded) as usize];
        out.extend(row.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
    }
    out
}

/// One F16 plane of the `(x*3 + y*5 + seed*97 + off) & 0x3ff` pattern (small half bit patterns: finite, lossless).
#[cfg(feature = "htj2k-backend")]
pub(crate) fn f16_plane(w: u32, h: u32, seed: u32, off: u32) -> Vec<u8> {
    (0..w as usize * h as usize)
        .flat_map(|i| {
            let (x, y) = (i as u32 % w, i as u32 / w);
            (((x.wrapping_mul(3) + y.wrapping_mul(5) + seed.wrapping_mul(97) + off) & 0x3ff) as u16).to_le_bytes()
        })
        .collect()
}

/// All-F16 image named `names` (plane `k` = `f16_plane(w, h, k, off)`), lossless-HTJ2K header (`codec_id 6`).
#[cfg(feature = "htj2k-backend")]
pub(crate) fn f16_image(w: u32, h: u32, tile_w: u32, tile_h: u32, names: &[&str], off: u32) -> Image {
    let chans: Vec<(&str, Dtype)> = names.iter().map(|&n| (n, Dtype::F16)).collect();
    let planes = (0..names.len() as u32).map(|k| f16_plane(w, h, k, off)).collect();
    Image::new(header(w, h, tile_w, tile_h, 6, Vec::new(), &chans), planes).expect("f16 image")
}

/// `img` with every F16 plane of tile `(tx, ty)` perturbed (low byte XOR 5: stays finite): exactly that tile differs,
/// so a clip of pokes has SKIP tiles everywhere else.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn poke_tile(img: &Image, tx: u32, ty: u32) -> Image {
    let h = &img.header;
    let (x0, y0, tw, th) = h.tile_rect(tx, ty);
    let mut planes = img.planes.clone();
    for p in &mut planes {
        for y in y0..y0 + th {
            for x in x0..x0 + tw {
                p[((y * h.width + x) * 2) as usize] ^= 5;
            }
        }
    }
    Image::new(h.clone(), planes).expect("poked image")
}

/// Assert the resident tile `(rx, ry)` of `atlas` equals `cpu` (full-size F16 planes of the same clip) in its first
/// `nch` channels, EVERY pixel (a resident plane reads back as the exact half value).
#[cfg(feature = "htj2k-backend")]
pub(crate) fn assert_tile(
    gpu: &jph_wgpu_rs::Gpu,
    atlas: &crate::ExvResidentAtlas,
    cpu: &[Vec<u8>],
    hdr: &Header,
    (rx, ry): (u32, u32),
    nch: usize,
    what: &str,
) {
    let (x0, y0, tw, th) = hdr.tile_rect(rx, ry);
    let planes = atlas.get(None, rx, ry).unwrap_or_else(|| panic!("{what}: tile ({rx},{ry}) is not resident"));
    for (ch, cpu_ch) in cpu.iter().enumerate().take(nch) {
        let plane = planes.plane(ch).unwrap_or_else(|| panic!("{what}: tile ({rx},{ry}) ch{ch} is not a resident plane"));
        let got = jph_wgpu_rs::texture::read_plane(gpu, plane, tw, th, plane.tex.format()).expect("read plane");
        for ty in 0..th {
            for tx in 0..tw {
                let i = ((y0 + ty) * hdr.width + x0 + tx) as usize * 2;
                let want = exv_format::half::half_to_f32(u16::from_le_bytes([cpu_ch[i], cpu_ch[i + 1]]));
                assert_eq!(got[(ty * tw + tx) as usize], want, "{what}: tile ({rx},{ry}) ch{ch} px({tx},{ty})");
            }
        }
    }
}

/// `R,G,B,A,Y` F16, tile 256 — the >4-channel N-plane fixture.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn synth_f16(w: u32, h: u32) -> Image {
    f16_image(w, h, 256, 256, &["R", "G", "B", "A", "Y"], 0)
}

/// Single-tile (tile == image) F16 `R,G,B`: the one tile blob is the whole-frame codestream.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn synth_f16_single_tile(w: u32, h: u32) -> Image {
    f16_image(w, h, w, h, &["R", "G", "B"], 0)
}

/// `R,G,B,A:F16 + Z:F32` (finite depth values so the reversible codestream round-trips bit-exact), tile 256.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn synth_f16_f32(w: u32, h: u32) -> Image {
    let z: Vec<u8> = (0..w as usize * h as usize)
        .flat_map(|i| {
            let (x, y) = (i as u32 % w, i as u32 / w);
            (((x % 97) as f32) * 0.5 + ((y % 13) as f32) * 3.25 + 1.0).to_le_bytes()
        })
        .collect();
    let chans = [("R", Dtype::F16), ("G", Dtype::F16), ("B", Dtype::F16), ("A", Dtype::F16), ("Z", Dtype::F32)];
    let planes = vec![f16_plane(w, h, 0, 0), f16_plane(w, h, 1, 0), f16_plane(w, h, 2, 0), f16_plane(w, h, 3, 0), z];
    Image::new(header(w, h, 256, 256, 6, Vec::new(), &chans), planes).expect("f16+f32 image")
}

/// Mirrors the CLI's `synth_image` (R,G,B:F16 + Z:F32 + id:U32), tile 256: the 32-bit capability-boundary fixture.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn synth_mc(w: u32, h: u32) -> Image {
    let n = w as usize * h as usize;
    let z: Vec<u8> = (0..n).flat_map(|i| ((i as u32 % w) as f32 * 0.1).to_le_bytes()).collect();
    let id: Vec<u8> = (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect();
    let chans = [("R", Dtype::F16), ("G", Dtype::F16), ("B", Dtype::F16), ("Z", Dtype::F32), ("id", Dtype::U32)];
    let planes = vec![f16_plane(w, h, 0, 0), f16_plane(w, h, 1, 0), f16_plane(w, h, 2, 0), z, id];
    Image::new(header(w, h, 256, 256, 6, Vec::new(), &chans), planes).expect("mc image")
}

/// Max abs difference of a composed `Rgba32Float` readback `got` vs the CPU planes: components 0,1,2 -> R,G,B
/// (F16 planes, `half_to_f32`) and the unmapped A slot == constant 1.0. Lossless F16 -> f32 is exact => 0.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn max_abs_rgb(got: &[f32], planes: &[Vec<u8>], w: u32, h: u32) -> f32 {
    let mut max_abs = 0f32;
    for idx in 0..w as usize * h as usize {
        let o = idx * 4;
        for (c, plane) in planes.iter().take(3).enumerate() {
            let bits = u16::from_le_bytes([plane[idx * 2], plane[idx * 2 + 1]]);
            max_abs = max_abs.max((got[o + c] - exv_format::half::half_to_f32(bits)).abs());
        }
        max_abs = max_abs.max((got[o + 3] - 1.0).abs());
    }
    max_abs
}

/// Resident atlas on `ctx` with the gates' standard budgets (512 MiB resident, [`crate::DECODE_BUDGET`] per decode) and
/// the jph-wgpu HTJ2K tile decode FORCED: the gates exist to pin the GPU decoders bit-exact (the default routing sends
/// full-resolution HTJ2K to the CPU, see `ExvResidentAtlas::htj2k_on_gpu`).
#[cfg(feature = "htj2k-backend")]
pub(crate) fn new_atlas(ctx: &GpuContext) -> crate::ExvResidentAtlas {
    crate::ExvResidentAtlas::new(ctx, 512 << 20, crate::DECODE_BUDGET).expect("atlas").with_htj2k_gpu(true)
}

/// Every `(rx, ry)` of an `nx` x `ny` tile grid, row-major.
#[cfg(feature = "htj2k-backend")]
pub(crate) fn tile_grid(nx: u32, ny: u32) -> Vec<(u32, u32)> {
    (0..ny).flat_map(|ry| (0..nx).map(move |rx| (rx, ry))).collect()
}
