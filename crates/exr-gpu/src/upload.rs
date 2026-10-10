//! Packing + GPU upload of one layer.
//!
//! `exr-core` hands us PLANAR channels (one byte-plane each); a packed texture
//! needs them INTERLEAVED (`[c0px0, c1px0, …, c0px1, …]`). `pack_layer` does that
//! interleave on the CPU (the cheap, parallel-stage transform — same idea as the
//! viewer's `to_rgba_f32`, generalised to N channels and native type without the
//! f32 force-promote). `upload_layer` then creates the texture and uploads via
//! `queue.write_texture` (synchronous; no staging buffer / 256-byte alignment
//! needed — that rule only bites buffer↔texture copies, used in Phase D readback).

use exr_core::attr::PixelType;
use exr_core::buf::ensure_len;

use crate::format::{select_format, LayerFormat, UploadLayout};
use crate::layer::Layer;
use crate::source::SourceImage;
use crate::{GpuError, GpuImage, Result};

/// Interleave a layer's planar channels into one packed, native-endian buffer
/// sized by the pre-validated `layout` (`width * height * texel_size` bytes),
/// reusing `buf`'s allocation across calls.
///
/// V3: `buf` is grown-or-truncated to exactly `layout.cpu_size` via
/// `exr_core::compressor::ensure_len` (grow-only allocation-wise: truncating
/// a longer `buf` just adjusts `len`, no realloc) instead of always
/// allocating a fresh `vec![0u8; layout.cpu_size]`. Because a reused buffer
/// is NOT guaranteed to be zeroed (unlike a fresh `vec![0u8; ..]`), the
/// vector/unknown-layer pad path — which used to rely on the buffer's
/// initial zero state — now explicitly zeroes its pad slot instead.
///
/// Pure + testable (no GPU). The allocation length comes from `layout.cpu_size`
/// — the same checked value the texture and readback paths use — so packing can
/// never disagree with the upload size. For a 3→4 padded colour layer the 4th
/// component is filled with opaque (1.0 / `u32::MAX`); vector/unknown layers get
/// an explicit 0.
fn pack_layer_into(src: &SourceImage, layer: &Layer, fmt: &LayerFormat, layout: &UploadLayout, buf: &mut Vec<u8>) {
    ensure_len(buf, layout.cpu_size);
    let pixels = (layer.width as usize) * (layer.height as usize);
    let comps = fmt.components as usize;
    let bpc = fmt.bytes_per_component as usize;

    // Resolve the source channel for each real slot once. `check_layer_belongs`
    // has already proved every index resolves and every plane is long enough;
    // `filter_map` keeps this pure function total even if called directly.
    let planes: Vec<&[u8]> = layer
        .channels
        .iter()
        .filter_map(|&i| src.channels.get(i).map(|c| c.bytes.as_slice()))
        .collect();

    for px in 0..pixels {
        for (slot, plane) in planes.iter().enumerate() {
            let src_off = px * bpc;
            let dst_off = (px * comps + slot) * bpc;
            buf[dst_off..dst_off + bpc].copy_from_slice(&plane[src_off..src_off + bpc]);
        }
    }

    // Colour-layer 3→4 pad: write opaque into slot 3. Vector/unknown pad is
    // an explicit zero-write (see doc comment above — no longer relying on
    // the buffer's initial state now that it may be reused/dirty).
    if fmt.padded {
        let pad: &[u8] = if is_color_layer(src, layer) {
            match layer.pixel_type {
                PixelType::Half => &0x3C00u16.to_ne_bytes(), // half(1.0)
                PixelType::Float => &1.0f32.to_ne_bytes(),
                PixelType::Uint => &u32::MAX.to_ne_bytes(),
                _ => &[],
            }
        } else {
            &[]
        };
        for px in 0..pixels {
            let dst_off = (px * comps + 3) * bpc;
            if pad.len() == bpc {
                buf[dst_off..dst_off + bpc].copy_from_slice(pad);
            } else {
                buf[dst_off..dst_off + bpc].fill(0);
            }
        }
    }
}

/// Thin wrapper over [`pack_layer_into`] for callers that don't have a
/// buffer to reuse. See [`upload_layer`] / [`upload_layer_into`] for the
/// GPU-facing equivalents.
#[cfg(test)]
fn pack_layer(src: &SourceImage, layer: &Layer, fmt: &LayerFormat, layout: &UploadLayout) -> Vec<u8> {
    let mut buf = Vec::new();
    pack_layer_into(src, layer, fmt, layout, &mut buf);
    buf
}

/// A layer is "colour-like" if its first canonical channel is R/red (canonical
/// ordering puts R in slot 0), so its missing alpha pad should be opaque.
///
/// The layer-suffix split is [`exr_core::attr::channel_suffix`] — the SAME rule
/// `ChannelList::layers()` / `group_layers` use. exr-gpu must never carry its own
/// copy of it (a second `rfind('.')` here silently disagreed with the core rule).
fn is_color_layer(src: &SourceImage, layer: &Layer) -> bool {
    let Some(name) = layer.channels.first().and_then(|&i| src.channels.get(i)).map(|c| &c.name)
    else {
        return false;
    };
    let suffix = exr_core::attr::channel_suffix(name);
    matches!(suffix.to_ascii_lowercase().as_str(), "r" | "red")
}

/// Verify the layer actually describes THIS source, before anything indexes it.
///
/// A [`Layer`] is a plain index list; nothing in the type system ties it to the
/// [`Part`](crate::Part) it came from, so a caller can hand `Part::load_layer` a
/// layer belonging to a different part (or a stale one from a re-read). Every
/// index is checked against `src.channels` here, together with the per-channel
/// plane length, so the packing loop below can never index out of bounds or read
/// a short plane. Empty layers are rejected by `select_format`.
fn check_layer_belongs(src: &SourceImage, layer: &Layer) -> Result<()> {
    let pixels = (layer.width as usize) * (layer.height as usize);
    for &i in &layer.channels {
        let ch = src.channels.get(i).ok_or_else(|| {
            GpuError::FormatMismatch(format!(
                "layer \"{}\" refers to channel index {i}, but this part has {} channel(s) \
                 — the layer belongs to a different part",
                layer.label(),
                src.channels.len()
            ))
        })?;
        // One texture is one sample grid: every channel must be sampled like
        // the layer (a layer from another part, or a hand-built one, may not be).
        if (ch.width, ch.height, ch.sampling) != (layer.width, layer.height, layer.sampling) {
            return Err(GpuError::FormatMismatch(format!(
                "channel \"{}\" is {}x{} sampled {:?}, layer \"{}\" is {}x{} sampled {:?}",
                ch.name,
                ch.width,
                ch.height,
                ch.sampling,
                layer.label(),
                layer.width,
                layer.height,
                layer.sampling
            )));
        }
        let px_size = exr_core::misc::pixel_type_size(ch.pixel_type).map_err(|e| {
            GpuError::FormatMismatch(format!("channel \"{}\" has an invalid pixel type: {e}", ch.name))
        })?;
        let want = pixels.saturating_mul(px_size);
        if ch.bytes.len() < want {
            return Err(GpuError::FormatMismatch(format!(
                "channel \"{}\" has {} bytes but {}x{} of {:?} needs {want}",
                ch.name,
                ch.bytes.len(),
                layer.width,
                layer.height,
                ch.pixel_type
            )));
        }
    }
    Ok(())
}

/// Verify every channel in the layer shares `layer.pixel_type` (one texture has
/// one format). A mixed-type layer is a hard error — never a silent coercion.
/// Runs after [`check_layer_belongs`], so every index resolves.
fn check_uniform_type(src: &SourceImage, layer: &Layer) -> Result<()> {
    for &i in &layer.channels {
        let Some(ch) = src.channels.get(i) else { continue };
        if ch.pixel_type != layer.pixel_type {
            return Err(GpuError::FormatMismatch(format!(
                "layer \"{}\" mixes pixel types ({:?} and {:?}) — split into same-type layers",
                layer.label(),
                layer.pixel_type,
                ch.pixel_type
            )));
        }
    }
    Ok(())
}

/// Pack and upload one layer to a fresh `wgpu::Texture` with the given `usage`.
///
/// The caller owns the returned [`GpuImage`] (texture + view) and drops it to free
/// VRAM. `usage` must include at least `TEXTURE_BINDING` (shader read) or
/// `COPY_SRC` (readback); `COPY_DST` is added automatically for the upload.
///
/// Thin wrapper over [`upload_layer_into`] for callers that don't keep a
/// scratch buffer across calls (one-shot loads). See `upload_layer_into`'s
/// doc for the buffer-reuse contract.
pub(crate) fn upload_layer(
    src: &SourceImage,
    layer: &Layer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    usage: wgpu::TextureUsages,
) -> Result<GpuImage> {
    let mut scratch = Vec::new();
    upload_layer_into(src, layer, device, queue, usage, &mut scratch)
}

/// Same as [`upload_layer`], but packs into a caller-owned `scratch` buffer
/// instead of allocating a fresh one every call.
///
/// V3: `scratch` is grown-or-truncated (never reallocated once big enough --
/// see [`pack_layer_into`]) so a caller uploading many layers in sequence
/// (e.g. the viewer stepping through a frame sequence) pays for at most one
/// realloc across the whole run, not one per upload.
pub(crate) fn upload_layer_into(
    src: &SourceImage,
    layer: &Layer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    usage: wgpu::TextureUsages,
    scratch: &mut Vec<u8>,
) -> Result<GpuImage> {
    check_layer_belongs(src, layer)?;
    check_uniform_type(src, layer)?;
    let fmt = select_format(layer.len(), layer.pixel_type)?;

    // Validate dimensions, device limit, and byte sizing BEFORE creating any wgpu
    // resource — an over-limit / overflowing image returns `GpuError::TooLarge`
    // here rather than tripping a raw wgpu validation panic in `create_texture`.
    let (w, h) = (layer.width.max(1), layer.height.max(1));
    let layout = UploadLayout::compute(w, h, fmt.texel_size(), device.limits().max_texture_dimension_2d)?;

    pack_layer_into(src, layer, &fmt, &layout, scratch);
    let buf = &scratch[..layout.cpu_size];

    let size = wgpu::Extent3d {
        width: layout.width,
        height: layout.height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("exr_gpu.layer.{}", layer.label())),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt.format,
        // The upload always writes the texture; the caller's intent (sample /
        // storage / readback) is OR-ed in by the public API.
        usage: usage | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // `write_texture` accepts a tightly-packed source (no 256-byte row rounding).
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        buf,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(layout.row_bytes),
            rows_per_image: Some(layout.height),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Channel names per GPU component; empty string marks the pad slot.
    let mut channel_names: Vec<String> = layer
        .channels
        .iter()
        .filter_map(|&i| src.channels.get(i).map(|c| c.name.clone()))
        .collect();
    while channel_names.len() < fmt.components as usize {
        channel_names.push(String::new());
    }

    log::debug!(
        "exr-gpu: uploaded layer \"{}\" {w}x{h} fmt={:?} channels=[{}]",
        layer.label(),
        fmt.format,
        channel_names.join(", ")
    );

    Ok(GpuImage {
        texture,
        view,
        format: fmt.format,
        width: layout.width,
        height: layout.height,
        layer_name: layer.name.clone(),
        channel_names,
        texel_size: layout.texel_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceChannel;

    /// 2x1 image, channel `name` planar bytes = the given per-pixel u16 (HALF).
    fn half_chan(name: &str, p0: u16, p1: u16) -> SourceChannel {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&p0.to_ne_bytes());
        bytes.extend_from_slice(&p1.to_ne_bytes());
        SourceChannel { name: name.into(), pixel_type: PixelType::Half, bytes, width: 2, height: 1, sampling: (1, 1) }
    }

    #[test]
    fn packs_rgb_interleaved_with_opaque_pad() {
        // R=[1,2] G=[3,4] B=[5,6] over 2 pixels → RGBA16 interleaved, A=half(1.0).
        let src = SourceImage {
            width: 2,
            height: 1,
            channels: vec![
                half_chan("R", 1, 2),
                half_chan("G", 3, 4),
                half_chan("B", 5, 6),
            ],
        };
        let layer = Layer {
            name: String::new(),
            channels: vec![0, 1, 2], // already canonical R,G,B
            pixel_type: PixelType::Half,
            width: 2,
            height: 1,
            sampling: (1, 1),
        };
        let fmt = select_format(3, PixelType::Half).unwrap();
        let layout = UploadLayout::compute(2, 1, fmt.texel_size(), 8192).unwrap();
        let buf = pack_layer(&src, &layer, &fmt, &layout);

        // 2 px * 4 comp * 2 B = 16 bytes. Read back as u16.
        let u16s: Vec<u16> = buf
            .as_chunks::<2>().0.iter()
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(u16s, [1, 3, 5, 0x3C00, 2, 4, 6, 0x3C00]); // px0 RGBA, px1 RGBA
    }

    /// V3 regression: a reused, DIRTY `scratch` buffer must not leak its old
    /// contents into a vector/unknown layer's zero pad slot.
    ///
    /// Mutation check: reverting `pack_layer_into`'s vector/unknown branch to
    /// the old "rely on initial zero" behavior (i.e. skipping the explicit
    /// `fill(0)` when `buf` is not already zeroed) makes this go red --
    /// confirmed by temporarily replacing the `buf[..].fill(0)` call with a
    /// no-op and re-running: the pad slot then reads back the garbage byte
    /// `0xAA` instead of 0, and the assertion fails. Reverted after
    /// confirming.
    #[test]
    fn vector_layer_pad_is_zero_even_with_a_dirty_reused_buffer() {
        // X=[1,2] Y=[3,4] Z=[5,6]: non-color channel names (no R/G/B suffix),
        // so `is_color_layer` is false and the pad must be an explicit zero,
        // not the opaque-alpha value.
        let src = SourceImage {
            width: 2,
            height: 1,
            channels: vec![
                half_chan("X", 1, 2),
                half_chan("Y", 3, 4),
                half_chan("Z", 5, 6),
            ],
        };
        let layer = Layer {
            name: String::new(),
            channels: vec![0, 1, 2],
            pixel_type: PixelType::Half,
            width: 2,
            height: 1,
            sampling: (1, 1),
        };
        let fmt = select_format(3, PixelType::Half).unwrap();
        let layout = UploadLayout::compute(2, 1, fmt.texel_size(), 8192).unwrap();
        assert!(fmt.padded);
        assert!(!is_color_layer(&src, &layer));

        // Pre-fill the scratch buffer with garbage, oversized (simulating
        // reuse after a bigger previous upload), before packing into it.
        let mut buf = vec![0xAAu8; layout.cpu_size + 64];
        pack_layer_into(&src, &layer, &fmt, &layout, &mut buf);

        let u16s: Vec<u16> = buf[..layout.cpu_size]
            .as_chunks::<2>().0.iter()
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(u16s, [1, 3, 5, 0, 2, 4, 6, 0]); // pad slot is 0, not 0xAAAA
    }

    /// Native-endian bytes of a HALF plane from raw u16 bit patterns.
    fn half_bytes(vals: &[u16]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    /// Full pipeline on a real device: upload a layer, read it back, and assert the
    /// bytes equal what `pack_layer` produced. This is THE end-to-end verification
    /// of upload + GPU copy + readback (the byte-level `pack_layer` test covers the
    /// interleave; this covers the wgpu texture/readback path). Gated on a GPU.
    #[test]
    #[ignore = "requires a GPU"]
    fn upload_then_readback_roundtrips() {
        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let Some(adapter) = adapters.into_iter().next() else {
            eprintln!("skip: no GPU adapter");
            return;
        };
        let (device, queue) =
            wgpu_info::request_max_device_blocking(&adapter, wgpu::Features::empty())
                .expect("device");

        // 2x2 RGB half image, planar per channel.
        let src = SourceImage {
            width: 2,
            height: 2,
            channels: vec![
                SourceChannel { name: "R".into(), pixel_type: PixelType::Half, bytes: half_bytes(&[1, 2, 3, 4]), width: 2, height: 2, sampling: (1, 1) },
                SourceChannel { name: "G".into(), pixel_type: PixelType::Half, bytes: half_bytes(&[5, 6, 7, 8]), width: 2, height: 2, sampling: (1, 1) },
                SourceChannel { name: "B".into(), pixel_type: PixelType::Half, bytes: half_bytes(&[9, 10, 11, 12]), width: 2, height: 2, sampling: (1, 1) },
            ],
        };
        let layer = Layer { name: String::new(), channels: vec![0, 1, 2], pixel_type: PixelType::Half, width: 2, height: 2, sampling: (1, 1) };

        let img = upload_layer(
            &src,
            &layer,
            &device,
            &queue,
            wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .expect("upload");
        assert_eq!(img.format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(img.texel_size, 8); // RGBA * 2 bytes

        let got = img.read_back(&device, &queue).expect("readback");
        let fmt = select_format(3, PixelType::Half).unwrap();
        let layout = UploadLayout::compute(2, 2, fmt.texel_size(), 8192).unwrap();
        let want = pack_layer(&src, &layer, &fmt, &layout);
        assert_eq!(got, want, "readback must equal the packed upload bytes");
    }

    #[test]
    fn mixed_type_layer_is_rejected() {
        let src = SourceImage {
            width: 1,
            height: 1,
            channels: vec![
                SourceChannel { name: "X".into(), pixel_type: PixelType::Float, bytes: vec![0; 4], width: 1, height: 1, sampling: (1, 1) },
                SourceChannel { name: "Y".into(), pixel_type: PixelType::Half, bytes: vec![0; 2], width: 1, height: 1, sampling: (1, 1) },
            ],
        };
        let layer = Layer { name: "m".into(), channels: vec![0, 1], pixel_type: PixelType::Float, width: 1, height: 1, sampling: (1, 1) };
        assert!(matches!(check_uniform_type(&src, &layer), Err(GpuError::FormatMismatch(_))));
    }

    /// A `Layer` from a DIFFERENT part (indices past this source's channel
    /// list) must be a clean error, not an out-of-bounds index.
    #[test]
    fn foreign_layer_is_rejected_not_indexed() {
        let src = SourceImage {
            width: 1,
            height: 1,
            channels: vec![SourceChannel {
                name: "R".into(),
                pixel_type: PixelType::Half,
                bytes: vec![0; 2],
                width: 1,
                height: 1,
                sampling: (1, 1),
            }],
        };
        let foreign = Layer {
            name: String::new(),
            channels: vec![0, 7], // 7 does not exist here
            pixel_type: PixelType::Half,
            width: 1,
            height: 1,
            sampling: (1, 1),
        };
        assert!(matches!(
            check_layer_belongs(&src, &foreign),
            Err(GpuError::FormatMismatch(_))
        ));
        // And the other helpers stay total on the same input (no panic).
        assert!(check_uniform_type(&src, &foreign).is_ok());
        assert!(is_color_layer(&src, &foreign));
    }

    /// A plane shorter than `width * height * sample_size` is a hard error — the
    /// packing loop must never read past a truncated channel buffer.
    #[test]
    fn short_channel_plane_is_rejected() {
        let src = SourceImage {
            width: 4,
            height: 4,
            channels: vec![SourceChannel {
                name: "R".into(),
                pixel_type: PixelType::Half,
                bytes: vec![0; 2], // needs 16 * 2 = 32
                width: 4,
                height: 4,
                sampling: (1, 1),
            }],
        };
        let layer =
            Layer { name: String::new(), channels: vec![0], pixel_type: PixelType::Half, width: 4, height: 4, sampling: (1, 1) };
        assert!(matches!(
            check_layer_belongs(&src, &layer),
            Err(GpuError::FormatMismatch(_))
        ));
    }

    /// The layer-suffix rule is exr-core's, not a local copy: a dotted name's
    /// suffix decides "colour-like", and an undotted `R` works too.
    #[test]
    fn color_layer_uses_the_shared_suffix_rule() {
        let src = SourceImage {
            width: 1,
            height: 1,
            channels: vec![
                SourceChannel { name: "diffuse.R".into(), pixel_type: PixelType::Half, bytes: vec![0; 2], width: 1, height: 1, sampling: (1, 1) },
                SourceChannel { name: "N.x".into(), pixel_type: PixelType::Half, bytes: vec![0; 2], width: 1, height: 1, sampling: (1, 1) },
                SourceChannel { name: "R".into(), pixel_type: PixelType::Half, bytes: vec![0; 2], width: 1, height: 1, sampling: (1, 1) },
            ],
        };
        let mk = |i: usize| Layer {
            name: String::new(),
            channels: vec![i],
            pixel_type: PixelType::Half,
            width: 1,
            height: 1,
            sampling: (1, 1),
        };
        assert!(is_color_layer(&src, &mk(0)), "diffuse.R → suffix R");
        assert!(!is_color_layer(&src, &mk(1)), "N.x → suffix x");
        assert!(is_color_layer(&src, &mk(2)), "bare R");
    }
}
