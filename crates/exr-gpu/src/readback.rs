//! Texture → CPU readback (the "process on GPU, get the pixels back" path).
//!
//! `copy_texture_to_buffer` REQUIRES `bytes_per_row` to be a multiple of
//! `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256) — unlike `write_texture` (the upload),
//! which accepts a tightly-packed source. So we copy into a row-padded staging
//! buffer, map it, then UN-pad rows back into a tight `width * height * texel_size`
//! buffer. Blocks on `device.poll(PollType::Wait { .. })` until the copy + map
//! callback complete (wgpu 29 API).

use std::sync::mpsc;

use exr_core::buf::ensure_len;

use crate::format::UploadLayout;
use crate::{GpuError, GpuImage, Result};

/// Copy `img`'s texture back to tightly-packed native-endian CPU bytes.
///
/// The texture must have been created with `COPY_SRC` usage (see
/// [`crate::Part::load_layer`]). Returns `width * height * texel_size` bytes,
/// row-major, no padding — the same layout `pack_layer` produced on upload.
///
/// Thin wrapper over [`read_back_into`] for callers that don't keep a
/// buffer across calls.
pub(crate) fn read_back(img: &GpuImage, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    read_back_into(img, device, queue, &mut out)?;
    Ok(out)
}

/// Same as [`read_back`], but un-pads into a caller-owned `out` buffer
/// instead of allocating a fresh one every call.
///
/// V3: `out` is grown-or-truncated via `exr_core::compressor::ensure_len`
/// (never reallocated once big enough) -- useful for a caller reading back
/// the same-sized texture repeatedly (e.g. a compute round-trip per frame).
pub(crate) fn read_back_into(
    img: &GpuImage,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    out: &mut Vec<u8>,
) -> Result<()> {
    // Reuse the same checked layout the upload used (dimensions already validated
    // when the texture was created, so this only re-derives the byte sizing).
    let layout = UploadLayout::for_texture(img.width, img.height, img.texel_size)?;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
    let unpadded_bpr = layout.row_bytes;
    // Round the row stride up to the 256-byte copy alignment.
    let padded_bpr = unpadded_bpr.div_ceil(align) * align;
    let buf_size = u64::from(padded_bpr) * u64::from(img.height);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exr_gpu.readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("exr_gpu.readback.encoder"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &img.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(img.height),
            },
        },
        wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);

    map_read_blocking(device, &staging)?;

    // Un-pad: copy `unpadded_bpr` bytes from each `padded_bpr`-strided row.
    let view = staging.slice(..).get_mapped_range()?; // wgpu 30: fallible
    let row = unpadded_bpr as usize;
    ensure_len(out, layout.cpu_size);
    for y in 0..img.height as usize {
        let src = y * padded_bpr as usize;
        let dst = y * row;
        out[dst..dst + row].copy_from_slice(&view[src..src + row]);
    }
    drop(view);
    staging.unmap();
    Ok(())
}

/// Map a `MAP_READ` staging buffer for reading and BLOCK until the GPU copy and
/// the map callback have completed (wgpu 29 `device.poll(PollType::Wait)`). Shared
/// by the texture readback (here) and the storage-buffer readback (`buffer.rs`) so
/// the map+poll dance lives in ONE place.
pub(crate) fn map_read_blocking(device: &wgpu::Device, staging: &wgpu::Buffer) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(|e| GpuError::Gpu(format!("device.poll: {e:?}")))?;
    match rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(GpuError::Gpu(format!("map_async failed: {e:?}"))),
        Err(e) => Err(GpuError::Gpu(format!("readback channel closed: {e}"))),
    }
}
