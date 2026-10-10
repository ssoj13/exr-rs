//! Planar storage-buffer escape hatch — all of a part's channels in ONE
//! `wgpu::Buffer` with a channel-offset table, for compute consumers that want
//! every AOV addressable by index (or that exceed the per-stage sampled-texture
//! limit).
//!
//! This is the cheapest upload: `exr-core` already hands us PLANAR channels, so we
//! concatenate the planes verbatim (NO interleave, NO RGBA pad, NO 256-byte row
//! rounding — those only apply to textures). Each channel starts on a 4-byte
//! boundary (`wgpu::COPY_BUFFER_ALIGNMENT`) so buffer↔buffer copies and aligned
//! shader access are valid. The buffer is a typed byte heap; `channels` tells a
//! consumer where each channel lives and how to interpret it.

use exr_core::attr::PixelType;
use exr_core::buf::ensure_len;

use crate::error::{GpuError, Result};
use crate::readback;
use crate::source::SourceImage;

/// Where one channel lives inside a [`GpuBuffer`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferChannel {
    /// Full EXR channel name (e.g. `"diffuse.R"`).
    pub name: String,
    /// Byte offset of the channel's first sample within the buffer (4-aligned).
    pub offset: u64,
    /// Channel length in bytes (`width * height * sample_size`, unpadded).
    pub len: u64,
    /// Sample type — how a consumer must read the bytes at `offset`.
    pub pixel_type: PixelType,
    /// Samples per row of this channel (the part width, or fewer when subsampled).
    pub width: u32,
    /// Sample rows of this channel.
    pub height: u32,
}

/// A part's channels uploaded planar into one GPU buffer, with an offset table.
pub struct GpuBuffer {
    /// The buffer holding all channels back-to-back (each 4-aligned).
    pub buffer: wgpu::Buffer,
    /// Per-channel location/type table, in source (header) order.
    pub channels: Vec<BufferChannel>,
    /// Data-window dimensions the planes are laid out over.
    pub width: u32,
    pub height: u32,
    /// Total buffer size in bytes (multiple of 4).
    pub size: u64,
}

impl GpuBuffer {
    /// Look up a channel's location by full name.
    pub fn channel(&self, name: &str) -> Option<&BufferChannel> {
        self.channels.iter().find(|c| c.name == name)
    }

    /// Copy the whole buffer back to CPU bytes (the buffer must have been created
    /// with `COPY_SRC`). Index it with the [`BufferChannel`] offsets. Blocks until
    /// the GPU copy + mapping complete.
    ///
    /// Thin wrapper over [`GpuBuffer::read_back_into`] for callers that don't
    /// keep a buffer across calls.
    pub fn read_back(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.read_back_into(device, queue, &mut out)?;
        Ok(out)
    }

    /// Same as [`GpuBuffer::read_back`], but copies into a caller-owned `out`
    /// buffer instead of allocating a fresh one every call.
    ///
    /// V3: `out` is grown-or-truncated via `exr_core::compressor::ensure_len`
    /// (never reallocated once big enough) and filled via `copy_from_slice`
    /// instead of `view.to_vec()`'s implicit fresh allocation -- useful for a
    /// caller reading back the same-sized buffer repeatedly.
    pub fn read_back_into(&self, device: &wgpu::Device, queue: &wgpu::Queue, out: &mut Vec<u8>) -> Result<()> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exr_gpu.buffer.readback"),
            size: self.size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exr_gpu.buffer.readback.encoder"),
        });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size);
        queue.submit([enc.finish()]);

        readback::map_read_blocking(device, &staging)?;
        let view = staging.slice(..).get_mapped_range()?; // wgpu 30: fallible
        ensure_len(out, self.size as usize);
        out.copy_from_slice(&view);
        drop(view);
        staging.unmap();
        Ok(())
    }
}

/// Concatenate a source's channels planar (4-aligned) into one buffer + offset
/// table, and upload. `usage` should include `STORAGE` (compute) and/or `COPY_SRC`
/// (readback); `COPY_DST` is added for the upload.
pub(crate) fn load_buffer(
    src: &SourceImage,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    usage: wgpu::BufferUsages,
) -> Result<GpuBuffer> {
    let align = wgpu::COPY_BUFFER_ALIGNMENT as usize; // 4

    let mut data: Vec<u8> = Vec::new();
    let mut channels = Vec::with_capacity(src.channels.len());
    for ch in &src.channels {
        let offset = data.len() as u64;
        data.extend_from_slice(&ch.bytes);
        channels.push(BufferChannel {
            name: ch.name.clone(),
            offset,
            len: ch.bytes.len() as u64,
            pixel_type: ch.pixel_type,
            width: ch.width,
            height: ch.height,
        });
        // Pad so the NEXT channel starts 4-aligned (and the total stays a multiple
        // of 4, required by copy_buffer_to_buffer).
        while !data.len().is_multiple_of(align) {
            data.push(0);
        }
    }

    let size = (data.len() as u64).max(align as u64);
    // Check the device limits BEFORE creating the buffer: a many-AOV 8K part can
    // exceed `max_buffer_size` (and, bound as STORAGE, the smaller
    // `max_storage_buffer_binding_size`), which is a wgpu validation failure —
    // i.e. a panic through the default uncaptured-error handler. Same rule as
    // `UploadLayout` on the texture path: a clean `TooLarge` instead.
    check_buffer_limits(
        size,
        usage.contains(wgpu::BufferUsages::STORAGE),
        device.limits().max_buffer_size,
        device.limits().max_storage_buffer_binding_size,
    )?;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exr_gpu.buffer"),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &data);

    log::debug!(
        "exr-gpu: uploaded planar buffer {}x{} {} bytes, {} channel(s)",
        src.width,
        src.height,
        size,
        channels.len()
    );

    Ok(GpuBuffer { buffer, channels, width: src.width, height: src.height, size })
}

/// Validate a planar-buffer allocation against the device's buffer limits.
///
/// Pure (no `Device`) so the limit arithmetic is unit-testable with synthetic
/// maximums — the buffer-path twin of `UploadLayout::compute`. When the buffer
/// will be bound as `STORAGE`, the binding limit applies as well and is usually
/// the smaller of the two.
fn check_buffer_limits(size: u64, storage: bool, max_buffer: u64, max_storage_binding: u64) -> Result<()> {
    if size > max_buffer {
        return Err(GpuError::TooLarge(format!(
            "planar buffer needs {size} bytes, over the device max_buffer_size ({max_buffer})"
        )));
    }
    if storage && size > max_storage_binding {
        return Err(GpuError::TooLarge(format!(
            "planar buffer needs {size} bytes, over the device \
             max_storage_buffer_binding_size ({max_storage_binding})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceChannel;

    #[test]
    fn buffer_limits_reject_over_sized_allocations() {
        // Fits both limits.
        assert!(check_buffer_limits(1 << 20, true, 1 << 30, 1 << 27).is_ok());
        // Over the storage BINDING limit only — rejected when bound as storage,
        // accepted when it is a plain copy/readback buffer.
        assert!(matches!(
            check_buffer_limits(1 << 28, true, 1 << 30, 1 << 27),
            Err(GpuError::TooLarge(_))
        ));
        assert!(check_buffer_limits(1 << 28, false, 1 << 30, 1 << 27).is_ok());
        // Over the absolute buffer limit — rejected either way.
        assert!(matches!(
            check_buffer_limits(1 << 31, false, 1 << 30, 1 << 27),
            Err(GpuError::TooLarge(_))
        ));
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn buffer_upload_readback_roundtrips() {
        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let Some(adapter) = adapters.into_iter().next() else {
            eprintln!("skip: no GPU adapter");
            return;
        };
        let (device, queue) =
            wgpu_info::request_max_device_blocking(&adapter, wgpu::Features::empty())
                .expect("device");

        // Two channels of different types/sizes to exercise 4-alignment padding.
        let r: Vec<u8> = [1u16, 2, 3].iter().flat_map(|v| v.to_ne_bytes()).collect(); // 6 bytes (HALF)
        let z: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_ne_bytes()).collect(); // 12 bytes
        let src = SourceImage {
            width: 3,
            height: 1,
            channels: vec![
                SourceChannel { name: "R".into(), pixel_type: PixelType::Half, bytes: r.clone(), width: 3, height: 1, sampling: (1, 1) },
                SourceChannel { name: "Z".into(), pixel_type: PixelType::Float, bytes: z.clone(), width: 3, height: 1, sampling: (1, 1) },
            ],
        };

        let gb = load_buffer(
            &src,
            &device,
            &queue,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
        .expect("load_buffer");

        // R at 0 (6 bytes), padded to 8; Z at 8 (12 bytes).
        assert_eq!(gb.channel("R").unwrap().offset, 0);
        assert_eq!(gb.channel("Z").unwrap().offset, 8);
        assert_eq!(gb.size % 4, 0);

        let bytes = gb.read_back(&device, &queue).expect("readback");
        let rc = gb.channel("R").unwrap();
        let zc = gb.channel("Z").unwrap();
        assert_eq!(&bytes[rc.offset as usize..(rc.offset + rc.len) as usize], &r[..]);
        assert_eq!(&bytes[zc.offset as usize..(zc.offset + zc.len) as usize], &z[..]);
    }
}
