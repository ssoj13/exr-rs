//! Device-limit gates of `GpuContext::from_wgpu` (needs a real adapter; `cargo test -p exv-gpu-kernels -- --ignored`).

use exv_gpu_kernels::{GpuContext, GpuError};
use gpu_info::wgpu;

/// A host device created with default (small) limits is a typed `Limit` error, not a pipeline-creation
/// panic: the default `max_compute_workgroup_storage_size` (16384) is below the 18432 B the batched rANS
/// kernels declare (`RANS_WG_BYTES`).
#[test]
#[ignore = "requires a GPU"]
fn from_wgpu_rejects_small_limits() {
    let ctx = GpuContext::new().expect("gpu ctx");
    let (device, queue) = gpu_info::wait::block_on(ctx.adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("small-limits"),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("default-limits device");
    match GpuContext::from_wgpu(device, queue, ctx.adapter.clone()) {
        Err(GpuError::Limit(m)) => assert!(m.contains("max_compute_workgroup_storage_size"), "{m}"),
        Err(e) => panic!("expected Limit, got {e}"),
        Ok(_) => panic!("a default-limits device must be rejected"),
    }
}

/// The context clone shares one pipeline set (no second compile per atlas/compose layer).
#[test]
#[ignore = "requires a GPU"]
fn context_clone_shares_pipelines() {
    let ctx = GpuContext::new().expect("gpu ctx");
    let c2 = ctx.clone();
    assert!(std::sync::Arc::ptr_eq(&ctx.pipelines, &c2.pipelines));
}

/// `check_fits` (the open-time tile gate over [`exv_tile::fits`]) accepts the format-default tiles and refuses a tile
/// past the device's texture edge with a typed `Limit`; `recommended_tile_size` agrees with it (the chosen size fits).
#[test]
#[ignore = "requires a GPU"]
fn check_fits_gates_tiles_and_agrees_with_the_recommendation() {
    use exv_format::{Channel, Dtype, Header};
    let ctx = GpuContext::new().expect("gpu ctx");
    let header = |tile: u32| Header {
        width: 1 << 20,
        height: 1 << 20,
        frame_count: 1,
        tile_w: tile,
        tile_h: tile,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![Channel { name: "R".into(), dtype: Dtype::F16 }; 4],
    };
    ctx.check_fits(&header(256).grid(), &[Dtype::F16; 4]).expect("256^2 tiles fit any device");
    let max_dim = ctx.device.limits().max_texture_dimension_2d;
    match ctx.check_fits(&header(max_dim.next_power_of_two() * 2).grid(), &[Dtype::F16; 4]) {
        Err(GpuError::Limit(m)) => assert!(m.contains("tile"), "{m}"),
        other => panic!("an over-limit tile must be a typed Limit, got {other:?}"),
    }
    let rec = ctx.recommended_tile_size(2, 4, 0).expect("the driver reports VRAM");
    ctx.check_fits(&header(rec).grid(), &[Dtype::F16; 4]).expect("the recommended tile size fits");
}

/// The DX12 backend builds the full pipeline set on a device of its own (Windows only). Guards the lock shape
/// that once needed a `windows` pin in this crate's manifest (wgpu-hal / gpu-allocator must agree on ONE
/// `windows` version or the DX12 backend does not even compile): this test compiles AND runs on DX12.
#[cfg(windows)]
#[test]
#[ignore = "requires a GPU with a DX12 driver"]
fn dx12_device_builds_pipelines() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = gpu_info::wait::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("a DX12 adapter");
    let (device, queue) =
        gpu_info::request_max_device_blocking(&adapter, wgpu::Features::empty()).expect("DX12 device");
    let ctx = GpuContext::from_wgpu(device, queue, adapter).expect("pipelines on DX12");
    assert_eq!(ctx.adapter.get_info().backend, wgpu::Backend::Dx12);
}
