//! `context.rs` — reusable GPU device/queue context, cached compute pipelines, VRAM tile types,
//! and the shared low-level buffer/readback helpers every decode module builds on.
//!
//! Holds: [`GpuContext`] (device+queue+cached `Pipelines`), [`GpuTile`]/[`TileEnc`] (a VRAM-resident
//! decoded plane + its word packing), [`read_back`] (test/export copy-out), and the low-level
//! plumbing reused across modules (now `pub` so `exv-gpu`'s dispatch layer can reuse it cross-crate) (`u32_bytes`, `storage_u32`, `e`, `map_read_blocking`,
//! `build_pipelines`, `Pipelines`, `RGBA_GATHER_SHADER`). Pure code motion from the old `gpu.rs`.

use crate::prelude::*;
// `kernels.rs` IS this crate's lib root, so its WGSL `const SHADER`/`*_BODY` sources and helpers
// (`from_elems`, ...) live at `crate::` directly (not under a `kernels` submodule).
use crate::*;
use std::sync::{Arc, OnceLock};

/// Typed failure of every GPU decode entry point. Callers MUST be able to tell "the file is bad" from
/// "this device cannot do it" (→ fall back to the CPU decoder) from "this path does not exist" (→ route
/// elsewhere); a single stringly-typed error forced them to treat a size limit as a corrupt file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// A device limit blocks the operation (dispatch dims, binding size, workgroup storage, texture
    /// size, out-of-memory). Not a stream defect: the CPU decoder is the correct fallback.
    Limit(String),
    /// Malformed/forged stream or inconsistent geometry — the CPU decoder rejects it too.
    Corrupt(String),
    /// This GPU path does not handle the input (dtype/codec/tag routed elsewhere) — a routing outcome.
    Unsupported(String),
    /// wgpu/driver failure: device lost, poll/map error, validation error captured by [`GpuContext::scoped`].
    Device(String),
    /// The caller's host decode budget ([`exv_format::DecodeLimits`]) refuses the bytes this decode would hold. Not a
    /// device limit: the CPU decoder obeys the same budget, so falling back to it would fail the same way.
    Budget(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Limit(m) => write!(f, "GPU limit: {m}"),
            GpuError::Corrupt(m) => write!(f, "corrupt stream: {m}"),
            GpuError::Unsupported(m) => write!(f, "unsupported on GPU: {m}"),
            GpuError::Device(m) => write!(f, "GPU device: {m}"),
            GpuError::Budget(m) => write!(f, "decode budget: {m}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl GpuError {
    /// Map a CPU-decoder failure (`what` names the tile/channel) onto the GPU error classes, keeping the
    /// distinction callers route on: a bad stream stays `Corrupt`, an unsupported codec/dtype stays
    /// `Unsupported`, a decode-budget or allocation refusal is a `Limit` (not the file's fault).
    pub fn from_decode(e: exv_format::DecodeError, what: &str) -> Self {
        use exv_format::DecodeError as D;
        let m = format!("{what}: {e}");
        match e {
            D::Corrupt(_) | D::Request(_) | D::Checksum(_) => GpuError::Corrupt(m),
            D::Unsupported(_) | D::Codec(_) => GpuError::Unsupported(m),
            D::Limit { .. } => GpuError::Budget(m),
            D::Alloc(_) => GpuError::Limit(m),
        }
    }
}

/// A failed wait/map of the shared `gpu_info::wait` cluster (device lost, map error, dropped callback) is a
/// device failure: the awaited work's results must not be read.
impl From<gpu_info::wait::WaitError> for GpuError {
    fn from(e: gpu_info::wait::WaitError) -> Self {
        GpuError::Device(format!("wait: {e}"))
    }
}

/// Shorthand for results of the GPU decode entry points.
pub type GpuResult<T> = Result<T, GpuError>;

/// The cooperative cancel checkpoint of every batched host loop: `Err` once `c` is cancelled. Always placed BEFORE a
/// stage's host work / submit, never after one (a finished stage's buffers just drop). A whole-still caller passes an
/// inert `Cancel::new()`; the resident atlas threads its job's token through this ONE parameter, so a seek stops a
/// multi-run batch at the next run / chunk / channel instead of at the end of the whole batch.
pub fn cancelled_gpu(c: &bufpool::Cancel) -> GpuResult<()> {
    c.check().map_err(|_| GpuError::Device("cancelled".to_string()))
}

/// `GpuError::Corrupt` constructor for `.ok_or_else(|| corrupt(..))` chains over untrusted blob bytes.
pub fn corrupt(m: &str) -> GpuError {
    GpuError::Corrupt(m.to_string())
}

// The shaders read storage as native-endian `u32` and the container is little-endian, so a zero-copy byte view of
// `&[u32]` IS the LE upload; a big-endian host would need a swap pass that nothing here provides.
const _: () = assert!(cfg!(target_endian = "little"), "storage uploads reinterpret u32 words as LE bytes");

/// Create a storage buffer from `&[u32]` (minimum 4 bytes so wgpu does not complain about zero size). The words are
/// uploaded through a zero-copy byte view: the batched hosts upload tens of MB of descriptors per frame, and the old
/// per-call `Vec<u8>` conversion was a measurable share of the host time.
pub fn storage_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u32],
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let size = (bytes.len().max(4)) as u64;
    let buf = new_buf(device, label, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra);
    if !bytes.is_empty() {
        queue.write_buffer(&buf, 0, bytes);
    }
    buf
}

/// One disjoint part of an upload staging view, handed to exactly one rayon task by [`fill_parts`].
struct Part<'a>(wgpu::WriteOnly<'a, [u8]>);

// SAFETY: wgpu implements `Send for WriteOnly<'_, T>` only for `T: Send` with an implicit `T: Sized`, which
// needlessly excludes slices; the type is documented as behaving like `&mut [T]` (Send for `[u8]`). Every `Part` is
// a DISJOINT range split off the one view (`split_at_checked`) and moves into exactly one task, so no memory is
// aliased across threads.
unsafe impl Send for Part<'_> {}

/// Create a STORAGE buffer laid out as consecutive `parts` (byte lengths) and fill every part IN PARALLEL straight
/// into upload staging memory: `fill(i, dst)` must write exactly `parts[i]` bytes into `dst`. The batched hosts
/// upload hundreds of MB per frame (streams, tables, descriptors); building them in a `Vec` first and then memcpy-ing
/// was a serial host-side double pass; the disjoint parts are filled concurrently by rayon.
/// The buffer is rounded up to a whole `u32` (min 4 B); the tail padding is left unwritten (never read).
///
/// **Staging:** a buffer of at least [`UP_MIN`] bytes is staged in a REUSED `MAP_WRITE` chunk of the device's
/// [`UploadPool`] (its mapped pages stay warm across frames: fresh `write_buffer_with` staging pays first-touch page
/// faults, 2.1-2.7 GB/s vs 3.6-4.5 GB/s measured, OPT.md M); the copy is submitted right away (ordered before every later
/// submit of this queue). A smaller buffer, or one the pool cannot serve within its byte cap (all chunks in flight),
/// takes fresh `write_buffer_with` staging with identical bytes ([`GpuContext::staging_stats`] counts the fallbacks).
pub fn storage_parts<F>(ctx: &GpuContext, label: &str, parts: &[usize], fill: F) -> GpuResult<wgpu::Buffer>
where
    F: Fn(usize, wgpu::WriteOnly<'_, [u8]>) + Sync,
{
    let total: usize = parts.iter().sum();
    let size = total.next_multiple_of(4).max(4) as u64;
    // COPY_SRC is free and lets tests / diagnostics read the uploaded bytes back.
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let buf = new_buf(&ctx.device, label, size, usage);
    if total == 0 {
        return Ok(buf);
    }
    let nz = wgpu::BufferSize::new(size).ok_or_else(|| GpuError::Device(format!("{label}: empty staging write")))?;
    let run = |view: wgpu::WriteOnly<'_, [u8]>| {
        let mut view = view;
        fill_parts(view.slice(..), parts, || (), |_, i, dst| -> GpuResult<()> {
            fill(i, dst);
            Ok(())
        })
    };
    match fill_pooled(ctx, label, size, run)? {
        Some(chunk) => {
            let mut enc = encoder(&ctx.device, "exv.upload");
            enc.copy_buffer_to_buffer(&chunk, 0, &buf, 0, size);
            Staged::Chunk(chunk).submit(ctx, enc);
        }
        None => {
            let mut view = ctx
                .queue
                .write_buffer_with(&buf, 0, nz)
                .ok_or_else(|| GpuError::Limit(format!("{label}: cannot allocate {size} B of upload staging")))?;
            run(view.slice(..))?;
            drop(view); // schedules the staging -> buffer copy for the next submit
        }
    }
    Ok(buf)
}

/// THE pooled-staging decision, shared by every upload ([`storage_parts`], [`stage_bytes`], [`write_pooled`]): when
/// `size` is worth a chunk (>= [`UP_MIN`]) and the [`UploadPool`] has one within its cap, fill it with `run` while it
/// is mapped and return it UNMAPPED (the bytes sit at offset 0; the caller copies out of it and recalls it through
/// [`Staged::submit`]); `None` = stage fresh (too small, or the pool is exhausted: counted in
/// [`GpuContext::staging_stats`]). A failed fill discards the half-written chunk (never reused). The pool lock covers only
/// take/return, never the fill, so concurrent uploads (atlas worker + a whole-frame decode + the UI thread's requests)
/// fill DIFFERENT chunks in parallel and nothing can block or re-enter.
fn fill_pooled<F>(ctx: &GpuContext, label: &str, size: u64, run: F) -> GpuResult<Option<wgpu::Buffer>>
where
    F: FnOnce(wgpu::WriteOnly<'_, [u8]>) -> GpuResult<()>,
{
    // (`FnOnce`: a caller that falls back to fresh staging calls its own `run` again, so it passes `&run`.)
    if size < UP_MIN {
        return Ok(None);
    }
    let up = &ctx.pipelines.up;
    let Some(chunk) = up.take(&ctx.device, size) else {
        up.fallbacks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(None);
    };
    let filled = match chunk.slice(..size).get_mapped_range_mut() {
        Ok(mut view) => run(view.slice(..)),
        Err(e) => Err(GpuError::Device(format!("{label}: staging chunk is not mapped: {e}"))),
    };
    if let Err(e) = filled {
        up.discard(&chunk); // half-written: never reuse
        return Err(e);
    }
    chunk.unmap();
    Ok(Some(chunk))
}

/// Host bytes staged as a COPY source ([`stage_bytes`]): a recycled pool chunk, or a fresh pooled `COPY_DST | COPY_SRC`
/// buffer filled through `write_buffer_with`. Either way [`Self::buf`] holds the bytes from offset 0 once the submit
/// that [`Self::submit`] performs has been queued.
pub enum Staged {
    /// A [`UploadPool`] chunk (mapped pages stay warm across uploads); recalled by [`Self::submit`].
    Chunk(wgpu::Buffer),
    /// The fresh fallback (small upload, or the pool exhausted): returns to the scratch pool when dropped.
    Fresh(Scratch),
}

impl Staged {
    /// The buffer holding the staged bytes at offset 0 (`COPY_SRC`).
    pub fn buf(&self) -> &wgpu::Buffer {
        match self {
            Staged::Chunk(b) => b,
            Staged::Fresh(s) => s,
        }
    }

    /// Submit `enc` (the copies out of [`Self::buf`]) and release the staging: a chunk is re-mapped when this submit
    /// completes and returns to the pool, a fresh buffer goes back to the scratch pool.
    pub fn submit(self, ctx: &GpuContext, enc: wgpu::CommandEncoder) {
        if let Staged::Chunk(chunk) = self {
            ctx.pipelines.up.recall_on_submit(&enc, chunk);
        }
        ctx.queue.submit([enc.finish()]);
    }
}

/// Stage `size` bytes written by `run` (given the mapped / staging view) as a COPY source: a pooled chunk when
/// [`fill_pooled`] takes it, else a fresh pooled `COPY_DST | COPY_SRC` buffer filled with `write_buffer_with`. The ONE
/// staging helper of the resident atlas's host-plane upload (bytes, alignment and pitch are the caller's: `run` lays
/// them out), next to [`storage_parts`]' own destination-direct fresh path.
pub fn stage_bytes<F>(ctx: &GpuContext, label: &str, size: u64, run: F) -> GpuResult<Staged>
where
    F: Fn(wgpu::WriteOnly<'_, [u8]>) -> GpuResult<()>,
{
    let nz = wgpu::BufferSize::new(size).ok_or_else(|| GpuError::Device(format!("{label}: empty staging write")))?;
    if let Some(chunk) = fill_pooled(ctx, label, size, &run)? {
        return Ok(Staged::Chunk(chunk));
    }
    let buf = ctx.pooled(label, size, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
    let mut view = ctx
        .queue
        .write_buffer_with(&buf, 0, nz)
        .ok_or_else(|| GpuError::Limit(format!("{label}: cannot allocate {size} B of upload staging")))?;
    run(view.slice(..))?;
    drop(view); // schedules the staging -> `buf` copy ahead of the caller's submit
    Ok(Staged::Fresh(buf))
}

/// Write `bytes` into `dst` at byte `offset` (both multiples of 4): a plain `queue.write_buffer` below [`UP_MIN`], else
/// through a pooled chunk ([`fill_pooled`]), copied in parallel 4 MiB parts and submitted at once (so it is ordered
/// before every later submit, like the queued write it replaces). The whole-frame decoders' CPU-fallback planes use it.
pub fn write_pooled(ctx: &GpuContext, label: &str, dst: &wgpu::Buffer, offset: u64, bytes: &[u8]) -> GpuResult<()> {
    let size = bytes.len() as u64;
    if size < UP_MIN {
        ctx.queue.write_buffer(dst, offset, bytes);
        return Ok(());
    }
    let parts: Vec<usize> = (0..bytes.len()).step_by(UP_PART).map(|at| UP_PART.min(bytes.len() - at)).collect();
    let copy = |view: wgpu::WriteOnly<'_, [u8]>| {
        fill_parts(view, &parts, || (), |_, i, mut d| -> GpuResult<()> {
            let at = i * UP_PART;
            d.copy_from_slice(bytes.get(at..at + d.len()).ok_or_else(|| corrupt("write_pooled: part out of range"))?);
            Ok(())
        })
    };
    match fill_pooled(ctx, label, size, copy)? {
        Some(chunk) => {
            let mut enc = encoder(&ctx.device, "exv.upload");
            enc.copy_buffer_to_buffer(&chunk, 0, dst, offset, size);
            Staged::Chunk(chunk).submit(ctx, enc);
        }
        None => ctx.queue.write_buffer(dst, offset, bytes),
    }
    Ok(())
}

/// Part size of [`write_pooled`]'s parallel copy into a pooled chunk.
const UP_PART: usize = 4 << 20;

/// Smallest upload staged through the [`UploadPool`]: smaller buffers (tables, lane spans, descriptors, tile records)
/// are cheap to stage fresh and would otherwise each occupy a whole pool chunk.
pub const UP_MIN: u64 = 4 << 20;

/// Pool of reusable host-visible `MAP_WRITE | COPY_SRC` upload chunks of one device (shared by every context clone
/// through `Arc<Pipelines>`). A chunk is ONE upload: sized `next_power_of_two(bytes)` so frames of similar size reuse
/// it, filled while mapped, unmapped, copied, then re-mapped when its copy submit completes
/// (`map_buffer_on_submit`, no blocking poll: it comes back through `rx`). The retained bytes (idle + in flight) are
/// HARD-capped (`cap` = `min(scratch-pool budget, 256 MiB)`, host memory is priority 2): at the cap, idle chunks of
/// other sizes are evicted, and if the rest are still in flight the caller stages fresh. `held` is the live total.
pub struct UploadPool {
    inner: std::sync::Mutex<UpInner>,
    tx: std::sync::mpsc::Sender<wgpu::Buffer>,
    cap: u64,
    held: std::sync::atomic::AtomicU64,
    fallbacks: std::sync::atomic::AtomicU64,
}

struct UpInner {
    /// Idle, currently mapped chunks.
    free: Vec<wgpu::Buffer>,
    rx: std::sync::mpsc::Receiver<wgpu::Buffer>,
}

impl UploadPool {
    fn new(cap: u64) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self { inner: std::sync::Mutex::new(UpInner { free: Vec::new(), rx }), tx, cap, held: 0.into(), fallbacks: 0.into() }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, UpInner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A MAPPED chunk of exactly `next_power_of_two(need)` bytes, reused or created within the cap; `None` = over the cap.
    fn take(&self, device: &wgpu::Device, need: u64) -> Option<wgpu::Buffer> {
        use std::sync::atomic::Ordering::Relaxed;
        let size = need.next_power_of_two();
        {
            let mut g = self.lock();
            while let Ok(b) = g.rx.try_recv() {
                g.free.push(b);
            }
            if let Some(i) = g.free.iter().position(|b| b.size() == size) {
                return Some(g.free.swap_remove(i));
            }
            // Evict idle chunks of other sizes while the new one would exceed the cap.
            while self.held.load(Relaxed) + size > self.cap {
                let old = g.free.pop()?;
                self.held.fetch_sub(old.size(), Relaxed);
            }
            self.held.fetch_add(size, Relaxed);
        }
        // Reserved above; created outside the lock (a 64 MiB mapped-at-creation buffer takes a while).
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exv.upload.chunk"),
            size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        }))
    }

    /// Give `chunk` back once the copy recorded in `enc` has run: it is re-mapped on that submit's completion and returns
    /// through the channel (`enc` MUST be submitted, else the chunk is lost to the pool until dropped).
    fn recall_on_submit(&self, enc: &wgpu::CommandEncoder, chunk: wgpu::Buffer) {
        let tx = self.tx.clone();
        let again = chunk.clone();
        enc.map_buffer_on_submit(&again, wgpu::MapMode::Write, .., move |r| {
            // A failed re-map (device lost) drops the chunk: `held` then over-counts, which only makes the cap stricter.
            if r.is_ok() {
                tx.send(chunk).ok();
            }
        });
    }

    /// Forget a chunk that must not be reused (half-filled by a failed upload).
    fn discard(&self, chunk: &wgpu::Buffer) {
        self.held.fetch_sub(chunk.size(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Split `view` (an upload staging view: `queue.write_buffer_with`) into consecutive DISJOINT parts of `parts[i]` bytes
/// and fill them IN PARALLEL: `fill(state, i, dst)` writes part `i`. `init` builds one per-rayon-split `state` (a reused
/// host scratch, e.g. the tile a CPU decoder writes before its rows are copied into the staging), so a batch of
/// thousands of parts allocates a handful of scratch buffers, not one per part. The first error wins; parts past it
/// may stay unwritten (the caller must not submit the copy then). Shared by [`storage_parts`] and the resident atlas's
/// host-plane upload: ONE owner of the `Send` wrapper below.
pub fn fill_parts<S, E, I, F>(mut view: wgpu::WriteOnly<'_, [u8]>, parts: &[usize], init: I, fill: F) -> Result<(), E>
where
    E: Send + From<GpuError>,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, wgpu::WriteOnly<'_, [u8]>) -> Result<(), E> + Sync + Send,
{
    use rayon::prelude::*;
    let mut rest = view.slice(..);
    let mut pieces = Vec::with_capacity(parts.len());
    for (i, &n) in parts.iter().enumerate() {
        let (head, tail) = rest.split_at_checked(n).map_err(|_| corrupt("fill_parts: part overruns the staging view"))?;
        pieces.push((i, Part(head)));
        rest = tail;
    }
    pieces.into_par_iter().try_for_each_init(init, |s, (i, Part(dst))| fill(s, i, dst))
}

/// Plain GPU buffer (never mapped at creation) — the ONE `create_buffer` spelling of the crate's hosts.
pub fn new_buf(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
}

/// Command encoder labelled `label` — the ONE `create_command_encoder` spelling of the crate's hosts.
pub fn encoder(device: &wgpu::Device, label: &str) -> wgpu::CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

/// `MAP_READ` staging buffer of `size` bytes: the target of every test/export copy-out.
pub fn staging(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    new_buf(device, label, size, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ)
}

/// Bind group of `p`'s group 0 over whole buffers, binding `i` = `bufs[i]` (every compute kernel of this crate
/// declares its storage bindings densely from 0). Texture-carrying groups build their entries by hand.
pub fn bind_bufs(device: &wgpu::Device, p: &wgpu::ComputePipeline, label: &str, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| e(i as u32, b)).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: &p.get_bind_group_layout(0), entries: &entries })
}

/// One `BindGroupEntry` binding a texture view (the counterpart of [`e`] for display passes).
pub fn et(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(view) }
}

/// 2-D single-mip, single-sample texture — the ONE `create_texture` spelling of the display/atlas hosts.
pub fn tex2d(
    device: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// A pooled intermediate storage buffer ([`GpuContext::scratch`]): derefs to the `wgpu::Buffer` and goes back
/// to the device's [`gpu_info::ResourcePool`] when dropped.
///
/// **Why:** the per-decode `sym` buffers (4 bytes per decoded byte: tens to hundreds of MB for a whole frame)
/// are the same size for every frame of a clip. Creating/dropping them per decode reaches the driver as
/// `vkAllocateMemory`/`vkFreeMemory`, which the sibling measured as multi-100 ms process-wide stalls on Windows.
///
/// **Safe because:** a scratch buffer is written in full by the kernel that fills it before any kernel reads it
/// (rANS writes every symbol of every lane), so stale contents are never observed; reuse on the same queue
/// needs no wait (wgpu orders a later pass after an earlier one); the guard is the buffer's only owner.
pub struct Scratch {
    buf: wgpu::Buffer,
    pl: Arc<Pipelines>,
}

impl std::ops::Deref for Scratch {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        &self.buf
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // The pool keeps this clone; the guard's own handle dies right after, so the pooled buffer has no
        // other user (the pool's "unshared" rule).
        self.pl.pool.put(self.buf.clone());
    }
}

/// Idle-VRAM budget of the scratch pool: 1/16 of the process VRAM budget (clamped 64 MiB..1 GiB), 256 MiB when
/// the OS does not report one. Memory is priority 2: the pool may speed up repeated decodes, never hoard VRAM.
fn pool_budget(adapter: &wgpu::Adapter) -> u64 {
    gpu_info::vram_budget_bytes(adapter).map_or(256 << 20, |v| (v / 16).clamp(64 << 20, 1 << 30))
}

/// Cached compute pipelines — shaders are compiled ONCE per device (rebuilding per tile dominated decode
/// time). The set every decode path needs is built eagerly; kernels that are experimental or specialised
/// ([`rans_b_rt_for`](Self::rans_b_rt_for), [`exvrt_inv_fast`](Self::exvrt_inv_fast),
/// [`exvrt_fused`](Self::exvrt_fused)) compile on first use, so a device pays only for what it runs.
/// `wgpu::ComputePipeline` is Arc-backed; the whole struct is shared behind an `Arc` by every
/// [`GpuContext`] clone (the atlas/compose layers no longer build a second set).
pub struct Pipelines {
    /// Device the lazy kernels compile on, and its adapter (the key of the persistent pipeline cache).
    device: wgpu::Device,
    info: wgpu::AdapterInfo,
    pub rans: wgpu::ComputePipeline,
    pub repack: wgpu::ComputePipeline,
    pub wav_cols: wgpu::ComputePipeline,
    pub wav_rows: wgpu::ComputePipeline,
    pub reorder: wgpu::ComputePipeline,
    // batched (descriptor-grid) wave decode
    pub rans_b: wgpu::ComputePipeline,
    pub repack_b: wgpu::ComputePipeline,
    /// Batched reorder+delta inverse (LosslessFast stage 2, `UNDELTA_B_BODY`).
    pub undelta_b: wgpu::ComputePipeline,
    pub wav_cols_b: wgpu::ComputePipeline,
    pub wav_rows_b: wgpu::ComputePipeline,
    // EXV-RT block codec inverse+emit (B-RT.3); the rANS stage reuses `rans_b`.
    pub exvrt_inv: wgpu::ComputePipeline,
    /// The cooperative shared-memory inverse (16 blocks per workgroup, 256 threads, ~17 KiB workgroup scratch).
    /// Selected for every block<=16 tile.
    pub exvrt_inv_shared: wgpu::ComputePipeline,
    /// EXV-RT descriptor prep (`rt_prep`): per-tile records -> per-block `Desc` + `RtDesc` rows on the GPU.
    pub rt_prep: wgpu::ComputePipeline,
    /// [`EXP2_TAB_256`] in VRAM — binding 3 of every EXV-RT inverse kernel (the lossy log decode's table).
    pub exp2_t: wgpu::Buffer,
    /// Hardware `BC6H_SF16` Mode 11 block + optional lossless residual compute decoder.
    pub bc6h: wgpu::ComputePipeline,
    /// Thread-dense fast rANS kernels, one per [`RT_LANE_SET`] lane count (`LANES` is compile-time so the
    /// per-lane loops unroll). Lazy: see [`rans_b_rt_for`](Self::rans_b_rt_for).
    rans_b_rt: [OnceLock<wgpu::ComputePipeline>; RT_LANE_SET.len()],
    /// Per-thread-scratch inverse (lane-count independent). Currently unreachable from routing
    /// (`block<=16` always takes the shared kernel) but kept for A/B experiments; lazy.
    exvrt_inv_fast: OnceLock<wgpu::ComputePipeline>,
    /// PATH B — FUSED single-pass decoder (rANS + inverse in one kernel). Measured ~4x slower, routing keeps
    /// it disabled (see `prepare_exvrt`); lazy and needs 8 storage buffers per stage.
    exvrt_fused: OnceLock<wgpu::ComputePipeline>,
    /// Interleaved RGBA32F gather (device-unified display path): reads up to 4 decoded per-channel
    /// VRAM planes (each in its [`GpuTile`] word rep), converts dtype→f32 EXACTLY as the CPU viewer,
    /// and writes an `Rgba32Float` STORAGE texture. Source = [`RGBA_GATHER_SHADER`]; entry
    /// [`GpuFrame::to_rgba_texture`].
    pub rgba_gather: wgpu::ComputePipeline,
    /// Single-plane device blit — FLOAT output. Reads ONE decoded per-channel VRAM plane (its
    /// [`GpuTile`] word rep) and writes an `R32Float` STORAGE texture, converting dtype→f32 EXACTLY
    /// as [`RGBA_GATHER_SHADER`]. Used by [`blit_tile_to_plane_texture`] for F16/Bf16/F32 planes
    /// (the no-readback resident-atlas path). Source = [`PLANE_BLIT_COMMON`]+[`PLANE_BLIT_F32_ENTRY`].
    pub plane_blit_f32: wgpu::ComputePipeline,
    /// Single-plane device blit — UINT output. Same as [`plane_blit_f32`](Self::plane_blit_f32) but
    /// writes the RAW element bits into an `R32Uint` STORAGE texture (U32 AOV planes — no float
    /// conversion). Source = [`PLANE_BLIT_COMMON`]+[`PLANE_BLIT_U32_ENTRY`].
    pub plane_blit_u32: wgpu::ComputePipeline,
    /// Single-plane device blit — F16 output into an `R16Float` STORAGE texture (half the VRAM of the `R32Float`
    /// one). Lazy and only on a device that passes [`GpuContext::r16_store`]: on any other device the shader's storage
    /// format is a pipeline-creation validation error. Source = [`PLANE_BLIT_COMMON`]+[`PLANE_BLIT_F16_ENTRY`].
    plane_blit_f16: OnceLock<wgpu::ComputePipeline>,
    /// [`GpuContext::r16_store`]'s probe result (asked of the adapter once).
    r16_store: OnceLock<bool>,
    /// Idle intermediate buffers reused across decodes ([`Scratch`]); one pool per device, shared by every
    /// context clone through this `Arc<Pipelines>`.
    pool: gpu_info::ResourcePool<wgpu::Buffer>,
    /// The pool's idle-VRAM budget, bytes (see `pool_budget`).
    pool_bytes: u64,
    /// Idle resident-plane textures recycled by the atlas ([`GpuContext::plane_tex`] / [`Pipelines::recycle_tex`]): an
    /// evicted 8k plane is 134 MB, and freeing + re-allocating three of them per request cost 7..15 ms of driver
    /// `vkFreeMemory`/`vkAllocateMemory` inside the next submit (OPT.md Q). Capped at `pool_bytes` of IDLE VRAM beside the
    /// atlas budget; the lock only guards take/put/trim (swapping the pool for an empty one drops every idle texture).
    tex: std::sync::RwLock<gpu_info::ResourcePool<wgpu::Texture>>,
    /// Persistent upload staging ([`storage_parts`]): chunks recycle across frames so their mapped pages stay warm.
    up: UploadPool,
}

/// Compile `src` (the constants [`prelude`] is prepended here — the ONE place a shader gets it) and build ONE
/// compute pipeline (auto bind-group layout) for `entry`. Goes through `gpu_info`'s persistent pipeline cache, so
/// a driver compiles each kernel once per adapter rather than once per process (a no-op where the device has none).
fn make_pipe(device: &wgpu::Device, info: &wgpu::AdapterInfo, label: &str, src: &str, entry: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl(src).into()),
    });
    gpu_info::create_compute_pipeline(
        device,
        info,
        &wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        },
    )
}

impl Pipelines {
    /// A `w x h` `format` texture of `usage` for a resident plane: an idle one of the EXACT same key from the pool (edge
    /// tiles have their own sizes), else a new one. **Contents are unspecified** (a recycled texture holds the previous
    /// tile's pixels): every taker writes the whole extent before anything samples it (the plane blit and the host-upload
    /// copy both cover `w x h` entirely).
    fn take_tex(&self, device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
        let key = gpu_info::TextureKey { width: w, height: h, format, usage };
        self.tex.read().unwrap_or_else(std::sync::PoisonError::into_inner).take_texture(device, key, label)
    }

    /// Give back a plane texture NOBODY else holds (the atlas's `PlaneTex` guard calls it from its `Drop`): kept for the
    /// next [`GpuContext::plane_tex`] of its key, the least recently returned ones freed beyond the idle cap.
    pub fn recycle_tex(&self, tex: wgpu::Texture) {
        self.tex.read().unwrap_or_else(std::sync::PoisonError::into_inner).put(tex);
    }

    /// The fast rANS pipeline specialized for `lanes`, or `None` if no pipeline exists for that lane
    /// count (the caller then routes to the slow `rans_b` kernel, which handles any N). Compiled on first use.
    pub fn rans_b_rt_for(&self, lanes: u32) -> Option<&wgpu::ComputePipeline> {
        let i = RT_LANE_SET.iter().position(|&x| x == lanes)?;
        Some(self.rans_b_rt[i].get_or_init(|| {
            make_pipe(&self.device, &self.info, &format!("rans_b_rt{lanes}.wgsl"), &format!("{DESC_STRUCT}{}", rans_b_rt_src(lanes)), "main")
        }))
    }

    /// The `R16Float` plane blit, compiled on first use. ONLY call it on a device where [`GpuContext::r16_store`] holds
    /// ([`blit_record`] does): elsewhere the pipeline cannot be created.
    pub fn plane_blit_f16(&self) -> &wgpu::ComputePipeline {
        self.plane_blit_f16.get_or_init(|| {
            let [_, _, _, (label, src, entry)] = display_srcs();
            make_pipe(&self.device, &self.info, label, &src, entry)
        })
    }

    /// Per-thread-scratch EXV-RT inverse (experiment kernel), compiled on first use.
    pub fn exvrt_inv_fast(&self) -> &wgpu::ComputePipeline {
        self.exvrt_inv_fast.get_or_init(|| {
            make_pipe(&self.device, &self.info, "exvrt_inv_fast.wgsl", &format!("{RT_DESC_STRUCT}{}", exvrt_inv_src()), "exvrt_inv_fast")
        })
    }

    /// Fused rANS+inverse EXV-RT kernel (experiment), compiled on first use. It binds 8 storage buffers, so
    /// a device below that limit gets a typed error instead of a pipeline-creation panic.
    pub fn exvrt_fused(&self) -> GpuResult<&wgpu::ComputePipeline> {
        let have = self.device.limits().max_storage_buffers_per_shader_stage;
        if have < 8 {
            return Err(GpuError::Limit(format!("fused EXV-RT kernel needs 8 storage buffers per stage, device has {have}")));
        }
        Ok(self.exvrt_fused.get_or_init(|| {
            make_pipe(
                &self.device,
                &self.info,
                "exvrt_fused.wgsl",
                &format!("{DESC_STRUCT}{RT_DESC_STRUCT}{}", exvrt_fused_src()),
                "exvrt_fused",
            )
        }))
    }
}

/// WGSL dtype→f32 conversion shared by [`RGBA_GATHER_SHADER`] and [`PLANE_BLIT_COMMON`] (spliced in for the
/// `//@TO_F32@` token by [`with_f32`]). **Determinism contract:** `half_to_f32` builds the f32 from bits
/// (normals: re-biased exponent + shifted mantissa; subnormals: `mant * 2^-24`, an exact power-of-two scale), so
/// every finite half converts bit-exactly to `exv-capi::half_to_f32` on any driver — the previous
/// `exp2(f32(exp - 15))` was only specified to a few ULP. NaN becomes the canonical quiet NaN with the sign kept.
/// Integer dtypes normalize with a WGSL division, which the spec allows 2.5 ULP: that display conversion is
/// NOT bit-exact vs the CPU `sample`; the bit-exact claim of this crate covers the decoded integer words.
const TO_F32_WGSL: &str = r#"
fn half_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1fu;
    let mant = h & 0x3ffu;
    if (exp == 0x1fu) {
        return bitcast<f32>(sign | select(0x7f800000u, 0x7fc00000u, mant != 0u));
    }
    if (exp == 0u) {
        // subnormal: mant * 2^-24 (0x33800000 = 2^-24) — exact; also yields +-0 for mant == 0.
        return bitcast<f32>(sign | bitcast<u32>(f32(mant) * bitcast<f32>(0x33800000u)));
    }
    return bitcast<f32>(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

// Raw element bits -> linear f32 by dtype code (`DT_*`, generated from `Dtype::code`), like `exv_codec::image::sample_f32`
// (ints normalized to [0,1] by a float division).
fn to_f32(bits: u32, dcode: u32) -> f32 {
    switch dcode {
        case DT_F16: { return half_to_f32(bits); }
        case DT_F32: { return bitcast<f32>(bits); }
        case DT_BF16: { return bitcast<f32>(bits << 16u); }
        case DT_U8: { return f32(bits) / 255.0; }
        case DT_U16: { return f32(bits) / 65535.0; }
        default: { return f32(bits) / 4294967295.0; }
    }
}
"#;

/// Splice [`TO_F32_WGSL`] into a shader source containing the `//@TO_F32@` token.
fn with_f32(src: &str) -> String {
    src.replace("//@TO_F32@", TO_F32_WGSL)
}

/// WGSL interleaved-RGBA gather: read up to four decoded per-channel VRAM planes, reconstruct each
/// element's bits (per its [`GpuTile`] `enc`/`esz`), convert dtype→f32 **byte-identically to the CPU
/// viewer/plugin** (`exv-capi::sample`), and write an `Rgba32Float` storage texture. A `None` rgb slot
/// yields 0.0, a `None` alpha yields 1.0. NO CPU readback — the whole gather runs on the GPU so the
/// output texture is ready to bind into a render pass. Params layout (`array<u32>`): `[0]=width`,
/// `[1]=height`, then 5 words per slot (R,G,B,A) at `2+slot*5`: `present, off_words, enc_kind
/// (0=BytePerWord,1=ElemPerWord), esz, dtype_code`.
const RGBA_GATHER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> p_r: array<u32>;
@group(0) @binding(1) var<storage, read> p_g: array<u32>;
@group(0) @binding(2) var<storage, read> p_b: array<u32>;
@group(0) @binding(3) var<storage, read> p_a: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var out_tex: texture_storage_2d<rgba32float, write>;

// Static binding indexing: WGSL cannot dynamically index an array-of-bindings, so switch on the slot.
fn plane_word(slot: u32, i: u32) -> u32 {
    switch slot {
        case 0u: { return p_r[i]; }
        case 1u: { return p_g[i]; }
        case 2u: { return p_b[i]; }
        default: { return p_a[i]; }
    }
}

// Reconstruct the raw element bits for pixel `idx` from `slot`'s plane (matches GpuTile enc/esz).
fn read_bits(slot: u32, idx: u32) -> u32 {
    let base = 2u + slot * 5u;
    let off = params[base + 1u];
    let enc = params[base + 2u];
    let esz = params[base + 3u];
    if (enc == 0u) {
        // BytePerWord: esz consecutive words, low byte of each (LE) — the LosslessFast word rep.
        var bits = 0u;
        var k = 0u;
        loop {
            if (k >= esz) { break; }
            let w = plane_word(slot, off + idx * esz + k);
            bits = bits | ((w & 0xffu) << (8u * k));
            k = k + 1u;
        }
        return bits;
    }
    // ElemPerWord: one word holds the esz-byte element zero-extended LE (the wavelet/BlockRt word rep).
    let w = plane_word(slot, off + idx);
    if (esz >= 4u) { return w; }
    return w & ((1u << (8u * esz)) - 1u);
}

//@TO_F32@

fn slot_value(slot: u32, idx: u32, dflt: f32) -> f32 {
    let base = 2u + slot * 5u;
    if (params[base] == 0u) { return dflt; } // absent channel → default (0 for rgb, 1 for alpha)
    return to_f32(read_bits(slot, idx), params[base + 4u]);
}

@compute @workgroup_size(WG2, WG2)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params[0];
    let height = params[1];
    if (gid.x >= width || gid.y >= height) { return; }
    let idx = gid.y * width + gid.x;
    let r = slot_value(0u, idx, 0.0);
    let g = slot_value(1u, idx, 0.0);
    let b = slot_value(2u, idx, 0.0);
    let a = slot_value(3u, idx, 1.0);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, a));
}
"#;

/// Shared WGSL for the single-plane device blit ([`blit_tile_to_plane_texture`]): read ONE decoded
/// VRAM plane's `u32` word rep (binding 0) per its `enc`/`esz` (params binding 1) and convert
/// dtype→f32. `half_to_f32`/`to_f32` come from the SAME snippet as [`RGBA_GATHER_SHADER`]
/// ([`TO_F32_WGSL`]; `read_bits` is the single-plane form of the gather's) so the numeric path is
/// identical to the gather AND to the CPU `read_back`+`upload_host_plane` — the whole safety net.
/// params layout (`array<u32>`): `[0]=width, [1]=height, [2]=off_words, [3]=enc (0=BytePerWord,
/// 1=ElemPerWord), [4]=esz, [5]=dtype_code`. The entry point + output binding (binding 2) are appended
/// per output format ([`PLANE_BLIT_F32_ENTRY`] / [`PLANE_BLIT_U32_ENTRY`]).
const PLANE_BLIT_COMMON: &str = r#"
@group(0) @binding(0) var<storage, read> plane: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

// Reconstruct the raw element bits for pixel `idx` (matches GpuTile enc/esz; == RGBA_GATHER_SHADER).
fn read_bits(idx: u32) -> u32 {
    let off = params[2];
    let enc = params[3];
    let esz = params[4];
    if (enc == 0u) {
        // BytePerWord: esz consecutive words, low byte of each (LE) — the LosslessFast word rep.
        var bits = 0u;
        var k = 0u;
        loop {
            if (k >= esz) { break; }
            let w = plane[off + idx * esz + k];
            bits = bits | ((w & 0xffu) << (8u * k));
            k = k + 1u;
        }
        return bits;
    }
    // ElemPerWord: one word holds the esz-byte element zero-extended LE (wavelet/BlockRt word rep).
    let w = plane[off + idx];
    if (esz >= 4u) { return w; }
    return w & ((1u << (8u * esz)) - 1u);
}

//@TO_F32@
"#;

/// FLOAT entry for the single-plane blit (F16/Bf16/F32 → `R32Float`): store `to_f32(bits, dcode)`.
const PLANE_BLIT_F32_ENTRY: &str = r#"
@group(0) @binding(2) var out_tex: texture_storage_2d<r32float, write>;
@compute @workgroup_size(WG2, WG2)
fn plane_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params[0];
    let height = params[1];
    if (gid.x >= width || gid.y >= height) { return; }
    let idx = gid.y * width + gid.x;
    let v = to_f32(read_bits(idx), params[5]);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(v, 0.0, 0.0, 0.0));
}
"#;

/// F16 entry for the single-plane blit (F16 → `R16Float`): the same `to_f32` as the `R32Float` entry, so the stored
/// value is the exact f16 (an f16 widened to f32 narrows back losslessly); only the texel width differs.
const PLANE_BLIT_F16_ENTRY: &str = r#"
@group(0) @binding(2) var out_tex: texture_storage_2d<r16float, write>;
@compute @workgroup_size(WG2, WG2)
fn plane_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params[0];
    let height = params[1];
    if (gid.x >= width || gid.y >= height) { return; }
    let idx = gid.y * width + gid.x;
    let v = to_f32(read_bits(idx), params[5]);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(v, 0.0, 0.0, 0.0));
}
"#;

/// UINT entry for the single-plane blit (U32 → `R32Uint`): store the RAW element bits (no float conv).
const PLANE_BLIT_U32_ENTRY: &str = r#"
@group(0) @binding(2) var out_tex: texture_storage_2d<r32uint, write>;
@compute @workgroup_size(WG2, WG2)
fn plane_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params[0];
    let height = params[1];
    if (gid.x >= width || gid.y >= height) { return; }
    let idx = gid.y * width + gid.x;
    let bits = read_bits(idx);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<u32>(bits, 0u, 0u, 0u));
}
"#;

/// The display-pass shader sources `(label, src, entry)` — gather + the two single-plane blits — with the
/// shared dtype conversion spliced in. ONE list for [`build_pipelines`] and the shader validation tests.
pub(crate) fn display_srcs() -> [(&'static str, String, &'static str); 4] {
    [
        ("rgba_gather.wgsl", with_f32(RGBA_GATHER_SHADER), "gather"),
        ("plane_blit_f32.wgsl", with_f32(&format!("{PLANE_BLIT_COMMON}{PLANE_BLIT_F32_ENTRY}")), "plane_f32"),
        ("plane_blit_u32.wgsl", with_f32(&format!("{PLANE_BLIT_COMMON}{PLANE_BLIT_U32_ENTRY}")), "plane_u32"),
        ("plane_blit_f16.wgsl", with_f32(&format!("{PLANE_BLIT_COMMON}{PLANE_BLIT_F16_ENTRY}")), "plane_f16"),
    ]
}

/// Build the eager decode pipelines once.
fn build_pipelines(device: &wgpu::Device, info: wgpu::AdapterInfo, pool_bytes: u64) -> GpuResult<Pipelines> {
    let p = |label: &str, src: &str, entry: &str| make_pipe(device, &info, label, src, entry);
    // batched variants: prepend the shared Desc struct to each body.
    let rans_b = format!("{DESC_STRUCT}{}", rans_b_src());
    let rep_b = format!("{DESC_STRUCT}{REPACK_B_BODY}");
    let wav_b = format!("{DESC_STRUCT}{}", wavelet_b_src());
    let wav_1 = wavelet_src();
    let rans_1 = rans_src();
    let rt_inv = format!("{RT_DESC_STRUCT}{}", exvrt_inv_src());
    let [(gl, gs, ge), (fl, fs, fe), (ul, us, ue), _] = display_srcs();
    Ok(Pipelines {
        device: device.clone(),
        info: info.clone(),
        rans: p("rans.wgsl", &rans_1, "main"),
        repack: p("repack.wgsl", REPACK_SHADER, "repack"),
        wav_cols: p("wav.wgsl", &wav_1, "inv_cols"),
        wav_rows: p("wav.wgsl", &wav_1, "inv_rows"),
        reorder: p("reorder.wgsl", REORDER_SHADER, "undelta"),
        rans_b: p("rans_b.wgsl", &rans_b, "main"),
        repack_b: p("repack_b.wgsl", &rep_b, "repack"),
        undelta_b: p("undelta_b.wgsl", &format!("{DESC_STRUCT}{UNDELTA_B_BODY}"), "undelta"),
        wav_cols_b: p("wav_b.wgsl", &wav_b, "inv_cols"),
        wav_rows_b: p("wav_b.wgsl", &wav_b, "inv_rows"),
        exvrt_inv: p("exvrt_inv.wgsl", &rt_inv, "exvrt_inv"),
        exvrt_inv_shared: p("exvrt_inv.wgsl", &rt_inv, "exvrt_inv_shared"),
        rt_prep: p("rt_prep.wgsl", &format!("{DESC_STRUCT}{RT_DESC_STRUCT}{TILE_DESC_STRUCT}{RT_PREP_BODY}"), "rt_prep"),
        exp2_t: {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("exv.exp2_t"),
                size: (EXP2_TAB_256.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            let bytes: Vec<u8> = EXP2_TAB_256.iter().flat_map(|b| b.to_le_bytes()).collect();
            buf.slice(..)
                .get_mapped_range_mut()
                .map_err(|e| GpuError::Device(format!("exp2 table upload: {e:?}")))?
                .copy_from_slice(&bytes);
            buf.unmap();
            buf
        },
        bc6h: p("bc6h.wgsl", BC6H_SHADER, "bc6h_decode"),
        rans_b_rt: Default::default(),
        exvrt_inv_fast: OnceLock::new(),
        exvrt_fused: OnceLock::new(),
        rgba_gather: p(gl, &gs, ge),
        plane_blit_f32: p(fl, &fs, fe),
        plane_blit_u32: p(ul, &us, ue),
        plane_blit_f16: OnceLock::new(),
        r16_store: OnceLock::new(),
        pool: gpu_info::ResourcePool::new(pool_bytes),
        pool_bytes,
        tex: std::sync::RwLock::new(gpu_info::ResourcePool::new(pool_bytes)),
        up: UploadPool::new(pool_bytes.min(256 << 20)),
    })
}

/// Device limits the eagerly built pipelines need. Host devices (viewer, plugin) are created by the
/// caller with whatever limits they asked for; failing here gives a typed error at startup instead of a
/// wgpu pipeline-creation panic. Workgroup storage is derived from the shaders' own array sizes
/// ([`RANS_WG_BYTES`], [`RT_INV_WG_BYTES`]) so the gate cannot drift from the WGSL.
fn check_device(device: &wgpu::Device) -> GpuResult<()> {
    check_limits(&device.limits())
}

/// [`check_device`] over plain limits (CPU-testable): every miss is listed in one [`GpuError::Limit`].
fn check_limits(l: &wgpu::Limits) -> GpuResult<()> {
    let mut miss: Vec<String> = Vec::new();
    let mut need = |name: &str, have: u64, want: u64| {
        if have < want {
            miss.push(format!("{name} {have} < {want}"));
        }
    };
    need("max_storage_buffers_per_shader_stage", l.max_storage_buffers_per_shader_stage as u64, 7);
    need("max_compute_workgroup_storage_size", l.max_compute_workgroup_storage_size as u64, RANS_WG_BYTES.max(RT_INV_WG_BYTES) as u64);
    // The cooperative EXV-RT inverse is the widest workgroup (`RT_INV_WG` threads); every other kernel needs `WG`.
    need("max_compute_invocations_per_workgroup", l.max_compute_invocations_per_workgroup as u64, RT_INV_WG.max(WG) as u64);
    need("max_compute_workgroup_size_x", l.max_compute_workgroup_size_x as u64, RT_INV_WG.max(WG) as u64);
    need("max_storage_textures_per_shader_stage", l.max_storage_textures_per_shader_stage as u64, 1);
    if miss.is_empty() {
        Ok(())
    } else {
        Err(GpuError::Limit(format!("device too small for the exv kernels: {}", miss.join(", "))))
    }
}

/// The largest storage buffer the device can both create and bind (`min(max_storage_buffer_binding_size,
/// max_buffer_size)`): the ONE definition behind [`GpuContext::check_buf`] and `default_sym_cap`.
fn buf_limit(device: &wgpu::Device) -> u64 {
    let l = device.limits();
    l.max_storage_buffer_binding_size.min(l.max_buffer_size)
}

/// Default [`GpuContext::sym_cap`]: the device's buffer limit, further bounded by a quarter of the process VRAM
/// budget (one scratch must not crowd out the composites it feeds) and by the `u32` word index the kernels use
/// (`sym_base + ..` is u32 arithmetic). Every term is a real device number, none invented.
fn default_sym_cap(device: &wgpu::Device, adapter: &wgpu::Adapter) -> u64 {
    let cap = buf_limit(device).min(u64::from(u32::MAX));
    gpu_info::vram_budget_bytes(adapter).map_or(cap, |v| cap.min(v / 4)).max(4)
}

/// One run of consecutive units of a batched decode that share one symbol scratch ([`plan_chunks`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Indices (into the planned item order) of the units of this chunk.
    pub units: std::ops::Range<usize>,
    /// Total symbol words of the chunk (<= the cap).
    pub words: usize,
}

/// Greedily cut `items` = `(group, symbol words)` into [`Chunk`]s of at most `cap_words` words that never mix
/// groups (a group = whatever must share bindings: an output channel, a kernel specialisation). A single unit larger
/// than the cap cannot be split and is a typed [`GpuError::Limit`] (the CPU decoder is the fallback). The batched
/// whole-frame hosts (wave, EXV-RT) use this so a frame whose symbol buffer exceeds the device limit decodes in
/// several chunks instead of failing (the old 8k LosslessWave `batch sym > 2 GiB` error).
pub fn plan_chunks(items: impl IntoIterator<Item = (u64, usize)>, cap_words: usize) -> GpuResult<Vec<Chunk>> {
    let mut out: Vec<Chunk> = Vec::new();
    let mut cur: Option<(u64, Chunk)> = None;
    for (i, (group, words)) in items.into_iter().enumerate() {
        if words > cap_words {
            return Err(GpuError::Limit(format!(
                "unit {i} needs a {} B symbol buffer > the {} B cap (device buffer limit) - it cannot be split",
                words as u64 * 4,
                cap_words as u64 * 4
            )));
        }
        match cur.as_mut() {
            Some((g, c)) if *g == group && c.words + words <= cap_words => {
                c.units.end = i + 1;
                c.words += words;
            }
            _ => {
                out.extend(cur.take().map(|(_, c)| c));
                cur = Some((group, Chunk { units: i..i + 1, words }));
            }
        }
    }
    out.extend(cur.map(|(_, c)| c));
    Ok(out)
}

/// Run `f` inside validation + out-of-memory error scopes of `device`: any wgpu error raised by the work it
/// records becomes an `Err` instead of reaching the (panicking) default uncaptured-error handler. The ONE
/// error-capture policy of the crate — see [`GpuContext::scoped`].
fn scope_dev<T>(device: &wgpu::Device, f: impl FnOnce() -> GpuResult<T>) -> GpuResult<T> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let val = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = f();
    // Scopes pop innermost first; the pop future resolves immediately on native backends.
    let verr = gpu_info::wait::block_on(val.pop());
    let oerr = gpu_info::wait::block_on(oom.pop());
    let out = out?;
    if let Some(e) = oerr {
        return Err(GpuError::Limit(format!("out of memory: {e}")));
    }
    if let Some(e) = verr {
        return Err(GpuError::Device(format!("validation: {e}")));
    }
    Ok(out)
}

/// Device + queue + shared compute pipelines. Cheap to clone (everything is Arc-backed), so the atlas and
/// compose layers hold a clone instead of compiling a second pipeline set.
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Compute pipelines, built once in [`GpuContext::from_wgpu`] and reused by every decode.
    pub pipelines: Arc<Pipelines>,
    /// We keep the adapter to query the VRAM budget (`gpu_info::vram_budget_bytes`) after creation.
    pub adapter: wgpu::Adapter,
    /// Largest decode-intermediate (symbol scratch) buffer the batched decoders allocate, in bytes: they split a
    /// frame into chunks below it ([`plan_chunks`]) instead of failing. Defaults to `default_sym_cap`; public so a
    /// test can inject a small cap and exercise the multi-chunk path without a multi-GiB frame.
    pub sym_cap: u64,
}

impl GpuContext {
    /// Create the context on its OWN wgpu device (prefers a discrete GPU, maximum limits). Standalone
    /// use (CLI, tests, benches). To SHARE an app's device instead, use [`from_wgpu`](Self::from_wgpu).
    /// A software adapter (WARP/llvmpipe) is refused unless opted into explicitly: call
    /// [`new_sw`](Self::new_sw), or set the environment variable `EXV_GPU_ALLOW_SW=1` (CI / WARP-only
    /// machines running the GPU test-suite, whose tests all go through `new()`).
    pub fn new() -> GpuResult<Self> {
        let allow_sw = std::env::var_os("EXV_GPU_ALLOW_SW").is_some_and(|v| v != "0");
        let (adapter, device, queue) = make_device(allow_sw)?;
        Self::from_wgpu(device, queue, adapter)
    }

    /// Like [`new`](Self::new) but accepts a CPU/software adapter when no GPU exists (CI machines).
    pub fn new_sw() -> GpuResult<Self> {
        let (adapter, device, queue) = make_device(true)?;
        Self::from_wgpu(device, queue, adapter)
    }

    /// Build the decode context on an EXISTING wgpu device/queue/adapter — **device unification**.
    ///
    /// # Why
    /// A host (the viewer, a plugin) owns ONE wgpu device for the whole app; decoding on a SECOND
    /// device would force a cross-device copy of every decoded plane/texture. `from_wgpu` builds our
    /// compute pipelines on the PASSED-IN device (via the shared [`build_pipelines`]), so decoded VRAM
    /// buffers and [`GpuFrame::to_rgba_texture`] outputs are usable directly by the caller's render
    /// pass. The adapter is stored for [`vram_budget`](Self::vram_budget) /
    /// [`recommended_tile_size`](Self::recommended_tile_size). Mirrors `jph_wgpu::Gpu::from_parts`.
    ///
    /// # Errors
    /// [`GpuError::Limit`] when the device's limits are below what the kernels need (host-created devices
    /// may use default/downlevel limits), [`GpuError::Device`] when pipeline creation fails.
    pub fn from_wgpu(device: wgpu::Device, queue: wgpu::Queue, adapter: wgpu::Adapter) -> GpuResult<Self> {
        check_device(&device)?;
        let pipelines = scope_dev(&device, || build_pipelines(&device, adapter.get_info(), pool_budget(&adapter)))?;
        let sym_cap = default_sym_cap(&device, &adapter);
        Ok(Self { device, queue, adapter, pipelines: Arc::new(pipelines), sym_cap })
    }

    /// Run `f` (which records/submits GPU work) inside the crate's single error-capture policy: wgpu
    /// validation and out-of-memory errors raised meanwhile are returned as `Err` instead of panicking the
    /// host through the default uncaptured-error handler. Every public decode entry point wraps its body in
    /// this; scopes nest, so composed entry points are fine.
    pub fn scoped<T>(&self, f: impl FnOnce() -> GpuResult<T>) -> GpuResult<T> {
        scope_dev(&self.device, f)
    }

    /// Err([`GpuError::Limit`]) if any of the dispatch dimensions exceeds `max_compute_workgroups_per_dimension`.
    pub fn check_dispatch(&self, what: &str, g: [u32; 3]) -> GpuResult<()> {
        let max = self.device.limits().max_compute_workgroups_per_dimension;
        match g.iter().find(|&&d| d > max) {
            Some(d) => Err(GpuError::Limit(format!(
                "{what}: dispatch {g:?} exceeds max_compute_workgroups_per_dimension {max} (dim {d})"
            ))),
            None => Ok(()),
        }
    }

    /// Err([`GpuError::Limit`]) if a `bytes`-long storage buffer cannot be bound/created on this device
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`).
    pub fn check_buf(&self, what: &str, bytes: u64) -> GpuResult<()> {
        let lim = buf_limit(&self.device);
        if bytes > lim {
            return Err(GpuError::Limit(format!(
                "{what}: buffer {bytes} B > device limit {lim} B (binding/max_buffer_size) — reduce the tile"
            )));
        }
        Ok(())
    }

    /// Split `groups` 1-D workgroups into a legal `[x, y, 1]` dispatch (`x` capped at the per-dimension
    /// limit). The kernel recovers the linear workgroup index as `wid.y * num_workgroups.x + wid.x` and must
    /// bounds-check against its element count (the grid may overshoot by < one row).
    pub fn split_1d(&self, what: &str, groups: u64) -> GpuResult<[u32; 3]> {
        let max = self.device.limits().max_compute_workgroups_per_dimension as u64;
        let x = groups.min(max).max(1);
        let y = groups.div_ceil(x).max(1);
        if y > max {
            return Err(GpuError::Limit(format!("{what}: {groups} workgroups exceed the {max}x{max} dispatch grid")));
        }
        Ok([x as u32, y as u32, 1])
    }

    /// Words of the symbol scratch a batched decode plans its chunks with, given its biggest unit (`biggest`
    /// words): the scratch pool's idle budget, so a repeated whole-frame decode REUSES one pooled scratch instead of
    /// `vkAllocateMemory`-ing (and the driver clearing) a fresh multi-hundred-MB buffer every frame — but never
    /// below one unit and never above [`sym_cap`](Self::sym_cap), the device's real buffer limit.
    pub fn chunk_words(&self, biggest: usize) -> usize {
        let soft = (self.pipelines.pool_bytes / 4) as usize;
        soft.max(biggest).min((self.sym_cap / 4) as usize)
    }

    /// Process VRAM budget (bytes) from the driver via gpu-info-rs; `None` if the OS does not report it.
    pub fn vram_budget(&self) -> Option<u64> {
        gpu_info::vram_budget_bytes(&self.adapter)
    }

    /// THIS device's tile limits for [`exv_tile::fits`] / [`exv_tile::pick_tile_size`]: per-binding
    /// (`max_storage_buffer_binding_size`), the decode VRAM bound `vram` and `max_texture_dimension_2d`.
    fn tile_limits(&self, vram: u64) -> exv_tile::DeviceLimits {
        let lim = self.device.limits();
        exv_tile::DeviceLimits {
            max_binding: lim.max_storage_buffer_binding_size,
            vram,
            max_tex_dim: lim.max_texture_dimension_2d,
        }
    }

    /// Recommended tile size for THIS device's limits ([`exv_tile::pick_tile_size`] over [`Self::tile_limits`]): a
    /// single budget contract for the decoder and the out-of-core tiler — the chosen tile is guaranteed to fit
    /// the decode buffers (closes M1).
    ///
    /// # Errors
    /// [`GpuError::Limit`] when the driver and the OS report no VRAM size (no invented default).
    pub fn recommended_tile_size(&self, esz: usize, channels: usize, halo: u32) -> GpuResult<u32> {
        let vram = self
            .vram_budget()
            .ok_or_else(|| GpuError::Limit("VRAM size unknown (neither the driver nor the OS reports it)".into()))?;
        let l = self.tile_limits(vram);
        Ok(exv_tile::pick_tile_size(l.max_binding, l.vram, l.max_tex_dim, esz, channels, halo))
    }

    /// Validate AT OPEN that the clip's tiles fit this device ([`exv_tile::fits`], the formula
    /// [`Self::recommended_tile_size`] uses): a file whose tiles the device cannot bind/decode is a typed
    /// [`GpuError::Limit`] up front, not a mid-decode failure. When the driver reports no VRAM the VRAM clause is
    /// not enforced (nothing known to enforce); the binding and texture-edge clauses always are.
    ///
    /// `dtypes` = the channels a tile decode actually materializes (every header channel, or the caller's channel
    /// subset: the resident atlas checks only the planes it will make resident).
    ///
    /// # Errors
    /// [`GpuError::Limit`] naming the violated limit.
    pub fn check_fits(&self, grid: &exv_format::TileGrid, dtypes: &[exv_format::Dtype]) -> GpuResult<()> {
        let l = self.tile_limits(self.vram_budget().unwrap_or(u64::MAX));
        exv_tile::fits(&l, grid, dtypes, 0).map_err(|e| GpuError::Limit(e.to_string()))
    }

    /// A storage buffer of `bytes` (rounded up to a whole `u32`) for a decode intermediate, taken from the
    /// device's scratch pool or created; returned to the pool when the guard drops. Contents are unspecified.
    pub fn scratch(&self, label: &str, bytes: u64) -> Scratch {
        self.pooled(label, bytes, wgpu::BufferUsages::STORAGE)
    }

    /// [`Self::scratch`] with an explicit `usage` (the pool keys on size AND usage): e.g. the resident atlas's
    /// `COPY_DST | COPY_SRC` host-upload buffer, filled through `write_buffer_with` and copied into textures. Same
    /// contract: contents unspecified, the guard returns it to the pool on drop.
    pub fn pooled(&self, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> Scratch {
        let key = gpu_info::BufferKey { size: bytes.next_multiple_of(4).max(4), usage };
        Scratch { buf: self.pipelines.pool.take_buffer(&self.device, key, label), pl: Arc::clone(&self.pipelines) }
    }

    /// Can a compute shader `textureStore` into an `R16Float` texture on this device? Not in the WebGPU baseline: it needs
    /// the device feature `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` (the viewer's device requests it, `exv-view`
    /// `gpu_ctx::FEATURES`; a standalone [`GpuContext::new`] device does not) AND the adapter reporting `STORAGE_WRITE_ONLY`
    /// with the usages the blit target needs for that format. Asked once per device.
    pub fn r16_store(&self) -> bool {
        *self.pipelines.r16_store.get_or_init(|| {
            use wgpu::{Features, TextureFormat, TextureFormatFeatureFlags as Ff};
            if !self.device.features().contains(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
                return false;
            }
            let f = self.adapter.get_texture_format_features(TextureFormat::R16Float);
            f.allowed_usages.contains(BLIT_USAGE) && f.flags.contains(Ff::STORAGE_WRITE_ONLY)
        })
    }

    /// [`blit_format`] for THIS device: the resident format a decoded plane of `dtype` gets, or `None` without an exact
    /// device blit.
    pub fn blit_fmt(&self, dtype: Dtype) -> Option<wgpu::TextureFormat> {
        blit_format(dtype, self.r16_store())
    }

    /// A resident-plane texture from the recycling pool, else a fresh one ([`Pipelines::take_tex`] has the contract:
    /// stale contents, the caller overwrites the whole extent). The ONE creation spelling of atlas plane textures; the
    /// owner hands it back through [`Pipelines::recycle_tex`] when the tile dies.
    pub fn plane_tex(&self, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
        self.pipelines.take_tex(&self.device, label, w, h, format, usage)
    }

    /// `(idle textures, idle bytes)` of the plane-texture pool: VRAM held beside the atlas budget (capped, see [`Pipelines`]).
    pub fn tex_pool_stats(&self) -> (usize, u64) {
        self.pipelines.tex.read().unwrap_or_else(std::sync::PoisonError::into_inner).idle_count()
    }

    /// The most idle bytes the plane-texture pool keeps ([`Self::tex_pool_stats`] never exceeds it; a texture bigger than
    /// the whole cap is freed instead of pooled).
    pub fn tex_pool_cap(&self) -> u64 {
        self.pipelines.pool_bytes
    }

    /// Free every idle plane texture (atlas teardown, memory pressure): the pool is swapped for an empty one.
    pub fn trim_tex_pool(&self) {
        *self.pipelines.tex.write().unwrap_or_else(std::sync::PoisonError::into_inner) = gpu_info::ResourcePool::new(self.pipelines.pool_bytes);
    }

    /// `(bytes retained by the upload pool, uploads that wanted it but staged fresh)`: the pool's host-memory cost and
    /// the debug counter of how often it was exhausted.
    pub fn staging_stats(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.pipelines.up.held.load(Relaxed), self.pipelines.up.fallbacks.load(Relaxed))
    }

    /// Block until ALL queued GPU work has finished — **no readback** (the data stays in VRAM).
    /// Used to fence/time the decode end-to-end (`decode_still_gpu` only queues work; it does not wait).
    pub fn drain(&self) -> GpuResult<()> {
        Ok(gpu_info::wait::wait_idle(&self.device, &self.queue)?)
    }
}

/// Preference rank of an adapter type (lower = better): discrete > integrated > virtual > unclassified >
/// software. `Other` (some GL/driver setups report real hardware this way) must still beat a `Cpu` adapter,
/// or `new()` would refuse "only a software adapter" while a hardware one exists.
fn rank(t: wgpu::DeviceType) -> u8 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

/// Get a GPU adapter (owned) + device with maximum limits (via gpu-info). The best-ranked adapter wins
/// (`enumerate_adapters` order is undefined); a software adapter (`DeviceType::Cpu`, e.g. WARP/llvmpipe)
/// is used ONLY when `allow_sw` — silently decoding on a CPU "GPU" would be a performance cliff nobody asked for.
fn make_device(allow_sw: bool) -> GpuResult<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let mut adapters = gpu_info::wait::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    adapters.sort_by_key(|a| rank(a.get_info().device_type));
    if adapters.is_empty() {
        return Err(GpuError::Device("no GPU adapter".to_string()));
    }
    if !allow_sw && adapters[0].get_info().device_type == wgpu::DeviceType::Cpu {
        return Err(GpuError::Device("only a software adapter is available (use GpuContext::new_sw to accept it)".to_string()));
    }
    let adapter = adapters.swap_remove(0);
    let (device, queue) = gpu_info::request_max_device_blocking(&adapter, wgpu::Features::empty())
        .map_err(|e| GpuError::Device(format!("request_device: {e:?}")))?;
    Ok((adapter, device, queue))
}

/// How a [`GpuTile`]'s `u32`-word storage buffer unpacks to native-endian packed plane bytes.
#[derive(Clone, Copy, Debug)]
pub enum TileEnc {
    /// One byte per `u32` word (the word's low byte) — rANS symbol/byte streams (LosslessFast).
    BytePerWord,
    /// One `esz`-byte element per `u32` word (zero-extended LE) — wavelet element planes.
    ElemPerWord(usize),
}

/// A decoded plane resident in VRAM — the GPU-first, no-readback output of the `_vram` decoders.
///
/// `buf` is a `STORAGE | COPY_SRC` storage buffer of `words` `u32` values; `enc` says how to unpack it
/// to packed plane bytes. The hot path consumes `buf` on the GPU (compositing/viewer); [`read_back`]
/// (test/export only) copies it to the CPU and applies `enc` to reproduce the CPU `decode_channel` bytes.
pub struct GpuTile {
    pub buf: wgpu::Buffer,
    /// Word offset of this plane WITHIN `buf` (0 for a dedicated buffer; non-zero when several planes
    /// share one combined composite buffer — the batched `decode_still_gpu` packs all channels into one).
    pub off_words: usize,
    pub words: usize,
    pub w: u32,
    pub h: u32,
    pub dtype: Dtype,
    pub enc: TileEnc,
}

impl GpuTile {
    /// `off_words` as the `u32` the WGSL word indexing uses; a tile placed beyond 2^32 words cannot be
    /// addressed by the shaders, so it is an error rather than a silent truncation.
    pub fn off_u32(&self) -> GpuResult<u32> {
        u32::try_from(self.off_words)
            .map_err(|_| GpuError::Limit(format!("tile offset {} words exceeds the u32 shader index range", self.off_words)))
    }
}

/// Copy a [`GpuTile`] out of VRAM and unpack it to packed native-endian plane bytes.
///
/// **Test/export only — NOT the hot path** (the GPU-first path keeps the tile in VRAM). Does its own
/// staging copy + submit + blocking map; the result is bit-for-bit == CPU `decode_channel`.
pub fn read_back(ctx: &GpuContext, tile: &GpuTile) -> GpuResult<Vec<u8>> {
    if tile.words == 0 {
        return Ok(Vec::new());
    }
    let device = &ctx.device;
    let size = (tile.words * 4) as u64;
    let off = (tile.off_words as u64) * 4;
    if off + size > tile.buf.size() {
        return Err(GpuError::Corrupt(format!(
            "read_back: tile range {off}+{size} B exceeds its {} B buffer",
            tile.buf.size()
        )));
    }
    let words = ctx.scoped(|| {
        let rb = staging(device, "exv.readback", size);
        let mut enc = encoder(device, "exv.readback.enc");
        enc.copy_buffer_to_buffer(&tile.buf, off, &rb, 0, size);
        ctx.queue.submit([enc.finish()]);
        Ok(words_of(&map_read_blocking(device, &rb)?))
    })?;
    Ok(match tile.enc {
        TileEnc::BytePerWord => words.iter().map(|&word| (word & 0xff) as u8).collect(),
        TileEnc::ElemPerWord(esz) => from_elems(&words, esz),
    })
}

/// **On-device blit of ONE decoded VRAM plane ([`GpuTile`]) into a single-channel resident texture**
/// — the no-readback replacement for `read_back` + host `queue.write_texture` in the resident atlas.
///
/// # Why
/// The resident GPU-decode path used to copy each decoded plane VRAM→host ([`read_back`], a BLOCKING
/// GPU→CPU map) then host→VRAM (`write_texture`) — a full round-trip + GPU stall PER plane PER tile PER
/// frame. This reads the plane's `u32` word buffer (respecting `off_words`/`enc`/`esz`) and
/// `textureStore`s it into a freshly created STORAGE texture ON THE GPU (encoder + dispatch + submit,
/// NO map, NO wait), converting dtype→f32 **byte-identically to [`RGBA_GATHER_SHADER`] and to the CPU
/// `read_back`+`upload_host_plane`** (the WGSL conversion is copied verbatim), so the resident plane is
/// bit-exact with the CPU path and interchangeable in `exv-gpu`'s plane-compose sampler.
///
/// # Formats (and the R16Float capability)
/// `F16` → **`R16Float`** where the device can store into it ([`GpuContext::r16_store`]), else **`R32Float`**;
/// `Bf16`/`F32` → `R32Float`; `U32` → `R32Uint`; anything else → a hard error (those dtypes never encode into a GPU
/// HTJ2K/wave/EXV-RT tile). `R16Float` is NOT a WebGPU-baseline STORAGE format: it needs the device feature
/// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` plus the adapter's `STORAGE_WRITE_ONLY` flag (the viewer's device has
/// both, a standalone context does not: [`blit_format`] is THE table). Either width is bit-exact when sampled —
/// an `R32Float` holding `half_to_f32(bits)` and an `R16Float` holding the f16 itself both read back as the SAME f32, so
/// GPU-blitted planes of either format and CPU-uploaded `R16Float` planes compose IDENTICALLY (the atlas may mix them
/// within one tile). The texture is `TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC`.
///
/// `dtype`/`w`/`h` equal `tile.dtype`/`tile.w`/`tile.h`; they are passed explicitly to mirror the atlas
/// call site (which already holds the tile rect + component dtype). Returns the owned texture, its
/// default view, and the chosen format (the caller wraps them, e.g. `ResidentTexture::new_plane`).
pub fn blit_tile_to_plane_texture(
    ctx: &GpuContext,
    tile: &GpuTile,
    dtype: Dtype,
    w: u32,
    h: u32,
) -> GpuResult<(wgpu::Texture, wgpu::TextureView, wgpu::TextureFormat)> {
    ctx.scoped(|| {
        let mut enc = encoder(&ctx.device, "exv.plane_blit.enc");
        let r = blit_record(ctx, &mut enc, tile, dtype, w, h)?;
        ctx.queue.submit([enc.finish()]);
        Ok(r)
    })
}

/// Usages of a blitted resident plane: sampled by compose, written by the blit shader, readable back by tests.
const BLIT_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::STORAGE_BINDING).union(wgpu::TextureUsages::COPY_SRC);

/// The resident texture format [`blit_record`] writes for a decoded plane of `dtype`, or `None` when the device blit
/// has no exact route for it (THE table; callers route such planes to a host decode + upload instead). Floats widen
/// to `R32Float` exactly (f16 / bf16 bit-built, `TO_F32_WGSL`), U32 keeps its raw bits in `R32Uint`. U8/U16 are
/// absent: their display value is a normalizing division the WGSL spec allows 2.5 ULP, not the CPU `sample_f32`.
///
/// `r16_store` = the device can `textureStore` into `R16Float` ([`GpuContext::r16_store`]; callers go through
/// [`GpuContext::blit_fmt`]): F16 then stays `R16Float` (half the VRAM, the same texel the CPU upload route holds, and
/// sampling it yields the exact f32 an `R32Float` copy would). BF16 does not fit f16, so it keeps `R32Float`.
pub fn blit_format(dtype: Dtype, r16_store: bool) -> Option<wgpu::TextureFormat> {
    match dtype {
        Dtype::F16 if r16_store => Some(wgpu::TextureFormat::R16Float),
        Dtype::F16 | Dtype::Bf16 | Dtype::F32 => Some(wgpu::TextureFormat::R32Float),
        Dtype::U32 => Some(wgpu::TextureFormat::R32Uint),
        Dtype::U8 | Dtype::U16 => None,
    }
}

/// Record the plane blit of [`blit_tile_to_plane_texture`] into the caller's `enc` WITHOUT submitting: a tile with
/// several GPU-decoded planes records all its blits into ONE encoder and pays ONE submit (the resident atlas used to
/// submit once per plane). The caller wraps the recording + submit in [`GpuContext::scoped`]; the texture, params
/// buffer and bind group stay alive through the recorded command buffer. The format comes from [`blit_format`].
pub fn blit_record(
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    tile: &GpuTile,
    dtype: Dtype,
    w: u32,
    h: u32,
) -> GpuResult<(wgpu::Texture, wgpu::TextureView, wgpu::TextureFormat)> {
    let format = ctx.blit_fmt(dtype).ok_or_else(|| GpuError::Unsupported(format!("plane blit: dtype {dtype:?}")))?;
    let pipe = match format {
        wgpu::TextureFormat::R32Uint => &ctx.pipelines.plane_blit_u32,
        wgpu::TextureFormat::R16Float => ctx.pipelines.plane_blit_f16(),
        _ => &ctx.pipelines.plane_blit_f32,
    };
    // enc_kind/esz mirror the RGBA gather's param packing (`GpuFrame::to_rgba_texture`).
    let (enc_kind, esz) = match tile.enc {
        TileEnc::BytePerWord => (0u32, dtype.size() as u32),
        TileEnc::ElemPerWord(esz) => (1u32, esz as u32),
    };
    let off = tile.off_u32()?;
    let max_dim = ctx.device.limits().max_texture_dimension_2d;
    if w > max_dim || h > max_dim {
        return Err(GpuError::Limit(format!("plane blit: {w}x{h} exceeds max_texture_dimension_2d {max_dim}")));
    }
    let params: Vec<u32> = vec![w, h, off, enc_kind, esz, dtype.code() as u32];
    // Recycled when the pool holds this exact size: the pass below writes every texel (`gid >= w/h` only guards the
    // dispatch overshoot), so the previous tile's pixels never survive.
    let tex = ctx.plane_tex("exv.atlas.plane_blit", w, h, format, BLIT_USAGE);
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let pbuf = storage_u32(&ctx.device, &ctx.queue, "exv.plane_blit.params", &params, wgpu::BufferUsages::empty());
    let layout = pipe.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("exv.plane_blit.bg"),
        layout: &layout,
        entries: &[e(0, &tile.buf), e(1, &pbuf), et(2, &view)],
    });
    pass(ctx, enc, "exv.plane_blit.pass", pipe, &bg, [w.div_ceil(WG2), h.div_ceil(WG2), 1])?;
    Ok((tex, view, format))
}

/// Record ONE compute pass dispatching `g` workgroups of `p` with bind group `bg`. The dispatch size is
/// checked against the device limit first, so an oversized grid is a [`GpuError::Limit`] instead of a wgpu
/// validation panic. Each dependent stage gets its own pass (wgpu inserts the storage barrier between them).
pub fn pass(
    ctx: &GpuContext,
    enc: &mut wgpu::CommandEncoder,
    label: &str,
    p: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    g: [u32; 3],
) -> GpuResult<()> {
    ctx.check_dispatch(label, g)?;
    let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
    c.set_pipeline(p);
    c.set_bind_group(0, bg, &[]);
    c.dispatch_workgroups(g[0], g[1], g[2]);
    Ok(())
}

/// One `BindGroupEntry` for a whole storage buffer (batched bind groups).
pub fn e(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
}

/// Map a `MAP_READ` staging buffer, block until the copy+mapping completes, and return its bytes (the buffer
/// is unmapped again). The ONE readback codepath: every test/export copy-out goes through here, so a failed
/// map is an `Err`, never a panic. Waits through `gpu_info::wait::map_read` (completion callback + non-blocking
/// polls) instead of a blocking `Device::poll(Wait)`: the latter holds wgpu's snatch lock and froze the
/// presenting UI thread of a host sharing this device (see the module docs of `gpu_info::wait`).
pub fn map_read_blocking(device: &wgpu::Device, staging: &wgpu::Buffer) -> GpuResult<Vec<u8>> {
    gpu_info::wait::map_read(device, &staging.slice(..))?;
    let bytes = {
        let view = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| GpuError::Device(format!("get_mapped_range: {e:?}")))?;
        view.to_vec()
    };
    staging.unmap();
    Ok(bytes)
}

/// LE bytes → `u32` words (readback buffers are `u32` storage).
pub(crate) fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_chunks_splits_by_cap_and_group() {
        let c = |a: usize, b: usize, w: usize| Chunk { units: a..b, words: w };
        // cap 10: 4+4 fit, +4 spills; a group change cuts even under the cap; empty input -> no chunks.
        let v = plan_chunks([(0, 4), (0, 4), (0, 4), (1, 1), (1, 2)], 10).unwrap();
        assert_eq!(v, [c(0, 2, 8), c(2, 3, 4), c(3, 5, 3)]);
        assert!(plan_chunks([], 10).unwrap().is_empty());
        // exactly-at-cap packs, one over cuts, a unit above the cap is a typed limit.
        assert_eq!(plan_chunks([(0, 5), (0, 5), (0, 5)], 10).unwrap(), [c(0, 2, 10), c(2, 3, 5)]);
        assert!(matches!(plan_chunks([(0, 11)], 10), Err(GpuError::Limit(_))));
    }

    #[test]
    fn unclassified_hardware_outranks_software() {
        use wgpu::DeviceType::*;
        let mut v = [Cpu, Other, VirtualGpu, IntegratedGpu, DiscreteGpu];
        v.sort_by_key(|t| rank(*t));
        assert_eq!(v, [DiscreteGpu, IntegratedGpu, VirtualGpu, Other, Cpu]);
    }

    /// Limits that satisfy every kernel; each single field dropped one below its bar must be a typed `Limit` naming it.
    #[test]
    fn check_limits_gates_every_kernel_requirement() {
        let ok = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 7,
            max_compute_workgroup_storage_size: RANS_WG_BYTES.max(RT_INV_WG_BYTES),
            max_compute_invocations_per_workgroup: RT_INV_WG.max(WG),
            max_compute_workgroup_size_x: RT_INV_WG.max(WG),
            max_storage_textures_per_shader_stage: 1,
            ..wgpu::Limits::default()
        };
        assert!(check_limits(&ok).is_ok());
        type Lower = fn(&mut wgpu::Limits);
        let cases: [(&str, Lower); 5] = [
            ("max_storage_buffers_per_shader_stage", |l| l.max_storage_buffers_per_shader_stage -= 1),
            ("max_compute_workgroup_storage_size", |l| l.max_compute_workgroup_storage_size -= 1),
            ("max_compute_invocations_per_workgroup", |l| l.max_compute_invocations_per_workgroup = RT_INV_WG - 1),
            ("max_compute_workgroup_size_x", |l| l.max_compute_workgroup_size_x = RT_INV_WG - 1),
            ("max_storage_textures_per_shader_stage", |l| l.max_storage_textures_per_shader_stage = 0),
        ];
        for (name, lower) in cases {
            let mut l = ok.clone();
            lower(&mut l);
            match check_limits(&l) {
                Err(GpuError::Limit(m)) => assert!(m.contains(name), "{name} not named in {m}"),
                other => panic!("{name} below the bar must be GpuError::Limit, got {other:?}"),
            }
        }
        // The inverse's shared bytes must be what the WGSL declares (sh_blk + sh_d), not just >= the rANS tables.
        assert_eq!(RT_INV_WG_BYTES, (RT_INV_SH + RT_INV_BLOCKS * RT_DESC_WORDS as u32) * 4);
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    /// With every pool chunk in flight (cap reached) `storage_parts` falls back to fresh staging (counted) with IDENTICAL
    /// contents instead of blocking or growing the pool; with room it is served from the pool. Needs a GPU.
    #[test]
    #[ignore = "needs a GPU"]
    fn exhausted_pool_falls_back_with_same_bytes() {
        let Ok(ctx) = GpuContext::new() else { return };
        let parts = [UP_MIN as usize, 24, 4096];
        let fill = |i: usize, mut d: wgpu::WriteOnly<'_, [u8]>| {
            let v: Vec<u8> = (0..d.len()).map(|k| (k * 7 + i * 31) as u8).collect();
            d.copy_from_slice(&v);
        };
        let read = |b: &wgpu::Buffer| {
            let st = staging(&ctx.device, "rd", b.size());
            let mut e = encoder(&ctx.device, "rd");
            e.copy_buffer_to_buffer(b, 0, &st, 0, b.size());
            ctx.queue.submit([e.finish()]);
            map_read_blocking(&ctx.device, &st).expect("map")
        };
        let pooled = storage_parts(&ctx, "pooled", &parts, fill).expect("pooled");
        assert_eq!(ctx.staging_stats().1, 0, "served from the pool");
        assert!(ctx.staging_stats().0 >= UP_MIN, "a chunk is retained");
        let want = read(&pooled);
        // Take every remaining byte of the cap, keeping the chunks in flight.
        let mut hog: Vec<_> = std::iter::from_fn(|| ctx.pipelines.up.take(&ctx.device, 64 << 20)).collect();
        hog.extend(std::iter::from_fn(|| ctx.pipelines.up.take(&ctx.device, UP_MIN)));
        let fresh = storage_parts(&ctx, "fresh", &parts, fill).expect("exhausted pool must not block");
        assert_eq!(ctx.staging_stats().1, 1);
        assert_eq!(read(&fresh), want);
        drop(hog);
    }
}
