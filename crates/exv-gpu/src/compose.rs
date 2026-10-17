//! C2 — **plane-compose pass**: assemble a tile's resident single-channel planes into a
//! linear-RGBA display texture for [`crate::compose::PlaneCompose`]'s downstream consumer,
//! `exv_view::color_pass` (exposure / gamma / channel-isolate).
//!
//! # What and why
//! [`ExvResidentAtlas`](crate::ExvResidentAtlas) makes a tile resident as N single-channel
//! [`ResidentTexture`](crate::ResidentTexture) planes (`decode_level_planes`, plane order ==
//! `header().channels`). The player, however, only ever *shows* ≤4 channels at once — full RGBA,
//! or one isolated channel drawn as grey. This pass takes a caller-supplied channel→RGBA map,
//! binds the **≤4 selected planes**, and writes one `Rgba32Float` output texture that
//! `color_pass` can consume UNCHANGED (it reads a `texture_2d<f32>` linear-RGBA source). We do NOT tonemap / gamma / sRGB-encode here —
//! that stays in `color_pass`; this pass only *composes* linear planes into an RGBA image.
//!
//! # Binding design (decisive: 4 fixed sampled-texture bindings, NO `binding_array`)
//! Exactly four `texture_2d<f32>` bindings = the R,G,B,A **source planes**, selected on the CPU
//! (Rust) side from the N resident planes via the `[u32;4]` map. Rationale: the display never needs
//! more than the ≤4 selected planes — the other N−4 stay resident in the atlas untouched, so a
//! layer-switch is just "rebind different planes + re-run compose", with **no re-decode**. Four
//! fixed sampled-texture bindings are supported on *every* wgpu backend with **zero feature gates**
//! (no `TEXTURE_BINDING_ARRAY`, no non-uniform-indexing, no MAX_PLANES fallback). An unmapped slot
//! (map entry [`UNMAPPED`] or out-of-range) binds a 1×1 dummy and is masked off in the shader
//! (`valid_mask`): R/G/B → 0.0, A → 1.0. A mapped channel the tile does NOT hold (the atlas keeps only the
//! requested channel set, [`crate::Want::chans`]) is a hard error, never a sampled missing texture. A plane whose `ResidentTexture` is `R16Float` sampled as
//! `texture_2d<f32>` still returns `f32` — fine. Integer (`R32Uint`/`R32Sint`, i.e. U32 AOVs) are
//! NOT displayable through this float path and are rejected with a hard error (no silent fallback).
//!
//! # Packed (<=4-channel) tiles
//! The default lossless HTJ2K tile (RGB / RGBA F16, <=4 components, no F32/U32) is stored by jph-wgpu
//! as ONE `Rgba16Float` texture whose texel components ARE the channels in header order (1 comp:
//! R=G=B=c0, A=1; 2: (c0,c1,0,1); 3: (c0,c1,c2,1); 4: all). For such a tile the four RGBA slots sample
//! the SAME texture (binding 1) and each slot picks the component `chan_map[slot]` (0..=3) — no
//! re-decode, no repack, the tile stays GPU-resident. The shader is told via `packed` + `comp`.
//!
//! # Geometry
//! One compute dispatch per resident tile. Each tile writes into the shared target at the global
//! pixel offset `(tile_x, tile_y) = (rx * tile_w, ry * tile_h)` (nominal tile size), over the tile's
//! ACTUAL plane dimensions (edge tiles are smaller). `textureLoad` (integer coords, no sampler).
//!
//! # Where used
//! C-track resident display path: `exv-view` opens a lossless-HTJ2K `.exv`, drives
//! [`ExvResidentAtlas::request`](crate::ExvResidentAtlas::request), then calls
//! [`PlaneCompose::compose`] to get the linear-RGBA texture it feeds into `color_pass`.

use crate::{ExvResidentAtlas, GpuContext, GpuError, GpuResult, TileTex};
use exv_gpu_kernels::{WG2, encoder as enc, et, new_buf, tex2d, wgsl};
use gpu_info::wgpu;

/// Sentinel channel-map entry meaning "this RGBA slot is unmapped": the shader substitutes a
/// constant (0.0 for R/G/B, 1.0 for A) instead of sampling a plane. Also used implicitly when a
/// map index is `>=` the tile's plane count.
pub const UNMAPPED: u32 = u32::MAX;

/// Output format of the composed linear-RGBA target: 32-bit float so an F32 AOV is never truncated to half.
/// It is the SOURCE `exv_view::color_pass` reads (any float `texture_2d<f32>`; the viewer's CPU path packs the
/// same `Rgba32Float` linear layout), not the display format.
pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Uniform byte size (std140): `tile_x,tile_y,tile_w,tile_h,valid_mask,packed` (6 `u32`, padded to
/// 32 B so the `vec4<u32> comp` is 16-aligned) + 16 B of `comp` → 48 bytes. Also the dynamic-offset
/// binding size.
const UNIFORM_SIZE: u64 = 48;

// ── Shader ──────────────────────────────────────────────────────────────────────────────────────
//
// Four fixed sampled planes + one storage output. Per invocation: load each MAPPED plane's `.x`
// (single-channel), substitute the constant for unmapped slots (masked via `valid_mask`), and store
// the linear RGBA at the global pixel. `select(f, t, cond)` evaluates BOTH operands, but a
// `textureLoad` on the 1×1 dummy is well-defined (in-range at (0,0), and out-of-range loads return
// zero per WGSL) — the result is discarded by `select` regardless, so binding a dummy is safe.
const SHADER: &str = r#"
struct Uniforms {
    tile_x:     u32,   // global pixel offset x = rx * nominal_tile_w
    tile_y:     u32,   // global pixel offset y = ry * nominal_tile_h
    tile_w:     u32,   // this tile's ACTUAL width  (plane width; edge tiles < nominal)
    tile_h:     u32,   // this tile's ACTUAL height
    valid_mask: u32,   // bit i (i=0:R,1:G,2:B,3:A) set => slot i samples its plane; else constant
    packed:     u32,   // 1 => plane_r is ONE RGBA16F-packed tile; slot i reads component comp[i] of it
    comp:       vec4<u32>, // packed mode: component (0..3) feeding slot i
}

@group(0) @binding(0) var<uniform> u:      Uniforms;
@group(0) @binding(1) var plane_r:         texture_2d<f32>;
@group(0) @binding(2) var plane_g:         texture_2d<f32>;
@group(0) @binding(3) var plane_b:         texture_2d<f32>;
@group(0) @binding(4) var plane_a:         texture_2d<f32>;
@group(0) @binding(5) var out_tex:         texture_storage_2d<rgba32float, write>;

fn comp_of(v: vec4<f32>, i: u32) -> f32 {
    if i == 0u { return v.x; }
    if i == 1u { return v.y; }
    if i == 2u { return v.z; }
    return v.w;
}

@compute @workgroup_size(WG2, WG2, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.tile_w || gid.y >= u.tile_h { return; }

    let lp = vec2<i32>(i32(gid.x), i32(gid.y)); // tile-local sample coord

    // Each slot's raw sample: its own plane's `.x`, or (packed tile) a component of the shared texel.
    // Every `textureLoad` is evaluated unconditionally (all bindings are valid textures); `select`
    // then discards the masked / wrong-mode values.
    let pk = textureLoad(plane_r, lp, 0);
    let sr = select(textureLoad(plane_r, lp, 0).x, comp_of(pk, u.comp.x), u.packed != 0u);
    let sg = select(textureLoad(plane_g, lp, 0).x, comp_of(pk, u.comp.y), u.packed != 0u);
    let sb = select(textureLoad(plane_b, lp, 0).x, comp_of(pk, u.comp.z), u.packed != 0u);
    let sa = select(textureLoad(plane_a, lp, 0).x, comp_of(pk, u.comp.w), u.packed != 0u);

    let r = select(0.0, sr, (u.valid_mask & 1u) != 0u);
    let g = select(0.0, sg, (u.valid_mask & 2u) != 0u);
    let b = select(0.0, sb, (u.valid_mask & 4u) != 0u);
    let a = select(1.0, sa, (u.valid_mask & 8u) != 0u);

    let gp = vec2<i32>(i32(u.tile_x + gid.x), i32(u.tile_y + gid.y)); // image-global write coord
    textureStore(out_tex, gp, vec4<f32>(r, g, b, a));
}
"#;

// ── Reusable output texture ───────────────────────────────────────────────────────────────────

/// The composed display target, cached and re-created only when the image dimensions change.
struct ComposeTarget {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// Everything a compose pass reads: the target holds exactly this, so an identical request is skipped
/// ([`PlaneCompose::compose`]). `atlas` is the atlas' address and `decodes` its decode counter: every change of a
/// resident tile's pixels (new frame node, merged channels, re-decode after eviction) is a decode, so the pair
/// stands for the atlas content. Hits and LRU eviction of tiles outside `tiles` leave the composed region intact.
#[derive(PartialEq, Eq)]
struct Stamp {
    atlas: usize,
    decodes: u64,
    level: Option<u32>,
    tiles: Vec<(u32, u32)>,
    dims: [u32; 4],
    chan_map: [u32; 4],
}

// ── PlaneCompose ────────────────────────────────────────────────────────────────────────────────

/// GPU plane-compose pass: pipeline + bind-group layout + a 1×1 dummy plane + a growable dynamic
/// uniform buffer + a cached output texture. Build once on a [`GpuContext`]; call
/// [`compose`](Self::compose) per frame.
pub struct PlaneCompose {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// 1×1 `R32Float` placeholder bound into any unmapped RGBA slot (its samples are masked off).
    dummy_view: wgpu::TextureView,
    _dummy: wgpu::Texture,
    /// Dynamic uniform buffer (one `UNIFORM_SIZE`-aligned slot per dispatched tile) + its capacity.
    uniform: Option<(wgpu::Buffer, u64)>,
    /// Cached output texture, re-created only on an image-dimension change.
    target: Option<ComposeTarget>,
    /// What `target` was last composed from (`None` = unknown / a failed pass): the dirty check of
    /// [`compose`](Self::compose).
    stamp: Option<Stamp>,
    /// Compose passes submitted since creation (a dirty-check skip does not count): lets the viewer / tests confirm
    /// an unchanged redraw composed nothing.
    passes: u64,
}

impl PlaneCompose {
    /// Build the compute pipeline, bind-group layout and dummy plane on `ctx`'s device.
    ///
    /// The plane bindings are declared `filterable: false` (we only `textureLoad`, never sample), so
    /// BOTH `R16Float` (filterable) and `R32Float` (non-filterable without `FLOAT32_FILTERABLE`)
    /// plane textures bind without requiring any extra device feature.
    pub fn new(ctx: &GpuContext) -> Result<Self, String> {
        let device = &ctx.device;

        // The kernels crate's constants prelude supplies `WG2` (one definition of the 2-D workgroup edge).
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exv_gpu.compose"),
            source: wgpu::ShaderSource::Wgsl(wgsl(SHADER).into()),
        });

        // binding 0: dynamic uniform; 1..=4: the four sampled planes; 5: the storage output.
        let uniform_min = wgpu::BufferSize::new(UNIFORM_SIZE)
            .ok_or_else(|| "compose: zero uniform size".to_string())?;
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: Some(uniform_min),
            },
            count: None,
        }];
        for b in 1..=4u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: TARGET_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exv_gpu.compose.bgl"),
            entries: &entries,
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("exv_gpu.compose.pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("exv_gpu.compose"),
            layout: Some(&pl_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 1×1 float placeholder for unmapped slots (contents irrelevant — masked in the shader).
        let dummy = tex2d(
            device,
            "exv_gpu.compose.dummy_plane",
            1,
            1,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self { pipeline, bgl, dummy_view, _dummy: dummy, uniform: None, target: None, stamp: None, passes: 0 })
    }

    /// Compose passes submitted since creation (requests skipped by the dirty check are not counted).
    pub fn passes(&self) -> u64 {
        self.passes
    }

    /// Borrow the current composed target texture + view, if [`compose`](Self::compose) has produced
    /// one for the current dimensions. Mainly for callers that want the view after the fact.
    pub fn target(&self) -> Option<(&wgpu::Texture, &wgpu::TextureView)> {
        self.target.as_ref().map(|t| (&t.tex, &t.view))
    }

    /// Compose the requested resident `tiles` of `atlas` (at `level`) into a linear-RGBA
    /// [`TARGET_FORMAT`] texture and return `(texture, view)`.
    ///
    /// * `ctx`               — the shared GPU context (device/queue).
    /// * `atlas`             — the resident tile cache; each requested tile must already be resident
    ///   (via [`ExvResidentAtlas::request`](crate::ExvResidentAtlas::request)) — a non-resident tile
    ///   is skipped (its output region keeps whatever was there).
    /// * `level`             — LOD (`None` = full resolution), matching what was `request`ed.
    /// * `tiles`             — the tile grid coords `(rx, ry)` to compose.
    /// * `img_w`, `img_h`    — full image (or LOD) dimensions → the target texture size.
    /// * `tile_w`, `tile_h`  — NOMINAL tile size (the header's `tile_w`/`tile_h`), used only to place
    ///   each tile at `(rx*tile_w, ry*tile_h)`; the per-tile ACTUAL extent comes from the plane.
    /// * `chan_map`          — `[r, g, b, a]` plane indices into the tile's planes; [`UNMAPPED`]
    ///   (or an out-of-range index) leaves that slot at its constant (R/G/B → 0.0, A → 1.0).
    ///
    /// Returns `Err` only for a real GPU/setup problem (e.g. a selected plane is an integer AOV that
    /// this float path cannot display) — never a silent fallback.
    ///
    /// **Dirty check:** a request identical to the one the target already holds (same `atlas` with the same decode
    /// count, `level`, `tiles`, sizes and `chan_map`) records nothing and returns the target as is, so an unchanged
    /// redraw (hover, a display-param change that only the colour pass reads) costs no compose pass. A
    /// `PlaneCompose` therefore belongs to ONE atlas for its lifetime (the viewer builds both together).
    #[allow(clippy::too_many_arguments)]
    pub fn compose(
        &mut self,
        ctx: &GpuContext,
        atlas: &ExvResidentAtlas,
        level: Option<u32>,
        tiles: &[(u32, u32)],
        img_w: u32,
        img_h: u32,
        tile_w: u32,
        tile_h: u32,
        chan_map: [u32; 4],
    ) -> Result<(&wgpu::Texture, &wgpu::TextureView), String> {
        let stamp = Stamp {
            atlas: std::ptr::from_ref(atlas) as usize,
            decodes: atlas.decode_count(),
            level,
            tiles: tiles.to_vec(),
            dims: [img_w, img_h, tile_w, tile_h],
            chan_map,
        };
        if self.target.is_none() || self.stamp.as_ref() != Some(&stamp) {
            // A failed pass leaves the target half-written: forget the stamp so the retry composes.
            self.stamp = None;
            // All GPU work runs inside the crate's error-capture scope: a wgpu validation/OOM error (e.g. a
            // target texture the device cannot allocate) is an `Err` for the viewer, not a panic.
            ctx.scoped(|| self.compose_into(ctx, atlas, level, tiles, img_w, img_h, tile_w, tile_h, chan_map))
                .map_err(|e| e.to_string())?;
            self.stamp = Some(stamp);
        }
        let t = self.target.as_ref().ok_or_else(|| "compose: no target".to_string())?;
        Ok((&t.tex, &t.view))
    }

    /// Body of [`compose`](Self::compose): (re)build the target, record + submit the per-tile dispatches.
    /// With nothing resident to draw the (possibly stale) target is left as is.
    #[allow(clippy::too_many_arguments)]
    fn compose_into(
        &mut self,
        ctx: &GpuContext,
        atlas: &ExvResidentAtlas,
        level: Option<u32>,
        tiles: &[(u32, u32)],
        img_w: u32,
        img_h: u32,
        tile_w: u32,
        tile_h: u32,
        chan_map: [u32; 4],
    ) -> GpuResult<()> {
        let bad = |m: &str| GpuError::Corrupt(m.to_string());
        if img_w == 0 || img_h == 0 {
            return Err(bad("compose: zero image dimensions"));
        }
        let device = &ctx.device;
        // The Rgba32Float target must fit the device's 2-D texture limit (16k² RGBA32F is already 4 GiB; the
        // OOM scope catches the rest) — a typed Limit error lets the viewer pick LOD/CPU instead of panicking.
        let max_dim = device.limits().max_texture_dimension_2d;
        if img_w > max_dim || img_h > max_dim {
            return Err(GpuError::Limit(format!("compose: target {img_w}x{img_h} exceeds max_texture_dimension_2d {max_dim}")));
        }
        let uniform_min = wgpu::BufferSize::new(UNIFORM_SIZE).ok_or_else(|| bad("compose: zero uniform size"))?;

        // 1) Ensure the cached output texture matches the requested image dimensions.
        self.ensure_target(device, img_w, img_h);

        // 2) Ensure the dynamic uniform buffer holds one aligned slot per requested tile.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = UNIFORM_SIZE.div_ceil(align.max(1)) * align.max(1);
        let needed = stride
            .checked_mul(tiles.len().max(1) as u64)
            .ok_or_else(|| bad("compose: uniform size overflow"))?;
        self.ensure_uniform(device, needed);

        // 3) Build the per-tile uniform payload + bind lists. A tile that is not resident (get ==
        //    None) is silently skipped — its output region simply is not written this pass.
        let target = self.target.as_ref().ok_or_else(|| bad("compose: no target"))?;
        let uniform_buf = self.uniform.as_ref().map(|(b, _)| b).ok_or_else(|| bad("compose: no uniform buffer"))?;

        // Collect (dynamic_offset, [4 plane views], valid_mask, tile_w, tile_h) for each resident tile.
        struct Job<'a> {
            offset: u32,
            views: [&'a wgpu::TextureView; 4],
            tw: u32,
            th: u32,
        }
        let mut jobs: Vec<Job> = Vec::with_capacity(tiles.len());
        let mut uni_bytes: Vec<u8> = Vec::with_capacity(tiles.len() * stride as usize);

        for &(rx, ry) in tiles {
            let planes = match atlas.get(level, rx, ry) {
                Some(p) => p,
                None => continue, // not resident → skip (caller controls residency)
            };
            // A <=4-channel GPU HTJ2K tile is ONE RGBA16F-packed texture whose components are the channels
            // (see the module docs): bind it once and let each slot pick its component.
            let (packed, first) = match &planes.tex {
                TileTex::Packed(t) => (true, t),
                TileTex::Planes(slots) => match slots.iter().flatten().next() {
                    Some(t) => (false, t),
                    None => continue, // no plane resident — nothing to draw
                },
            };
            let (tw, th) = (first.tex.width(), first.tex.height());

            // Resolve the 4 RGBA slots to plane views (or the dummy), building the valid mask.
            let mut views: [&wgpu::TextureView; 4] =
                [&self.dummy_view, &self.dummy_view, &self.dummy_view, &self.dummy_view];
            let mut valid_mask: u32 = 0;
            let mut comp = [0u32; 4];
            if packed {
                views[0] = &first.view;
            }
            for (slot, &idx) in chan_map.iter().enumerate() {
                if idx == UNMAPPED {
                    continue;
                }
                if packed {
                    // Channel `idx` of the packed texel (a header channel index < 4 by construction).
                    if idx < 4 {
                        comp[slot] = idx;
                        valid_mask |= 1u32 << slot;
                    }
                    continue;
                }
                let TileTex::Planes(slots) = &planes.tex else { continue };
                let plane = match slots.get(idx as usize) {
                    None => continue, // out-of-range index → treat as unmapped
                    Some(Some(p)) => p,
                    // A routed channel the atlas does not hold: the caller requested a different channel set than it
                    // composes. Never sample a missing texture, never draw a silent black slot.
                    Some(None) => {
                        return Err(GpuError::Unsupported(format!(
                            "compose: channel {idx} of tile ({rx},{ry}) is not resident (request it in `Want::chans`)"
                        )));
                    }
                };
                // This float compose path cannot display an integer AOV (U32 → R32Uint/R32Sint):
                // binding it to a `texture_2d<f32>` slot is invalid. Hard-error (no silent fallback).
                match plane.tex.format() {
                    wgpu::TextureFormat::R16Float | wgpu::TextureFormat::R32Float => {}
                    other => {
                        return Err(GpuError::Unsupported(format!(
                            "compose: plane {idx} format {other:?} is not float-displayable (integer AOVs need a dedicated path)"
                        )));
                    }
                }
                views[slot] = &plane.view;
                valid_mask |= 1u32 << slot;
            }

            let offset = uni_bytes.len() as u32;
            // Pack the uniform slot (UNIFORM_SIZE bytes, remainder padded to `stride`).
            let tile_x = rx.saturating_mul(tile_w);
            let tile_y = ry.saturating_mul(tile_h);
            let mut slot_buf = [0u8; UNIFORM_SIZE as usize];
            slot_buf[0..4].copy_from_slice(&tile_x.to_le_bytes());
            slot_buf[4..8].copy_from_slice(&tile_y.to_le_bytes());
            slot_buf[8..12].copy_from_slice(&tw.to_le_bytes());
            slot_buf[12..16].copy_from_slice(&th.to_le_bytes());
            slot_buf[16..20].copy_from_slice(&valid_mask.to_le_bytes());
            slot_buf[20..24].copy_from_slice(&u32::from(packed).to_le_bytes());
            for (i, c) in comp.iter().enumerate() {
                slot_buf[32 + 4 * i..36 + 4 * i].copy_from_slice(&c.to_le_bytes());
            }
            uni_bytes.extend_from_slice(&slot_buf);
            uni_bytes.resize(uni_bytes.len() + (stride as usize - UNIFORM_SIZE as usize), 0);

            jobs.push(Job { offset, views, tw, th });
        }

        if jobs.is_empty() {
            // Nothing resident to draw; the (possibly stale) target is returned rather than erroring.
            return Ok(());
        }

        ctx.queue.write_buffer(uniform_buf, 0, &uni_bytes);

        // 4) One dispatch per tile, each with its own bind group (distinct plane views) + dynamic
        //    uniform offset. All uniform slots were written above (pre-submit), so every dynamic
        //    offset reads its own correct region — no write-after-read hazard on a shared buffer.
        let mut encoder = enc(device, "exv_gpu.compose.enc");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("exv_gpu.compose.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for job in &jobs {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("exv_gpu.compose.bg"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: uniform_buf,
                                offset: 0,
                                size: Some(uniform_min),
                            }),
                        },
                        et(1, job.views[0]),
                        et(2, job.views[1]),
                        et(3, job.views[2]),
                        et(4, job.views[3]),
                        et(5, &target.view),
                    ],
                });
                pass.set_bind_group(0, &bg, &[job.offset]);
                pass.dispatch_workgroups(job.tw.div_ceil(WG2), job.th.div_ceil(WG2), 1);
            }
        }
        ctx.queue.submit(Some(encoder.finish()));
        self.passes += 1;
        Ok(())
    }

    /// (Re)create the output texture only when the requested dimensions differ from the cache.
    fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let ok = matches!(&self.target, Some(t) if t.width == width && t.height == height);
        if ok {
            return;
        }
        // STORAGE_BINDING: written by this pass. TEXTURE_BINDING: sampled by color_pass.
        // COPY_SRC: read back by the verification test.
        let tex = tex2d(
            device,
            "exv_gpu.compose.target",
            width,
            height,
            TARGET_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.target = Some(ComposeTarget { tex, view, width, height });
    }

    /// Grow (never shrink) the dynamic uniform buffer to hold at least `needed` bytes.
    fn ensure_uniform(&mut self, device: &wgpu::Device, needed: u64) {
        let ok = matches!(&self.uniform, Some((_, cap)) if *cap >= needed);
        if ok {
            return;
        }
        let buf = new_buf(
            device,
            "exv_gpu.compose.uniforms",
            needed.max(UNIFORM_SIZE),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.uniform = Some((buf, needed.max(UNIFORM_SIZE)));
    }
}
