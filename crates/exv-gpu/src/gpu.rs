//! `exv-gpu` — wgpu/WGSL decode + streaming upload into VRAM.
//!
//! # What and why
//! GPU-native hot path: per tile (workgroup) — interleaved-rANS decode (N lanes ↔ N threads) →
//! inverse transform (tile-local) → de-planarize into a texture/storage buffer. Everything integer →
//! **GPU == CPU reference bit-for-bit** (Phase 5 gate).
//!
//! # Invariants (see `PLAN.md` §3.1, §3.4–3.5)
//! - **GPU-native, NO readback in the hot path:** the result stays in VRAM and is consumed by the GPU
//!   (viewer/processing). Readback (GPU→CPU) — only explicit export/save and bit-exact verification
//!   (test-only, as here in [`decode_block_gpu`]).
//! - WGSL rANS — `RansConfig` from `rans-rs` (32-bit state, pow2 `M`, ≤256 alphabet). The decoders are the
//!   shared-memory/scalar kernels (source of truth); a subgroup fast-path was planned but is not implemented.
//! - wgpu comes from `gpu_info::wgpu` (a single version, 30, across the whole graph).
//!
//! # Status
//! Single-tile and whole-image decode (LosslessFast, LosslessWave, EXV-RT) plus the resident atlas/compose path,
//! all verified bit-for-bit against the CPU reference (`cargo test -p exv-gpu -- --ignored` on a real adapter).
//! Errors are typed ([`GpuError`] / [`GpuDecodeError`]): device limits are `Limit`, not `Corrupt`.

// The pure GPU decode kernels + `GpuContext` now live in the sibling `exv-gpu-kernels` crate and are
// re-exported below (public surface stays byte-identical); this crate keeps the orchestration layer.
mod dispatch;
#[cfg(feature = "htj2k-backend")]
pub(crate) mod wgpu_tile_atlas;
#[cfg(feature = "htj2k-backend")]
mod htj2k;
#[cfg(feature = "htj2k-backend")]
pub mod jph_wgpu_rs {
    pub use crate::htj2k::jph_compat::*;
}
#[cfg(feature = "htj2k-backend")]
mod atlas;
// Async half of the atlas (plan / worker decode / install); same gate, same lockstep.
#[cfg(feature = "htj2k-backend")]
mod atlas_job;
// C2 resident-plane -> linear-RGBA compose pass; gated with the resident-decode backend.
#[cfg(feature = "htj2k-backend")]
pub mod compose;
// Shared fixtures of the in-crate GPU gates below.
#[cfg(test)]
mod testkit;

/// Crate-internal prelude: re-exported so each module can `use crate::prelude::*` (a glob, hence
/// warning-exempt) instead of repeating these imports. Pure plumbing for the module split.
pub(crate) mod prelude {
    pub(crate) use gpu_info::wgpu;
    pub(crate) use exv_format::{BlobKind, CodecId, DecodeLimits, Dtype, bufpool, classify_blob};
    pub(crate) use std::sync::Arc;
    #[cfg(feature = "htj2k-backend")]
    pub(crate) use crate::{jph_wgpu_rs, wgpu_tile_atlas};
}

// Public API re-exports — the crate's surface is byte-identical to the pre-split monolith, so
// exv-view / exv-cli / exv-capi need ZERO changes.
pub use exv_gpu_kernels::{
    ExvrtWarmTile, GpuContext, GpuError, GpuResult, GpuTile, TileEnc, decode_bc6h_into, decode_block_gpu,
    decode_tile_bc6h_vram, decode_tile_exvrt_vram, decode_tile_gpu_fast, decode_tile_gpu_fast_vram,
    decode_tile_gpu_wave, decode_tile_gpu_wave_vram, inverse_wavelet_gpu, prepare_tile_exvrt_vram, read_back,
};
pub use dispatch::{
    GpuDecodeError, GpuDecoder, GpuFrame, GpuImage, RgbaFormat, decode_still_gpu, decode_wave_batched,
    read_back_image,
};
#[cfg(feature = "htj2k-backend")]
pub use htj2k::{
    ContentKey, DECODE_BUDGET, DisplayKey, Htj2kGpuDecoder, ResidentTexture, decode_htj2k_resident,
    decode_htj2k_still_planes_gpu,
};
#[cfg(feature = "htj2k-backend")]
pub use atlas::{AtlasStats, Clip, ExvResidentAtlas, PlaneTex, Planes, TileTex, Want};
#[cfg(feature = "htj2k-backend")]
pub use atlas_job::{Bytes, Done, Plan, Wake};
/// The channel-set type of [`Want::chans`] (the same type `exv_video` / `exv_codec` decode with).
#[cfg(feature = "htj2k-backend")]
pub use exv_codec::frame::Chans;

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::testkit::*;
    use exv_gpu_kernels::*;
    use crate::dispatch::*;
    #[cfg(feature = "htj2k-backend")]
    use crate::atlas::*;
    #[cfg(feature = "htj2k-backend")]
    use crate::Chans;
    #[cfg(feature = "htj2k-backend")]
    use crate::htj2k::*;
    use exv_codec::image::sample_f32;
    use exv_format::half::f32_to_half;
    #[cfg(feature = "htj2k-backend")]
    use exv_format::half::half_to_f32;
    use rans::{RansConfig, decode_capped, encode, prepare};

    /// The kernels crate's compile-time `exp2` table is a copy of the codec's (kernels may not depend on
    /// exv-codec): pin them bit-for-bit, so the GPU lossy decode cannot drift from `hdr::decode_value`.
    #[test]
    fn exp2_table_matches_hdr() {
        let cpu: Vec<u32> = exv_codec::hdr::exp2_table(256).iter().map(|f| f.to_bits()).collect();
        assert_eq!(cpu, EXP2_TAB_256.to_vec());
    }

    /// The kernels' copies of the lossy-log constants (scale shift, table length, code saturation) equal the
    /// codec's — the WGSL `HDR_*` prelude constants derive from them, so a codec change must fail here.
    #[test]
    fn hdr_consts_match_codec() {
        use exv_codec::hdr::{DEFAULT_SCALE, encode_value};
        assert_eq!(1u32 << HDR_SHIFT, DEFAULT_SCALE);
        assert_eq!(EXP2_N, DEFAULT_SCALE as usize);
        assert_eq!(encode_value(f32::INFINITY, DEFAULT_SCALE), HDR_MAX_CODE);
    }

    /// `AtlasStats` renders the status-bar line for a bounded and an unbounded (budget 0) atlas.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    fn atlas_stats_display() {
        let mib = 1usize << 20;
        let s = AtlasStats { resident_bytes: 3 * mib, budget_bytes: 8 * mib, decodes: 5, reused: 2, planes: 15, uploaded_bytes: 6 * mib as u64, inflight: 0, pooled_bytes: 0 };
        assert_eq!(s.to_string(), "atlas 3.0 MiB / 8.0 MiB, 5 decodes, 2 reused, 15 planes, 6.0 MiB up");
        let u = AtlasStats { budget_bytes: 0, ..s };
        assert_eq!(u.to_string(), "atlas 3.0 MiB / unbounded, 5 decodes, 2 reused, 15 planes, 6.0 MiB up");
        let f = AtlasStats { inflight: 12, ..s };
        assert_eq!(f.to_string(), "atlas 3.0 MiB / 8.0 MiB, 5 decodes, 2 reused, 15 planes, 6.0 MiB up, 12 in flight");
        let p = AtlasStats { pooled_bytes: 4 * mib as u64, ..s };
        assert_eq!(p.to_string(), "atlas 3.0 MiB / 8.0 MiB, 5 decodes, 2 reused, 15 planes, 6.0 MiB up, 4.0 MiB pooled");
    }

    /// G5 gate: GPU HTJ2K (jph-wgpu) decode of a lossless **all-16-bit** multi-component `.exv` ==
    /// CPU `read_still`, bit-for-bit. 16-bit (F16) components ride the binary-complement NLT — this
    /// exercises the GPU entropy + iDWT + per-component NLT-inverse end-to-end on the GPU-decodable
    /// set. (32-bit components need jph-wgpu's not-yet-implemented 64-bit HT path; see the next test.)
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_htj2k_f16_planes_match_cpu() {
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};
        // Non-multiple-of-tile dims -> edge tiles; all-F16 channels (the GPU-decodable set).
        let img = synth_f16(300, 200);
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = decode_htj2k_still_planes_gpu(&ctx, &bytes, &DecodeLimits::default()).expect("gpu htj2k decode");
        assert_eq!(gpu.len(), cpu.planes.len(), "channel count");
        for (ci, (g, c)) in gpu.iter().zip(&cpu.planes).enumerate() {
            assert_eq!(g, c, "channel {ci} ({}) GPU != CPU", img.header.channels[ci].name);
        }
    }

    /// Capability boundary: a `.exv` with 32-bit components (F32/U32) can require jph-wgpu's 64-bit
    /// HT magnitude path, which the GPU kernel does not implement. The bridge must surface that as a
    /// clean `Err` (so the caller CPU-falls-back) and NEVER panic / silently corrupt.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_htj2k_32bit_falls_back_cleanly() {
        use exv_codec::Profile;
        use exv_video::write_still;
        let img = synth_mc(300, 200); // contains Z:F32 + id:U32
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let ctx = GpuContext::new().expect("gpu ctx");
        // Returns Err (e.g. "64-bit magnitude path unsupported"); the caller uses CPU read_still.
        assert!(
            decode_htj2k_still_planes_gpu(&ctx, &bytes, &DecodeLimits::default()).is_err(),
            "32-bit-component HTJ2K must cleanly defer to CPU, not decode on the GPU kernel"
        );
    }

    /// N5 gate: BF16 and unsigned 8/16-bit HTJ2K components are CPU-decode only. jph-wgpu's resident pack reads a
    /// 16-bit signed component as an IEEE half, so a BF16 file used to take the RGBA16F path and DISPLAY WRONG
    /// NUMBERS silently (the same hazard the U8/U16 gate exists for): the still-planes GPU entry must refuse it
    /// (`exv_codec::htj2k_gpu_decodable`), and the atlas, even with the GPU route FORCED, must take its CPU route and
    /// show the exact CPU display values (`atlas_widened_dtypes_match_cpu_display` pins every pixel).
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_htj2k_bf16_and_unsigned_are_cpu_only() {
        use exv_codec::Profile;
        use exv_video::write_still;
        use jph_wgpu_rs::{ContentKey, DisplayKey};
        let ctx = GpuContext::new().expect("gpu ctx");
        for dt in [Dtype::Bf16, Dtype::U8, Dtype::U16] {
            let n = 64 * 64 * dt.size();
            let planes: Vec<Vec<u8>> =
                (0..3u8).map(|c| (0..n).map(|i| (i as u8).wrapping_mul(7).wrapping_add(c)).collect()).collect();
            let img = exv_codec::image::Image::new(header(64, 64, 64, 64, 6, Vec::new(), &[("R", dt), ("G", dt), ("B", dt)]), planes)
                .expect("image");
            let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
            assert!(decode_htj2k_still_planes_gpu(&ctx, &bytes, &DecodeLimits::default()).is_err(), "{dt:?}: still planes must refuse the GPU");
            let mut atlas = new_atlas(&ctx);
            let idx = exv_video::VideoIndex::parse(&bytes).expect("idx");
            let r = atlas.request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &tile_grid(1, 1), level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0));
            assert!(r.is_ok(), "{dt:?}: the forced-GPU atlas must route to its CPU decode: {r:?}");
            let lod = atlas.request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &tile_grid(1, 1), level: Some(1), chans: Chans::All }, ContentKey(1), DisplayKey(0));
            assert!(lod.is_err(), "{dt:?}: an HTJ2K LOD is a jph-wgpu decode: refused for CPU-only components");
        }
    }

    /// Trap 1 gate: every dtype a valid clip can hold is resident-displayable. BF16/U8/U16 channels of HTJ2K (default and
    /// forced-GPU route), EXV-RT (RAW sub-blobs) and lossless wave (rANS sub-blobs: U8/U16 have no exact device blit, so
    /// they take the CPU route; BF16 takes the GPU wave + blit) clips decode into resident planes whose EVERY texel equals
    /// the CPU display value `sample_f32` of `read_still` (bit compare), and `ExvResidentAtlas::supports` admits them at
    /// full resolution while refusing what the atlas cannot do (LOD outside HTJ2K, a codec with no resident route).
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_widened_dtypes_match_cpu_display() {
        use exv_codec::Profile;
        use jph_wgpu_rs::{ContentKey, DisplayKey};
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = jph_wgpu_rs::Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let (w, h, tile) = (300u32, 200u32, 128u32);
        let mut key = 10u64;
        for dt in [Dtype::Bf16, Dtype::U8, Dtype::U16] {
            let n = (w * h) as usize * dt.size();
            // Smooth-ish ramps (so rANS shrinks them) plus every byte value; BF16 stays finite (sign/exponent bytes bounded).
            let planes: Vec<Vec<u8>> = (0..3u8)
                .map(|c| (0..n).map(|i| if dt == Dtype::Bf16 && i % 2 == 1 { 0x3c + (i / 512 % 4) as u8 } else { ((i / 7) as u8).wrapping_add(c) }).collect())
                .collect();
            let chans = [("R", dt), ("G", dt), ("B", dt)];
            for (what, cid, profile, forced) in [
                ("htj2k", 6u32, Profile::htj2k(), false),
                ("htj2k forced", 6, Profile::htj2k(), true),
                ("exv-rt", 4, Profile::exv_rt_dwt(16, 0), false),
                ("wave", 2, Profile::lossless_wave(), false),
            ] {
                let img = exv_codec::image::Image::new(header(w, h, tile, tile, cid, profile.params(), &chans), planes.clone()).expect("image");
                let bytes = exv_video::write_still(&img, &profile).expect("encode .exv");
                let cpu = exv_video::read_still(&bytes).expect("cpu read").planes;
                let idx = exv_video::VideoIndex::parse(&bytes).expect("idx");
                let dts = [dt; 3];
                ExvResidentAtlas::supports(&profile, &dts, None).unwrap_or_else(|e| panic!("{what} {dt:?}: supports: {e}"));
                if what != "htj2k" {
                    assert!(ExvResidentAtlas::supports(&profile, &dts, Some(1)).is_err(), "{what} {dt:?}: LOD must be refused");
                }
                let mut atlas = ExvResidentAtlas::new(&ctx, 512 << 20, DECODE_BUDGET).expect("atlas").with_htj2k_gpu(forced);
                let grid = tile_grid(w.div_ceil(tile), h.div_ceil(tile));
                key += 1;
                atlas
                    .request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &grid, level: None, chans: Chans::All }, ContentKey(key), DisplayKey(0))
                    .unwrap_or_else(|e| panic!("{what} {dt:?}: request: {e}"));
                for &(rx, ry) in &grid {
                    let (x0, y0, tw, th) = img.header.tile_rect(rx, ry);
                    let p = atlas.get(None, rx, ry).expect("resident");
                    for (ci, cpu_ch) in cpu.iter().enumerate() {
                        let plane = p.plane(ci).unwrap_or_else(|| panic!("{what} {dt:?}: ch{ci} not a plane"));
                        assert_eq!(plane.tex.format(), wgpu::TextureFormat::R32Float, "{what} {dt:?}: widened / blitted to f32");
                        let got = jph_wgpu_rs::texture::read_plane(&gpu, plane, tw, th, plane.tex.format()).expect("read plane");
                        for ty in 0..th {
                            for tx in 0..tw {
                                let want = sample_f32(cpu_ch, dt, ((y0 + ty) * w + x0 + tx) as usize);
                                let g = got[(ty * tw + tx) as usize];
                                assert_eq!(g.to_bits(), want.to_bits(), "{what} {dt:?}: tile ({rx},{ry}) ch{ci} px({tx},{ty}): {g} vs {want}");
                            }
                        }
                    }
                }
            }
        }
        // A codec with no resident route is refused up front, with the reason.
        assert!(ExvResidentAtlas::supports(&exv_codec::Profile::htj2k_ldr(16), &[Dtype::U8], None).is_err());
    }

    /// Trap 3 gate: the GPU entry points obey the CALLER's `DecodeLimits`. A flipped byte inside a RAW (CPU-fallback)
    /// sub-blob is a checksum error under `Verify::On` and decodes (to the flipped value) under `Verify::Off`, on the
    /// whole-still path AND the resident atlas (whose policy rides on the clip's `VideoIndex::with_limits`); a budget
    /// below the CPU-fallback host plane / one tile's upload is the typed budget error, not a device limit.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_entry_points_obey_caller_limits() {
        use exv_codec::Profile;
        use exv_format::Verify;
        use jph_wgpu_rs::{ContentKey, DisplayKey};
        let ctx = GpuContext::new().expect("gpu ctx");
        let (w, h) = (256u32, 128u32);
        let n = (w * h) as usize;
        // R:F16 (GPU block codec) + id:U32 (RAW -> CPU fallback). The id plane holds unique values, so one is findable.
        let id: Vec<u8> = (0..n as u32).flat_map(|i| (0xC0DE_0000u32 | i).to_le_bytes()).collect();
        let img = exv_codec::image::Image::new(
            header(w, h, 128, 128, 4, Profile::exv_rt_dwt(16, 0).params(), &[("R", Dtype::F16), ("id", Dtype::U32)]),
            vec![f16_plane(w, h, 0, 0), id],
        )
        .expect("image");
        let good = exv_video::write_still(&img, &Profile::exv_rt_dwt(16, 0)).expect("encode");
        let needle = (0xC0DE_0000u32 | 5).to_le_bytes();
        let at = good.windows(4).position(|c| c == needle).expect("raw id value in the container");
        assert_eq!(good.windows(4).filter(|c| *c == needle).count(), 1, "unique needle");
        let mut bad = good.clone();
        bad[at] ^= 0x77;
        let on = DecodeLimits::default();
        let off = DecodeLimits::default().with_verify(Verify::Off);

        // Whole still.
        assert!(decode_still_gpu(&ctx, &good, &on).is_ok());
        match decode_still_gpu(&ctx, &bad, &on) {
            Err(GpuDecodeError::Corrupt(m)) => assert!(m.to_lowercase().contains("checksum") || m.contains("CRC"), "{m}"),
            other => panic!("Verify::On must reject the flipped blob: {:?}", other.err()),
        }
        let img_off = decode_still_gpu(&ctx, &bad, &off).expect("Verify::Off decodes the flipped blob");
        let planes = read_back_image(&ctx, &img_off).expect("readback");
        assert_eq!(&planes[1][20..24], &[needle[0] ^ 0x77, needle[1], needle[2], needle[3]], "the flipped byte is decoded as-is");
        match decode_still_gpu(&ctx, &good, &DecodeLimits::new(4096)) {
            Err(GpuDecodeError::Budget(m)) => assert!(m.contains("CPU-fallback plane"), "{m}"),
            other => panic!("a 4 KiB budget must refuse the U32 host plane: {:?}", other.err()),
        }

        // Resident atlas: the policy is the clip index's.
        let req = |data: &[u8], lim: DecodeLimits, k: u64| {
            let idx = exv_video::VideoIndex::parse(data).expect("idx").with_limits(lim);
            let mut atlas = ExvResidentAtlas::new(&ctx, 512 << 20, DECODE_BUDGET).expect("atlas");
            atlas.request(Clip { data, idx: &idx }, 0, Want { tiles: &tile_grid(2, 1), level: None, chans: Chans::All }, ContentKey(k), DisplayKey(0))
        };
        assert!(req(&good, on, 1).is_ok());
        let e = req(&bad, on, 2).expect_err("Verify::On must reject the flipped blob");
        assert!(e.to_lowercase().contains("checksum") || e.contains("CRC"), "{e}");
        assert!(req(&bad, off, 3).is_ok(), "Verify::Off decodes the flipped blob");
        let e = req(&good, DecodeLimits::new(4096), 4).expect_err("budget below one tile's upload");
        assert!(e.contains("decode budget"), "{e}");
    }

    /// `ExvResidentAtlas` gate — the out-of-core viewer path for big lossless-HTJ2K images: decode
    /// ONLY the requested (visible) tiles into the atlas, bit-exact per tile vs CPU `read_still`; a
    /// re-request is a cache HIT (no re-decode); unrequested tiles are never resident. This is the
    /// 16k–65k mechanism (we never decode the whole image — only the viewport's tiles).
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn exv_resident_atlas_decodes_only_visible_tiles_bit_exact() {
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};
        use jph_wgpu_rs::texture::read_plane;
        use jph_wgpu_rs::{ContentKey, DisplayKey, Gpu};

        let (w, h) = (600u32, 400u32);
        let (tw_full, th_full) = (256u32, 256u32); // synth_f16 tiling -> 3x2 = 6 tiles
        let img = synth_f16(w, h);
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());

        let mut atlas = new_atlas(&ctx);
        // The clip index is parsed ONCE and reused by every request (the per-redraw viewer pattern).
        let idx = exv_video::VideoIndex::parse(&bytes).expect("idx");
        // Request a SUBSET of the 6 tiles (the "visible" ones), full resolution.
        let want = [(0u32, 0u32), (2u32, 1u32)];
        atlas.request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &want, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("request");
        assert_eq!(atlas.decode_count(), want.len() as u64, "only the requested tiles are decoded");
        // The stats snapshot agrees with the individual counters and renders for the status bar.
        let st = atlas.stats();
        assert_eq!(st.decodes, atlas.decode_count());
        assert!(st.to_string().contains("decodes"), "stats display: {st}");

        // Bit-exact per requested tile. synth_f16 is 5×F16 (>4 channels) → the N-plane path: each
        // component is its OWN single-channel resident texture (read per its format). Compare R,G,B.
        for &(rx, ry) in &want {
            let x0 = rx * tw_full;
            let y0 = ry * th_full;
            let tw = (w - x0).min(tw_full);
            let th = (h - y0).min(th_full);
            let planes = atlas.get(None, rx, ry).expect("requested tile is resident");
            assert!((0..5).all(|ci| planes.plane(ci).is_some()), "tile({rx},{ry}) has one texture per component");
            for ch in 0..3usize {
                let plane = planes.plane(ch).expect("plane");
                let vals = read_plane(&gpu, plane, tw, th, plane.tex.format()).expect("read plane");
                for &(tx, ty) in &[(0u32, 0u32), (tw - 1, th - 1), (tw / 2, th / 2)] {
                    let img_idx = ((y0 + ty) * w + (x0 + tx)) as usize;
                    let bits = u16::from_le_bytes([cpu.planes[ch][img_idx * 2], cpu.planes[ch][img_idx * 2 + 1]]);
                    let want_v = half_to_f32(bits);
                    let got = vals[(ty * tw + tx) as usize];
                    assert_eq!(got, want_v, "tile({rx},{ry}) px({tx},{ty}) ch{ch}");
                }
            }
        }

        // An unrequested tile is NOT resident (we never decoded the whole image).
        assert!(atlas.get(None, 1, 0).is_none(), "unrequested tile must not be resident");

        // Re-requesting the same tiles is a cache HIT — no additional decodes.
        atlas.request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &want, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("re-request");
        assert_eq!(atlas.decode_count(), want.len() as u64, "re-request is a cache hit (no re-decode)");
        // ... and an all-hit request does not even open the frame (no index re-parse per redraw).
        let opens = FRAME_OPENS.load(std::sync::atomic::Ordering::Relaxed);
        for _ in 0..5 {
            atlas.request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &want, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("hit");
        }
        assert_eq!(FRAME_OPENS.load(std::sync::atomic::Ordering::Relaxed), opens, "all-hit requests must not re-open the frame");
    }

    /// C2 gate: [`PlaneCompose`](crate::compose::PlaneCompose) composes a `>4`-channel (N-plane)
    /// lossless-HTJ2K still into a linear-RGBA `Rgba32Float` texture that is **bit-exact** vs CPU
    /// `read_still`. Maps components 0,1,2 -> R,G,B; the A slot is left [`UNMAPPED`] (=> constant
    /// 1.0). Lossless F16 -> f32 is exact, so the max abs diff must be exactly 0. Skips (no fail) if
    /// no GPU is present.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn compose_gt4ch_rgb_matches_cpu_read_still() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let (tw, th) = (256u32, 256u32); // synth_f16 tiling -> 3x2 = 6 tiles (edge tiles smaller)
        let img = synth_f16(w, h); // 5x F16 -> N-plane resident path (>4 channels)
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP compose_gt4ch_rgb_matches_cpu_read_still: no GPU ({e})");
                return;
            }
        };

        let mut atlas = new_atlas(&ctx);
        let tiles = tile_grid(3, 2);
        atlas
            .request(Clip { data: &bytes, idx: &exv_video::VideoIndex::parse(&bytes).expect("idx") }, 0, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0))
            .expect("request");

        let mut compose = PlaneCompose::new(&ctx).expect("compose");
        // ch0 -> R, ch1 -> G, ch2 -> B, A unmapped (=> 1.0).
        let map = [0u32, 1, 2, UNMAPPED];
        let (tex, _view) = compose
            .compose(&ctx, &atlas, None, &tiles, w, h, tw, th, map)
            .expect("compose");
        let got = readback_rgba32f(&ctx, tex, w, h);

        let max_abs = max_abs_rgb(&got, &cpu.planes, w, h);
        eprintln!("compose_gt4ch_rgb_matches_cpu_read_still max_abs_diff = {max_abs}");
        assert_eq!(
            max_abs, 0.0,
            "composed RGBA must be bit-exact vs CPU read_still (lossless F16)"
        );
    }

    /// Default-codec gate (GPU-M7 / NEW-4): a 3-channel F16 lossless-HTJ2K still (the common RGB case)
    /// stays GPU-resident as ONE RGBA16F-packed texture per tile and composes **bit-exact** vs CPU
    /// `read_still`, for the identity map AND a channel-swapped map (proves the per-slot component pick);
    /// the unmapped A slot is the constant 1.0. Multi-tile with edge tiles.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn compose_packed_rgb_matches_cpu_read_still() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let n = (w * h) as usize;
        let img = f16_image(w, h, 256, 256, &["R", "G", "B"], 0);
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP compose_packed_rgb_matches_cpu_read_still: no GPU ({e})");
                return;
            }
        };
        let mut atlas = new_atlas(&ctx);
        let tiles = tile_grid(3, 2);
        atlas.request(Clip { data: &bytes, idx: &exv_video::VideoIndex::parse(&bytes).expect("idx") }, 0, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("request");
        let packed = atlas.get(None, 0, 0).expect("resident");
        assert!(
            matches!(&packed.tex, TileTex::Packed(t) if t.tex.format() == wgpu::TextureFormat::Rgba16Float),
            "a 3-channel F16 tile must be one RGBA16F-packed texture (the path under test)"
        );

        let mut compose = PlaneCompose::new(&ctx).expect("compose");
        // identity, channel swap, and a grey broadcast (one component feeding R,G,B — what a 1-channel layer maps to)
        for map in [[0u32, 1, 2, UNMAPPED], [2, 1, 0, UNMAPPED], [1, 1, 1, UNMAPPED]] {
            let (tex, _view) = compose.compose(&ctx, &atlas, None, &tiles, w, h, 256, 256, map).expect("compose");
            let got = readback_rgba32f(&ctx, tex, w, h);
            for i in 0..n {
                for (slot, &src) in map.iter().take(3).enumerate() {
                    let b = &cpu.planes[src as usize][i * 2..i * 2 + 2];
                    let want = half_to_f32(u16::from_le_bytes([b[0], b[1]]));
                    assert_eq!(got[i * 4 + slot], want, "map {map:?} px {i} slot {slot}");
                }
                assert_eq!(got[i * 4 + 3], 1.0, "unmapped A must be 1.0 (px {i})");
            }
        }
    }

    /// Dirty check of `PlaneCompose::compose`: an identical request composes nothing (no pass, same pixels), a changed
    /// channel map / tile set / atlas decode count composes again, and the recomposed pixels stay the CPU's.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn compose_skips_an_unchanged_request() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::write_still;
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let img = f16_image(w, h, 256, 256, &["R", "G", "B"], 0);
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let idx = exv_video::VideoIndex::parse(&bytes).expect("idx");
        let Ok(ctx) = GpuContext::new() else {
            eprintln!("SKIP compose_skips_an_unchanged_request: no GPU");
            return;
        };
        let mut atlas = new_atlas(&ctx);
        let all = tile_grid(3, 2);
        let clip = || Clip { data: &bytes, idx: &idx };
        let want = |tiles| Want { tiles, level: None, chans: Chans::All };
        atlas.request(clip(), 0, want(&all[..1]), ContentKey(1), DisplayKey(0)).expect("request one");

        let mut compose = PlaneCompose::new(&ctx).expect("compose");
        let map = [0u32, 1, 2, UNMAPPED];
        let run = |c: &mut PlaneCompose, a: &ExvResidentAtlas, tiles: &[(u32, u32)], map| {
            c.compose(&ctx, a, None, tiles, w, h, 256, 256, map).expect("compose");
            c.passes()
        };
        assert_eq!(run(&mut compose, &atlas, &all[..1], map), 1);
        assert_eq!(run(&mut compose, &atlas, &all[..1], map), 1, "identical request: skipped");
        assert_eq!(run(&mut compose, &atlas, &all[..1], [2, 1, 0, UNMAPPED]), 2, "channel map changed");
        assert_eq!(run(&mut compose, &atlas, &all[..1], [2, 1, 0, UNMAPPED]), 2);
        // A new tile set whose extra tile the atlas decodes: both the tile set and the decode count changed.
        atlas.request(clip(), 0, want(&all), ContentKey(1), DisplayKey(0)).expect("request all");
        assert_eq!(run(&mut compose, &atlas, &all, map), 3, "tile set / decodes changed");
        assert_eq!(run(&mut compose, &atlas, &all, map), 3);
        // Same tile set, more decodes (a second content key re-decodes) => dirty although the request is the same.
        let before = atlas.decode_count();
        atlas.request(clip(), 0, want(&all), ContentKey(2), DisplayKey(0)).expect("request key 2");
        assert!(atlas.decode_count() > before, "a new content key must decode");
        assert_eq!(run(&mut compose, &atlas, &all, map), 4, "atlas content changed");
    }

    /// The DEFAULT HTJ2K route (measured: CPU decode + plane upload, see `ExvResidentAtlas::htj2k_on_gpu`) composes to the
    /// same pixels as `read_still` and as the forced jph-wgpu route, although its tiles are N single-channel planes
    /// instead of one RGBA16F-packed texture; a coarser LOD request still takes the GPU path.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_htj2k_cpu_route_matches_cpu_and_gpu_route() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let img = f16_image(w, h, 256, 256, &["R", "G", "B"], 0);
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let idx = exv_video::VideoIndex::parse(&bytes).expect("idx");
        let cpu = read_still(&bytes).expect("cpu read_still");
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP atlas_htj2k_cpu_route_matches_cpu_and_gpu_route: no GPU ({e})");
                return;
            }
        };
        let tiles = tile_grid(3, 2);
        let mut cpu_route = ExvResidentAtlas::new(&ctx, 512 << 20, DECODE_BUDGET).expect("atlas");
        let mut gpu_route = new_atlas(&ctx);
        let mut compose = PlaneCompose::new(&ctx).expect("compose");
        let map = [0u32, 1, 2, UNMAPPED];
        let mut shots = Vec::new();
        for atlas in [&mut cpu_route, &mut gpu_route] {
            atlas.request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("request");
            let (tex, _view) = compose.compose(&ctx, atlas, None, &tiles, w, h, 256, 256, map).expect("compose");
            shots.push(readback_rgba32f(&ctx, tex, w, h));
        }
        let planes = cpu_route.get(None, 0, 0).expect("resident");
        assert!((0..3).all(|ci| planes.plane(ci).is_some_and(|p| p.tex.format() == wgpu::TextureFormat::R16Float)), "CPU route = N planes");
        assert!(matches!(gpu_route.get(None, 0, 0).expect("resident").tex, TileTex::Packed(_)), "forced GPU route = one packed texture");
        assert_eq!(shots[0], shots[1], "CPU route == GPU route, every pixel");
        for i in 0..(w * h) as usize {
            for slot in 0..3 {
                let b = &cpu.planes[slot][i * 2..i * 2 + 2];
                assert_eq!(shots[0][i * 4 + slot], half_to_f32(u16::from_le_bytes([b[0], b[1]])), "px {i} slot {slot}");
            }
        }
        // A coarser LOD is the GPU path's job (resident RGBA16F-packed), whatever the default route says.
        cpu_route.request(Clip { data: &bytes, idx: &idx }, 0, Want { tiles: &tiles, level: Some(1), chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("lod request");
        assert!(matches!(&cpu_route.get(Some(1), 0, 0).expect("lod resident").tex, TileTex::Packed(t) if t.tex.format() == wgpu::TextureFormat::Rgba16Float));
    }

    /// **GPU-resident VIDEO gate:** DISPLAY frame `k>0` of a multi-frame all-INTRA (`gop=1`)
    /// lossless-HTJ2K `.exv` is made resident by [`ExvResidentAtlas::request`]`(.., frame=1, ..)` —
    /// opened via [`exv_video::frame_body`] like a still — composed (ch0,1,2 -> R,G,B), and is
    /// **bit-exact** vs the CPU [`exv_video::decode_frame`]`(.., 1)` for the SAME channels (lossless
    /// F16 -> f32 is exact, so max abs diff == 0). The two frames carry DIFFERENT content and the
    /// resident tile's `node` stamp is the frame it was decoded from, so this proves frame 1 is decoded from
    /// frame 1's body (not a reuse of frame 0's resident tiles). Mirrors `compose_gt4ch_rgb_matches_cpu_read_still`
    /// but on a >=2-frame clip. Skips (no fail) if no GPU is present.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn compose_frame1_matches_cpu_decode_frame() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::{TemporalClass, VideoIndex, decode_frame, temporal_class, write_video};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let (tw, th) = (256u32, 256u32); // 3x2 = 6 tiles (edge tiles smaller)
        // Two DISTINCT 5xF16 (>4-channel) frames — a per-frame value offset makes frame 1 != frame 0,
        // so a bit-exact match vs `decode_frame(1)` can only come from decoding frame 1's own body.
        let mk = |foff: u32| f16_image(w, h, tw, th, &["R", "G", "B", "A", "Y"], foff);
        // gop=1 (search=0, b_frames=0) => EVERY frame is INTRA => each decodes standalone.
        let bytes =
            write_video(&[mk(0), mk(123)], &Profile::htj2k(), 1, 0, 0).expect("encode gop=1 .exv");
        let idx = VideoIndex::parse(&bytes).expect("index");
        assert_eq!(temporal_class(&bytes, &idx, TemporalClass::Bidir), Ok(TemporalClass::Intra), "gop=1 clip must be all-INTRA");
        assert_eq!(idx.frame_count(), 2, "two-frame clip");
        let cpu = decode_frame(&bytes, &idx, 1, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("cpu decode_frame(1)");
        let cpu0 = decode_frame(&bytes, &idx, 0, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("cpu decode_frame(0)");
        assert_ne!(cpu.planes, cpu0.planes, "fixture frames must differ (else the test is vacuous)");

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP compose_frame1_matches_cpu_decode_frame: no GPU ({e})");
                return;
            }
        };

        let mut atlas = new_atlas(&ctx);
        let tiles = tile_grid(3, 2);
        // Decode DISPLAY frame 1 (the content key is stable per clip; the frame identity is the tile stamp).
        atlas
            .request(Clip { data: &bytes, idx: &exv_video::VideoIndex::parse(&bytes).expect("idx") }, 1, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0))
            .expect("request frame 1 tiles");
        assert_eq!(atlas.decode_count(), tiles.len() as u64, "each requested tile decoded once");

        let mut compose = PlaneCompose::new(&ctx).expect("compose");
        // ch0 -> R, ch1 -> G, ch2 -> B, A unmapped (=> 1.0).
        let map = [0u32, 1, 2, UNMAPPED];
        let (tex, _view) = compose
            .compose(&ctx, &atlas, None, &tiles, w, h, tw, th, map)
            .expect("compose");
        let got = readback_rgba32f(&ctx, tex, w, h);

        let max_abs = max_abs_rgb(&got, &cpu.planes, w, h);
        eprintln!("compose_frame1_matches_cpu_decode_frame max_abs_diff = {max_abs}");
        assert_eq!(
            max_abs, 0.0,
            "composed frame 1 must be bit-exact vs CPU decode_frame(1) (lossless F16)"
        );
    }

    /// **Track A gate:** a `>4`-channel **lossless-WAVE (Wavelet53)** still is made resident by
    /// [`ExvResidentAtlas::request`] through the GPU 5/3 wave kernel, composed (ch0,1,2 -> R,G,B), and
    /// is **bit-exact** vs CPU `read_still`. Lossless-wave F16 -> f32 is exact, so max abs diff == 0.
    /// Asserts the fixture's first tile/channel is rANS-tagged so the GPU wave path is TRULY exercised
    /// (not a silent CPU fallback). Skips (no fail) when no GPU is present.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn compose_gt4ch_wave_matches_cpu_read_still() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::{read_still, still_frame_body, write_still};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let (tw, th) = (256u32, 256u32); // synth_f16 tiling -> 3x2 = 6 tiles (edge tiles smaller)
        let img = synth_f16(w, h); // 5x F16 -> >4-channel N-plane resident path
        let bytes = write_still(&img, &Profile::lossless_wave()).expect("encode lossless-wave .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");

        // Guarantee the GPU wave path is exercised: tile 0 / channel 0 must be a rANS sub-blob (GpuHot
        // picks rANS whenever it shrinks). Assert it so a silent CPU fallback would fail LOUDLY here.
        {
            let (hdr, body) = still_frame_body(&bytes).expect("frame body");
            let reader = exv_codec::frame::FrameReader::open(body, hdr).expect("frame reader");
            let sub = reader.tile_channel_blob(0, 0).expect("tile 0 ch 0 sub-blob");
            assert!(
                matches!(classify_blob(sub), Some(BlobKind::Rans(_))),
                "fixture tile 0 ch 0 must be rANS-tagged to exercise the GPU wave path"
            );
        }

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP compose_gt4ch_wave_matches_cpu_read_still: no GPU ({e})");
                return;
            }
        };

        let mut atlas = new_atlas(&ctx);
        let tiles = tile_grid(3, 2);
        atlas
            .request(Clip { data: &bytes, idx: &exv_video::VideoIndex::parse(&bytes).expect("idx") }, 0, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0))
            .expect("request wave tiles");
        assert_eq!(atlas.decode_count(), tiles.len() as u64, "each requested tile decoded once");

        let mut compose = PlaneCompose::new(&ctx).expect("compose");
        // ch0 -> R, ch1 -> G, ch2 -> B, A unmapped (=> 1.0).
        let map = [0u32, 1, 2, UNMAPPED];
        let (tex, _view) = compose
            .compose(&ctx, &atlas, None, &tiles, w, h, tw, th, map)
            .expect("compose");
        let got = readback_rgba32f(&ctx, tex, w, h);

        let max_abs = max_abs_rgb(&got, &cpu.planes, w, h);
        eprintln!("compose_gt4ch_wave_matches_cpu_read_still max_abs_diff = {max_abs}");
        assert_eq!(
            max_abs, 0.0,
            "composed RGBA must be bit-exact vs CPU read_still (lossless wave F16)"
        );
    }

    /// **EXV-RT atlas gate:** a `>4`-channel **EXV-RT (`blockrt_dwt`, `Transform::BlockRt`)** still is
    /// made resident by [`ExvResidentAtlas::request`] through the GPU EXV-RT block kernel
    /// ([`decode_tile_exvrt_planes`] → [`decode_tile_exvrt_vram`]), composed (ch0,1,2 -> R,G,B), and is
    /// **bit-exact** vs CPU `read_still` at **q=0** (the pure-integer inverse path: 5/3 lift + clamp, NO
    /// `exp2`), so max abs diff == 0. The fixture uses smooth F16 ramps so the block codec engages
    /// (TAG_EXV_RT) on every tile; tile 0 / channel 0 is asserted `BlobKind::BlockRt` so a silent CPU
    /// fallback would fail LOUDLY here. Multi-tile (300×200, 128 tile grid → 3×2 = 6 tiles, edge tiles).
    /// Skips (no fail) when no GPU is present.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn compose_gt4ch_exvrt_matches_cpu_read_still() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, still_frame_body, write_still};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (300u32, 200u32);
        let (tw, th) = (128u32, 128u32); // 300×200 / 128 → 3×2 = 6 tiles (edge tiles smaller)
        // 5× F16 SMOOTH low-frequency ramps (>4 channels) so the block codec engages on every tile.
        let plane = |slope: f32, off: f32| -> Vec<u8> {
            (0..h).flat_map(|y| (0..w).flat_map(move |x| f16_le(((x + y) as f32) * slope + off))).collect()
        };
        let planes = vec![
            plane(0.010, 0.10),
            plane(0.008, 0.20),
            plane(0.006, 0.30),
            plane(0.004, 0.40),
            plane(0.012, 0.50),
        ];
        let prof = Profile::exv_rt_dwt(16, 0); // block=16, q=0 (lossless lifting = bit-exact)
        let chans = [("c0", Dtype::F16), ("c1", Dtype::F16), ("c2", Dtype::F16), ("c3", Dtype::F16), ("c4", Dtype::F16)];
        let header = header(w, h, tw, th, prof.codec_id().0, prof.params(), &chans);
        let img = Image::new(header, planes).expect("img");
        let bytes = write_still(&img, &prof).expect("encode EXV-RT .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");

        // Guarantee the GPU EXV-RT branch is exercised: tile 0 / channel 0 must be a BlockRt sub-blob.
        {
            let (hdr, body) = still_frame_body(&bytes).expect("frame body");
            let reader = exv_codec::frame::FrameReader::open(body, hdr).expect("frame reader");
            let sub = reader.tile_channel_blob(0, 0).expect("tile 0 ch 0 sub-blob");
            assert!(
                matches!(classify_blob(sub), Some(BlobKind::BlockRt(_))),
                "fixture tile 0 ch 0 must be BlockRt-tagged to exercise the GPU EXV-RT path"
            );
        }

        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP compose_gt4ch_exvrt_matches_cpu_read_still: no GPU ({e})");
                return;
            }
        };

        let mut atlas = new_atlas(&ctx);
        let tiles = tile_grid(3, 2);
        atlas
            .request(Clip { data: &bytes, idx: &exv_video::VideoIndex::parse(&bytes).expect("idx") }, 0, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0))
            .expect("request EXV-RT tiles");
        assert_eq!(atlas.decode_count(), tiles.len() as u64, "each requested tile decoded once");

        let mut compose = PlaneCompose::new(&ctx).expect("compose");
        // ch0 -> R, ch1 -> G, ch2 -> B, A unmapped (=> 1.0).
        let map = [0u32, 1, 2, UNMAPPED];
        let (tex, _view) = compose
            .compose(&ctx, &atlas, None, &tiles, w, h, tw, th, map)
            .expect("compose");
        let got = readback_rgba32f(&ctx, tex, w, h);

        let max_abs = max_abs_rgb(&got, &cpu.planes, w, h);
        eprintln!("compose_gt4ch_exvrt_matches_cpu_read_still max_abs_diff = {max_abs}");
        assert_eq!(
            max_abs, 0.0,
            "composed RGBA must be bit-exact vs CPU read_still (EXV-RT q=0)"
        );
    }

    /// P1-9 gate: on a device that can store into `R16Float` (feature `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`, what the
    /// viewer requests) the GPU-blitted F16 planes of the wave and EXV-RT routes are resident as `R16Float` = HALF the
    /// VRAM of the standalone context's `R32Float`, and `compose` over them is bit-exact vs CPU `read_still`, exactly like
    /// the `R32Float` planes of the plain context (the standalone path keeps working and keeps its format).
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_blit_f16_is_r16float_with_the_device_feature_and_bit_exact() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::{VideoIndex, read_still, write_still};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let plain = GpuContext::new().expect("gpu ctx");
        let feat = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let (device, queue) = gpu_info::request_max_device_blocking(&plain.adapter, feat).expect("device with the format feature");
        let rich = GpuContext::from_wgpu(device, queue, plain.adapter.clone()).expect("context over the feature device");
        assert!(!plain.r16_store(), "a standalone context requests no features");
        if !rich.r16_store() {
            eprintln!("SKIP: the adapter cannot store into R16Float");
            return;
        }
        assert_eq!(rich.blit_fmt(Dtype::F16), Some(wgpu::TextureFormat::R16Float));
        assert_eq!(plain.blit_fmt(Dtype::F16), Some(wgpu::TextureFormat::R32Float));
        assert_eq!(rich.blit_fmt(Dtype::F32), Some(wgpu::TextureFormat::R32Float), "only F16 narrows");

        let (w, h, tw, th) = (600u32, 400u32, 256u32, 256u32);
        let img = f16_image(w, h, tw, th, &["R", "G", "B"], 0);
        let tiles = tile_grid(3, 2);
        for (what, prof) in [("EXV-RT", Profile::exv_rt_dwt(16, 0)), ("wave", Profile::lossless_wave())] {
            let bytes = write_still(&img, &prof).expect("encode");
            let cpu = read_still(&bytes).expect("cpu read_still");
            let idx = VideoIndex::parse(&bytes).expect("idx");
            let mut resident = Vec::new();
            for (name, ctx) in [("R32Float", &plain), ("R16Float", &rich)] {
                let mut atlas = new_atlas(ctx);
                let want = Want { tiles: &tiles, level: None, chans: Chans::All };
                atlas.request(Clip { data: &bytes, idx: &idx }, 0, want, ContentKey(1), DisplayKey(0)).unwrap_or_else(|e| panic!("{what} {name}: {e}"));
                let fmt = atlas.get(None, 0, 0).and_then(|p| p.plane(0)).map(|p| p.tex.format());
                let expect = if name == "R16Float" { wgpu::TextureFormat::R16Float } else { wgpu::TextureFormat::R32Float };
                assert_eq!(fmt, Some(expect), "{what} {name}: GPU-blitted F16 format");
                let mut compose = PlaneCompose::new(ctx).expect("compose");
                let (tex, _view) = compose.compose(ctx, &atlas, None, &tiles, w, h, tw, th, [0, 1, 2, UNMAPPED]).expect("compose");
                let got = readback_rgba32f(ctx, tex, w, h);
                assert_eq!(max_abs_rgb(&got, &cpu.planes, w, h), 0.0, "{what} {name}: compose is bit-exact vs CPU read_still");
                resident.push(atlas.stats().resident_bytes);
            }
            assert_eq!(resident[0], 2 * resident[1], "{what}: R16Float halves the resident VRAM");
            eprintln!("{what}: RGB F16 600x400 resident {} B (R32Float) -> {} B (R16Float)", resident[0], resident[1]);
        }
    }

    /// A2 gate: a `>4`-channel tile that ALSO carries an **F32** AOV decodes to per-plane resident
    /// textures bit-exact vs CPU `read_still`. The F32 plane rides jph-wgpu's separate 32-bit float
    /// magnitude codepath (an F16-only pass does NOT exercise it), so this is its own gate. Same
    /// visible-subset residency contract as the F16 atlas test.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn exv_resident_atlas_planes_gt4ch_f32_bit_exact() {
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};
        use jph_wgpu_rs::texture::read_plane;
        use jph_wgpu_rs::{ContentKey, DisplayKey, Gpu};

        let (w, h) = (600u32, 400u32);
        let (tw_full, th_full) = (256u32, 256u32); // 3×2 = 6 tiles (edge tiles smaller)
        let img = synth_f16_f32(w, h);
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());

        let mut atlas = new_atlas(&ctx);
        let want = [(0u32, 0u32), (2u32, 1u32)];
        atlas
            .request(Clip { data: &bytes, idx: &exv_video::VideoIndex::parse(&bytes).expect("idx") }, 0, Want { tiles: &want, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0))
            .expect("request");
        assert_eq!(atlas.decode_count(), want.len() as u64, "only the requested tiles are decoded");

        let dtypes: Vec<Dtype> = img.header.channels.iter().map(|c| c.dtype).collect();
        for &(rx, ry) in &want {
            let x0 = rx * tw_full;
            let y0 = ry * th_full;
            let tw = (w - x0).min(tw_full);
            let th = (h - y0).min(th_full);
            let planes = atlas.get(None, rx, ry).expect("requested tile is resident");
            assert!((0..dtypes.len()).all(|ci| planes.plane(ci).is_some()), "one resident plane per component");
            for (ch, plane) in (0..dtypes.len()).filter_map(|ci| planes.plane(ci)).enumerate() {
                let vals = read_plane(&gpu, plane, tw, th, plane.tex.format()).expect("read plane");
                for &(tx, ty) in &[(0u32, 0u32), (tw - 1, th - 1), (tw / 2, th / 2)] {
                    let img_idx = ((y0 + ty) * w + (x0 + tx)) as usize;
                    let got = vals[(ty * tw + tx) as usize];
                    // Compare on bit patterns: exact for the F32 plane AND for each F16→f32 plane
                    // (all values here are finite, so a lossless decode is bit-identical).
                    let want_v = sample_f32(&cpu.planes[ch], dtypes[ch], img_idx);
                    assert_eq!(
                        got.to_bits(),
                        want_v.to_bits(),
                        "tile({rx},{ry}) px({tx},{ty}) ch{ch} ({:?})",
                        dtypes[ch]
                    );
                }
            }
        }

        // The N-plane path preserves the visible-subset residency contract.
        assert!(atlas.get(None, 1, 0).is_none(), "unrequested tile must not be resident");
    }

    /// P2 resident gate + measurement: GPU resident decode of a single-tile lossless HTJ2K f16 frame
    /// == CPU `read_still` (bit-exact, via the resident RGBA16F texel == half_to_f32(cpu half bits)),
    /// and reports the warm resident decode GB/s (NO readback in the timed region) for 30 GB/s tracking.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_htj2k_resident_matches_cpu_and_reports_gbps() {
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};
        use jph_wgpu_rs::Gpu;
        use jph_wgpu_rs::texture::read_texture;
        use std::time::Instant;

        let (w, h) = (512u32, 512u32);
        let img = synth_f16_single_tile(w, h);
        let bytes = write_still(&img, &Profile::htj2k()).expect("encode .exv");
        let cpu = read_still(&bytes).expect("cpu read_still");
        let ctx = GpuContext::new().expect("gpu ctx");

        // Correctness: resident texel == half_to_f32(cpu half bits), per RGB channel.
        let tex = decode_htj2k_resident(&ctx, &bytes, 0, &DecodeLimits::default()).expect("resident decode");
        let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let rgba = read_texture(&gpu, &tex, w, h).expect("read_texture");
        for &(px, py) in &[(0u32, 0u32), (w - 1, h - 1), (123, 200), (511, 0)] {
            let pi = (py * w + px) as usize;
            for ch in 0..3usize {
                let bits = u16::from_le_bytes([cpu.planes[ch][pi * 2], cpu.planes[ch][pi * 2 + 1]]);
                let want = half_to_f32(bits);
                let got = rgba[pi * 4 + ch];
                assert_eq!(got, want, "resident ch{ch} px({px},{py}) {got} != {want}");
            }
        }

        let raw = (w as u64 * h as u64 * 3 * 2) as f64; // f16 RGB raw bytes
        let iters = 20u32;

        // Baseline: one-shot path (rebuilds the jph-wgpu pipeline/pool/tables every call).
        for _ in 0..3 {
            let _ = decode_htj2k_resident(&ctx, &bytes, 0, &DecodeLimits::default()).expect("warm one-shot");
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _t = decode_htj2k_resident(&ctx, &bytes, 0, &DecodeLimits::default()).expect("timed one-shot");
        }
        let oneshot = t.elapsed().as_secs_f64() / iters as f64;

        // T1: persistent decoder reuses the cached GPU objects (no per-frame rebuild) -> warm steady-state.
        let mut dec = Htj2kGpuDecoder::new(&ctx).expect("persistent decoder");
        // The persistent path must still be bit-exact vs the CPU half bit patterns.
        let wtex = dec.decode(&bytes, 0, &DecodeLimits::default()).expect("persistent decode");
        let wrgba = read_texture(&gpu, &wtex, w, h).expect("read_texture warm");
        for &(px, py) in &[(0u32, 0u32), (w - 1, h - 1), (123, 200), (511, 0)] {
            let pi = (py * w + px) as usize;
            for ch in 0..3usize {
                let bits = u16::from_le_bytes([cpu.planes[ch][pi * 2], cpu.planes[ch][pi * 2 + 1]]);
                assert_eq!(wrgba[pi * 4 + ch], half_to_f32(bits), "warm ch{ch} px({px},{py})");
            }
        }
        for _ in 0..3 {
            let _ = dec.decode(&bytes, 0, &DecodeLimits::default()).expect("warm persistent");
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _t = dec.decode(&bytes, 0, &DecodeLimits::default()).expect("timed persistent");
        }
        let warm = t.elapsed().as_secs_f64() / iters as f64;

        eprintln!(
            "resident HTJ2K f16 {w}x{h}: one-shot {:.2} GB/s ({:.2} ms/frame) | persistent {:.2} GB/s ({:.2} ms/frame), no readback",
            raw / 1e9 / oneshot,
            oneshot * 1e3,
            raw / 1e9 / warm,
            warm * 1e3
        );
        // T1 invariant: reusing cached GPU objects must be materially faster than rebuilding each call.
        assert!(
            warm < oneshot,
            "persistent decode ({:.2} ms) should beat one-shot ({:.2} ms)",
            warm * 1e3,
            oneshot * 1e3
        );
    }

    /// Spike A gate: GPU decode == CPU reference byte-for-byte, across different configs.
    /// Requires a GPU: `cargo test -p exv-gpu -- --ignored`.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_matches_cpu() {
        let ctx = GpuContext::new().expect("gpu ctx");
        let mut seed = 0xA5A5_1234_DEAD_0001u64;
        // skewed data (realistic for pixel residuals)
        let data: Vec<u8> = (0..50_000)
            .map(|_| {
                let r = xorshift(&mut seed) % 100;
                if r < 70 { 0u8 } else if r < 90 { 1u8 } else { (xorshift(&mut seed) & 0xff) as u8 }
            })
            .collect();

        for &(lanes, tl) in &[(1u8, 12u8), (16, 12), (32, 12), (32, 8), (64, 15), (65, 12), (128, 12), (255, 12)] {
            let cfg = RansConfig::new(lanes, tl);
            // sanity: CPU round-trip
            assert_eq!(decode_capped(&encode(&data, cfg).expect("encode"), data.len()).unwrap(), data, "CPU rt (lanes={lanes}, tl={tl})");
            // GPU decode of the same prepared chunks
            let prep = prepare(&data, cfg).expect("prepare");
            let gpu = decode_block_gpu(&ctx, &prep).expect("gpu decode");
            assert_eq!(gpu, data, "GPU != CPU (lanes={lanes}, tl={tl})");
        }
    }

    /// Phase 5 gate: GPU inverse 5/3 wavelet == CPU byte-for-byte (esz≤2).
    /// `cargo test -p exv-gpu -- --ignored`.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_wavelet_matches_cpu() {
        let ctx = GpuContext::new().expect("gpu ctx");
        for &(w, h, dt) in &[
            (64u32, 48u32, Dtype::F16),
            (33, 17, Dtype::U16),
            (1, 9, Dtype::U16),
            (9, 1, Dtype::U16),
            (16, 16, Dtype::U16),
        ] {
            let n = (w * h) as usize;
            let plane: Vec<u8> =
                (0..n).flat_map(|i| ((i as u32 * 13 % 4096) as u16).to_le_bytes()).collect();
            let coeffs = exv_codec::wavelet::forward_2d(&plane, dt, w, h).expect("forward");
            let cpu = exv_codec::wavelet::inverse_2d(&coeffs, dt, w, h).expect("inverse");
            assert_eq!(cpu, plane, "CPU wavelet sanity {w}x{h}");
            let gpu = inverse_wavelet_gpu(&ctx, &coeffs, dt, w, h).expect("gpu wavelet");
            assert_eq!(gpu, plane, "GPU wavelet inverse != CPU ({w}x{h} {dt:?})");
        }
    }

    /// Phase 5 gate (full GPU decode): a LosslessWave rANS tile → rANS+repack+wavelet on the GPU,
    /// through [`decode_tile_gpu_wave_vram`] + [`read_back`], == the original pixel bit-for-bit.
    ///
    /// The fixture is a rANS tile built DIRECTLY (transform → rANS → `TAG_RANS`): `encode_channel`
    /// auto-picks the smaller of rANS/deflate and chooses DEFLATE on smooth ramps (a CPU-only tile),
    /// so it cannot be used to exercise the GPU rANS path. We assert the tag to fail loudly if the
    /// fixture ever stops being rANS.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_full_tile_decode_wave() {
        use exv_codec::{Profile, decode_channel, wavelet};
        let ctx = GpuContext::new().expect("gpu ctx");
        for &(w, h, dt) in &[(64u32, 64u32, Dtype::F16), (48, 72, Dtype::U16), (33, 17, Dtype::F16)] {
            let n = (w * h) as usize;
            let plane: Vec<u8> =
                (0..n).flat_map(|i| ((i as u32 * 7 % 4096) as u16).to_le_bytes()).collect();
            let prof = Profile::lossless_wave();
            let decorrelated = wavelet::forward_2d(&plane, dt, w, h).expect("forward");
            let mut blob = vec![exv_format::TAG_RANS];
            blob.extend_from_slice(&encode(&decorrelated, prof.rans).expect("rans encode"));
            assert!(matches!(classify_blob(&blob), Some(BlobKind::Rans(_))), "fixture must be rANS");
            // sanity: CPU decode of the same rANS blob
            assert_eq!(decode_channel(&blob, dt, w, h, &prof).unwrap(), plane, "cpu {w}x{h}");
            // full GPU decode (vram + readback) of the same blob
            let gpu = decode_tile_gpu_wave(&ctx, &blob, dt, w, h).expect("gpu tile decode");
            assert_eq!(gpu, plane, "GPU full-tile decode != original ({w}x{h} {dt:?})");
        }
    }

    /// **EXV-RT B-RT.3 Increment 1 gate:** GPU decode of ONE EXV-RT tile at q=0 is BIT-EXACT vs the CPU
    /// oracle `exv_codec::blockrt_dwt::decode_plane`. Pure integer at q=0 (5/3 inverse + clamp, no exp2),
    /// so equality is byte-for-byte — NOT a tolerance. Covers a block-multiple size (64×64) and two
    /// non-multiples of 16 (40×24, 80×48) to exercise mirror-pad edge blocks + the crop.
    #[test]
    #[ignore = "requires a GPU"]
    fn exvrt_gpu_q0_bit_exact() {
        let ctx = GpuContext::new().expect("gpu ctx");
        // 1024×1024 (block=16 → 4096 blocks) forces the thread-dense FAST path (`nblk >= FAST_MIN_BLOCKS`);
        // the small sizes stay on the 1-block-per-workgroup path. Both must be byte-exact vs the CPU.
        for (w, h) in [(64u32, 64u32), (40u32, 24u32), (80u32, 48u32), (1024u32, 1024u32)] {
            // A smooth ramp so the block codec engages (small adaptive widths). q=0 is lossless of the f16 BITS.
            let src: Vec<u8> = (0..w * h).flat_map(|i| f16_le((i % 97) as f32 * 0.05 + 0.1)).collect();
            // Sweep the rANS interleave width: N=1 is the serial path, N∈{2,4,8} exercise each specialized
            // N-lane-per-thread fast kernel. All must be BIT-EXACT vs the CPU (rANS is exact for any N).
            for lanes in [1u8, 2, 4, 8, 16, 65, 128, 255] {
                // encode_plane yields the tag-less blob; decode_plane (oracle) and decode_tile_exvrt_vram
                // both consume exactly this payload (the GPU reads `lanes` from the blob header).
                let payload = exv_codec::blockrt_dwt::encode_plane(&src, w, h, 16, 0, lanes).expect("encode_plane");
                let cpu = exv_codec::blockrt_dwt::decode_plane(&payload, w, h).expect("cpu decode");
                let tile = decode_tile_exvrt_vram(&ctx, &payload, Dtype::F16, w, h).expect("gpu decode");
                let gpu = read_back(&ctx, &tile).expect("readback");
                assert_eq!(gpu, cpu, "EXV-RT q=0 GPU != CPU ({w}x{h}, lanes={lanes})");
            }
        }
    }

    /// **EXV-RT B-RT.3 Increment 2 gate:** GPU decode of ONE EXV-RT tile at q>0 is BIT-EXACT vs the CPU oracle
    /// `blockrt_dwt::decode_plane`. Everything (detail dequant, inverse 5/3, code clamp, `hdr::decode_value`
    /// as a table fetch + exact exponent add, `f32_to_half`) is integer or exactly-specified float math — the
    /// hardware `exp2` is not used. We still print the max ULP and the count of differing pixels
    /// (`--nocapture`). Covers a block-multiple (64×64) and two non-multiples of 16 (48×80, 40×24) across
    /// q ∈ {1, 2, 4}.
    #[test]
    #[ignore = "requires a GPU"]
    #[allow(non_snake_case)] // `qN` reads as "q sub N" (the quant level) — clearer than `q_n` here.
    fn exvrt_gpu_qN_ulp() {
        let ctx = GpuContext::new().expect("gpu ctx");
        let mut worst_ulp = 0i32;
        for &q in &[1u32, 2, 4] {
            // 1024×1024 (block=16 → 4096 blocks) forces the thread-dense FAST inverse kernel through the
            // q>0 hdr log path (`decode_value`→`f32_to_half`); the small sizes stay on the slow kernel.
            for &(w, h) in &[(64u32, 64u32), (48u32, 80u32), (40u32, 24u32), (1024u32, 1024u32)] {
                // Smooth HDR ramp in LOG space (2^-4 .. 2^4 ≈ 0.0625 .. 16, a few stops each axis) so the
                // block codec engages AND `exp2` is exercised across many distinct codes/exponents. A
                // per-column sign flip drives NEGATIVE HDR values too, so `decode_value(c<0)` + the
                // `f32_to_half` sign-bit path are exercised (not just the positive branch).
                let src: Vec<u8> = (0..w * h)
                    .flat_map(|i| {
                        let fx = (i % w) as f32 / w as f32;
                        let fy = (i / w) as f32 / h as f32;
                        let sign = if (i % w) % 2 == 0 { 1.0 } else { -1.0 };
                        f16_le(sign * 2f32.powf(fx * 4.0 + fy * 4.0 - 4.0))
                    })
                    .collect();
                // Default N=4 interleave → exercises the N-lane fast kernel through the q>0 hdr log path.
                let payload =
                    exv_codec::blockrt_dwt::encode_plane(&src, w, h, 16, q, exv_codec::blockrt_dwt::RT_LANES).expect("encode_plane");
                let cpu = exv_codec::blockrt_dwt::decode_plane(&payload, w, h).expect("cpu decode");
                let tile = decode_tile_exvrt_vram(&ctx, &payload, Dtype::F16, w, h).expect("gpu decode");
                let gpu = read_back(&ctx, &tile).expect("readback");
                assert_eq!(gpu.len(), cpu.len(), "length mismatch ({w}x{h} q={q})");

                let mut max_ulp = 0i32;
                let mut ndiff = 0usize;
                for (px, (gc, cc)) in gpu.as_chunks::<2>().0.iter().zip(cpu.as_chunks::<2>().0.iter()).enumerate() {
                    let g = u16::from_le_bytes([gc[0], gc[1]]);
                    let c = u16::from_le_bytes([cc[0], cc[1]]);
                    // Sign bit (bit 15) must be identical: the sign comes from the integer code alone, which
                    // is bit-exact on both sides. A sign flip = a real bug, not exp2 noise.
                    assert_eq!(g & 0x8000, c & 0x8000, "sign flip at px {px} ({w}x{h} q={q}): gpu={g:#06x} cpu={c:#06x}");
                    // Signs equal → (g - c) cancels the sign bit → |g - c| is the magnitude-code ULP distance.
                    let d = (g as i32 - c as i32).abs();
                    if d != 0 {
                        ndiff += 1;
                    }
                    max_ulp = max_ulp.max(d);
                    // The log decode is a table fetch + exact exponent add (no hardware `exp2`), so GPU == CPU
                    // exactly — any difference is a transliteration bug.
                    assert_eq!(d, 0, "px {px} not bit-exact ({w}x{h} q={q}): gpu={g:#06x} cpu={c:#06x}");
                }
                worst_ulp = worst_ulp.max(max_ulp);
                let total = (w * h) as usize;
                println!("EXV-RT q={q} {w}x{h}: max_ulp={max_ulp} diff_px={ndiff}/{total}");
            }
        }
        println!("EXV-RT q>0 overall worst ULP = {worst_ulp} (contract: 0, bit-exact)");
    }

    /// Phase 5 gate (LosslessFast on GPU): a rANS tile → rANS+reorder-undelta on the GPU, through
    /// [`decode_tile_gpu_fast_vram`] + [`read_back`], == the original. Fixture built directly as rANS
    /// (see [`gpu_full_tile_decode_wave`] for why `encode_channel` can't be used here).
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_full_tile_decode_fast() {
        use exv_codec::{Profile, decode_channel, reorder};
        let ctx = GpuContext::new().expect("gpu ctx");
        for &(w, h, dt) in &[
            (64u32, 64u32, Dtype::F16),
            (33, 17, Dtype::U8),
            (40, 30, Dtype::U32),
            (48, 72, Dtype::F32),
        ] {
            let n = (w * h) as usize * dt.size();
            let plane: Vec<u8> = (0..n).map(|i| (i * 37 % 256) as u8).collect();
            let prof = Profile::lossless_fast();
            let decorrelated = reorder::split_delta(&plane, dt.size());
            let mut blob = vec![exv_format::TAG_RANS];
            blob.extend_from_slice(&encode(&decorrelated, prof.rans).expect("rans encode"));
            assert!(matches!(classify_blob(&blob), Some(BlobKind::Rans(_))), "fixture must be rANS");
            assert_eq!(decode_channel(&blob, dt, w, h, &prof).unwrap(), plane, "cpu {w}x{h}");
            let gpu = decode_tile_gpu_fast(&ctx, &blob, dt, w, h).expect("gpu fast tile");
            assert_eq!(gpu, plane, "GPU LosslessFast tile decode != original ({w}x{h} {dt:?})");
        }
    }

    /// Report which blob tags appear across all `(tile, channel)` blobs of a `.exv`:
    /// `(has_rANS, has_deflate, has_raw)`. Test helper for the G3 mixed-tag gate.
    fn scan_blob_tags(data: &[u8]) -> (bool, bool, bool) {
        let (mut rans, mut deflate, mut raw) = (false, false, false);
        for_blobs(data, |k| match k {
            Some(BlobKind::Rans(_)) => rans = true,
            Some(BlobKind::Deflate(_)) => deflate = true,
            Some(BlobKind::Raw(_)) => raw = true,
            _ => {} // Htj2k / None: residency scan tracks rANS/deflate/raw only
        });
        (rans, deflate, raw)
    }

    /// G3 gate: decode a whole multi-tile `.exv` on the GPU (no readback), compositing into VRAM, ==
    /// the ORIGINAL planes bit-for-bit. The fixture is engineered so the per-(tile,channel) blobs cover
    /// ALL THREE merge paths — rANS (GPU) + deflate + raw (CPU-fallback upload) — asserted via the tag
    /// scan; dims are non-multiples of the tile (edge tiles). Comparing vs the ORIGINAL (not only
    /// `read_still`) means a shared tile-geometry bug cannot hide.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_decode_still_mixed_tags() {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, write_still};
        // Default LosslessFast = GpuHot entropy policy. Channels are engineered so the per-tile blobs
        // cover all three merge paths; dims are non-multiples of the tile → edge tiles.
        let (w, h) = (300u32, 200u32);
        let n = (w * h) as usize;
        let mut seed = 0x1234_5678_9abc_def1u64;
        // A (U8): smooth ramp → split_delta ≈ constant → rANS shrinks → GpuHot keeps rANS → GPU path.
        let a: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        // B (U8): running sum of (i%256) → split_delta is a period-256 ramp (uniform histogram, rANS
        // can't shrink; periodic, deflate LZ can) → GpuHot rescues with deflate → CPU fallback.
        let b: Vec<u8> = {
            let mut acc = 0u8;
            (0..n).map(|i| { acc = acc.wrapping_add((i % 256) as u8); acc }).collect()
        };
        // C (U32): pseudo-random → neither backend shrinks → raw (no-shrink) → CPU fallback.
        let c: Vec<u8> = (0..n).flat_map(|_| (xorshift(&mut seed) as u32).to_le_bytes()).collect();
        let header = header(w, h, 128, 128, 1, Vec::new(), &[("A", Dtype::U8), ("B", Dtype::U8), ("C", Dtype::U32)]);
        let img = Image::new(header, vec![a, b, c]).expect("img");
        let prof = Profile::lossless_fast(); // GpuHot (default)
        let bytes = write_still(&img, &prof).expect("encode .exv");

        // the fixture MUST actually cover rANS + deflate + raw, else the gate is green-but-empty.
        let tags = scan_blob_tags(&bytes);
        assert!(tags.0 && tags.1 && tags.2, "fixture must cover rANS+deflate+raw, got {tags:?}");

        let ctx = GpuContext::new().expect("gpu ctx");
        let gimg = decode_still_gpu(&ctx, &bytes, &DecodeLimits::default()).expect("gpu still decode");
        let planes = read_back_image(&ctx, &gimg).expect("readback");
        assert_eq!(planes, img.planes, "GPU whole-image decode != original planes");
        let cpu = read_still(&bytes).expect("cpu read");
        assert_eq!(planes, cpu.planes, "GPU != CPU read_still");
    }

    /// Lane count of every rANS blob in a `.exv` (test helper: proves the fixture really carries the lane
    /// count under test — a silently different count would make the parity gate meaningless).
    fn rans_lanes(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for_blobs(data, |k| {
            if let Some(BlobKind::Rans(p)) = k {
                out.push(rans::parse(p).expect("rans blob parses").cfg.lanes);
            }
        });
        out
    }

    /// **CLASS-B / old "bug #16" gate:** a whole multi-tile **LosslessWave** (Wavelet53) `.exv` decodes on the
    /// GPU through the batched kernel == the ORIGINAL planes and == CPU `read_still`, for lane counts across
    /// the 64-thread workgroup boundary. The batched rANS kernel used to decode only lanes 0..63, so every
    /// default-encoded (128-lane) tile came out wrong; it now strides lanes over its threads. Each fixture is
    /// asserted to really contain rANS blobs with the lane count under test, and `table_log = 15` covers the
    /// CPU-fallback arm (`> MAX_GPU_TABLE_LOG`) of the same decoder.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_decode_still_wave_batched_lanes() {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, write_still};
        let (w, h) = (300u32, 200u32);
        let n = (w * h) as usize;
        let mut seed = 0xbeef_1234_5678_9abcu64;
        // A (U8 smooth), B (U16 smooth) → wavelet+rANS → GPU; C (F32 random) → CPU fallback (esz4).
        let a: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..n).flat_map(|i| ((i as u32 * 7 % 4096) as u16).to_le_bytes()).collect();
        let c: Vec<u8> = (0..n).flat_map(|_| (xorshift(&mut seed) as u32).to_le_bytes()).collect();
        let header = header(w, h, 128, 128, 2, Vec::new(), &[("A", Dtype::U8), ("B", Dtype::U16), ("C", Dtype::F32)]);
        let img = Image::new(header, vec![a, b, c]).expect("img");
        let ctx = GpuContext::new().expect("gpu ctx");
        for &(lanes, tl) in &[(1u8, 12u8), (64, 12), (65, 12), (128, 12), (255, 12), (128, 15)] {
            let mut prof = Profile::lossless_wave();
            prof.rans = RansConfig::new(lanes, tl);
            let bytes = write_still(&img, &prof).expect("encode .exv");
            let seen = rans_lanes(&bytes);
            assert!(
                !seen.is_empty() && seen.iter().all(|&l| l == lanes),
                "fixture must carry rANS tiles with lanes={lanes}, got {seen:?}"
            );
            let gimg = decode_still_gpu(&ctx, &bytes, &DecodeLimits::default()).unwrap_or_else(|e| panic!("gpu wave decode lanes={lanes}: {e}"));
            let planes = read_back_image(&ctx, &gimg).expect("readback");
            assert_eq!(planes, img.planes, "GPU batched wave != original (lanes={lanes}, table_log={tl})");
            assert_eq!(planes, read_still(&bytes).expect("cpu read").planes, "GPU != CPU read_still (lanes={lanes})");
        }
    }

    /// **Chunking gate (limit injection):** a whole-frame wave AND EXV-RT decode whose symbol scratch would exceed the
    /// device buffer limit is split into chunks that reuse one scratch — the decoded planes stay == CPU `read_still`.
    /// `GpuContext::sym_cap` is the injected stand-in for a real >2 GiB frame (8k LosslessWave used to fail with
    /// `batch sym > 2 GiB`): a cap of 1-4 units forces several chunks per channel; a cap below ONE unit is a typed
    /// `DeviceLimit` (a unit cannot be split), never a wgpu validation panic or garbage pixels.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_batched_decode_splits_below_sym_cap() {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, write_still};
        let (w, h) = (300u32, 200u32);
        let n = (w * h) as usize;
        let mut ctx = GpuContext::new().expect("gpu ctx");
        for prof in [Profile::lossless_wave(), Profile::exv_rt_dwt(16, 0)] {
            let header = header(w, h, 128, 128, prof.codec_id().0, prof.params(), &RGBA_F16);
            let img = Image::new(header, rgba_f16_planes(n)).expect("img");
            let bytes = write_still(&img, &prof).expect("encode .exv");
            let cpu = read_still(&bytes).expect("cpu read");
            // Whole frame in ONE chunk (the device cap) vs forced multi-chunk: identical planes.
            let whole = ctx.sym_cap;
            for cap_words in [u64::MAX, 100_000, 70_000] {
                ctx.sym_cap = (cap_words.saturating_mul(4)).min(whole);
                let before = BATCH_CHUNKS.load(std::sync::atomic::Ordering::Relaxed);
                let gimg = decode_still_gpu(&ctx, &bytes, &DecodeLimits::default()).unwrap_or_else(|e| panic!("{:?} cap {cap_words}: {e}", prof.transform));
                let planes = read_back_image(&ctx, &gimg).expect("readback");
                assert_eq!(planes, cpu.planes, "{:?}: chunked GPU decode != CPU (cap {cap_words} words)", prof.transform);
                let chunks = BATCH_CHUNKS.load(std::sync::atomic::Ordering::Relaxed) - before;
                if cap_words != u64::MAX && prof.gpu_kind() == Some(exv_codec::GpuKind::RansWave) {
                    assert!(chunks > RGBA_F16.len(), "{:?}: cap {cap_words} must split a channel into several chunks, got {chunks}", prof.transform);
                }
            }
            // A unit larger than the cap cannot be split: typed limit.
            ctx.sym_cap = 4;
            match decode_still_gpu(&ctx, &bytes, &DecodeLimits::default()) {
                Err(GpuDecodeError::DeviceLimit(_)) => {}
                Err(e) => panic!("{:?}: expected DeviceLimit for an unsplittable unit, got {e}", prof.transform),
                Ok(_) => panic!("{:?}: a 4-byte symbol cap must not decode", prof.transform),
            }
            ctx.sym_cap = whole;
        }
    }

    /// **EXV-RT whole-frame batch gate:** over a thousand small tiles with DISTINCT rANS tables (so the fast kernel's
    /// per-workgroup table + the padding rows are exercised: 15 blocks per tile pads each table run to 64) pushed into
    /// ONE batch, built with a symbol cap that forces several chunks, decode == the CPU `decode_plane` of each tile
    /// at its origin. Asserts the fast path really ran and the cap really split the frame.
    #[test]
    #[ignore = "requires a GPU"]
    fn exvrt_batch_padding_and_chunks_match_cpu() {
        let mut ctx = GpuContext::new().expect("gpu ctx");
        // (80x48: 1200 tiles x 15 blocks = 18000 blocks, far above FAST_MIN_BLOCKS) and (40x24: tile edges NOT a multiple of
        // the 16 px block -> every interior tile has overhanging blocks that must be cropped to the TILE, not the plane,
        // or a block overwrites its neighbour's pixels in a dispatch-order-dependent way).
        // Caps: one chunk, (80x48 only) two ~9000-block fast chunks, and 3-4 tiles per chunk (slow kernel).
        let geometries: [(u32, u32, u32, u32, &[u64]); 2] = [(80, 48, 40, 30, &[u64::MAX, 3_000_000, 20_000]), (40, 24, 35, 20, &[u64::MAX, 20_000])];
        for (tw, th, nx, ny, caps) in geometries {
            check_rt_batch(&mut ctx, tw, th, nx, ny, caps);
        }
    }

    /// One geometry of [`exvrt_batch_padding_and_chunks_match_cpu`].
    fn check_rt_batch(ctx: &mut GpuContext, tw: u32, th: u32, nx: u32, ny: u32, caps: &[u64]) {
        use exv_codec::blockrt_dwt::{RT_LANES, decode_plane, encode_plane};
        let (fw, fh) = (tw * nx, th * ny);
        let blobs: Vec<Vec<u8>> = (0..nx * ny)
            .map(|i| {
                // A different gain per tile -> a different frequency table per tile.
                let src: Vec<u8> = (0..tw * th).flat_map(|p| f16_le(((p % 61) as f32 * 0.03 + 0.2) * (1.0 + (i % 17) as f32 * 0.37))).collect();
                encode_plane(&src, tw, th, 16, 0, RT_LANES).expect("encode_plane")
            })
            .collect();
        let out_words = (fw * fh) as usize;
        for &cap_words in caps {
            ctx.sym_cap = cap_words.saturating_mul(4).min(ctx.device.limits().max_storage_buffer_binding_size);
            let out = new_buf(&ctx.device, "t.out", out_words as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let mut batch = RtBatch::new();
            for (i, blob) in blobs.iter().enumerate() {
                let (x0, y0) = ((i as u32 % nx) * tw, (i as u32 / nx) * th);
                batch.push(RtTile::parse(blob, tw, th).expect("parse"), RtDest::plane(0, x0, y0, fw, fh)).expect("push");
            }
            let res = ctx.scoped(|| batch.build(ctx, &[&out], &bufpool::Cancel::new())).expect("build");
            ctx.scoped(|| res.dispatch(ctx)).expect("dispatch");
            if cap_words >= 3_000_000 {
                assert!(res.is_fast(), "thousands of block16/4-lane blocks per chunk must take the thread-dense fast kernel");
            }
            if cap_words != u64::MAX {
                assert!(res.chunks() > 1, "cap {cap_words} words must split the frame, got {} chunk(s)", res.chunks());
            }
            let gpu = read_back(ctx, &GpuTile { buf: out, off_words: 0, words: out_words, w: fw, h: fh, dtype: Dtype::F16, enc: TileEnc::ElemPerWord(2) })
                .expect("readback");
            for (i, blob) in blobs.iter().enumerate() {
                let cpu = decode_plane(blob, tw, th).expect("cpu decode");
                let (x0, y0) = ((i as u32 % nx) * tw, (i as u32 / nx) * th);
                for row in 0..th {
                    let g = &gpu[(((y0 + row) * fw + x0) * 2) as usize..][..(tw * 2) as usize];
                    assert_eq!(g, &cpu[(row * tw * 2) as usize..][..(tw * 2) as usize], "tile {i} row {row} (cap {cap_words})");
                }
            }
        }
    }

    /// **Build checkpoint gate (P1-24 a):** `RtBatch::build` under a cancelled token fails with a typed error before any
    /// upload or submit (the checkpoints sit before the frame-wide uploads and before every chunk's staging), and the same
    /// batch built under a live token decodes (the token is the only difference).
    #[test]
    #[ignore = "requires a GPU"]
    fn exvrt_build_stops_on_a_cancelled_token() {
        use exv_codec::blockrt_dwt::{RT_LANES, encode_plane};
        let ctx = GpuContext::new().expect("gpu ctx");
        let (w, h) = (64u32, 64u32);
        let src: Vec<u8> = (0..w * h).flat_map(|p| f16_le((p % 61) as f32 * 0.03 + 0.2)).collect();
        let blob = encode_plane(&src, w, h, 16, 0, RT_LANES).expect("encode_plane");
        let out = new_buf(&ctx.device, "t.out", u64::from(w * h) * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let batch = || {
            let mut b = RtBatch::new();
            b.push(RtTile::parse(&blob, w, h).expect("parse"), RtDest::plane(0, 0, 0, w, h)).expect("push");
            b
        };
        let dead = bufpool::Cancel::new();
        dead.cancel();
        let err = ctx.scoped(|| batch().build(&ctx, &[&out], &dead)).err().expect("a cancelled token must stop the build");
        assert!(err.to_string().contains("cancelled"), "{err}");
        ctx.scoped(|| batch().build(&ctx, &[&out], &bufpool::Cancel::new())?.dispatch(&ctx)).expect("a live token decodes");
    }

    /// M1 gate: a forged EXV-RT blob whose first lane state is out of range is REJECTED on the GPU path
    /// exactly like the CPU (`decode_plane` -> `None`), instead of decoding garbage. Also covers a lane
    /// length overflowing the block and an offset index pointing past the region.
    #[test]
    #[ignore = "requires a GPU"]
    fn exvrt_gpu_rejects_forged_blobs() {
        let ctx = GpuContext::new().expect("gpu ctx");
        let (w, h) = (64u32, 64u32);
        let src: Vec<u8> = (0..w * h).flat_map(|i| (((i % 97) * 13) as u16).to_le_bytes()).collect();
        let good = exv_codec::blockrt_dwt::encode_plane(&src, w, h, 16, 0, 4).expect("encode_plane");
        assert!(decode_tile_exvrt_vram(&ctx, &good, Dtype::F16, w, h).is_ok(), "fixture must decode");
        // layout: 8 B header, 512 B freq, (nblk+1)*4 offsets, nblk widths, then the substream region.
        let nblk = 16usize; // 64x64 / 16x16 blocks
        let region = 8 + 512 + (nblk + 1) * 4 + nblk;
        let off0 = u32::from_le_bytes(good[520..524].try_into().unwrap()) as usize;
        let lanes = good[3] as usize;
        let first_state = region + off0 + (lanes - 1) * 4; // block 0, lane 0 initial state
        let mut zero_state = good.clone();
        zero_state[first_state..first_state + 4].copy_from_slice(&[0, 0, 0, 0]);
        let mut huge_lane = good.clone();
        huge_lane[region + off0..region + off0 + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut bad_index = good.clone();
        bad_index[520 + 4..520 + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        for (name, blob) in [("zero state", zero_state), ("lane length overflow", huge_lane), ("offset index OOB", bad_index)] {
            assert!(exv_codec::blockrt_dwt::decode_plane(&blob, w, h).is_none(), "CPU must reject {name}");
            match decode_tile_exvrt_vram(&ctx, &blob, Dtype::F16, w, h) {
                Err(GpuError::Corrupt(_)) => {}
                Err(e) => panic!("{name}: expected GpuError::Corrupt, got {e}"),
                Ok(_) => panic!("{name}: GPU accepted a blob the CPU rejects"),
            }
        }
    }

    /// H2 gate: a single LosslessWave tile above `WG * max_compute_workgroups_per_dimension` elements (the
    /// 1-D repack dispatch used to exceed the per-dimension cap = wgpu validation panic) now decodes via the
    /// 2-D grid, == the original. 2048x2050 = 4.2 M elements > 65535 * 64 = 4,194,240.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_wave_tile_beyond_1d_dispatch() {
        use exv_codec::{Profile, wavelet};
        let ctx = GpuContext::new().expect("gpu ctx");
        let (w, h, dt) = (2048u32, 2050u32, Dtype::U16);
        let max = ctx.device.limits().max_compute_workgroups_per_dimension as u64;
        assert!(
            (w as u64 * h as u64).div_ceil(WG as u64) > max,
            "fixture must exceed the 1-D dispatch cap ({max}) to exercise the 2-D grid"
        );
        let plane: Vec<u8> =
            (0..w as usize * h as usize).flat_map(|i| ((i as u32 * 7 % 4096) as u16).to_le_bytes()).collect();
        let prof = Profile::lossless_wave();
        let mut blob = vec![exv_format::TAG_RANS];
        blob.extend_from_slice(&encode(&wavelet::forward_2d(&plane, dt, w, h).expect("forward"), prof.rans).expect("rans encode"));
        let gpu = decode_tile_gpu_wave(&ctx, &blob, dt, w, h).expect("gpu tile decode > 1-D dispatch cap");
        assert!(gpu == plane, "GPU wave tile ({w}x{h}) != original");
    }

    /// Error-scope policy: a wgpu validation error raised inside `GpuContext::scoped` is an `Err`, not a panic.
    #[test]
    #[ignore = "requires a GPU"]
    fn scoped_turns_validation_error_into_err() {
        let ctx = GpuContext::new().expect("gpu ctx");
        let r: GpuResult<()> = ctx.scoped(|| {
            // empty usage is a wgpu validation error
            let _b = new_buf(&ctx.device, "bad", 4, wgpu::BufferUsages::empty());
            Ok(())
        });
        assert!(matches!(r, Err(GpuError::Device(_))), "expected a captured validation error, got {r:?}");
    }

    /// `split_1d` yields a legal grid covering every workgroup; absurd counts are a typed `Limit`.
    #[test]
    #[ignore = "requires a GPU"]
    fn split_1d_grid() {
        let ctx = GpuContext::new().expect("gpu ctx");
        let max = ctx.device.limits().max_compute_workgroups_per_dimension as u64;
        let [x, y, _] = ctx.split_1d("t", max + 1).expect("split");
        assert!(x as u64 <= max && y as u64 <= max && x as u64 * y as u64 > max);
        assert_eq!(ctx.split_1d("t", 5).expect("small"), [5, 1, 1]);
        assert!(matches!(ctx.split_1d("t", max * max + 1), Err(GpuError::Limit(_))));
    }

    /// **EXV-RT B-RT.3 Increment 3 gate:** decode a WHOLE multi-tile EXV-RT `.exv` on the GPU (per-channel
    /// VRAM composite, no readback on the hot path) and assert it is BYTE-FOR-BYTE == the CPU
    /// `exv_codec::read_still` at q=0. q=0 is the pure-integer inverse path (5/3 lift + clamp, NO `exp2`),
    /// so equality is strict — NOT a tolerance (the q>0 ≤1-ULP behaviour is covered by `exvrt_gpu_qN_ulp`).
    /// The fixture is multi-tile with EDGE tiles (300×200, 128-tile grid → dims are a multiple of neither
    /// the 128 tile nor the 16 block): two F16 channels through the GPU block decoder + one U32 channel
    /// through the RAW/CPU-composite arm (a EXV-RT image stores non-F16 channels as `TAG_RAW`). Both arms
    /// are asserted present so the gate can never be green-but-empty.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_decode_still_exvrt() {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, write_still};
        let (w, h) = (300u32, 200u32);
        let n = (w * h) as usize;
        // A, B: smooth F16 ramps → the block codec engages (small adaptive widths) → TAG_EXV_RT → GPU path.
        let a: Vec<u8> = (0..n).flat_map(|i| f16_le((i % 97) as f32 * 0.05 + 0.1)).collect();
        let b: Vec<u8> = (0..n).flat_map(|i| f16_le((i % 61) as f32 * 0.03 + 0.2)).collect();
        // C: U32 → the block codec never touches a non-F16 dtype → stored RAW → CPU-composite arm.
        let mut seed = 0x51ed_0a17_c0ff_ee42u64;
        let c: Vec<u8> = (0..n).flat_map(|_| (xorshift(&mut seed) as u32).to_le_bytes()).collect();
        let prof = Profile::exv_rt_dwt(16, 0); // block=16, q=0 (lossless lifting = bit-exact)
        let chans = [("A", Dtype::F16), ("B", Dtype::F16), ("C", Dtype::U32)];
        let header = header(w, h, 128, 128, prof.codec_id().0, prof.params(), &chans);
        let img = Image::new(header, vec![a, b, c]).expect("img");
        let bytes = write_still(&img, &prof).expect("encode .exv");

        // The fixture MUST exercise BOTH the GPU BlockRt arm (≥1 TAG_EXV_RT tile) and the RAW/CPU arm
        // (the U32 channel), else the gate is green-but-empty.
        let (mut has_blockrt, mut has_raw) = (false, false);
        for_blobs(&bytes, |k| match k {
            Some(BlobKind::BlockRt(_)) => has_blockrt = true,
            Some(BlobKind::Raw(_)) => has_raw = true,
            _ => {}
        });
        assert!(has_blockrt, "fixture must contain a TAG_EXV_RT tile to exercise the GPU block decoder");
        assert!(has_raw, "fixture must contain a RAW tile (the U32 channel) to exercise the CPU-composite arm");

        let ctx = GpuContext::new().expect("gpu ctx");
        let gimg = decode_still_gpu(&ctx, &bytes, &DecodeLimits::default()).expect("gpu exv-rt still decode");
        let planes = read_back_image(&ctx, &gimg).expect("readback");
        // q=0 → strictly bit-exact vs the CPU decode of the SAME on-disk blobs (both decode identically).
        let cpu = read_still(&bytes).expect("cpu read");
        assert_eq!(planes, cpu.planes, "GPU EXV-RT whole-image decode != CPU read_still (q=0 must be bit-exact)");
    }

    /// The four F16 `R,G,B,A` planes shared by the gather gates (negative `B` exercises the sign path).
    fn rgba_f16_planes(n: usize) -> Vec<Vec<u8>> {
        let f16p = |f: &dyn Fn(usize) -> f32| -> Vec<u8> { (0..n).flat_map(|i| f16_le(f(i))).collect() };
        vec![
            f16p(&|i| (i % 97) as f32 * 0.05 + 0.1),
            f16p(&|i| (i % 61) as f32 * 0.03 + 0.2),
            f16p(&|i| (i % 53) as f32 * 0.07 - 1.3),
            f16p(&|i| (i % 13) as f32 * 0.05),
        ]
    }

    /// Channels of the gather gates: F16 `R,G,B,A`.
    const RGBA_F16: [(&str, Dtype); 4] = [("R", Dtype::F16), ("G", Dtype::F16), ("B", Dtype::F16), ("A", Dtype::F16)];

    /// **Task-A gather gate (HEADLESS):** decode a small multi-channel still with BOTH LosslessFast
    /// (ReorderDelta → `BytePerWord` planes) and EXV-RT (BlockRt → `ElemPerWord` planes), gather it into
    /// an `Rgba32Float` texture ON THE GPU via [`GpuFrame::to_rgba_texture`], read the texture back, and
    /// compare to the CPU reference gather (`read_still` → the same channel-pick + dtype→f32 the
    /// viewer/capi do). Proves the WGSL gather pass is correct with NO window. q=0 BlockRt + finite f16
    /// → the `exp2` in `half_to_f32` is exact, so ≤1 ULP is the (loose) bound; in practice bit-exact.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_frame_rgba_gather_matches_cpu() {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, write_still};
        let (w, h) = (64u32, 48u32); // w*16 = 1024 is 256-aligned → no padded-row edge case
        let n = (w * h) as usize;
        for prof in [Profile::lossless_fast(), Profile::exv_rt_dwt(16, 0)] {
            let header = header(w, h, 128, 128, prof.codec_id().0, prof.params(), &RGBA_F16);
            let img = Image::new(header, rgba_f16_planes(n)).expect("img");
            let bytes = write_still(&img, &prof).expect("encode .exv");

            let dec = GpuDecoder::new().expect("decoder");
            let frame = dec.decode_still(&bytes, &DecodeLimits::default()).expect("decode_still");
            assert_eq!((frame.width, frame.height), (w, h));
            let tex = frame.to_rgba_texture([Some(0), Some(1), Some(2)], Some(3)).expect("gather");
            dec.context().drain().expect("drain");
            let gpu = readback_rgba32f(dec.context(), &tex, w, h);

            // CPU reference: exactly the viewer/capi gather (read_still → dtype→f32 per channel).
            let cpu = read_still(&bytes).expect("cpu read");
            let mut max_ulp = 0i64;
            for i in 0..n {
                for (ci, plane) in cpu.planes.iter().enumerate() {
                    let want = sample_f32(plane, Dtype::F16, i);
                    let got = gpu[i * 4 + ci];
                    let ulp = (want.to_bits() as i64 - got.to_bits() as i64).abs();
                    max_ulp = max_ulp.max(ulp);
                    assert!(ulp <= 1, "px {i} ch {ci}: cpu {want} gpu {got} ({ulp} ULP), {:?}", prof.transform);
                }
            }
            eprintln!("gather {:?}: max {max_ulp} ULP over {} px × 4 ch", prof.transform, n);
        }
    }

    /// `GpuFrame` inspection/export API: `channel_count`/`channel_buffer` (in range `Some` with the right dtype,
    /// beyond `None`) and the CPU gather `read_back_rgba` (F32 and F16) == the reference gather of `read_still`
    /// planes (`sample_f32` + `f32_to_half`), incl. the absent-channel defaults (rgb 0.0, alpha 1.0).
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_frame_read_back_rgba_and_channels() {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::{read_still, write_still};
        let (w, h) = (64u32, 48u32);
        let n = (w * h) as usize;
        let prof = Profile::lossless_fast();
        let header = header(w, h, 128, 128, prof.codec_id().0, prof.params(), &RGBA_F16);
        let img = Image::new(header, rgba_f16_planes(n)).expect("img");
        let bytes = write_still(&img, &prof).expect("encode .exv");
        let cpu = read_still(&bytes).expect("cpu read");
        let dec = GpuDecoder::new().expect("decoder");
        let frame = dec.decode_still(&bytes, &DecodeLimits::default()).expect("decode_still");

        assert_eq!(frame.channel_count(), 4);
        for i in 0..frame.channel_count() {
            let t = frame.channel_buffer(i).expect("plane in range");
            assert_eq!((t.dtype, t.w, t.h), (Dtype::F16, w, h), "plane {i} metadata");
        }
        assert!(frame.channel_buffer(4).is_none(), "plane beyond channel_count is None");

        // slot -> source channel; `None` = absent (rgb 0.0 / alpha 1.0).
        let slots: [Option<usize>; 4] = [Some(0), None, Some(2), None];
        let want = |slot: usize, i: usize| match slots[slot] {
            Some(c) => sample_f32(&cpu.planes[c], Dtype::F16, i),
            None => if slot == 3 { 1.0 } else { 0.0 },
        };
        let f32s = frame.read_back_rgba([slots[0], slots[1], slots[2]], slots[3], RgbaFormat::F32, 0).expect("f32 rgba");
        let f16s = frame.read_back_rgba([slots[0], slots[1], slots[2]], slots[3], RgbaFormat::F16, 0).expect("f16 rgba");
        assert_eq!((f32s.len(), f16s.len()), (n * 16, n * 8));
        for i in 0..n {
            for slot in 0..4 {
                let v = want(slot, i);
                let o32 = (i * 4 + slot) * 4;
                assert_eq!(f32s[o32..o32 + 4], v.to_le_bytes(), "f32 px {i} slot {slot}");
                let o16 = (i * 4 + slot) * 2;
                assert_eq!(f16s[o16..o16 + 2], f32_to_half(v).to_le_bytes(), "f16 px {i} slot {slot}");
            }
        }
    }

    /// Task-A NoGpuDecoder gate: a lossy (WaveletLossy) `.exv` has no GPU decoder here —
    /// [`decode_still_gpu`] returns `NoGpuDecoder` (a routing outcome), NOT a panic and NOT a silent
    /// CPU-mimicked result. The caller (viewer) then falls back to CPU `read_still`.
    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_lossy_returns_no_gpu_decoder() {
        use exv_codec::Profile;
        use exv_codec::image::Image;
        use exv_video::write_still;
        let (w, h) = (64u32, 48u32);
        let n = (w * h) as usize;
        let plane: Vec<u8> = (0..n).flat_map(|i| f16_le((i % 50) as f32 * 0.1)).collect();
        let prof = Profile::lossy_wave(8);
        let header = header(w, h, 128, 128, prof.codec_id().0, prof.params(), &[("R", Dtype::F16)]);
        let img = Image::new(header, vec![plane]).expect("img");
        let bytes = write_still(&img, &prof).expect("encode .exv");
        let dec = GpuDecoder::new().expect("decoder");
        match dec.decode_still(&bytes, &DecodeLimits::default()) {
            Err(GpuDecodeError::NoGpuDecoder { .. }) => {}
            Err(e) => panic!("expected NoGpuDecoder for a lossy .exv, got other error: {e}"),
            Ok(_) => panic!("expected NoGpuDecoder for a lossy .exv, got Ok(frame)"),
        }
    }

    /// **Temporal Tier A gate:** clips with SKIP frames (identical frames, partial changes, a return to an old frame)
    /// show EVERY frame bit-exact vs the CPU `decode_frame`, in ascending, reverse and shuffled seek order — for the
    /// lossy wave (CPU tile decode + upload), EXV-RT (GPU block kernel) and a lossless HTJ2K clip of identical frames
    /// (all SKIP) — and SKIP tiles are carried over without a decode.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_skip_clip_matches_cpu_every_frame() {
        use exv_codec::Profile;
        use exv_video::{TemporalClass, VideoIndex, decode_frame, temporal_class, write_video};
        use jph_wgpu_rs::{ContentKey, DisplayKey, Gpu};

        let (w, h) = (600u32, 400u32); // 3x2 tiles
        let a = f16_image(w, h, 256, 256, &["R", "G", "B", "A", "Y"], 0);
        let b = poke_tile(&a, 2, 1);
        let c = poke_tile(&b, 0, 0);
        let partial = vec![a.clone(), a.clone(), b.clone(), b.clone(), c.clone(), a.clone(), c.clone()];
        let held = vec![a.clone(), a.clone(), a.clone(), a.clone()];
        let cases: [(&str, Profile, &Vec<exv_codec::image::Image>); 3] = [
            ("lossy wave", Profile::lossy_wave(8), &partial),
            ("EXV-RT", Profile::exv_rt_dwt(16, 0), &partial),
            ("lossless HTJ2K, held frames", Profile::htj2k(), &held),
        ];
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let tiles = tile_grid(3, 2);
        for (name, prof, frames) in cases {
            let bytes = write_video(frames, &prof, 8, 0, 0).expect("encode");
            let idx = VideoIndex::parse(&bytes).expect("idx");
            assert_eq!(temporal_class(&bytes, &idx, TemporalClass::Bidir), Ok(TemporalClass::SkipOnly), "{name}: INTRA + SKIP");
            let cpu: Vec<_> = (0..frames.len()).map(|k| decode_frame(&bytes, &idx, k, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("cpu").planes).collect();
            let mut atlas = new_atlas(&ctx);
            let mut seed = 0x9E37_79B9_7F4A_7C15u64;
            let mut shuffled: Vec<usize> = (0..frames.len()).cycle().take(3 * frames.len()).collect();
            for i in (1..shuffled.len()).rev() {
                shuffled.swap(i, (xorshift(&mut seed) % (i as u64 + 1)) as usize);
            }
            let asc: Vec<usize> = (0..frames.len()).collect();
            let desc: Vec<usize> = asc.iter().rev().copied().collect();
            for (pass, order) in [asc, desc, shuffled].iter().enumerate() {
                for &k in order {
                    atlas
                        .request(Clip { data: &bytes, idx: &idx }, k, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0))
                        .unwrap_or_else(|e| panic!("{name} pass {pass} frame {k}: {e}"));
                    for &t in &tiles {
                        assert_tile(&gpu, &atlas, &cpu[k], &idx.header, t, 3, &format!("{name} pass {pass} frame {k}"));
                    }
                }
            }
            let st = atlas.stats();
            assert!(st.reused > 0, "{name}: SKIP tiles must be carried over without a decode: {st}");
        }
    }

    /// **Channel-subset gate (audit item 5):** for every resident route (lossless HTJ2K = CPU decode + batched upload,
    /// lossless wave = GPU F16 + CPU F32/U32, EXV-RT = GPU F16 + RAW F32/U32) over the R,G,B F16 + Z F32 + id U32 fixture:
    /// a routed `R,G,B` request makes exactly 3 planes per tile resident (budget = their bytes, Z/id absent) and they are
    /// bit-identical to a full decode AND to the CPU `read_still`; a second request adding Z decodes ONLY Z (one plane per
    /// tile, the RGB textures kept, not re-created); the union then hits without opening the container; compose refuses a
    /// routed channel that is not resident.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_channel_subset_decodes_and_merges_only_the_delta() {
        use crate::compose::{PlaneCompose, UNMAPPED};
        use exv_codec::Profile;
        use exv_video::{VideoIndex, read_still, write_still};
        use jph_wgpu_rs::texture::read_plane;
        use jph_wgpu_rs::{ContentKey, DisplayKey, Gpu};
        use std::sync::atomic::Ordering::Relaxed;
        use wgpu_tile_atlas::TilePayload;

        let (w, h) = (600u32, 400u32); // 3x2 tiles of 256 (edge tiles 88 x 144: padded upload rows)
        let img = synth_mc(w, h);
        let dtypes: Vec<Dtype> = img.header.channels.iter().map(|c| c.dtype).collect();
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let tiles = tile_grid(3, 2);
        let n = tiles.len() as u64;
        // Every pixel of plane `ci` of tile `t` (both atlases) == the CPU decode, on bit patterns.
        let check = |atlas: &ExvResidentAtlas, cpu: &[Vec<u8>], (rx, ry): (u32, u32), ci: usize, what: &str| {
            let (x0, y0, tw, th) = img.header.tile_rect(rx, ry);
            let p = atlas.get(None, rx, ry).and_then(|p| p.plane(ci)).unwrap_or_else(|| panic!("{what}: ({rx},{ry}) ch{ci} absent"));
            let got = read_plane(&gpu, p, tw, th, p.tex.format()).expect("read plane");
            for ty in 0..th {
                for tx in 0..tw {
                    let want = sample_f32(&cpu[ci], dtypes[ci], ((y0 + ty) * w + x0 + tx) as usize);
                    assert_eq!(got[(ty * tw + tx) as usize].to_bits(), want.to_bits(), "{what}: ({rx},{ry}) ch{ci} px({tx},{ty})");
                }
            }
        };
        for (name, prof) in [("htj2k", Profile::htj2k()), ("wave", Profile::lossless_wave()), ("exv-rt", Profile::exv_rt_dwt(16, 0))] {
            let bytes = write_still(&img, &prof).expect("encode");
            let cpu = read_still(&bytes).expect("cpu").planes;
            let idx = VideoIndex::parse(&bytes).expect("idx");
            let clip = Clip { data: &bytes, idx: &idx };
            let rgb = Want { tiles: &tiles, level: None, chans: Chans::Only(&[2, 0, 1]) };

            let mut full = ExvResidentAtlas::new(&ctx, 512 << 20, DECODE_BUDGET).expect("atlas");
            full.request(clip, 0, Want { chans: Chans::All, ..rgb }, ContentKey(1), DisplayKey(0)).expect("full");
            let mut sub = ExvResidentAtlas::new(&ctx, 512 << 20, DECODE_BUDGET).expect("atlas");
            sub.request(clip, 0, rgb, ContentKey(1), DisplayKey(0)).expect("rgb");
            let st = sub.stats();
            assert_eq!((st.decodes, st.planes), (n, 3 * n), "{name}: RGB = 3 planes per tile: {st}");
            let rgb_bytes: usize = tiles.iter().map(|&(rx, ry)| sub.get(None, rx, ry).expect("resident").bytes()).sum();
            assert_eq!(st.resident_bytes, rgb_bytes, "{name}: the budget charges the present planes only");
            for &t in &tiles {
                let p = sub.get(None, t.0, t.1).expect("resident");
                assert!(p.plane(3).is_none() && p.plane(4).is_none(), "{name}: Z / id were not requested, so not resident");
                for ci in 0..3 {
                    check(&sub, &cpu, t, ci, &format!("{name} subset"));
                    check(&full, &cpu, t, ci, &format!("{name} full"));
                }
            }
            // Both equal the CPU decode bit for bit (checked above), so the subset's RGB == the full decode's RGB.
            assert!(full.stats().resident_bytes > st.resident_bytes, "{name}: the full decode holds more planes");

            // Add Z: only the delta decodes; the RGB textures are kept (same wgpu texture objects).
            let before: Vec<wgpu::Texture> = tiles.iter().map(|&(rx, ry)| sub.get(None, rx, ry).and_then(|p| p.plane(0)).expect("R").tex.clone()).collect();
            let opens = crate::atlas::FRAME_OPENS.load(Relaxed);
            sub.request(clip, 0, Want { chans: Chans::Only(&[0, 1, 2, 3]), ..rgb }, ContentKey(1), DisplayKey(0)).expect("rgb+z");
            let st2 = sub.stats();
            assert_eq!(crate::atlas::FRAME_OPENS.load(Relaxed), opens + 1, "{name}: the delta is a miss");
            assert_eq!((st2.decodes, st2.planes), (2 * n, 4 * n), "{name}: exactly one new plane per tile: {st2}");
            for (&t, r) in tiles.iter().zip(&before) {
                assert!(sub.get(None, t.0, t.1).and_then(|p| p.plane(0)).is_some_and(|p| p.tex == *r), "{name}: R kept, not re-uploaded");
                check(&sub, &cpu, t, 3, &format!("{name} merged Z"));
            }
            // Z (F32: HTJ2K CPU route, wave esz > 2, EXV-RT RAW) is CPU-decoded on every route: the upload grew by exactly the
            // Z planes (tight 1024-B rows, padded edge rows).
            let z: u64 = tiles
                .iter()
                .map(|&(rx, ry)| {
                    let (_, _, tw, th) = img.header.tile_rect(rx, ry);
                    (tw as u64 * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64) * th as u64
                })
                .sum();
            assert_eq!(st2.uploaded_bytes - st.uploaded_bytes, z, "{name}: only Z crossed the bus");
            // Asking for any subset of what is resident is a pure hit (no frame open, no decode).
            sub.request(clip, 0, Want { chans: Chans::Only(&[3, 1]), ..rgb }, ContentKey(1), DisplayKey(0)).expect("hit");
            assert_eq!(crate::atlas::FRAME_OPENS.load(Relaxed), opens + 1, "{name}: a resident subset is a hit");
            assert_eq!(sub.stats().decodes, 2 * n);

            // Compose: routed resident channels work; a routed channel that is not resident (id) is a typed error.
            let mut compose = PlaneCompose::new(&ctx).expect("compose");
            compose.compose(&ctx, &sub, None, &tiles, w, h, 256, 256, [0, 1, 2, UNMAPPED]).expect("compose rgb");
            let err = compose.compose(&ctx, &sub, None, &tiles, w, h, 256, 256, [4, 1, 2, UNMAPPED]).expect_err("id is absent");
            assert!(err.contains("not resident"), "{name}: {err}");
        }
    }

    /// Ascending playback decodes exactly one tile per distinct INTRA node: SKIP tiles cost nothing at any GOP depth.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_sequential_playback_decodes_each_node_once() {
        use exv_codec::Profile;
        use exv_video::{TileChains, VideoIndex, write_video};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let a = f16_image(w, h, 256, 256, &["R", "G", "B", "A", "Y"], 0);
        let frames = vec![a.clone(), a.clone(), poke_tile(&a, 2, 1), poke_tile(&a, 2, 1), a.clone()];
        let bytes = write_video(&frames, &Profile::lossy_wave(8), 8, 0, 0).expect("encode");
        let idx = VideoIndex::parse(&bytes).expect("idx");
        let mut walk = TileChains::new(&bytes, &idx);
        let mut last = [usize::MAX; 6];
        let mut want = 0u64;
        for di in 0..frames.len() {
            for t in 0..6u32 {
                let base = walk.chain(di, t).expect("chain").base;
                if last[t as usize] != base {
                    last[t as usize] = base;
                    want += 1;
                }
            }
        }
        let ctx = GpuContext::new().expect("gpu ctx");
        let mut atlas = new_atlas(&ctx);
        for k in 0..frames.len() {
            atlas.request(Clip { data: &bytes, idx: &idx }, k, Want { tiles: &tile_grid(3, 2), level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("request");
        }
        assert_eq!(atlas.decode_count(), want, "{}", atlas.stats());
        assert!(want < 6 * frames.len() as u64);
    }

    /// Random seeks with a TINY atlas budget (about two tiles; only two tiles are visible at a time): LRU eviction
    /// drops payloads together with their node stamps, so a re-requested tile is re-decoded from the right ancestor,
    /// never served stale. Bit-exact vs the CPU `decode_frame` on every request.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_skip_clip_tiny_budget_random_seeks() {
        use exv_codec::Profile;
        use exv_video::{VideoIndex, decode_frame, write_video};
        use jph_wgpu_rs::{ContentKey, DisplayKey, Gpu};

        let (w, h) = (600u32, 400u32);
        let mut frames = vec![f16_image(w, h, 256, 256, &["R", "G", "B", "A", "Y"], 0)];
        for t in [(2, 1), (0, 0), (1, 1), (2, 0), (0, 1), (1, 0), (2, 1)] {
            let next = poke_tile(&frames[frames.len() - 1], t.0, t.1);
            frames.push(next.clone());
            frames.push(next); // an identical frame: all SKIP
        }
        let bytes = write_video(&frames, &Profile::lossy_wave(8), 16, 0, 0).expect("encode");
        let idx = VideoIndex::parse(&bytes).expect("idx");
        let cpu: Vec<_> = (0..frames.len()).map(|k| decode_frame(&bytes, &idx, k, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("cpu").planes).collect();
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        let tile_bytes = 5 * 256 * 256 * 2; // five R16Float planes of a full tile
        let mut atlas = ExvResidentAtlas::new(&ctx, 2 * tile_bytes + tile_bytes / 2, DECODE_BUDGET).expect("atlas");
        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        for step in 0..80 {
            let k = (xorshift(&mut seed) % frames.len() as u64) as usize;
            let (a, b) = ((xorshift(&mut seed) % 3) as u32, (xorshift(&mut seed) % 2) as u32);
            let (c, d) = ((xorshift(&mut seed) % 3) as u32, (xorshift(&mut seed) % 2) as u32);
            let want = [(a, b), (c, d)];
            atlas
                .request(Clip { data: &bytes, idx: &idx }, k, Want { tiles: &want, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0))
                .unwrap_or_else(|e| panic!("step {step} frame {k}: {e}"));
            for &t in &want {
                assert_tile(&gpu, &atlas, &cpu[k], &idx.header, t, 3, &format!("step {step} frame {k}"));
            }
        }
        let st = atlas.stats();
        assert!(st.resident_bytes <= st.budget_bytes + tile_bytes, "eviction keeps the atlas near budget: {st}");
        // A visible set bigger than the budget is a hard error, not a silent VRAM overshoot.
        let all = tile_grid(3, 2);
        let err = atlas.request(Clip { data: &bytes, idx: &idx }, 3, Want { tiles: &all, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).unwrap_err();
        assert!(err.contains("atlas budget"), "{err}");
    }

    /// **Plane-texture pool gate (P1-28):** an evicted / replaced tile's textures are recycled for the next tile of the SAME
    /// extent (edge tiles are their own size) and a recycled texture never shows the previous tile's pixels. On every
    /// resident route (GPU blit: wave + EXV-RT, CPU decode + upload: HTJ2K) over a 2-frame all-INTRA clip whose frames differ
    /// everywhere: with a budget of ~1.5 edge tiles, the 88x256 tile (2,0) is evicted by the next plan and its planes are
    /// the very objects the next decode of (2,1) lands in (`==`), bit-exact vs the CPU decode of the OTHER frame; the old
    /// (2,1) planes recycled by that replace then carry (2,0) the same way; a full 256x256 tile never takes an edge-sized
    /// texture. The idle bytes stay under the cap and a dropped atlas leaves none.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_plane_pool_recycles_without_stale_texels() {
        use exv_codec::Profile;
        use exv_video::{VideoIndex, decode_frame, write_video};
        use jph_wgpu_rs::{ContentKey, DisplayKey, Gpu};

        let (w, h) = (600u32, 512u32); // column 2 = two 88 x 256 edge tiles
        let names = ["R", "G", "B"];
        let frames = [f16_image(w, h, 256, 256, &names, 0), f16_image(w, h, 256, 256, &names, 300)];
        let ctx = GpuContext::new().expect("gpu ctx");
        let gpu = Gpu::from_parts(ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone());
        for (name, prof) in [("wave", Profile::lossless_wave()), ("htj2k cpu upload", Profile::htj2k()), ("exv-rt", Profile::exv_rt_dwt(16, 0))] {
            let bytes = write_video(&frames, &prof, 1, 0, 0).expect("encode all-INTRA clip");
            let idx = VideoIndex::parse(&bytes).expect("idx");
            let clip = Clip { data: &bytes, idx: &idx };
            let cpu: Vec<_> = (0..2).map(|k| decode_frame(&bytes, &idx, k, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("cpu").planes).collect();
            let req = |a: &mut ExvResidentAtlas, k: usize, t: (u32, u32)| {
                a.request(clip, k, Want { tiles: &[t], level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).unwrap_or_else(|e| panic!("{name} frame {k} tile {t:?}: {e}"));
            };
            // The resident bytes of ONE edge tile on this route (R32 blit vs R16 upload planes), unbounded probe atlas.
            let mut probe = ExvResidentAtlas::new(&ctx, 0, DECODE_BUDGET).expect("probe atlas");
            req(&mut probe, 0, (2, 0));
            let edge = probe.stats().resident_bytes;
            drop(probe);
            ctx.trim_tex_pool();

            let mut atlas = ExvResidentAtlas::new(&ctx, edge + edge / 2, DECODE_BUDGET).expect("atlas");
            let planes_of = |a: &ExvResidentAtlas, t: (u32, u32)| -> Vec<wgpu::Texture> {
                (0..3).map(|ci| a.get(None, t.0, t.1).and_then(|p| p.plane(ci)).expect("resident plane").tex.clone()).collect()
            };
            req(&mut atlas, 0, (2, 0));
            let held = planes_of(&atlas, (2, 0)); // handles of what will be evicted
            req(&mut atlas, 0, (2, 1)); // over budget now: (2,0) is no longer visible, evicted by the NEXT plan
            let old21 = planes_of(&atlas, (2, 1));
            req(&mut atlas, 1, (2, 1)); // evicts (2,0) -> pool; the new node's decode takes its textures
            let new21 = planes_of(&atlas, (2, 1));
            assert!(atlas.get(None, 2, 0).is_none(), "{name}: (2,0) was evicted");
            assert!(new21.iter().all(|t| held.contains(t)), "{name}: the replaced tile landed in the evicted tile's textures");
            assert_tile(&gpu, &atlas, &cpu[1], &idx.header, (2, 1), 3, &format!("{name} recycled (2,1) frame 1"));
            // The replaced (2,1) planes were recycled by the insert: (2,0) of frame 1 lands in them.
            req(&mut atlas, 1, (2, 0));
            let new20 = planes_of(&atlas, (2, 0));
            assert!(new20.iter().all(|t| old21.contains(t)), "{name}: (2,0) reuses the planes the replace recycled");
            assert_tile(&gpu, &atlas, &cpu[1], &idx.header, (2, 0), 3, &format!("{name} recycled (2,0) frame 1"));
            // A 256x256 tile cannot take an 88x256 texture (exact-extent key): fresh textures.
            // (An unbounded second atlas: a full tile alone is over `atlas`'s ~1.5-edge-tile budget.)
            let mut big = ExvResidentAtlas::new(&ctx, 0, DECODE_BUDGET).expect("unbounded atlas");
            req(&mut atlas, 1, (2, 1)); // a pure hit, but its viewport update evicts (2,0): edge textures sit idle in the pool
            assert!(ctx.tex_pool_stats().0 >= 3, "{name}: the evicted tile's planes are pooled");
            req(&mut big, 1, (0, 0));
            let full = planes_of(&big, (0, 0));
            assert!(full.iter().all(|t| !held.contains(t) && !old21.contains(t)), "{name}: a differently sized tile never reuses an edge texture");
            assert_tile(&gpu, &big, &cpu[1], &idx.header, (0, 0), 3, &format!("{name} fresh (0,0) frame 1"));
            let (n, bytes_idle) = ctx.tex_pool_stats();
            assert!(bytes_idle <= ctx.tex_pool_cap(), "{name}: {n} idle textures hold {bytes_idle} B over the {} B cap", ctx.tex_pool_cap());
            assert_eq!(atlas.stats().pooled_bytes, bytes_idle);
            drop(big);
            drop(atlas);
            assert_eq!(ctx.tex_pool_stats(), (0, 0), "{name}: a dropped atlas leaves no pooled VRAM");
        }
    }

    /// **Pool cap gate (P1-28):** the idle plane textures never exceed `tex_pool_cap`; a texture over the whole cap is
    /// freed, not pooled; `trim_tex_pool` frees all; a take of the exact key returns the recycled object.
    #[test]
    #[ignore = "requires a GPU"]
    fn plane_tex_pool_cap_exact_key_and_trim() {
        let ctx = GpuContext::new().expect("gpu ctx");
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC;
        let fmt = wgpu::TextureFormat::R32Float;
        ctx.trim_tex_pool();
        let cap = ctx.tex_pool_cap();
        // Three textures of ~cap/3 each + one over the whole cap (rows chosen from the cap, whatever the device's budget).
        let rows = ((cap / 3 / 4 / 4096) as u32).clamp(1, 8192);
        let one = 4096u64 * u64::from(rows) * 4;
        let texs: Vec<_> = (0..5).map(|_| ctx.plane_tex("pool.gate", 4096, rows, fmt, usage)).collect();
        let first = texs[0].clone();
        for t in texs {
            ctx.pipelines.recycle_tex(t);
        }
        let (n, idle) = ctx.tex_pool_stats();
        assert!(idle <= cap && idle == n as u64 * one, "{n} idle textures, {idle} B, cap {cap} B");
        // Exact key only: another size or usage is a fresh texture; the same key gives back a pooled object.
        let other = ctx.plane_tex("pool.gate", 4096, rows + 1, fmt, usage);
        assert!(other != first);
        let again = ctx.plane_tex("pool.gate", 4096, rows, fmt, usage);
        assert_eq!(ctx.tex_pool_stats().0, n - 1, "a pooled texture of the exact key was taken");
        drop(again);
        let huge = ctx.plane_tex("pool.gate.huge", 8192, 8192, wgpu::TextureFormat::Rgba32Float, usage);
        let before = ctx.tex_pool_stats();
        ctx.pipelines.recycle_tex(huge);
        let after = ctx.tex_pool_stats();
        assert!(after.1 <= cap, "{after:?} over the cap {cap}");
        if 8192u64 * 8192 * 16 > cap {
            assert_eq!(after, before, "a texture over the whole cap is dropped, not pooled");
        }
        ctx.trim_tex_pool();
        assert_eq!(ctx.tex_pool_stats(), (0, 0));
    }

    /// Temporal classes the tile-local GPU path does not decode are TYPED errors, never a guessed frame: lossless PRED
    /// tiles, and SKIP tiles under a non-zero MV (a shifted copy). Frame 0 (INTRA) still shows.
    #[cfg(feature = "htj2k-backend")]
    #[test]
    #[ignore = "requires a GPU"]
    fn atlas_rejects_pred_and_motion_chains() {
        use exv_codec::Profile;
        use exv_video::{TemporalClass, VideoIndex, motion, temporal_class, write_video};
        use jph_wgpu_rs::{ContentKey, DisplayKey};

        let (w, h) = (600u32, 400u32);
        let ctx = GpuContext::new().expect("gpu ctx");
        let tiles = tile_grid(3, 2);
        let a = f16_image(w, h, 256, 256, &["R", "G", "B", "A", "Y"], 0);

        // ONE changed pixel in tile (1,1): its residual vs the reference is nearly all zero, so PRED beats INTRA.
        let mut one = a.planes.clone();
        one[0][((300 * w + 300) * 2) as usize] ^= 1;
        let b = exv_codec::image::Image::new(a.header.clone(), one).expect("one-pixel frame");
        let pred = write_video(&[a.clone(), b], &Profile::lossless_fast(), 8, 0, 0).expect("encode");
        let idx = VideoIndex::parse(&pred).expect("idx");
        assert_eq!(temporal_class(&pred, &idx, TemporalClass::Bidir), Ok(TemporalClass::Pred));
        let mut atlas = new_atlas(&ctx);
        atlas.request(Clip { data: &pred, idx: &idx }, 0, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).expect("I-frame is fine");
        let err = atlas.request(Clip { data: &pred, idx: &idx }, 1, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).unwrap_err();
        assert!(err.contains("PRED"), "{err}");

        let pan: Vec<_> = (0..3i32)
            .map(|i| {
                let planes = a
                    .planes
                    .iter()
                    .zip(&a.header.channels)
                    .map(|(p, c)| motion::shift_plane(p, w, h, c.dtype, i * 3, i * 2).expect("shift"))
                    .collect();
                exv_codec::image::Image::new(a.header.clone(), planes).expect("pan frame")
            })
            .collect();
        let moved = write_video(&pan, &Profile::lossless_fast(), 4, 8, 0).expect("encode");
        let idx = VideoIndex::parse(&moved).expect("idx");
        assert_eq!(temporal_class(&moved, &idx, TemporalClass::Bidir), Ok(TemporalClass::Motion));
        let mut atlas = new_atlas(&ctx);
        let err = atlas.request(Clip { data: &moved, idx: &idx }, 1, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).unwrap_err();
        assert!(err.contains("motion vector"), "{err}");
        let oob = atlas.request(Clip { data: &moved, idx: &idx }, 9, Want { tiles: &tiles, level: None, chans: Chans::All }, ContentKey(1), DisplayKey(0)).unwrap_err();
        assert!(oob.contains("out of range"), "{oob}");
    }

    #[test]
    fn bc6h_gpu_matches_cpu_lossy_and_lossless() {
        use exv_codec::Profile;
        use exv_video::{read_still, write_still};

        let Ok(ctx) = GpuContext::new() else {
            return;
        };
        let img = f16_image(37, 23, 16, 16, &["R", "G", "B", "A"], 7);
        for (prof, is_lossless) in [(Profile::bc6h_lossy(), false), (Profile::bc6h_lossless(), true)] {
            let bytes = write_still(&img, &prof).expect("encode bc6h");
            let cpu = read_still(&bytes).expect("cpu decode bc6h");
            let gpu = decode_still_gpu(&ctx, &bytes, &DecodeLimits::default()).expect("gpu decode bc6h");
            let gpu_planes: Vec<Vec<u8>> = gpu
                .planes
                .iter()
                .map(|t| read_back(&ctx, t).expect("read_back bc6h plane"))
                .collect();
            assert_eq!(gpu_planes, cpu.planes, "GPU BC6H (lossless={is_lossless}) must match CPU bit-for-bit");
            if is_lossless {
                assert_eq!(gpu_planes, img.planes, "BC6H LosslessResidual must reconstruct input bit-exact");
            }
        }
    }
}

