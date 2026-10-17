//! Tier-0 QUALITY GATE (`docs/DESIGN_LOG.md` D-09): does the EXV-RT block codec (`blockrt`) hold quality on
//! real HDR VFX content? Reads a 1-frame `.exv` still (env `EXV_QUALITY_FILE`, default the robat frame), and for
//! each float channel sweeps (block size × index bits) reporting bpp + relative error (rms / max). This
//! tells us whether 8-bpp-class block compression is visually-lossless on the corpus, or whether a
//! per-block transform (DWT) is needed before Tier-2 ships.
//!
//! Run: `EXV_QUALITY_FILE=C:/temp/exv-rs/robat.0001.exv cargo test -p exv-codec --test blockrt_quality -- --ignored --nocapture`

use exv_codec::blockrt;
use exv_codec::{Profile, decode_channel, encode_channel};
use exv_format::half::f32_to_half as f32_to_f16;
use exv_video::read_still;
use exv_format::Dtype;

#[test]
#[ignore = "needs a .exv corpus file (EXV_QUALITY_FILE)"]
fn blockrt_quality_sweep() {
    let path = std::env::var("EXV_QUALITY_FILE")
        .unwrap_or_else(|_| "C:/temp/exv-rs/robat.0001.exv".to_string());
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: cannot read {path}: {e}");
            return;
        }
    };
    let img = read_still(&bytes).expect("read_still");
    let (w, h) = (img.header.width, img.header.height);
    eprintln!("file: {path}  {w}x{h}  {} channels", img.header.channels.len());

    // Convert any float channel to f16 LE (the realtime path is f16); skip integer/U* channels.
    let f16_of = |plane: &[u8], dt: Dtype| -> Option<Vec<u8>> {
        match dt {
            Dtype::F16 => Some(plane.to_vec()),
            Dtype::F32 => Some(
                plane
                    .as_chunks::<4>().0.iter()
                    .flat_map(|c| {
                        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                        f32_to_f16(v).to_le_bytes()
                    })
                    .collect(),
            ),
            _ => None,
        }
    };

    // Sweep a few channels (cap at 6 for output brevity) over block sizes and index bits.
    let combos = [(4u32, 2u32), (4, 3), (4, 4), (8, 3), (8, 4), (16, 4)];
    eprintln!("  (rms_rel / max_rel  per channel; bpp = bits/pixel/channel)");
    for (ci, ch) in img.header.channels.iter().enumerate().take(6) {
        let Some(src) = f16_of(&img.planes[ci], ch.dtype) else {
            eprintln!("  [{}] {:?} — skipped (not float)", ch.name, ch.dtype);
            continue;
        };
        for &log in &[false, true] {
            eprint!("  [{}]{} ", ch.name, if log { " LOG" } else { " lin" });
            for &(block, idx_bits) in &combos {
                let enc = blockrt::encode_plane_mode(&src, w, h, block, idx_bits, log);
                let dec = blockrt::decode_plane_mode(&enc, w, h, block, idx_bits, log);
                let (rms, maxr) = blockrt::quality(&src, &dec);
                eprint!(
                    "| {}b/{}i {:.1}bpp {:.3}/{:.2} ",
                    block, idx_bits, blockrt::bpp(block, idx_bits), rms, maxr
                );
            }
            eprintln!();
        }
    }
    eprintln!("note: rms_rel<~0.02 and max_rel modest = visually-lossless-ish; pick the lowest bpp that holds.");

    // --- Transform comparison: the EXISTING whole-plane wavelet-lossy path (5/3 DWT + quant + rANS) ---
    // This answers the decisive question CHEAPLY: does a TRANSFORM fix the colour quality where the
    // endpoint-lerp model failed? (Entropy backend is irrelevant to transform QUALITY; the GPU-parallel
    // block-transform version comes next IF this confirms the model.) bpp = encoded bytes * 8 / pixels.
    eprintln!("\n  TRANSFORM (wavelet-lossy, whole-plane; bpp = actual encoded size):");
    let qs = [2u8, 4, 8, 16];
    for (ci, ch) in img.header.channels.iter().enumerate().take(6) {
        let Some(src) = f16_of(&img.planes[ci], ch.dtype) else { continue; };
        eprint!("  [{}] ", ch.name);
        for &q in &qs {
            let prof = Profile::lossy_wave(q);
            // Encode the f16 plane through the real channel codec, decode, measure quality + size.
            let blob = encode_channel(&src, Dtype::F16, w, h, &prof).expect("encode");
            let bpp = blob.len() as f32 * 8.0 / (w as f32 * h as f32);
            match decode_channel(&blob, Dtype::F16, w, h, &prof) {
                Ok(dec) => {
                    let (rms, maxr) = blockrt::quality(&src, &dec);
                    eprint!("| q{} {:.1}bpp {:.3}/{:.2} ", q, bpp, rms, maxr);
                }
                Err(e) => eprint!("| q{} decode-fail ({e}) ", q),
            }
        }
        eprintln!();
    }
    eprintln!("  compare: transform rms vs endpoint-lerp rms at similar bpp -> does the transform fix colour?");

    // --- BLOCK-TRANSFORM sweep: the transform applied PER N×N BLOCK (the GPU-parallel decode unit) vs
    // whole-plane. Answers "does blocking the transform cost quality, and at which block size?" with
    // data. Each block goes through the SAME wavelet-lossy channel codec (proven); bpp = sum of block
    // blob bytes. Smaller block = less decorrelation (slightly worse) but must fit GPU shared memory
    // (16=4KB / 32=16KB / 64=64KB-too-big for f32×4ch). Container tile stays 256 (out-of-core), separate.
    eprintln!("\n  BLOCK-TRANSFORM (wavelet-lossy per N×N block, q=4; vs whole-plane above):");
    let blocks = [16u32, 32, 64, 128];
    for (ci, ch) in img.header.channels.iter().enumerate().take(6) {
        let Some(src) = f16_of(&img.planes[ci], ch.dtype) else { continue; };
        eprint!("  [{}] ", ch.name);
        for &n in &blocks {
            let (bpp, rms, maxr) = block_transform_quality(&src, w, h, n, 4);
            eprint!("| {}px {:.1}bpp {:.3}/{:.2} ", n, bpp, rms, maxr);
        }
        eprintln!();
    }
    eprintln!("  pick: smallest block whose rms ~matches whole-plane (that's the GPU transform-block size).");

    // --- SHARED-TABLE block codec: per-block transform+quant, but ALL blocks' coefficients pooled under
    // ONE rANS table (the fix for per-block table overhead). This is the real Tier-2 ratio: each block is
    // independently transformed/quantized (GPU-parallel decode), the entropy table is frame-wide. bpp here
    // = one rANS over the pooled coeff bytes / pixels. Expect bpp ~constant across block sizes (≈ whole-plane),
    // unlike the per-block-table 16bpp blowup above. q=4.
    eprintln!("\n  SHARED-TABLE (per-block transform+quant, ONE pooled rANS table, q=4):");
    for (ci, ch) in img.header.channels.iter().enumerate().take(6) {
        let Some(src) = f16_of(&img.planes[ci], ch.dtype) else { continue; };
        eprint!("  [{}] ", ch.name);
        for &n in &blocks {
            let (bpp, rms, maxr) = shared_table_quality(&src, w, h, n, 4);
            eprint!("| {}px {:.1}bpp {:.3}/{:.2} ", n, bpp, rms, maxr);
        }
        eprintln!();
    }
    eprintln!("  => shared-table bpp ~flat across block sizes confirms: small GPU block + frame table = ratio kept.");
}

/// Per-`n`×`n`-block transform+quant (independent per block — GPU-parallel) but ALL blocks' coefficients
/// entropy-coded under ONE shared rANS table. Returns `(bpp, rms_rel, max_rel)`. This is the real Tier-2
/// codec shape (block-parallel transform + frame-wide entropy table); the GPU decoder does per-block
/// rANS(state+offset, shared table) → dequant → inverse DWT.
fn shared_table_quality(src: &[u8], w: u32, h: u32, n: u32, q: u8) -> (f32, f32, f32) {
    use exv_codec::lossy::{coeffs_to_bytes, forward_quant, inverse_dequant};
    let mut pooled = Vec::new(); // all blocks' coeff bytes -> one rANS table
    let mut recon = vec![0u8; src.len()];
    let mut by = 0u32;
    while by < h {
        let bh = n.min(h - by);
        let mut bx = 0u32;
        while bx < w {
            let bw = n.min(w - bx);
            let mut sub = vec![0u8; (bw * bh) as usize * 2];
            for ry in 0..bh {
                let so = (((by + ry) * w + bx) as usize) * 2;
                let do_ = (ry * bw) as usize * 2;
                sub[do_..do_ + bw as usize * 2].copy_from_slice(&src[so..so + bw as usize * 2]);
            }
            // Independent per-block transform+quant (what the GPU inverts per block).
            let coeffs = forward_quant(&sub, Dtype::F16, bw, bh, q as u32);
            pooled.extend_from_slice(&coeffs_to_bytes(&coeffs));
            // Reconstruct this block for the quality metric.
            let dec = inverse_dequant(&coeffs, Dtype::F16, bw, bh, q as u32);
            for ry in 0..bh {
                let dofs = (((by + ry) * w + bx) as usize) * 2;
                let sofs = (ry * bw) as usize * 2;
                recon[dofs..dofs + bw as usize * 2].copy_from_slice(&dec[sofs..sofs + bw as usize * 2]);
            }
            bx += n;
        }
        by += n;
    }
    // ONE rANS over the pooled coefficient bytes = the shared-table ratio.
    let coded = rans::encode(&pooled, rans::RansConfig::default()).expect("encode");
    let bpp = coded.len() as f32 * 8.0 / (w as f32 * h as f32);
    let (rms, maxr) = blockrt::quality(src, &recon);
    (bpp, rms, maxr)
}

/// Encode `src` (f16 LE, w×h) PER N×N block through the wavelet-lossy channel codec, decode, reassemble,
/// and return `(bpp, rms_rel, max_rel)` vs the source. Edge blocks use their clamped dims. This measures
/// the quality cost of BLOCKING the transform (the GPU-parallel decode unit) at block size `n`.
fn block_transform_quality(src: &[u8], w: u32, h: u32, n: u32, q: u8) -> (f32, f32, f32) {
    let prof = Profile::lossy_wave(q);
    let mut recon = vec![0u8; src.len()];
    let mut total_bytes = 0usize;
    let mut by = 0u32;
    while by < h {
        let bh = n.min(h - by);
        let mut bx = 0u32;
        while bx < w {
            let bw = n.min(w - bx);
            // Extract the sub-block plane (row-major, f16 LE).
            let mut sub = vec![0u8; (bw * bh) as usize * 2];
            for ry in 0..bh {
                let src_off = (((by + ry) * w + bx) as usize) * 2;
                let dst_off = (ry * bw) as usize * 2;
                sub[dst_off..dst_off + bw as usize * 2]
                    .copy_from_slice(&src[src_off..src_off + bw as usize * 2]);
            }
            let blob = encode_channel(&sub, Dtype::F16, bw, bh, &prof).expect("encode");
            total_bytes += blob.len();
            if let Ok(dec) = decode_channel(&blob, Dtype::F16, bw, bh, &prof) {
                for ry in 0..bh {
                    let dst_off = (((by + ry) * w + bx) as usize) * 2;
                    let src_off = (ry * bw) as usize * 2;
                    recon[dst_off..dst_off + bw as usize * 2]
                        .copy_from_slice(&dec[src_off..src_off + bw as usize * 2]);
                }
            }
            bx += n;
        }
        by += n;
    }
    let (rms, maxr) = blockrt::quality(src, &recon);
    let bpp = total_bytes as f32 * 8.0 / (w as f32 * h as f32);
    (bpp, rms, maxr)
}

/// GO/NO-GO for the FROZEN EXV-RT v3 SHIPPING shape (before B-RT.2 locks): the earlier `shared_table`
/// number used ONE 128-lane rANS over the WHOLE PLANE. The shipping shape is different and must be
/// re-measured: **per-256px-tile pooled table + per-16×16-block single/K-lane HEADERLESS substreams +
/// fixed-i16 coeffs + a per-block offset index.** This exercises exactly the `rans::{build_table,
/// encode_block, decode_block}` primitives the GPU decoder will use, and reports true bpp (table +
/// index + substream overhead all counted) + rms. If bpp stays ~flat vs whole-plane q4, GO.
///
/// Run: `EXV_QUALITY_FILE=...robatframe.exv cargo test -p exv-codec --test blockrt_quality -- blockrt_shipping_shape --ignored --nocapture`
#[test]
#[ignore = "needs a .exv corpus file (EXV_QUALITY_FILE)"]
fn blockrt_shipping_shape() {
    use exv_codec::lossy::{forward_quant, inverse_dequant};
    let path = std::env::var("EXV_QUALITY_FILE")
        .unwrap_or_else(|_| "C:/temp/exv-rs/robat.0001.exv".to_string());
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skip: cannot read {path}");
        return;
    };
    let img = read_still(&bytes).expect("read_still");
    let (w, h) = (img.header.width, img.header.height);
    const TILE: u32 = 256;
    const BLK: u32 = 16;
    const TL: u8 = 12; // rANS table_log (M=4096)
    let q = 4u32;
    eprintln!("\n=== EXV-RT SHIPPING SHAPE (per-{TILE}-tile shared table + per-{BLK}-block substreams, q={q}) ===");
    eprintln!("file {w}x{h}; bpp counts table(512B/tile)+index(4B/blk)+substreams. cmp vs whole-plane q4.");

    for lanes in [1u8, 4] {
        eprintln!("  -- {lanes} lane(s)/block --");
        for (ci, ch) in img.header.channels.iter().enumerate().take(6) {
            let src = match ch.dtype {
                Dtype::F16 => img.planes[ci].clone(),
                Dtype::F32 => img.planes[ci]
                    .as_chunks::<4>().0.iter()
                    .flat_map(|c| f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
                    .collect(),
                _ => continue,
            };
            let mut recon = vec![0u8; src.len()];
            let (mut total, mut i16_overflow) = (0usize, 0u64);
            let mut ty = 0u32;
            while ty < h {
                let th = TILE.min(h - ty);
                let mut tx = 0u32;
                while tx < w {
                    let tw = TILE.min(w - tx);
                    // Gather this tile's blocks: forward_quant each 16x16 -> fixed-i16 bytes; pool them.
                    let mut blocks: Vec<(Vec<u8>, u32, u32, u32, u32)> = Vec::new(); // (i16bytes, bx, by, bw, bh)
                    let mut pool: Vec<u8> = Vec::new();
                    let mut by = 0u32;
                    while by < th {
                        let bh = BLK.min(th - by);
                        let mut bx = 0u32;
                        while bx < tw {
                            let bw = BLK.min(tw - bx);
                            let mut sub = vec![0u8; (bw * bh) as usize * 2];
                            for ry in 0..bh {
                                let so = (((ty + by + ry) * w + tx + bx) as usize) * 2;
                                let d = (ry * bw) as usize * 2;
                                sub[d..d + bw as usize * 2].copy_from_slice(&src[so..so + bw as usize * 2]);
                            }
                            let coeffs = forward_quant(&sub, Dtype::F16, bw, bh, q);
                            // i32 (4 bytes/coeff): the reversible 5/3 grows |coeff| up to ~4× the hdr
                            // code bound (≈65536) — exceeds i16 on pathological blocks, so a FROZEN format
                            // must not `as i16`. High bytes are ~always 0x00/0xFF → order-0 rANS eats them.
                            let mut i16b = Vec::with_capacity(coeffs.len() * 4);
                            for &c in &coeffs {
                                if c < i32::MIN as i64 || c > i32::MAX as i64 {
                                    i16_overflow += 1; // must be 0 (i32 covers the proven ±65536 bound)
                                }
                                i16b.extend_from_slice(&(c as i32).to_le_bytes());
                            }
                            pool.extend_from_slice(&i16b);
                            blocks.push((i16b, tx + bx, ty + by, bw, bh));
                            bx += BLK;
                        }
                        by += BLK;
                    }
                    // ONE shared table per tile; encode each block headerless; count table+index+subs.
                    let table = rans::build_table(&pool, TL);
                    let rebuilt = rans::Model::from_freqs(&table.freqs_le(), TL as u32).unwrap();
                    total += 512 + 4 * (blocks.len() + 1); // freq table + offset index (nblk+1)
                    for (i16b, ax, ay, bw, bh) in &blocks {
                        let sub_enc = rans::encode_block(i16b, &table, lanes).expect("encode");
                        total += sub_enc.len();
                        // Decode back -> i16 -> i64 -> inverse transform -> scatter (verify quality).
                        let n = (bw * bh) as usize;
                        let dec = rans::decode_block(&sub_enc, &rebuilt, n * 4, lanes).unwrap();
                        let coeffs: Vec<i64> = dec
                            .as_chunks::<4>().0.iter()
                            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
                            .collect();
                        let plane = inverse_dequant(&coeffs, Dtype::F16, *bw, *bh, q);
                        for ry in 0..*bh {
                            let d = (((*ay + ry) * w + *ax) as usize) * 2;
                            let s = (ry * bw) as usize * 2;
                            recon[d..d + *bw as usize * 2].copy_from_slice(&plane[s..s + *bw as usize * 2]);
                        }
                    }
                    tx += TILE;
                }
                ty += TILE;
            }
            let bpp = total as f32 * 8.0 / (w as f32 * h as f32);
            let (rms, maxr) = blockrt::quality(&src, &recon);
            let ov = if i16_overflow > 0 { format!("  !!i32-OVERFLOW={i16_overflow}") } else { String::new() };
            eprintln!("    [{}] {:.2} bpp  rms {:.3}/{:.2}{}", ch.name, bpp, rms, maxr, ov);
        }
    }
    eprintln!("  GO if bpp ~flat vs whole-plane q4 (A~1.4 B~3.4 G~3.3 GI.B~0.6 GI.G~1.1 GI.R~1.4) AND no i16-overflow.");
}

/// EXV-RT B-RT.3-prep LAYOUT MEASUREMENT: does a per-block **zigzag + adaptive fixed-width (1..4 bytes)**
/// coeff serialization cut the GPU rANS symbol count (and hold/improve ratio) vs the shipped fixed-i32?
///
/// WHY: the GPU decoder cost is dominated by SERIAL rANS symbols/block (perf review: v0 fixed-i32 = 1024
/// sym/block = 2 sym/output-byte → ~10 GB/s). Order-0 rANS is permutation-invariant, so plain byte-plane
/// SoA is inert; the real lever is FEWER BYTES/COEFF. Zigzag maps small signed coeffs to small unsigned
/// (so smooth blocks need 1 byte, not 4); a per-block width header keeps it GPU-uniform (256*w symbols,
/// w known from the first symbol). This measures avg bytes/coeff (= symbol-reduction factor vs 4) AND bpp
/// (table+index+width-headers+substreams all counted) on the real corpus, so the layout choice is DATA-led.
///
/// Run: `EXV_QUALITY_FILE=...robatframe.exv cargo test -p exv-codec --test blockrt_quality -- blockrt_adaptive_width --ignored --nocapture`
#[test]
#[ignore = "needs a .exv corpus file (EXV_QUALITY_FILE)"]
fn blockrt_adaptive_width() {
    use exv_codec::lossy::{forward_quant, inverse_dequant};
    let path = std::env::var("EXV_QUALITY_FILE")
        .unwrap_or_else(|_| "C:/temp/exv-rs/robat.0001.exv".to_string());
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skip: cannot read {path}");
        return;
    };
    let img = read_still(&bytes).expect("read_still");
    let (w, h) = (img.header.width, img.header.height);
    const TILE: u32 = 256;
    const BLK: u32 = 16;
    const TL: u8 = 12;
    let q = 4u32;
    // zigzag: signed i64 -> unsigned u64 (small |x| -> small code). Inverse: (z>>1) ^ -(z&1).
    let zz = |x: i64| -> u64 { ((x << 1) ^ (x >> 63)) as u64 };
    let unzz = |z: u64| -> i64 { ((z >> 1) as i64) ^ -((z & 1) as i64) };
    let bytes_for = |m: u64| -> u32 { (1..=8).find(|&b| m < (1u64 << (8 * b))).unwrap_or(8).max(1) };

    eprintln!("\n=== EXV-RT ADAPTIVE-WIDTH LAYOUT (zigzag + per-block 1..4B) vs fixed-i32 (q={q}) ===");
    eprintln!("file {w}x{h}; reports: bpp | avg B/coeff (i32=4.00) | sym-cut vs i32 | rms.");
    for (ci, ch) in img.header.channels.iter().enumerate().take(6) {
        let src = match ch.dtype {
            Dtype::F16 => img.planes[ci].clone(),
            Dtype::F32 => img.planes[ci]
                .as_chunks::<4>().0.iter()
                .flat_map(|c| f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
                .collect(),
            _ => continue,
        };
        let mut recon = vec![0u8; src.len()];
        let (mut total, mut payload_syms, mut ncoeff) = (0usize, 0u64, 0u64);
        let mut ty = 0u32;
        while ty < h {
            let th = TILE.min(h - ty);
            let mut tx = 0u32;
            while tx < w {
                let tw = TILE.min(w - tx);
                // Per block: forward_quant -> zigzag -> pick width w=max bytes over the block -> [w][coeffs*w].
                let mut blocks: Vec<(Vec<u8>, u32, u32, u32, u32)> = Vec::new();
                let mut pool: Vec<u8> = Vec::new();
                let mut by = 0u32;
                while by < th {
                    let bh = BLK.min(th - by);
                    let mut bx = 0u32;
                    while bx < tw {
                        let bw = BLK.min(tw - bx);
                        let mut sub = vec![0u8; (bw * bh) as usize * 2];
                        for ry in 0..bh {
                            let so = (((ty + by + ry) * w + tx + bx) as usize) * 2;
                            let d = (ry * bw) as usize * 2;
                            sub[d..d + bw as usize * 2].copy_from_slice(&src[so..so + bw as usize * 2]);
                        }
                        let coeffs = forward_quant(&sub, Dtype::F16, bw, bh, q);
                        let codes: Vec<u64> = coeffs.iter().map(|&c| zz(c)).collect();
                        let bw_bytes = bytes_for(codes.iter().copied().max().unwrap_or(0)); // 1..8 (real: 1..3)
                        let mut blk = Vec::with_capacity(1 + codes.len() * bw_bytes as usize);
                        blk.push(bw_bytes as u8); // per-block width header
                        for &z in &codes {
                            blk.extend_from_slice(&z.to_le_bytes()[..bw_bytes as usize]);
                        }
                        payload_syms += 1 + codes.len() as u64 * bw_bytes as u64;
                        ncoeff += codes.len() as u64;
                        pool.extend_from_slice(&blk);
                        blocks.push((blk, tx + bx, ty + by, bw, bh));
                        bx += BLK;
                    }
                    by += BLK;
                }
                let table = rans::build_table(&pool, TL);
                let rebuilt = rans::Model::from_freqs(&table.freqs_le(), TL as u32).unwrap();
                total += 512 + 4 * (blocks.len() + 1);
                for (blk, ax, ay, bw, bh) in &blocks {
                    let enc = rans::encode_block(blk, &table, 1).expect("encode");
                    total += enc.len();
                    let dec = rans::decode_block(&enc, &rebuilt, blk.len(), 1).unwrap();
                    let bwb = dec[0] as usize;
                    let n = (bw * bh) as usize;
                    let coeffs: Vec<i64> = (0..n)
                        .map(|k| {
                            let mut z = [0u8; 8];
                            z[..bwb].copy_from_slice(&dec[1 + k * bwb..1 + (k + 1) * bwb]);
                            unzz(u64::from_le_bytes(z))
                        })
                        .collect();
                    let plane = inverse_dequant(&coeffs, Dtype::F16, *bw, *bh, q);
                    for ry in 0..*bh {
                        let d = (((*ay + ry) * w + *ax) as usize) * 2;
                        let s = (ry * bw) as usize * 2;
                        recon[d..d + *bw as usize * 2].copy_from_slice(&plane[s..s + *bw as usize * 2]);
                    }
                }
                tx += TILE;
            }
            ty += TILE;
        }
        let bpp = total as f32 * 8.0 / (w as f32 * h as f32);
        let avg_bpc = payload_syms as f32 / ncoeff.max(1) as f32; // rANS symbols/coeff (incl. ~1/256 header)
        let sym_cut = 4.0 * ncoeff as f32 / payload_syms.max(1) as f32; // i32=4/coeff vs adaptive incl. headers
        let (rms, maxr) = blockrt::quality(&src, &recon);
        eprintln!("    [{}] {:.2} bpp  {:.2} B/coeff  {:.2}x fewer sym  rms {:.3}/{:.2}", ch.name, bpp, avg_bpc, sym_cut, rms, maxr);
    }
    eprintln!("  vs fixed-i32 (1-lane): A 1.40 B 4.76 G 4.71 GI.B 1.13 GI.G 1.82 GI.R 2.10 bpp @ 4.00 B/coeff.");
    eprintln!("  WIN if bpp <= i32 AND sym-cut >~2x (GPU serial-rANS speedup). Then adaptive-width replaces i32 in blockrt_dwt.");
}

