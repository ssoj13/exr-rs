//! EXV-RT **shipping** payload (frozen v0, `ver=1`): the realtime, GPU-parallel LOSSY block codec.
//!
//! # What and why
//! The quality gate (`tests/blockrt_quality.rs`) proved the endpoint-lerp model ([`crate::blockrt`])
//! fails colour on real HDR VFX (rms 0.07–0.18). The frozen replacement, measured GO on the robat
//! corpus, is: each channel plane is mirror-padded to a multiple of `block`, tiled into `block`×`block`
//! squares; each block goes through the reversible **signed 5/3 wavelet + quant** ([`crate::lossy`]),
//! its coefficients **zigzag-mapped and packed at a per-block adaptive width (1..4 bytes)** — a smooth
//! block needs only 1 byte/coeff; all blocks' code bytes are POOLED to build ONE shared rANS frequency
//! table, and each block is a HEADERLESS single-lane rANS substream against that table
//! ([`rans::build_table`]/[`rans::encode_block`]). A per-block offset index + width byte make every block
//! **independently addressable** → the GPU decodes thousands of blocks in parallel (B-RT.3, WGSL).
//!
//! # Why this shape
//! - **Shared table** (one per plane, stored once) keeps the whole-plane ratio while every block stays
//!   independently decodable — the fix for per-block table overhead.
//! - **zigzag + per-block adaptive width (1..4 B)**: the GPU decode cost is the SERIAL rANS symbol count;
//!   fixed-i32 wasted ~3 dead sign-extension bytes/coeff = 4 symbols/coeff. Zigzag makes small signed
//!   coeffs small unsigned, and a per-block width byte packs them in the fewest bytes — measured on robat:
//!   **~1.1–1.3 B/coeff (↑3× fewer rANS symbols) AND ~30% better ratio** than fixed-i32, with identical
//!   reconstruction. Still safe (width=4 covers the full i32 code range) and GPU-uniform (all coeffs in a
//!   block share one width, known from the side array before decoding → fixed `256·w` trip count).
//! - **mirror-pad to `block`**: every block is a uniform `block`×`block` → the GPU decode loop has a
//!   fixed trip count; the padded border is discarded on decode.
//!
//! # Blob layout (WITHOUT the leading `TAG_EXV_RT`, which `encode_channel` prepends), LE
//! ```text
//! block:u8 | q:u8 | table_log:u8 | lanes:u8 | nbx:u16 | nby:u16 | freq[256×u16]
//!   | off[(nbx·nby)+1 × u32]  (byte offsets into the substream region; last = total)
//!   | width[nbx·nby × u8]     (per-block code width, 1..4 bytes/coeff)
//!   | substreams…            (one headerless `lanes`-way interleaved rANS block each; block bi has
//!                             `block²·width[bi]` byte-symbols, split across `lanes` interleaved lanes)
//! ```
//! `lanes` (per-plane, shared by every block) is the rANS interleave width. `lanes=1` is a serial
//! substream (no lane header); `lanes=N` gives N independent dependent-chains per block so the GPU can
//! decode N states with N-way instruction-level parallelism (hides the per-symbol renorm latency). It is
//! an ENTROPY-layer knob only — the coefficients, widths, offset index and reconstruction are identical
//! for any `lanes` (rANS is exact regardless of lane count).
//! `block`/`q`/`table_log` are self-describing (decode reads them from the blob, not the profile);
//! `w`/`h` (the tile rect) come from the container. Bit-exact for `q==0` (lossless lifting); `q>0` is
//! the intended lossy reconstruction. `decode_plane` returns `None` (never panics) on any malformed
//! **blob content** for a given `w`/`h` — short header, bad table, non-monotonic/overrun offset index.
//!
//! NOTE: the working/output size is `O(w·h)` (the tile rect), like every codec's per-tile decode. The blob
//! cannot size it: `nbx`/`nby` must equal `ceil(w/block)`/`ceil(h/block)` (checked before any allocation),
//! so the padded plane is derived from the caller's validated `w`/`h` only; the container bounds the tile
//! (`MAX_TILE_AREA` in `Header::read`, `DecodeLimits` in `frame`).

use std::borrow::Cow;

use crate::lossy;
use exv_format::Dtype;

/// rANS table_log for the shared per-plane table (M = 4096). One value, stored in the blob.
const TABLE_LOG: u8 = 12;

/// Default rANS interleave width for the shipping EXV-RT encoder: **1** (single-lane). An `N`-lane
/// substream would let the GPU decoder run `N` states per thread for ILP, but the PATH-A finding
/// (2026-07-02) is that N-lane ILP does NOT break the ~7-8 GB/s warm wall (flat at q0, ~12% LOSS at q4
/// from register pressure): the bottleneck is the INVERSE pass (spilled `array<i32,256>` scratch), not
/// the rANS pass — proven by q4 being SLOWER than q0 despite FEWER symbols. The N-lane substrate is kept
/// (tested for N∈{1,2,4,8}; the GPU fast path is specialized for those) but stays OFF pending a use that
/// actually benefits. Entropy-only: any value reconstructs the SAME pixels.
pub const RT_LANES: u8 = 1;

/// Reflect an index into `[0, n)` with 101-style folding (no edge repeat), robust when the pad exceeds
/// `n` (tiny edge tiles). `i` may exceed `2n`; folding by the `2(n-1)` period keeps it in range.
fn mirror(i: i64, n: i64) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut r = i % period;
    if r < 0 {
        r += period;
    }
    if r >= n {
        r = period - r;
    }
    r as usize
}

/// Mirror-pad an f16 LE plane `w×h` up to `pw×ph` (pw≥w, ph≥h). The padded border is reflected so the
/// wavelet sees no artificial edge; it is discarded on decode. A plane that already is a whole number of
/// blocks is BORROWED (no whole-plane copy: that copy was most of EXV-RT encode's 2.1x-raw peak).
fn mirror_pad(src: &[u8], w: u32, h: u32, pw: u32, ph: u32) -> Cow<'_, [u8]> {
    if (w, h) == (pw, ph) {
        return Cow::Borrowed(src);
    }
    // usize products: pw*ph can reach 2^32 at the max tile (65536²), which overflows u32 (→ 0-len alloc → OOB).
    let (wu, pwu) = (w as usize, pw as usize);
    let mut out = vec![0u8; pwu * ph as usize * 2];
    // Source column per padded column, folded once instead of an i64 `%` per pixel.
    let xs: Vec<usize> = (0..pwu).map(|x| mirror(x as i64, w as i64)).collect();
    for (y, row) in out.chunks_exact_mut(pwu * 2).enumerate() {
        let sy = mirror(y as i64, h as i64);
        let srow = &src[sy * wu * 2..(sy + 1) * wu * 2];
        row[..wu * 2].copy_from_slice(srow); // columns < w map to themselves
        for (d, &sx) in row.as_chunks_mut::<2>().0.iter_mut().zip(&xs).skip(wu) {
            d.copy_from_slice(&srow[sx * 2..sx * 2 + 2]);
        }
    }
    Cow::Owned(out)
}

/// Zigzag map signed→unsigned so a small |x| becomes a small code (−1→1, 1→2, −2→3…), which the
/// per-block adaptive width then stores in the fewest bytes. Inverse: [`unzigzag`].
fn zigzag(x: i64) -> u64 {
    ((x << 1) ^ (x >> 63)) as u64
}
fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}
/// Minimal bytes (1..=8) to hold `z` little-endian. Coeffs are i32-bounded (|coeff| ≤ ~262144 → zigzag ≤
/// u32::MAX → ≤4), so a smooth block is 1 and a real block ≤3; returning the TRUE width (not a silent cap)
/// means a transform-invariant violation (width>4) is rejected LOUDLY by decode's `width∈1..=4` check
/// rather than silently truncating a coefficient — per the repo's no-silent-fallback rule.
fn code_width(z: u64) -> u32 {
    (64 - z.leading_zeros()).div_ceil(8).max(1)
}

/// Zigzag codes of `coeffs`, each stored as `wb` little-endian bytes. `1..=4` (every valid stream) are const-width
/// loops (a runtime-width copy was an out-of-line memcpy per coefficient); a wider code is an invariant
/// violation that [`code_width`] reports truthfully and decode rejects, so it only needs the plain loop.
fn pack_into(coeffs: &[i64], wb: usize, out: &mut Vec<u8>) {
    fn run<const W: usize>(coeffs: &[i64], out: &mut Vec<u8>) {
        let at = out.len();
        out.resize(at + coeffs.len() * W, 0);
        // `W <= 4`: the low `W` bytes of the code are those of its `u32` truncation.
        for (o, &c) in out.get_mut(at..).unwrap_or_default().as_chunks_mut::<W>().0.iter_mut().zip(coeffs) {
            *o = crate::temporal::put::<W>(zigzag(c) as u32);
        }
    }
    match wb {
        1 => run::<1>(coeffs, out),
        2 => run::<2>(coeffs, out),
        3 => run::<3>(coeffs, out),
        4 => run::<4>(coeffs, out),
        _ => out.extend(coeffs.iter().flat_map(|&c| zigzag(c).to_le_bytes().into_iter().take(wb))),
    }
}

/// Inverse of [`pack_into`] for the widths decode accepts (`wb ∈ 1..=4`, checked by the caller), into `g`
/// (`bytes.len() / wb` coefficients).
fn unpack_into(bytes: &[u8], wb: usize, g: &mut [i64]) {
    fn run<const W: usize>(bytes: &[u8], g: &mut [i64]) {
        for (o, ch) in g.iter_mut().zip(bytes.as_chunks::<W>().0) {
            *o = unzigzag(u64::from(crate::temporal::get::<W>(ch)));
        }
    }
    match wb {
        1 => run::<1>(bytes, g),
        2 => run::<2>(bytes, g),
        3 => run::<3>(bytes, g),
        _ => run::<4>(bytes, g),
    }
}

/// Encode an f16 LE channel plane (`w×h`) into the EXV-RT payload (WITHOUT the tag byte). `Err` only if the
/// entropy coder refuses a block (cannot happen for in-spec geometry: blocks are ≤ 64²·4 symbols).
/// `block` = block edge (16 for v0; ≤64 for the GPU dim cap), `q` = wavelet detail quant (0 = lossless),
/// `lanes` = rANS interleave width (≥1; see [`RT_LANES`]). All blocks in the plane share `lanes`; it is an
/// entropy-only knob (the reconstructed pixels are identical for any `lanes`).
pub fn encode_plane(src: &[u8], w: u32, h: u32, block: u32, q: u32, lanes: u8) -> Result<Vec<u8>, rans::EncodeError> {
    let mut out = Vec::new();
    encode_plane_into(src, w, h, block, q, lanes, &mut out)?;
    Ok(out)
}

/// [`encode_plane`] APPENDING the payload to `out`: the blob is assembled in place (header, an offset index patched
/// once the substreams are known, widths, then each block's substream straight from the rANS encoder), so there is
/// no per-block `Vec`, no `subs` buffer and no final assembly copy. On `Err` the caller truncates `out`.
pub fn encode_plane_into(src: &[u8], w: u32, h: u32, block: u32, q: u32, lanes: u8, out: &mut Vec<u8>) -> Result<(), rans::EncodeError> {
    let lanes = lanes.max(1);
    let nbx = w.div_ceil(block);
    let nby = h.div_ceil(block);
    let (pw, ph) = (nbx * block, nby * block);
    let padded = mirror_pad(src, w, h, pw, ph);
    let bpx = block as usize; // block edge in pixels
    let blk_bytes = bpx * bpx * 2; // f16 bytes per block
    let nblk = (nbx * nby) as usize;

    // Per block: forward transform+quant → zigzag codes → per-block adaptive fixed width (1..4 bytes =
    // max bytes any code in the block needs). Smooth blocks → tiny codes → 1 byte/coeff (~3× fewer rANS
    // symbols + ~30% better ratio than fixed-i32; measured on robat). All blocks' code bytes are POOLED (one
    // buffer, `ends[bi]` = where block `bi` stops) for the shared table.
    let mut sub = vec![0u8; blk_bytes];
    let mut coeffs = vec![0i64; bpx * bpx];
    let mut widths: Vec<u8> = Vec::with_capacity(nblk);
    let mut ends: Vec<usize> = Vec::with_capacity(nblk);
    // The pooled block codes live in the per-thread growable scratch (`with_vec`): a plane-sized `Vec` per channel
    // per tile was most of the encoder's allocation churn.
    crate::with_vec(|pool| -> Result<(), rans::EncodeError> {
        pool.reserve(nblk * bpx * bpx);
        for byi in 0..nby {
            for bxi in 0..nbx {
                for (ry, d) in sub.chunks_exact_mut(bpx * 2).enumerate() {
                    let sy = byi as usize * bpx + ry;
                    let so = (sy * pw as usize + bxi as usize * bpx) * 2;
                    if let Some(s) = padded.get(so..so + bpx * 2) {
                        d.copy_from_slice(s);
                    }
                }
                lossy::forward_quant_into(&sub, Dtype::F16, block, block, q, &mut coeffs);
                // |c| ≤ ~4·65535 (raw F16 bit patterns at q=0) or ≤ ~4·MAX_CODE (hdr codes, q>0) ≪ i32::MAX →
                // zigzag < 2^20 (≤3 bytes); the debug_assert catches a transform regression that would overflow
                // the code width. The width is the minimal one holding the LARGEST code (it is monotonic in the value).
                debug_assert!(coeffs.iter().all(|&c| c >= i32::MIN as i64 && c <= i32::MAX as i64), "coeff exceeds i32");
                let wbytes = code_width(coeffs.iter().fold(0, |m, &c| m.max(zigzag(c))));
                pack_into(&coeffs, wbytes as usize, pool);
                widths.push(wbytes as u8);
                ends.push(pool.len());
            }
        }

        let table = rans::build_table(pool, TABLE_LOG);

        // Blob header, then the offset index (zeros, patched below), then the widths; the substreams follow.
        out.reserve(8 + 512 + (nblk + 1) * 4 + nblk + pool.len());
        out.extend_from_slice(&[block as u8, q as u8, TABLE_LOG, lanes]); // `lanes`: rANS interleave width (u32-aligns the table)
        out.extend_from_slice(&(nbx as u16).to_le_bytes());
        out.extend_from_slice(&(nby as u16).to_le_bytes());
        out.extend_from_slice(&table.freqs_le());
        let idx_at = out.len();
        out.resize(idx_at + (nblk + 1) * 4, 0);
        out.extend_from_slice(&widths); // per-block code width (u8×nblk), read before each block's rANS

        // Encode each block headerless against the shared table as an `lanes`-way interleaved substream; build
        // the offset index. `lanes` is the ENTROPY interleave (independent of `width`, which is bytes/coeff);
        // the substream carries its own `lanes-1` lane-length header (last lane implied by the offset index).
        let subs_at = out.len();
        let mut offsets: Vec<u32> = Vec::with_capacity(nblk + 1);
        let mut rest = pool.as_slice();
        let mut prev = 0usize;
        for &end in &ends {
            offsets.push((out.len() - subs_at) as u32);
            let Some((block_syms, tail)) = rest.split_at_checked(end - prev) else { break };
            rans::encode_block_into(block_syms, &table, lanes, out)?;
            (rest, prev) = (tail, end);
        }
        offsets.push((out.len() - subs_at) as u32);
        if let Some(idx) = out.get_mut(idx_at..idx_at + offsets.len() * 4) {
            for (slot, o) in idx.as_chunks_mut::<4>().0.iter_mut().zip(&offsets) {
                slot.copy_from_slice(&o.to_le_bytes());
            }
        }
        Ok(())
    })
}

/// Decode a EXV-RT v0 payload (from [`encode_plane`], tag byte already stripped) back to f16 LE `w×h`.
/// `block`/`q` are read from the blob (self-describing) and cross-checked against `w`/`h`. `None` on any
/// corruption (short header, geometry mismatch, bad table/offset index, out-of-range rANS state). The output
/// is allocated only AFTER the layout is proven ([`Layout::parse`]: the offset-index floor ties the plane size
/// to the blob size), so a tiny forged blob cannot make this allocate a huge plane.
pub fn decode_plane(blob: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    rans::with_model(|model| {
        let layout = Layout::parse(blob, w, h, model)?;
        let mut out = vec![0u8; layout.out_len];
        layout.decode(&mut out)?;
        Some(out)
    })
}

/// [`decode_plane`] into `out` (exactly `w*h*2` bytes, every one overwritten). Blocks decode straight into `out`
/// when the plane is a whole number of blocks, else into a reused padded scratch that is cropped into `out`; the
/// per-block symbol / coefficient / pixel buffers are allocated once per plane, not once per block.
pub fn decode_plane_into(blob: &[u8], w: u32, h: u32, out: &mut [u8]) -> Option<()> {
    rans::with_model(|model| {
        let layout = Layout::parse(blob, w, h, model)?;
        if layout.out_len != out.len() {
            return None;
        }
        layout.decode(out)
    })
}

/// A validated EXV-RT payload for a `w×h` tile: the shared table, the block offset index, widths and substream
/// region, with every bound proven before any output buffer exists.
struct Layout<'a> {
    /// The plane's shared table, loaded into the per-thread decode model ([`rans::with_model`]).
    model: &'a rans::Model,
    /// The `(nbx·nby)+1` little-endian `u32` substream offsets, read in place ([`off`]): a decoded copy was one heap
    /// `Vec` per plane.
    idx: &'a [u8],
    widths: &'a [u8],
    region: &'a [u8],
    lanes: u8,
    geo: Geo,
    /// Tile size in pixels and the output length in bytes (`w*h*2`).
    w: usize,
    h: usize,
    out_len: usize,
}

impl<'a> Layout<'a> {
    /// Parse + validate `blob` against the tile rect `w×h`, loading its table into `model` (reused, no per-plane
    /// allocation); `None` on any malformed content.
    fn parse(blob: &'a [u8], w: u32, h: u32, model: &'a mut rans::Model) -> Option<Self> {
        let block = *blob.first()? as u32;
        let q = *blob.get(1)? as u32;
        let table_log = *blob.get(2)?;
        let lanes = *blob.get(3)?; // rANS interleave width (per-plane, shared by every block)
        // block must be ≥1 (div_ceil/geometry) and ≤64 (GPU dim cap); table_log validated by from_freqs.
        // lanes must be ≥1 (0 = malformed; the encoder always writes ≥1).
        if block == 0 || block > 64 || lanes == 0 {
            return None;
        }
        let nbx = u16::from_le_bytes([*blob.get(4)?, *blob.get(5)?]) as u32;
        let nby = u16::from_le_bytes([*blob.get(6)?, *blob.get(7)?]) as u32;
        // Geometry must match the container's tile rect exactly (else a forged header could mis-slice).
        if nbx != w.div_ceil(block) || nby != h.div_ceil(block) {
            return None;
        }
        let (wu, hu) = (w as usize, h as usize);
        let out_len = wu.checked_mul(hu)?.checked_mul(2)?;

        let mut p = 8usize;
        model.load_freqs(blob.get(p..p + 512)?, table_log as u32)?;
        let model: &'a rans::Model = model;
        p += 512;

        let nblk = (nbx as usize) * (nby as usize);
        let idx = blob.get(p..p + (nblk + 1) * 4)?;
        p += (nblk + 1) * 4;
        let widths = blob.get(p..p + nblk)?; // per-block code width (u8×nblk)
        p += nblk;
        let region = blob.get(p..)?;
        // Offset index must be monotonic with every substream at least `min_sub` long and end exactly at the
        // region length (no OOB slice, no trailing bytes). A block's substream holds `lanes` rANS states
        // (4 B each) plus `lanes-1` lane-length words, so anything shorter is corrupt — and this floor is what
        // stops a tiny blob from declaring a huge plane (decode work/allocation ≤ ~2048× the blob).
        let min_sub = 8 * lanes as u32 - 4;
        let mut offs = idx.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b));
        let mut last = offs.next()?;
        for o in offs {
            if o < last || o - last < min_sub {
                return None;
            }
            last = o;
        }
        if last as usize != region.len() {
            return None;
        }

        let bpx = block as usize;
        // usize products (u32 pw*ph overflows at the 65536² max tile → 0-len alloc → OOB panic).
        let pw = nbx as usize * bpx;
        let geo = Geo { nbx: nbx as usize, nby: nby as usize, bpx, pw, q };
        Some(Self { model, idx, widths, region, lanes, geo, w: wu, h: hu, out_len })
    }

    /// Decode into `out` (`out_len` bytes; every one overwritten).
    fn decode(&self, out: &mut [u8]) -> Option<()> {
        let g = &self.geo;
        let ph = g.nby * g.bpx;
        let dec = |dst: &mut [u8]| decode_blocks(self.region, self.idx, self.widths, self.model, self.lanes, g, dst);
        // Nothing to crop: decode straight into the output (no whole-plane copy).
        if (g.pw, ph) == (self.w, self.h) {
            return dec(out);
        }
        // Crop the mirror-padded plane back to w×h (all products in usize — u32 would overflow at max tile).
        let row_bytes = self.w * 2;
        crate::with_bytes(g.pw * ph * 2, |padded| {
            dec(padded)?;
            // `.max(1)`: a zero-width tile has an empty `out` and nothing to crop (chunk size 0 would panic).
            for (y, row) in out.chunks_exact_mut(row_bytes.max(1)).enumerate() {
                row.copy_from_slice(padded.get(y * g.pw * 2..y * g.pw * 2 + row_bytes)?);
            }
            Some(())
        })
        .ok()?
    }
}

/// Geometry of a plane's block grid, shared by the per-plane decode helper.
struct Geo {
    nbx: usize,
    nby: usize,
    /// Block edge in pixels.
    bpx: usize,
    /// Padded plane width in pixels (`nbx * bpx`).
    pw: usize,
    q: u32,
}

/// Entry `i` of the substream offset index `idx` (little-endian `u32`s); `None` past its end.
fn off(idx: &[u8], i: usize) -> Option<usize> {
    let b = idx.get(i.checked_mul(4)?..)?.first_chunk::<4>()?;
    Some(u32::from_le_bytes(*b) as usize)
}

/// Largest block edge (`Layout::parse` rejects more): sizes the stack scratch of [`decode_blocks`].
const MAX_BLOCK: usize = 64;

/// Decode every block of a plane into `dst` (the padded plane, `pw*ph*2` bytes). `None` on any malformed block.
fn decode_blocks(region: &[u8], idx: &[u8], widths: &[u8], model: &rans::Model, lanes: u8, g: &Geo, dst: &mut [u8]) -> Option<()> {
    let ncoeff = g.bpx * g.bpx;
    // Stack scratch, no heap: the largest block symbol run (width 4) and one decoded f16 block. `bpx <= MAX_BLOCK`.
    let mut scratch = [0u8; MAX_BLOCK * MAX_BLOCK * 6];
    let (syms, pix) = scratch.split_at_mut_checked(ncoeff * 4)?;
    let pix = pix.get_mut(..ncoeff * 2)?;
    lossy::with_coeffs(ncoeff, |coeffs| {
        for byi in 0..g.nby {
            for bxi in 0..g.nbx {
                let bi = byi * g.nbx + bxi;
                let wbytes = *widths.get(bi)? as usize; // 1..=4 (else corrupt → None)
                if !(1..=4).contains(&wbytes) {
                    return None;
                }
                let n_sym = ncoeff * wbytes; // block-uniform: every coeff is `wbytes` byte-symbols
                let sub = region.get(off(idx, bi)?..off(idx, bi + 1)?)?;
                let syms = syms.get_mut(..n_sym)?;
                rans::decode_block_into(sub, model, syms, lanes)?;
                unpack_into(syms, wbytes, coeffs);
                lossy::inverse_into(coeffs, Dtype::F16, g.bpx as u32, g.bpx as u32, g.q, pix);
                for (ry, src) in pix.chunks_exact(g.bpx * 2).enumerate() {
                    let d = ((byi * g.bpx + ry) * g.pw + bxi * g.bpx) * 2;
                    dst.get_mut(d..d + g.bpx * 2)?.copy_from_slice(src);
                }
            }
        }
        Some(())
    })
    .ok()?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32 → f16 LE bytes (the shared `hdr` conversion) for building test planes.
    fn f16(v: f32) -> [u8; 2] {
        crate::hdr::f32_to_half(v).to_le_bytes()
    }

    fn ramp_plane(w: u32, h: u32) -> Vec<u8> {
        (0..w * h).flat_map(|i| f16((i % 97) as f32 * 0.05 + 0.1)).collect()
    }

    /// q==0 is the LOSSLESS lifting path → decode must reproduce the source f16 bits EXACTLY, incl.
    /// mirror-padded edge blocks (size not a multiple of the block).
    #[test]
    fn dwt_lossless_q0_bit_exact() {
        for (w, h) in [(16u32, 16u32), (32, 32), (40, 24), (17, 31), (5, 5), (259, 130)] {
            let src = ramp_plane(w, h);
            let blob = encode_plane(&src, w, h, 16, 0, RT_LANES).expect("encode");
            let dec = decode_plane(&blob, w, h).expect("decode");
            assert_eq!(dec, src, "q=0 must be bit-exact ({w}x{h})");
        }
    }

    /// The N-lane rANS interleave is entropy-only: `encode_plane(N)`→`decode_plane` must reproduce the
    /// original for every supported lane count, at q=0 (bit-exact lossless) AND q=4 (lossy but
    /// deterministic — the same lane count decodes to the same bytes). This is the CPU ORACLE the GPU
    /// N-lane decoder is checked against. A size not a multiple of 16 exercises mirror-pad + crop.
    #[test]
    fn dwt_nlane_roundtrip() {
        for (w, h) in [(16u32, 16u32), (40, 24), (100, 70), (259, 130)] {
            let src = ramp_plane(w, h);
            for lanes in [1u8, 2, 4, 8] {
                // q=0 is bit-exact vs the source regardless of lane count.
                let blob0 = encode_plane(&src, w, h, 16, 0, lanes).expect("encode");
                let dec0 = decode_plane(&blob0, w, h).expect("decode q0");
                assert_eq!(dec0, src, "q=0 N={lanes} must be bit-exact ({w}x{h})");
                // q>0 is lossy, but the SAME lane count must round-trip to a stable result AND the pixels
                // must be independent of the lane count (entropy layer never touches the coefficients).
                let blob4 = encode_plane(&src, w, h, 16, 4, lanes).expect("encode");
                let dec4 = decode_plane(&blob4, w, h).expect("decode q4");
                assert_eq!(dec4.len(), src.len(), "q=4 N={lanes} length");
                let ref4 = decode_plane(&encode_plane(&src, w, h, 16, 4, 1).expect("encode"), w, h).expect("decode q4 ref");
                assert_eq!(dec4, ref4, "q=4 N={lanes} pixels must equal the 1-lane result");
            }
        }
    }

    /// q>0 (lossy) round-trips through the container to the SAME bytes as a second decode (determinism)
    /// and yields the right length; a size not a multiple of 16 exercises mirror-pad + crop.
    #[test]
    fn dwt_lossy_roundtrip_deterministic() {
        let (w, h) = (100u32, 70u32);
        let src = ramp_plane(w, h);
        let blob = encode_plane(&src, w, h, 16, 4, RT_LANES).expect("encode");
        assert_eq!(encode_plane(&src, w, h, 16, 4, RT_LANES).expect("encode"), blob, "encode deterministic");
        let a = decode_plane(&blob, w, h).expect("decode");
        assert_eq!(a.len(), src.len());
        assert_eq!(decode_plane(&blob, w, h).unwrap(), a, "decode deterministic");
    }

    /// A tiny blob must not be able to declare a huge plane: a forged 65536² tile geometry with a VALID
    /// table whose offset index gives every block an impossible (< one rANS state) substream is rejected
    /// before the multi-GB padded plane is allocated — `None`, not an abort. (If this regresses the test
    /// itself tries to allocate 16 GiB.)
    #[test]
    fn dwt_forged_huge_tile_is_none() {
        let (w, h, block) = (65536u32, 65536u32, 64u32);
        let (nbx, nby) = (w / block, h / block);
        let nblk = (nbx * nby) as usize;
        let mut head = vec![block as u8, 0, TABLE_LOG, 1];
        head.extend_from_slice(&(nbx as u16).to_le_bytes());
        head.extend_from_slice(&(nby as u16).to_le_bytes());
        head.extend_from_slice(&rans::build_table(&(0..=255u8).collect::<Vec<_>>(), TABLE_LOG).freqs_le());
        for sub_len in [0u32, 1, 3] {
            let mut blob = head.clone();
            (0..=nblk as u32).for_each(|i| blob.extend_from_slice(&(i * sub_len).to_le_bytes()));
            blob.extend_from_slice(&vec![1u8; nblk]); // widths
            blob.extend_from_slice(&vec![0u8; nblk * sub_len as usize]); // substream region
            assert!(decode_plane(&blob, w, h).is_none(), "sub_len {sub_len}");
        }
    }

    /// Corruption never panics and is rejected: truncated blob, wrong geometry, garbage offsets.
    #[test]
    fn dwt_decode_corruption_is_none() {
        let (w, h) = (40u32, 24u32);
        let blob = encode_plane(&ramp_plane(w, h), w, h, 16, 4, RT_LANES).expect("encode");
        assert!(decode_plane(&blob[..4], w, h).is_none()); // truncated header
        assert!(decode_plane(&blob, w + 16, h).is_none()); // geometry mismatch (nbx 3→4)
        assert!(decode_plane(&blob, w, h + 16).is_none()); // geometry mismatch (nby 2→3)
        assert!(decode_plane(&[], w, h).is_none());
        // Truncated substream region (drop the last byte) → offset-index end ≠ region len → None.
        assert!(decode_plane(&blob[..blob.len() - 1], w, h).is_none());
        // Per-block width side-array corruption: forged width outside 1..=4, and truncation inside the
        // width slice, must be rejected (not panic). Layout: [8 hdr][512 freq][(nblk+1)*4 off][nblk width].
        let (nbx, nby) = ((w as usize).div_ceil(16), (h as usize).div_ceil(16));
        let woff = 8 + 512 + (nbx * nby + 1) * 4; // first width byte
        for bad in [0u8, 5, 255] {
            let mut b = blob.clone();
            b[woff] = bad;
            assert!(decode_plane(&b, w, h).is_none(), "width={bad} must be rejected");
        }
        assert!(decode_plane(&blob[..woff + 1], w, h).is_none()); // truncated inside the width slice
    }

    /// A zero-width tile whose height is not a whole number of blocks takes the crop path with an EMPTY output: it
    /// decodes to an empty plane instead of panicking on a zero chunk size.
    #[test]
    fn dwt_zero_width_tile_decodes_empty() {
        let mut blob = vec![16u8, 0, TABLE_LOG, 1, 0, 0, 1, 0]; // block 16, q 0, nbx 0, nby 1 (h = 5)
        blob.extend_from_slice(&rans::build_table(&[1, 2, 3], TABLE_LOG).freqs_le());
        blob.extend_from_slice(&0u32.to_le_bytes()); // offset index: just the final offset
        assert_eq!(decode_plane(&blob, 0, 5), Some(Vec::new()));
    }
}
