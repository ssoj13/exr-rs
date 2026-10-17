//! HTJ2K (High-Throughput JPEG 2000) interop backend via `jph-rs` (an OpenJPH port).
//!
//! **CPU-only by design** (see `docs/archive/PLAN.md` §6, `docs/DESIGN_LOG.md` D-12: tier-2 JPEG 2000 parsing is serial; we do not
//! do a GPU port). One **multi-component** reversible (5/3) codestream per tile carries every channel
//! ([`encode_into`], the main path, mirroring `exr-core`'s HT encoder; [`decode_into`] is its decoder): a
//! single-channel blob is the same engine over a one-element slice, so there is exactly one encoder and one
//! decoder configuration to keep in sync. [`decode_htj2k_multi_level`] is the reduced-resolution decoder.
//!
//! Every [`Dtype`] is supported: F16/BF16 (16-bit signed + NLT), F32 (32-bit signed + NLT), U32 (32-bit
//! unsigned) — exactly the set verified by `exr-core` (Half/Float/Uint) — plus U8/U16 as plain UNSIGNED 8/16-bit
//! components (no NLT, zero-extended — never sign-extended like the F16 bit pattern). A tile may mix them:
//! J2K SIZ carries a bit depth + signedness per component, and decode verifies the stream's SIZ against the
//! dtype it is asked to produce (a mismatching stream is corrupt, never silently reinterpreted).
//!
//! # Hostile codestreams
//! A tile's codestream is attacker-controlled bytes, and `jph-rs` sizes its working storage and loops from the
//! stream's own markers (a sub-sampled component spins its line puller forever; a 4x4 code-block exponent
//! makes it allocate hundreds of MiB). So decode NEVER hands bytes to jph before [`check_header`] has parsed
//! the main + tile-part headers on the host side and proven they say exactly what [`encode_mc_core`] writes
//! for the tile (extent, 1x1 components, depth/sign/NLT per component, 64x64 HT code-blocks, 5 reversible
//! decompositions, one layer, no precincts, plain QCD/QCC); any other marker is corruption. See FORMAT.md 4.6.
//!
//! # Error boundary
//! `jph-rs` reports codec errors as `OjphResult` (its port of C++ `throw`). Encoding propagates them
//! as [`CodecError::Backend`] (an encoder failure must be loud, never a silent RAW/empty tile);
//! decoding maps them to a typed [`DecodeError`]. Every jph call runs inside [`with_cs`], whose
//! [`std::panic::catch_unwind`] turns a panic (a bug, not a codec error) into an error as well — an
//! error boundary, NOT an `unwrap`, mirroring C++ `try/catch(...)`; without it a panic across a rayon
//! worker boundary aborts. A codec that panicked is discarded, never returned to the pool.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use exv_format::{DecodeError, Dtype};
use jph_rs::{Codestream, MemInfile, Point, Ui32, VecOutfile};

use crate::{CodecError, Dst, Htj2kMode, hdr, htj2k_scale, ldr_step};

thread_local! {
    /// Per-worker reused codec pool (B0 — cuts allocation churn). [`Codestream::restart`] resets the
    /// codec state but KEEPS jph-rs's fixed/elastic allocator storage (`mem_fixed_allocator::alloc`
    /// only re-allocs the coefficient buffer when it must *grow*), so the uniform tiles of one image
    /// reuse the buffer instead of re-malloc'ing it per tile. `None` until first use (lazy).
    static CODESTREAM: RefCell<Option<Codestream>> = const { RefCell::new(None) };
}

/// Largest tile (samples = `w*h*components`) whose codec stays in the per-worker pool. jph-rs keeps the
/// allocator storage of the biggest tile it ever served, and that storage lives as long as the thread: one
/// huge tile would pin hundreds of MiB on every rayon worker for the process life. Tiles above this
/// (~134 MB of i32 samples) get a throw-away codec instead; typical 256..2048 px tiles stay pooled.
const POOL_MAX_SAMPLES: usize = 1 << 25;

/// Run `f` with this worker's pooled [`Codestream`] (`restart()`ed first: resets state, keeps the
/// allocator storage) and catch a panic. The codec returns to the pool after a normal return — even an
/// `Err` one, because the next call `restart()`s it back to a clean state — when `samples` is within
/// [`POOL_MAX_SAMPLES`], but is DISCARDED after a panic: a codec abandoned mid-codestream is never observed
/// again (a fresh one costs one allocation after a rare fault). `None` = jph-rs panicked. Soundness of the
/// kept codec between calls: its `infile`/`outfile` are raw pointers that `restart()` nulls before any use
/// and that `LocalCodestream` never dereferences at drop (it has no `Drop`), so a stale pointer is inert.
fn with_cs<T>(samples: usize, f: impl FnOnce(&mut Codestream) -> T) -> Option<T> {
    let mut cs = CODESTREAM.with(|cell| cell.borrow_mut().take()).unwrap_or_default();
    cs.restart();
    match catch_unwind(AssertUnwindSafe(|| f(&mut cs))) {
        Ok(v) => {
            if samples <= POOL_MAX_SAMPLES {
                CODESTREAM.with(|cell| *cell.borrow_mut() = Some(cs));
            }
            Some(v)
        }
        Err(_) => {
            tracing::error!("jph-rs panicked; codestream discarded");
            None
        }
    }
}

/// Map a jph-rs error (or any `Display`) to [`CodecError::Backend`].
fn be(e: impl std::fmt::Display) -> CodecError {
    CodecError::Backend(e.to_string())
}

/// The error for a caught jph panic (see [`with_cs`]).
fn panicked() -> CodecError {
    CodecError::Backend("jph-rs panicked".into())
}

/// A jph decode failure as a typed error; the jph text goes to the log (a `DecodeError` carries static text).
fn jph_err(e: impl std::fmt::Display) -> DecodeError {
    tracing::debug!("jph-rs decode error: {e}");
    DecodeError::Corrupt("HTJ2K codestream rejected by the decoder")
}

/// The decode-side error for a caught jph panic (see [`with_cs`]).
const PANICKED: DecodeError = DecodeError::Corrupt("HTJ2K decoder panicked");

/// Force jph-rs's SIMD dispatch initialisation (colour + wavelet transforms) and return the active
/// x86 instruction tier name (e.g. `"avx2"`). Codestream construction already runs these inits
/// lazily; calling this once at startup locks the `Once`-guarded dispatch deterministically (off the
/// first-frame critical path) and lets a binary verify the tier — a silent fall to scalar/SSE would
/// otherwise surface only as slow decode. (B0 pre-warm + tier check.)
pub fn warmup_simd() -> &'static str {
    use jph_rs::openjph::ojph_arch::{cpu_ext_level_name, get_cpu_ext_level};
    jph_rs::transform::ojph_colour::init_colour_transform_functions();
    jph_rs::transform::ojph_transform::init_wavelet_transform_functions();
    cpu_ext_level_name(get_cpu_ext_level())
}

/// `OJPH_NLT_BINARY_COMPLEMENT_NLT` — NLT for all non-UINT components (as in OpenEXR).
const NLT_BINARY_COMPLEMENT: u8 = 3;

/// `(bit_depth, signed, esz)` of a dtype's J2K component. `signed` doubles as "rides the binary-complement
/// NLT": float bit patterns are signed-magnitude and need it; unsigned integers (U8/U16/U32) do not.
fn jph_params(dtype: Dtype) -> (Ui32, bool, usize) {
    match dtype {
        Dtype::F16 | Dtype::Bf16 => (16, true, 2),
        Dtype::F32 => (32, true, 4),
        Dtype::U32 => (32, false, 4),
        Dtype::U16 => (16, false, 2),
        Dtype::U8 => (8, false, 1),
    }
}

/// What a decoder requires of one component of the tile's codestream: SIZ bit depth + signedness and whether
/// it carries the binary-complement NLT. Built from the dtype by [`comp_fmt`], compared by [`check_header`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Comp {
    bits: Ui32,
    signed: bool,
    nlt: bool,
}

/// The [`Comp`] a decoder requires for a `dtype` channel: the hdr log-code component (`codes`) is 16-bit
/// signed with NO NLT, everything else follows [`jph_params`] (signed = floats = binary-complement NLT).
fn comp_fmt(dtype: Dtype, codes: bool) -> Comp {
    if codes {
        Comp { bits: CODE_BITS, signed: true, nlt: false }
    } else {
        let (bits, signed, _) = jph_params(dtype);
        Comp { bits, signed, nlt: signed }
    }
}

/// Bytes → i32 samples (like `bytes_to_linebuf` in exr-core: HALF=i16→i32, 32-bit=i32 LE). `signed` picks
/// sign- vs zero-extension for 16-bit data (F16/BF16 bit patterns vs U16 values); U8 is always zero-extended.
fn to_i32(dst: &mut [i32], src: &[u8], esz: usize, signed: bool) {
    match esz {
        1 => dst.iter_mut().zip(src).for_each(|(d, &b)| *d = b as i32),
        2 if signed => {
            for (d, c) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = i16::from_le_bytes(*c) as i32;
            }
        }
        2 => {
            for (d, c) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = u16::from_le_bytes(*c) as i32;
            }
        }
        _ => {
            for (d, c) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = i32::from_le_bytes(*c);
            }
        }
    }
}

/// Store the low `N` bytes of each mapped sample LE (`N` a const: the copy is a plain load/store pair that
/// auto-vectorises — a runtime `esz` slice copy was an out-of-line `memcpy` call PER SAMPLE, ~23% of HTJ2K
/// decode CPU).
fn put<const N: usize>(dst: &mut [u8], src: &[i32], f: impl Fn(i32) -> i32) {
    for (c, &s) in dst.as_chunks_mut::<N>().0.iter_mut().zip(src) {
        c.copy_from_slice(&f(s).to_le_bytes()[..N]);
    }
}

/// i32 samples → bytes (the inverse of [`to_i32`]: it keeps the low `esz` bytes, which is exact for the
/// sign- and zero-extended forms alike), written straight into a row of the final plane. `max` clamps to
/// `[0, max]` first ([`Wavelet::Irrev97`] overshoot: the low-byte truncation would wrap 256 to 0).
fn from_i32(dst: &mut [u8], src: &[i32], esz: usize, max: Option<i32>) {
    match (esz, max) {
        (1, None) => put::<1>(dst, src, |s| s),
        (2, None) => put::<2>(dst, src, |s| s),
        (_, None) => put::<4>(dst, src, |s| s),
        (1, Some(m)) => put::<1>(dst, src, |s| s.clamp(0, m)),
        (2, Some(m)) => put::<2>(dst, src, |s| s.clamp(0, m)),
        (_, Some(m)) => put::<4>(dst, src, |s| s.clamp(0, m)),
    }
}

/// Bit-depth for HDR log codes ([`crate::hdr`]): codes are signed, |code| <= 16384 (15-bit
/// magnitude) -> 16-bit signed components, no NLT (they are already plain signed values, not the
/// binary-complement-ordered float bits the lossless path uses).
const CODE_BITS: Ui32 = 16;

/// The wavelet of a tile codestream: reversible 5/3 (lossless) or irreversible 9/7 (scalar-quantised). One per
/// codestream (the main COD), so a tile cannot mix them; [`check_header`] pins which one a decoder accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wavelet {
    Rev53,
    Irrev97,
}

/// Encode-side wavelet choice: [`Wavelet`] plus the 9/7 quantizer step (jph `delta`, normalised to the sample
/// range, see `crate::ldr_step`). The decoder never needs the step: it is in the codestream's QCD.
#[derive(Clone, Copy, Debug)]
enum Xform {
    Rev53,
    Irrev97 { step: f32 },
}

impl Xform {
    fn wavelet(self) -> Wavelet {
        match self {
            Xform::Rev53 => Wavelet::Rev53,
            Xform::Irrev97 { .. } => Wavelet::Irrev97,
        }
    }
}

/// Where one component's samples come from. Both forms are converted a ROW at a time straight into jph's line
/// buffer: no `n`-sample i32 intermediate, no zero-fill, no second copy.
enum Samples<'a> {
    /// The channel's raw bytes widened to i32.
    Bytes { bytes: &'a [u8], esz: usize, signed: bool },
    /// A float channel mapped to hdr log codes at `scale` ([`crate::hdr`]; the 16-bit signed code component).
    Hdr { bytes: &'a [u8], dtype: Dtype, scale: u32 },
}

impl Samples<'_> {
    /// Row `y` (`w = dst.len()` samples) into `dst` (jph's line buffer). [`specs`] has checked that every source
    /// holds exactly `w*h` samples, so the row slice exists (a short one would leave `dst` stale, never panic).
    fn row(&self, dst: &mut [i32], y: usize) {
        let w = dst.len();
        match *self {
            Samples::Bytes { bytes, esz, signed } => {
                if let Some(src) = bytes.get(y * w * esz..(y + 1) * w * esz) {
                    to_i32(dst, src, esz, signed);
                }
            }
            Samples::Hdr { bytes, dtype, scale } => {
                let esz = dtype.size();
                if let Some(src) = bytes.get(y * w * esz..(y + 1) * w * esz) {
                    hdr::to_codes_into(src, dtype, scale, dst);
                }
            }
        }
    }
}

/// One component's encode spec for [`encode_mc_core`]: its samples (`w*h`, row-major; every component is
/// 1×1-subsampled so all share `w`/`h`), its JPEG-2000 bit depth, signedness, and whether the
/// binary-complement NLT applies (signed floats use it; integers and the lossy 16-bit codes do not).
struct CompSpec<'a> {
    samples: Samples<'a>,
    bit_depth: Ui32,
    signed: bool,
    nlt: bool,
}

/// The [`CompSpec`] of one channel under `mode`: HdrCodes FLOAT channels become the 16-bit signed log-code component
/// (no NLT; the loss lives only in the hdr step); every other channel is coded DIRECTLY (signed floats ride the
/// binary-complement NLT; integers - ids must never be quantized - and the 9/7 U8/U16 channels are plain unsigned).
fn spec<'a>(mode: Htj2kMode, bytes: &'a [u8], dtype: Dtype, n: usize) -> Result<CompSpec<'a>, CodecError> {
    let (bit_depth, signed, esz) = jph_params(dtype);
    if n.checked_mul(esz) != Some(bytes.len()) {
        return Err(CodecError::Dims("channel length != w*h*esz"));
    }
    Ok(match mode {
        Htj2kMode::HdrCodes { q } if dtype.is_float() => CompSpec {
            samples: Samples::Hdr { bytes, dtype, scale: htj2k_scale(q) },
            bit_depth: CODE_BITS,
            signed: true,
            nlt: false,
        },
        _ => CompSpec { samples: Samples::Bytes { bytes, esz, signed }, bit_depth, signed, nlt: signed },
    })
}

/// Encode prepared components into ONE multi-component reversible (5/3) or irreversible (9/7)
/// J2K codestream APPENDED to `out`. Shared engine for EVERY encode in this module: the modes differ ONLY in
/// how each [`CompSpec`] is built ([`spec`]).
///
/// # Errors
/// [`CodecError::Dims`] on an empty/zero-dim input,
/// [`CodecError::Backend`] on any jph error or a jph panic.
fn encode_mc_core(specs: &[CompSpec<'_>], w: u32, h: u32, x: Xform, out: &mut Vec<u8>) -> Result<(), CodecError> {
    if specs.is_empty() || w == 0 || h == 0 {
        return Err(CodecError::Dims("empty tile or channel list"));
    }
    let (wu, hu) = (w as usize, h as usize);
    let n = wu.checked_mul(hu).ok_or(CodecError::Dims("w*h overflows"))?;
    let nch = specs.len();
    with_cs(nch.saturating_mul(n), |cs| -> Result<(), CodecError> {
        cs.access_siz().set_num_components(nch as Ui32);
        for (ci, s) in specs.iter().enumerate() {
            cs.access_siz().set_comp_info(ci as Ui32, Point::new(1, 1), s.bit_depth, s.signed);
            if s.nlt {
                cs.access_nlt_mut().set_nonlinear_transform(ci as Ui32, NLT_BINARY_COMPLEMENT).map_err(be)?;
            }
        }
        cs.set_planar(false);
        cs.access_siz().set_image_offset(Point::new(0, 0));
        cs.access_siz().set_image_extent(Point::new(w as Ui32, h as Ui32));
        {
            let cod = cs.access_cod();
            cod.employ_color_transform(0); // independent AOVs -> NO RCT
            cod.set_reversible(x.wavelet() == Wavelet::Rev53);
            // 64x64 codeblock: same max area (4096) as 128x32, but the 128-wide block breaks the GPU
            // decode kernel's <=64 per-dim cap. 64x64 unifies the CPU/GPU codeblock with no ratio or
            // throughput loss (`docs/FORMAT.md` §4.6), readying the format for the jph-wgpu backend.
            cod.set_block_dims(64, 64).map_err(be)?;
            cod.set_num_decompositions(5).map_err(be)?;
        }
        // Irreversible 9/7: install the scalar quantizer step (jph normalises it to the sample range).
        if let Xform::Irrev97 { step } = x {
            cs.access_qcd_mut().set_delta(step);
        }
        // The codestream is written STRAIGHT into the caller's blob (`VecOutfile` appends to `out`, seeks back
        // inside it to patch markers): no raw-tile-sized `MemOutfile` per tile and no copy out of it.
        let mut outfile = VecOutfile::new(out);
        cs.write_headers_slice(&mut outfile, &[]).map_err(be)?;

        // Non-planar interleaved feed: the codec consumes line-by-line across all
        // components (line y of comp 0, comp 1, …, comp nch-1, then line y+1). We
        // drive that order explicitly; the `next_comp` returned by
        // `fill_and_exchange` confirms it (mirrors exr-core's interleaved encode).
        let mut next_comp: Ui32 = 0;
        let mut cur = cs.exchange(std::ptr::null_mut(), &mut next_comp).map_err(be)?;
        for y in 0..hu {
            for s in specs {
                cs.fill_and_exchange(&mut cur, &mut next_comp, |lb, _| s.samples.row(lb.get_i32_mut(wu), y))
                .map_err(be)?;
            }
        }
        cs.flush().map_err(be)?;
        Ok(())
    })
    .ok_or_else(panicked)?
}

/// Encode a tile's channels under `mode` into ONE multi-component codestream APPENDED to `out` (the whole tile
/// blob payload; the caller writes the tag). `channels[ci]` is `w*h*esz` bytes of `dtypes[ci]`. Lossless: signed
/// floats ride the binary-complement NLT. HdrCodes: float channels become log codes ([`spec`]), integer channels
/// stay exact. Irrev97: U8/U16 only, quantized by `ldr_step(q)`. This ONE function backs the single-channel
/// blob (a 1-element slice), the tile blob and every `encode_htj2k_*` wrapper, so there is exactly one encoder
/// configuration to keep in sync with [`check_header`].
///
/// # Errors
/// [`CodecError::Dims`] (zero dims / a length mismatch), [`CodecError::Dtype`] (9/7 on a non-U8/U16 channel),
/// [`CodecError::Backend`] (jph error or panic). On `Err` the contents of `out` past its entry length are unspecified.
pub fn encode_into(mode: Htj2kMode, channels: &[&[u8]], dtypes: &[Dtype], w: u32, h: u32, out: &mut Vec<u8>) -> Result<(), CodecError> {
    if channels.len() != dtypes.len() {
        return Err(CodecError::Dims("channel count != dtype count"));
    }
    if let Htj2kMode::Irrev97 { .. } = mode
        && let Some(&dtype) = dtypes.iter().find(|d| !is_ldr(**d))
    {
        return Err(CodecError::Dtype { codec: "HTJ2K 9/7", dtype });
    }
    let n = (w as usize).checked_mul(h as usize).ok_or(CodecError::Dims("w*h overflows"))?;
    let specs = channels.iter().zip(dtypes).map(|(bytes, &dt)| spec(mode, bytes, dt, n)).collect::<Result<Vec<_>, _>>()?;
    let x = match mode {
        Htj2kMode::Irrev97 { q } => Xform::Irrev97 { step: ldr_step(q) },
        Htj2kMode::Lossless | Htj2kMode::HdrCodes { .. } => Xform::Rev53,
    };
    encode_mc_core(&specs, w, h, x, out)
}

// ---------------------------------------------------------------------------
// Host-side codestream header validation (runs BEFORE jph-rs sees the bytes; see the module docs).
// ---------------------------------------------------------------------------

/// J2K marker codes (big-endian u16) that a tile codestream may contain.
mod mk {
    pub const SOC: u16 = 0xFF4F;
    pub const SIZ: u16 = 0xFF51;
    pub const CAP: u16 = 0xFF50;
    pub const COD: u16 = 0xFF52;
    pub const QCD: u16 = 0xFF5C;
    pub const QCC: u16 = 0xFF5D;
    pub const NLT: u16 = 0xFF76;
    pub const COM: u16 = 0xFF64;
    pub const SOT: u16 = 0xFF90;
    pub const SOD: u16 = 0xFF93;
    pub const EOC: u16 = 0xFFD9;
}

/// Decomposition levels of every tile (`set_num_decompositions(5)` in [`encode_mc_core`]).
const DECOMPOSITIONS: u8 = 5;
/// Code-block exponent minus 2 (`set_block_dims(64, 64)`: 2^(4+2)).
const CB_EXP: u8 = 4;
/// Code-block style byte: HT only (bit 6), no other style flag.
const CB_STYLE_HT: u8 = 0x40;
/// Subbands of a 5-level 5/3 decomposition (`3 * levels + 1`): the length of a QCD/QCC exponent list.
const SUBBANDS: usize = 3 * DECOMPOSITIONS as usize + 1;

/// Big-endian u16 at `p` (`None` past the end).
fn be16(d: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_be_bytes(d.get(p..p.checked_add(2)?)?.try_into().ok()?))
}

/// Big-endian u32 at `p` (`None` past the end).
fn be32(d: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_be_bytes(d.get(p..p.checked_add(4)?)?.try_into().ok()?))
}

/// Validate a quantization segment body for the codestream's wavelet: reversible = style 0,
/// `[guard<<5|0][exponent<<3 ; SUBBANDS]`; irreversible = scalar expounded style 2,
/// `[guard<<5|2][exponent<<11|mantissa : u16 ; SUBBANDS]` (any step: it sizes nothing).
fn check_qnt(body: &[u8], wav: Wavelet) -> Result<(), &'static str> {
    let (&s, exps) = body.split_first().ok_or("HTJ2K QCD: empty")?;
    let (style, entry) = match wav {
        Wavelet::Rev53 => (0, 1),
        Wavelet::Irrev97 => (2, 2),
    };
    if s & 0x1F != style || s >> 5 == 0 {
        return Err("HTJ2K QCD: quantization style differs from the wavelet");
    }
    if exps.len() != SUBBANDS * entry || (wav == Wavelet::Rev53 && exps.iter().any(|e| e & 7 != 0)) {
        return Err("HTJ2K QCD: wrong subband count");
    }
    Ok(())
}

/// Prove that `data`'s main header and tile-part headers describe EXACTLY the single-tile codestream
/// [`encode_mc_core`] writes for a `w x h` tile with components `comps` - and nothing else, so jph-rs is never
/// asked to size storage or loop from a hostile marker (N1/N2: a YRsiz flip made `pull` spin forever, a
/// code-block exponent flip made it allocate 358 MiB). Whitelist, not blacklist: every marker outside
/// SIZ/CAP/COD/QCD/QCC/NLT/COM before SOT, and anything but SOD inside a tile-part header, is corruption.
fn check_header(data: &[u8], comps: &[Comp], w: u32, h: u32, wav: Wavelet) -> Result<(), &'static str> {
    const BAD: &str = "HTJ2K codestream header truncated";
    let nch = comps.len();
    if be16(data, 0) != Some(mk::SOC) {
        return Err("HTJ2K: missing SOC");
    }
    let (mut siz, mut cap, mut cod, mut qcd) = (false, false, false, false);
    // Per component: the NLT `Snlt` byte, and whether a QCC was seen (a repeat is corrupt).
    let mut nlt: Vec<Option<u8>> = vec![None; nch];
    let mut qcc = vec![false; nch];
    let mut p = 2usize;
    // Main header: segments until the first SOT. Every iteration advances >= 4 bytes, so it terminates.
    let sot = loop {
        let m = be16(data, p).ok_or(BAD)?;
        if m == mk::SOT {
            break p;
        }
        let len = be16(data, p + 2).ok_or(BAD)? as usize;
        if len < 2 {
            return Err(BAD);
        }
        let body = data.get(p + 4..p + 2 + len).ok_or(BAD)?; // `len` counts its own 2 bytes
        match m {
            mk::SIZ if !siz && p == 2 => {
                siz = true;
                // Rsiz:u16 | X Y XO YO XT YT XTO YTO:u32 | Csiz:u16 | (SSiz XRsiz YRsiz) per component
                if body.len() != 36 + 3 * nch
                    || be16(body, 34) != u16::try_from(nch).ok()
                    || be32(body, 2) != Some(w)
                    || be32(body, 6) != Some(h)
                    || (be32(body, 10), be32(body, 14)) != (Some(0), Some(0)) // image offset
                    || (be32(body, 18), be32(body, 22)) != (Some(w), Some(h)) // ONE tile = the whole image
                    || (be32(body, 26), be32(body, 30)) != (Some(0), Some(0)) // tile offset
                {
                    return Err("HTJ2K SIZ: extent / tiling / component count differs from the tile");
                }
                for (c, want) in body[36..].as_chunks::<3>().0.iter().zip(comps) {
                    if u32::from(c[0] & 0x7F) + 1 != want.bits || (c[0] & 0x80 != 0) != want.signed {
                        return Err("HTJ2K SIZ: component bit depth / signedness differs from the dtype");
                    }
                    if (c[1], c[2]) != (1, 1) {
                        return Err("HTJ2K SIZ: sub-sampled component (must be 1x1)");
                    }
                }
            }
            mk::CAP if !cap && siz => {
                cap = true;
                // Part-15 only (Pcap bit 17), exactly one Ccap word.
                if body.len() != 6 || be32(body, 0) != Some(0x0002_0000) {
                    return Err("HTJ2K CAP: not a plain Part-15 capability");
                }
            }
            mk::COD if !cod && siz => {
                cod = true;
                // Scod | prog | layers:u16 | mct | levels | xcb | ycb | style | transform  (no precinct list)
                let b = body;
                if b.len() != 10
                    || b[0] != 0 // no precincts, no SOP/EPH
                    || b[1] > 4 // progression order (any valid one decodes; memory-neutral)
                    || be16(b, 2) != Some(1) // one quality layer
                    || b[4] != 0 // no multi-component transform (independent AOVs)
                    || b[5] != DECOMPOSITIONS
                    || (b[6], b[7]) != (CB_EXP, CB_EXP) // 64x64 code-blocks
                    || b[8] != CB_STYLE_HT
                    || b[9] != u8::from(wav == Wavelet::Rev53) // transform: 1 = reversible 5/3, 0 = irreversible 9/7
                {
                    return Err("HTJ2K COD: not the 64x64 5-level single-layer HT layout of this wavelet");
                }
            }
            mk::QCD if !qcd && siz => {
                qcd = true;
                check_qnt(body, wav)?;
            }
            mk::QCC if siz => {
                let cb = if nch < 257 { 1 } else { 2 };
                let c = if cb == 1 { body.first().map(|&v| usize::from(v)) } else { be16(body, 0).map(usize::from) }.ok_or(BAD)?;
                let seen = qcc.get_mut(c).ok_or("HTJ2K QCC: component out of range")?;
                if std::mem::replace(seen, true) {
                    return Err("HTJ2K QCC: duplicate component");
                }
                check_qnt(body.get(cb..).ok_or(BAD)?, wav)?;
            }
            mk::NLT if siz => {
                // Znlt:u16 | Snlt | Tnlt (3 = binary complement). `Znlt` 0xFFFF ("all") is never written.
                if body.len() != 4 || body[3] != NLT_BINARY_COMPLEMENT {
                    return Err("HTJ2K NLT: not a binary-complement entry");
                }
                let slot = nlt.get_mut(usize::from(be16(body, 0).ok_or(BAD)?)).ok_or("HTJ2K NLT: component out of range")?;
                if slot.replace(body[2]).is_some() {
                    return Err("HTJ2K NLT: duplicate component");
                }
            }
            mk::COM if siz => {}
            _ => return Err("HTJ2K: unexpected marker in the main header"),
        }
        p += 2 + len;
    };
    if !(siz && cap && cod && qcd) {
        return Err("HTJ2K: SIZ/CAP/COD/QCD missing");
    }
    // NLT must be present exactly on the signed-float components, echoing that component's SSiz.
    for (slot, want) in nlt.iter().zip(comps) {
        let ssiz = (want.bits - 1) as u8 | if want.signed { 0x80 } else { 0 };
        let ok = if want.nlt { *slot == Some(ssiz) } else { slot.is_none() };
        if !ok {
            return Err("HTJ2K NLT: present/absent differently from the dtype's component");
        }
    }
    // Tile-parts: each `SOT(Lsot=10) | Isot=0 | Psot | TPsot | TNsot` then an EMPTY tile-part header (SOD).
    let mut p = sot;
    loop {
        if be16(data, p) != Some(mk::SOT) || be16(data, p + 2) != Some(10) || be16(data, p + 4) != Some(0) {
            return Err("HTJ2K: malformed tile-part (SOT)");
        }
        if be16(data, p + 12) != Some(mk::SOD) {
            return Err("HTJ2K: tile-part header is not empty");
        }
        let psot = be32(data, p + 6).ok_or(BAD)? as usize;
        if psot == 0 {
            return Ok(()); // runs to the end of the codestream
        }
        p = p.checked_add(psot).filter(|&n| psot >= 14 && n <= data.len()).ok_or("HTJ2K: tile-part length")?;
        if be16(data, p) == Some(mk::EOC) {
            return Ok(());
        }
    }
}

/// `check_header` as a typed decode error.
fn verify(data: &[u8], comps: &[Comp], w: u32, h: u32, wav: Wavelet) -> Result<(), DecodeError> {
    check_header(data, comps, w, h, wav).map_err(DecodeError::Corrupt)
}

/// Dtypes the 9/7 profile takes: plain unsigned integers of at most 16 bits.
fn is_ldr(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::U8 | Dtype::U16)
}

/// Where one decoded component goes: the caller's destination ([`Dst`]: a tile plane, rows of a larger plane, or
/// skipped), filled a row at a time as jph hands the lines over (no `n`-sample i32 intermediate, no second
/// conversion pass, no plane `Vec` of our own).
struct Sink<'d, 'a> {
    dst: &'d mut Dst<'a>,
    kind: SinkKind,
}

/// How a component's jph line becomes destination bytes.
#[derive(Clone, Copy)]
enum SinkKind {
    /// The channel's bytes of this element size (`esz`).
    Bytes { esz: usize },
    /// An hdr log-code component converted straight to the float channel `dtype` at `scale`
    /// ([`crate::hdr::from_codes_into`], one table load per sample).
    Hdr { dtype: Dtype, scale: u32 },
}

impl SinkKind {
    /// Bytes per decoded sample.
    fn esz(self) -> usize {
        match self {
            SinkKind::Bytes { esz } => esz,
            SinkKind::Hdr { dtype, .. } => dtype.size(),
        }
    }
}

/// Pull `w`x`h` samples of the components out of a created codestream into the `sinks`, one jph line at a time.
/// Under [`Wavelet::Irrev97`] each line is clamped to its component's unsigned range on the way (quantization
/// noise overshoots saturated samples, 255 -> 256, jph does not clamp, and the byte conversion keeps the LOW bytes,
/// so an unclamped overshoot would wrap to 0; clamping is what makes the error a small +-LSB class instead). Every
/// byte of every non-skipped sink is written or the call fails.
///
/// `planar` must be what the codestream was created with ([`pull_order`]): interleaved = line `y` of every
/// component in turn; planar = all lines of component 0, then 1, ... and the pull STOPS after the last wanted
/// component, so jph never entropy-decodes / inverse-transforms the components behind it (its decode is lazy per line).
fn pull_planes(
    cs: &mut Codestream,
    w: usize,
    h: usize,
    sinks: &mut [Sink<'_, '_>],
    comps: &[Comp],
    wav: Wavelet,
    planar: bool,
) -> Result<(), DecodeError> {
    const SHORT: DecodeError = DecodeError::Request("HTJ2K output plane length != w*h*esz");
    if sinks.iter().any(|s| !s.dst.fits(w, h, s.kind.esz())) {
        return Err(SHORT);
    }
    let mut next_comp: Ui32 = 0;
    let mut line = |cs: &mut Codestream, sink: &mut Sink<'_, '_>, c: &Comp, ci: usize, y: usize| -> Result<(), DecodeError> {
        let line = cs.pull_ref(&mut next_comp).map_err(jph_err)?.get_i32(w);
        if next_comp as usize != ci {
            return Err(DecodeError::Corrupt("HTJ2K component order"));
        }
        let esz = sink.kind.esz();
        let Some(row) = sink.dst.row(y, w * esz) else { return Ok(()) }; // a skipped component: pulled, discarded
        match sink.kind {
            SinkKind::Bytes { .. } => {
                let max = (wav == Wavelet::Irrev97).then(|| ((1u64 << c.bits) - 1) as i32);
                from_i32(row, line, esz, max);
            }
            SinkKind::Hdr { dtype, scale } => hdr::from_codes_into(line, dtype, scale, row),
        }
        Ok(())
    };
    if planar {
        let last = sinks.iter().rposition(|s| !s.dst.is_skip()).map_or(0, |i| i + 1);
        for (ci, (sink, c)) in sinks.iter_mut().zip(comps).enumerate().take(last) {
            for y in 0..h {
                line(cs, sink, c, ci, y)?;
            }
        }
    } else {
        for y in 0..h {
            for (ci, (sink, c)) in sinks.iter_mut().zip(comps).enumerate() {
                line(cs, sink, c, ci, y)?;
            }
        }
    }
    Ok(())
}

/// `true` (planar pull) iff some component is skipped: then the pull can stop after the last wanted one. With every
/// component wanted the interleaved order is kept (one line of each component live at a time, the measured path).
fn pull_order(sinks: &[Sink<'_, '_>]) -> bool {
    sinks.iter().any(|s| s.dst.is_skip())
}

/// Decode ONE multi-component J2K codestream into the `sinks` (one per component). Shared by every decoder here,
/// which differ only in what they ask each component to become. The stream's header is proven to match `w`x`h` and
/// `comps` ([`check_header`]) BEFORE jph-rs sees it, so nothing the stream declares sizes a buffer or drives a
/// loop here. `w`x`h` must come from a validated header; the caller's `DecodeLimits` bound the tile.
fn decode_mc_core(data: &[u8], comps: &[Comp], w: u32, h: u32, wav: Wavelet, sinks: &mut [Sink<'_, '_>]) -> Result<(), DecodeError> {
    let nch = comps.len();
    if nch == 0 || sinks.len() != nch {
        return Err(DecodeError::Request("HTJ2K decode without channels"));
    }
    if w == 0 || h == 0 {
        return Ok(());
    }
    let (wu, hu) = (w as usize, h as usize);
    let n = wu.checked_mul(hu).ok_or(DecodeError::Request("tile size overflows"))?;
    tracing::info_span!("htj2k_verify").in_scope(|| verify(data, comps, w, h, wav))?;
    with_cs(nch.saturating_mul(n), |cs| -> Result<(), DecodeError> {
        let mut infile = MemInfile::default();
        infile.open(data);
        let planar = pull_order(sinks);
        tracing::info_span!("htj2k_open").in_scope(|| -> Result<(), DecodeError> {
            cs.read_headers(&mut infile).map_err(jph_err)?;
            cs.set_planar(planar);
            cs.create().map_err(jph_err)
        })?;
        // jph's entropy + inverse-DWT work happens lazily inside the line pulls, converted into `sinks` per row.
        tracing::info_span!("htj2k_pull").in_scope(|| pull_planes(cs, wu, hu, sinks, comps, wav, planar))?;
        infile.close_infile();
        Ok(())
    })
    .ok_or(PANICKED)?
}

/// The wavelet and the (HdrCodes) code `scale` a decode under `mode` uses.
fn decode_plan(mode: Htj2kMode) -> (Wavelet, Option<u32>) {
    match mode {
        Htj2kMode::Lossless => (Wavelet::Rev53, None),
        Htj2kMode::HdrCodes { q } => (Wavelet::Rev53, Some(htj2k_scale(q))),
        Htj2kMode::Irrev97 { .. } => (Wavelet::Irrev97, None),
    }
}

/// The `(Comp, Sink)` pair of every channel of a decode under `mode`: HdrCodes FLOAT channels are the 16-bit code
/// component converted back to the float channel ([`SinkKind::Hdr`]); everything else is the dtype's own component
/// ([`comp_fmt`]) written as bytes. Mirrors [`spec`].
fn decode_parts<'d, 'a>(mode: Htj2kMode, dtypes: &[Dtype], out: &'d mut [Dst<'a>]) -> (Vec<Comp>, Vec<Sink<'d, 'a>>) {
    let (_, scale) = decode_plan(mode);
    dtypes
        .iter()
        .zip(out.iter_mut())
        .map(|(&dt, dst)| match scale {
            Some(scale) if dt.is_float() => (comp_fmt(dt, true), Sink { dst, kind: SinkKind::Hdr { dtype: dt, scale } }),
            _ => (comp_fmt(dt, false), Sink { dst, kind: SinkKind::Bytes { esz: jph_params(dt).2 } }),
        })
        .unzip()
}

/// Decode one tile's multi-component codestream under `mode` into `out[ci]` (one [`Dst`] per channel: a `w*h*esz`
/// plane or `h` rows of `w*esz` bytes, every byte overwritten; [`Dst::Skip`] components are not converted, and the
/// components behind the last wanted one are not decoded at all): the decoder of [`encode_into`]. The 9/7 reconstruction is rounded and
/// clamped to the integer sample range by jph and may differ by +-1 LSB between independent decoders (FORMAT.md
/// section 7).
///
/// # Errors
/// [`DecodeError::Corrupt`] on a dimension / component-count / component-format mismatch ([`check_header`]) or an
/// OpenJPH error/panic, [`DecodeError::Unsupported`] for a non-U8/U16 dtype under [`Htj2kMode::Irrev97`],
/// [`DecodeError::Request`] when `out` does not match `dtypes`/`w`/`h`.
pub fn decode_into(mode: Htj2kMode, data: &[u8], dtypes: &[Dtype], w: u32, h: u32, out: &mut [Dst<'_>]) -> Result<(), DecodeError> {
    let (wav, _) = decode_plan(mode);
    if wav == Wavelet::Irrev97 && !dtypes.iter().all(|&d| is_ldr(d)) {
        return Err(DecodeError::Unsupported("HTJ2K 9/7 carries U8/U16 channels only"));
    }
    if out.len() != dtypes.len() {
        return Err(DecodeError::Request("HTJ2K output plane count != channel count"));
    }
    let (comps, mut sinks) = decode_parts(mode, dtypes, out);
    decode_mc_core(data, &comps, w, h, wav, &mut sinks)
}

/// Decode a multi-component HTJ2K codestream of wavelet `wav` at reduced resolution
/// `level` (J2K resolution scalability; 0 = full, no mip pyramid; hdr-code tiles are not a pixel pyramid). `w`/`h` = the tile's FULL extent,
/// which the codestream's SIZ must match (a forged SIZ must not size our buffers); the reduced
/// dimensions are then `ceil(full/2^level)`, verified against what jph reports. Returns
/// `(per-channel reduced planes, reduced_w, reduced_h)`.
///
/// # Errors
/// [`DecodeError::Request`] for a level beyond the decomposition count, otherwise as [`decode_into`].
pub fn decode_htj2k_multi_level(
    data: &[u8],
    dtypes: &[Dtype],
    w: u32,
    h: u32,
    level: u32,
    wav: Wavelet,
) -> Result<(Vec<Vec<u8>>, u32, u32), DecodeError> {
    if dtypes.is_empty() {
        return Err(DecodeError::Request("HTJ2K decode without channels"));
    }
    if level > u32::from(DECOMPOSITIONS) {
        return Err(DecodeError::Request("resolution level beyond the decomposition count"));
    }
    let comps: Vec<_> = dtypes.iter().map(|&dt| comp_fmt(dt, false)).collect();
    let nch = dtypes.len();
    // Reduced extent per J2K: the header check pins zero image offset and 1x1-subsampled components, so the
    // factor is exactly 2^level (jph-rs `get_recon_size` = div_ceil(Xsiz, 2^level) - div_ceil(XOsiz, ..)).
    let f = 1u64 << level;
    let (rw, rh) = ((w as u64).div_ceil(f) as u32, (h as u64).div_ceil(f) as u32);
    if rw == 0 || rh == 0 {
        return Ok(((0..nch).map(|_| Vec::new()).collect(), rw, rh));
    }
    let (rwu, rhu) = (rw as usize, rh as usize);
    let n = rwu.checked_mul(rhu).ok_or(DecodeError::Request("tile size overflows"))?;
    verify(data, &comps, w, h, wav)?;
    let mut planes = dtypes
        .iter()
        .map(|&dt| Ok(exv_format::limits::alloc_zeroed(n.checked_mul(dt.size()).ok_or(DecodeError::Request("plane size overflows"))?)?))
        .collect::<Result<Vec<_>, DecodeError>>()?;
    with_cs(nch.saturating_mul(n), |cs| -> Result<(), DecodeError> {
        let mut infile = MemInfile::default();
        infile.open(data);
        cs.read_headers(&mut infile).map_err(jph_err)?;
        // Resolution scalability: restrict AFTER read_headers, BEFORE create(). Level 0 skips it (same
        // bytes as the full-res decode).
        if level > 0 {
            cs.restrict_input_resolution(level, level).map_err(jph_err)?;
        }
        cs.set_planar(false);
        cs.create().map_err(jph_err)?;
        let siz = cs.access_siz();
        if (siz.get_recon_width(0), siz.get_recon_height(0)) != (rw, rh) {
            return Err(DecodeError::Corrupt("HTJ2K reduced extent"));
        }
        let mut dsts: Vec<Dst<'_>> = planes.iter_mut().map(|p| Dst::Plane(p.as_mut_slice())).collect();
        let mut sinks: Vec<_> =
            dsts.iter_mut().zip(dtypes).map(|(dst, &dt)| Sink { dst, kind: SinkKind::Bytes { esz: jph_params(dt).2 } }).collect();
        pull_planes(cs, rwu, rhu, &mut sinks, &comps, wav, false)?;
        infile.close_infile();
        Ok(())
    })
    .ok_or(PANICKED)??;
    Ok((planes, rw, rh))
}

/// Number of components in a J2K codestream (its SIZ component count). Used by
/// the GPU mc-decode path / jph-wgpu (which needs the component count before
/// decode) and by tests asserting a tile blob is ONE multi-component codestream
/// rather than per-channel sub-blobs. `None` on a corrupt/short codestream.
pub fn num_components(data: &[u8]) -> Option<usize> {
    with_cs(0, |cs| -> Option<usize> {
        let mut infile = MemInfile::default();
        infile.open(data);
        cs.read_headers(&mut infile).ok()?;
        let n = cs.access_siz().get_num_components() as usize;
        infile.close_infile();
        Some(n)
    })
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Old-API-shaped wrappers over `encode_into` / `decode_into` so the cases below read as before.
    fn enc_mode(mode: Htj2kMode, ch: &[&[u8]], dts: &[Dtype], w: u32, h: u32) -> Result<Vec<u8>, CodecError> {
        let mut v = Vec::new();
        encode_into(mode, ch, dts, w, h, &mut v)?;
        Ok(v)
    }
    fn mode_of(wav: Wavelet) -> Htj2kMode {
        match wav {
            Wavelet::Rev53 => Htj2kMode::Lossless,
            Wavelet::Irrev97 => Htj2kMode::Irrev97 { q: 0 },
        }
    }
    fn dec_mode(mode: Htj2kMode, data: &[u8], dts: &[Dtype], w: u32, h: u32) -> Result<Vec<Vec<u8>>, DecodeError> {
        let mut planes = dts
            .iter()
            .map(|d| exv_format::limits::alloc_zeroed((w as usize) * (h as usize) * d.size()))
            .collect::<Result<Vec<_>, _>>()?;
        decode_into(mode, data, dts, w, h, &mut planes.iter_mut().map(|p| Dst::Plane(p)).collect::<Vec<_>>())?;
        Ok(planes)
    }
    fn encode_htj2k_multi(ch: &[&[u8]], dts: &[Dtype], w: u32, h: u32) -> Result<Vec<u8>, CodecError> {
        enc_mode(Htj2kMode::Lossless, ch, dts, w, h)
    }
    fn encode_htj2k(plane: &[u8], dt: Dtype, w: u32, h: u32) -> Result<Vec<u8>, CodecError> {
        encode_htj2k_multi(&[plane], &[dt], w, h)
    }
    fn encode_htj2k_multi_ldr(ch: &[&[u8]], dts: &[Dtype], w: u32, h: u32, q: u8) -> Result<Vec<u8>, CodecError> {
        enc_mode(Htj2kMode::Irrev97 { q }, ch, dts, w, h)
    }
    fn encode_htj2k_multi_lossy(ch: &[&[u8]], dts: &[Dtype], w: u32, h: u32, q: u8) -> Result<Vec<u8>, CodecError> {
        enc_mode(Htj2kMode::HdrCodes { q }, ch, dts, w, h)
    }
    fn decode_htj2k_multi(data: &[u8], dts: &[Dtype], w: u32, h: u32, wav: Wavelet) -> Result<Vec<Vec<u8>>, DecodeError> {
        dec_mode(mode_of(wav), data, dts, w, h)
    }
    fn decode_htj2k_multi_lossy(data: &[u8], dts: &[Dtype], w: u32, h: u32, q: u8) -> Result<Vec<Vec<u8>>, DecodeError> {
        dec_mode(Htj2kMode::HdrCodes { q }, data, dts, w, h)
    }
    fn decode_htj2k(data: &[u8], dt: Dtype, w: u32, h: u32) -> Result<Vec<u8>, DecodeError> {
        decode_htj2k_multi(data, &[dt], w, h, Wavelet::Rev53)?.pop().ok_or(DecodeError::Corrupt("HTJ2K plane count"))
    }
    fn decode_htj2k_level(data: &[u8], dt: Dtype, w: u32, h: u32, level: u32) -> Result<(Vec<u8>, u32, u32), DecodeError> {
        let (mut planes, rw, rh) = decode_htj2k_multi_level(data, &[dt], w, h, level, Wavelet::Rev53)?;
        Ok((planes.pop().ok_or(DecodeError::Corrupt("HTJ2K plane count"))?, rw, rh))
    }

    /// Reversible HTJ2K → bit-exact lossless for the supported types.
    #[test]
    fn roundtrip_lossless() {
        // incl. BF16; data (i*37)%256 covers 0x00..0xFF (extremes/sign/NaN bits)
        for dt in [Dtype::F16, Dtype::Bf16, Dtype::F32, Dtype::U32, Dtype::U8, Dtype::U16] {
            let esz = jph_params(dt).2;
            for &(w, h) in &[(64u32, 48u32), (33u32, 17u32)] {
                let plane: Vec<u8> =
                    (0..(w * h) as usize * esz).map(|i| ((i * 37) % 256) as u8).collect();
                let enc = encode_htj2k(&plane, dt, w, h).expect("encode");
                let dec = decode_htj2k(&enc, dt, w, h).expect("decode");
                assert_eq!(dec, plane, "htj2k lossless {dt:?} {w}x{h}");
            }
        }
    }

    /// Every encode failure is a typed error, never a silent empty/RAW result.
    #[test]
    fn encode_errors_are_typed() {
        let e = |r: Result<Vec<u8>, CodecError>| r.unwrap_err();
        assert!(matches!(e(encode_htj2k(&[0u8; 31], Dtype::F16, 4, 4)), CodecError::Dims(_)));
        assert!(matches!(e(encode_htj2k(&[], Dtype::F16, 0, 4)), CodecError::Dims(_)));
        assert!(matches!(e(encode_htj2k_multi_lossy(&[&[0u8; 6]], &[Dtype::F16], 2, 2, 4)), CodecError::Dims(_)));
        let (a, b) = ([0u8; 32], [0u8; 16]);
        assert!(matches!(e(encode_htj2k_multi(&[&a, &b], &[Dtype::F16], 4, 4)), CodecError::Dims(_)));
    }

    /// LOD level 0 must be byte-identical to the full-resolution [`decode_htj2k`] (it skips the
    /// `restrict_input_resolution` call), and must report the full dimensions.
    #[test]
    fn lod_level0_equals_full() {
        for dt in [Dtype::F16, Dtype::F32, Dtype::U32, Dtype::U8, Dtype::U16] {
            let esz = jph_params(dt).2;
            for &(w, h) in &[(64u32, 48u32), (100u32, 70u32)] {
                let plane: Vec<u8> =
                    (0..(w * h) as usize * esz).map(|i| ((i * 37) % 256) as u8).collect();
                let enc = encode_htj2k(&plane, dt, w, h).expect("encode");
                let full = decode_htj2k(&enc, dt, w, h).expect("decode full");
                assert_eq!(
                    decode_htj2k_level(&enc, dt, w, h, 0),
                    Ok((full, w, h)),
                    "level0 must equal full-res {dt:?} {w}x{h}"
                );
            }
        }
    }

    /// A constant-valued plane stays exactly constant at every reduced level: the reversible 5/3
    /// low-pass of a constant is that same constant (integer-exact lifting), and the monotonic NLT used
    /// on float components is a per-sample bijection, so the inverse reproduces V bit-for-bit. Also pins
    /// the reduced dimensions to `ceil(dim / 2^level)` (jph-rs `get_recon_size` rounding) on both a
    /// clean power-of-two tile and a non-power tile.
    #[test]
    fn lod_constant_tile_is_constant_at_each_level() {
        // Exactly-representable constants per dtype: half 4.0 = 0x4400, f32 4.0, u32 7.
        let f16v = 0x4400u16.to_le_bytes();
        let f32v = 4.0f32.to_le_bytes();
        let u32v = 7u32.to_le_bytes();
        let cases: &[(Dtype, &[u8])] = &[(Dtype::F16, &f16v), (Dtype::F32, &f32v), (Dtype::U32, &u32v)];
        for &(dt, v) in cases {
            let esz = jph_params(dt).2;
            assert_eq!(v.len(), esz, "constant width matches dtype size");
            for &(w, h) in &[(64u32, 64u32), (96u32, 40u32)] {
                let plane: Vec<u8> = v.iter().copied().cycle().take((w * h) as usize * esz).collect();
                let enc = encode_htj2k(&plane, dt, w, h).expect("encode");
                for level in 0..=2u32 {
                    let (bytes, rw, rh) = decode_htj2k_level(&enc, dt, w, h, level).expect("decode level");
                    let f = 1u32 << level;
                    assert_eq!(rw, w.div_ceil(f), "recon w {dt:?} {w}x{h} L{level}");
                    assert_eq!(rh, h.div_ceil(f), "recon h {dt:?} {w}x{h} L{level}");
                    assert_eq!(bytes.len(), rw as usize * rh as usize * esz, "len {dt:?} L{level}");
                    for chunk in bytes.chunks_exact(esz) {
                        assert_eq!(chunk, v, "constant preserved {dt:?} {w}x{h} L{level}");
                    }
                }
            }
        }
    }

    /// Forged geometry never sizes our buffers: a codestream decoded against a tile rect it does not
    /// match (including an absurd 2^20×2^20 rect that would be a ~4 TB allocation from a tiny input)
    /// is `None` — for the full decode AND the LOD decoder (A6) — with no abort.
    #[test]
    fn forged_dims_are_rejected_without_allocating() {
        let (w, h) = (64u32, 48u32);
        let plane = vec![0u8; (w * h) as usize * 2];
        let enc = encode_htj2k(&plane, Dtype::F16, w, h).expect("encode");
        assert!(decode_htj2k(&enc, Dtype::F16, w + 1, h).is_err(), "extent mismatch");
        assert!(decode_htj2k(&enc, Dtype::F16, 1 << 20, 1 << 20).is_err(), "alloc bomb");
        assert!(decode_htj2k_multi(&enc, &[Dtype::F16, Dtype::F16], w, h, Wavelet::Rev53).is_err(), "component mismatch");
        assert!(decode_htj2k_level(&enc, Dtype::F16, w + 1, h, 1).is_err(), "LOD extent mismatch");
        assert!(decode_htj2k_level(&enc, Dtype::F16, 1 << 20, 1 << 20, 1).is_err(), "LOD alloc bomb");
        assert!(decode_htj2k_level(&enc, Dtype::F16, w, h, 99).is_err(), "level beyond decomposition count");
        assert!(decode_htj2k(&enc[..enc.len() / 2], Dtype::F16, w, h).is_err(), "truncated");
        assert!(decode_htj2k(&[0xff; 64], Dtype::F16, w, h).is_err(), "garbage");
    }

    /// A panic/err inside jph leaves the pool usable: decode of garbage then a valid decode succeeds.
    #[test]
    fn pool_survives_bad_input() {
        let plane = vec![7u8; 16 * 16 * 4];
        let enc = encode_htj2k(&plane, Dtype::F32, 16, 16).expect("encode");
        assert!(decode_htj2k(&[1, 2, 3], Dtype::F32, 16, 16).is_err());
        assert_eq!(decode_htj2k(&enc, Dtype::F32, 16, 16).as_deref(), Ok(plane.as_slice()));
    }

    /// Deterministic per-channel test bytes (covers the full byte range, incl. sign / NaN-ish float bits).
    fn pattern(n: usize, esz: usize, seed: usize) -> Vec<u8> {
        (0..n * esz).map(|i| ((i * (31 + 2 * seed) + seed * 17) % 256) as u8).collect()
    }

    /// U8 and U16 are real unsigned components: a multi-component tile mixing every dtype round-trips
    /// bit-exact (lossless, lossy-ints, LOD level 0) across odd sizes and degenerate 1-pixel tiles; U16
    /// values above 0x7FFF are zero-extended (never sign-extended like the F16 bit pattern path).
    #[test]
    fn mixed_unsigned_and_float_tiles_roundtrip() {
        let dts = [Dtype::U8, Dtype::F16, Dtype::U16, Dtype::U32, Dtype::F32, Dtype::Bf16];
        for &(w, h) in &[(1u32, 1u32), (3, 2), (17, 5), (64, 48), (33, 70)] {
            let n = (w * h) as usize;
            let chans: Vec<Vec<u8>> = dts.iter().enumerate().map(|(i, d)| pattern(n, d.size(), i)).collect();
            let refs: Vec<&[u8]> = chans.iter().map(Vec::as_slice).collect();
            let enc = encode_htj2k_multi(&refs, &dts, w, h).expect("encode");
            assert_eq!(num_components(&enc), Some(dts.len()));
            assert_eq!(decode_htj2k_multi(&enc, &dts, w, h, Wavelet::Rev53).as_ref(), Ok(&chans), "lossless {w}x{h}");
            let (lv0, rw, rh) = decode_htj2k_multi_level(&enc, &dts, w, h, 0, Wavelet::Rev53).expect("level 0");
            assert_eq!((rw, rh), (w, h));
            assert_eq!(lv0, chans, "level 0 == full {w}x{h}");
            // Lossy profile: integer channels stay exact, floats go through hdr codes (checked elsewhere).
            let ints = [Dtype::U8, Dtype::U16, Dtype::U32];
            let int_refs: Vec<&[u8]> = [&chans[0], &chans[2], &chans[3]].iter().map(|c| c.as_slice()).collect();
            let lossy = encode_htj2k_multi_lossy(&int_refs, &ints, w, h, 4).expect("lossy ints");
            let back = decode_htj2k_multi_lossy(&lossy, &ints, w, h, 4).expect("decode lossy ints");
            assert_eq!(back, vec![chans[0].clone(), chans[2].clone(), chans[3].clone()], "lossy profile keeps ints exact");
        }
        let hi: Vec<u8> = [0xFFFFu16, 0x8000, 0x7FFF, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let enc = encode_htj2k(&hi, Dtype::U16, 2, 2).expect("encode");
        assert_eq!(decode_htj2k(&enc, Dtype::U16, 2, 2).as_deref(), Ok(hi.as_slice()), "U16 high values");
    }

    /// B4: the codestream's component bit depth / signedness must match the dtype it is decoded as — a
    /// stream written as U8 is NOT silently reinterpreted as U16/F16/U32, and vice versa.
    #[test]
    fn decode_rejects_component_format_mismatch() {
        let (w, h) = (16u32, 8u32);
        let n = (w * h) as usize;
        let all = [Dtype::U8, Dtype::U16, Dtype::F16, Dtype::Bf16, Dtype::U32, Dtype::F32];
        for &enc_dt in &all {
            let plane = pattern(n, enc_dt.size(), 3);
            let enc = encode_htj2k(&plane, enc_dt, w, h).expect("encode");
            for &dec_dt in &all {
                let compatible = jph_params(enc_dt).0 == jph_params(dec_dt).0 && jph_params(enc_dt).1 == jph_params(dec_dt).1;
                let got = decode_htj2k(&enc, dec_dt, w, h);
                assert_eq!(got.is_ok(), compatible, "encoded {enc_dt:?} decoded as {dec_dt:?}");
                assert_eq!(decode_htj2k_level(&enc, dec_dt, w, h, 0).is_ok(), compatible, "LOD {enc_dt:?}->{dec_dt:?}");
            }
        }
        // hdr-code components are 16-bit signed: they are not decodable as plain U16 ints.
        let f16 = [1u8, 0x3c, 2, 0x3c, 3, 0x3c, 4, 0x3c];
        let codes = encode_htj2k_multi_lossy(&[&f16], &[Dtype::F16], 2, 2, 4).expect("codes");
        assert!(decode_htj2k_multi_lossy(&codes, &[Dtype::F16], 2, 2, 4).is_ok());
        assert!(decode_htj2k(&codes, Dtype::U16, 2, 2).is_err());
    }
    // ---- hostile codestream headers (N1 / N2) -------------------------------------------------------

    /// A valid mixed multi-component tile (both NLT and non-NLT components) and its channel data.
    fn mixed_tile() -> (Vec<u8>, Vec<Dtype>, u32, u32) {
        let dts = vec![Dtype::F16, Dtype::U8, Dtype::F32, Dtype::U32, Dtype::Bf16];
        let (w, h) = (64u32, 64u32);
        let chans: Vec<Vec<u8>> = dts.iter().enumerate().map(|(i, d)| pattern((w * h) as usize, d.size(), i)).collect();
        let refs: Vec<&[u8]> = chans.iter().map(Vec::as_slice).collect();
        (encode_htj2k_multi(&refs, &dts, w, h).expect("encode"), dts, w, h)
    }

    /// Decode `data` on a fresh thread with a hard deadline: the pre-fix decoder spun forever on a sub-sampled
    /// component, which would otherwise hang the test run instead of failing it. `Err(())` = timed out.
    fn timed(data: Vec<u8>, dts: &[Dtype], w: u32, h: u32) -> Result<Result<Vec<Vec<u8>>, DecodeError>, ()> {
        let (tx, rx) = std::sync::mpsc::channel();
        let dts = dts.to_vec();
        std::thread::spawn(move || {
            // the receiver may have given up (timeout): a failed send then is expected
            tx.send(decode_htj2k_multi(&data, &dts, w, h, Wavelet::Rev53)).ok();
        });
        rx.recv_timeout(std::time::Duration::from_secs(30)).map_err(|_| ())
    }

    /// Offset of the first `marker` segment (marker bytes) in the main header.
    fn find_marker(enc: &[u8], marker: u16) -> usize {
        let mut p = 2;
        while be16(enc, p) != Some(marker) {
            assert_ne!(be16(enc, p), Some(mk::SOT), "marker {marker:#x} not in the main header");
            p += 2 + be16(enc, p + 2).expect("len") as usize;
        }
        p
    }

    /// N1: a flipped sub-sampling byte of ANY component made `read_still` spin forever (jph-rs `pull` never
    /// finishes a line of a sub-sampled component). Now it is a typed error before jph sees the stream.
    #[test]
    fn subsampled_component_is_rejected_not_spun() {
        let (enc, dts, w, h) = mixed_tile();
        let siz = find_marker(&enc, mk::SIZ);
        for ci in 0..dts.len() {
            for field in [1usize, 2] {
                // components start 36 bytes into the SIZ body (marker + Lsiz = 4 bytes)
                let at = siz + 4 + 36 + 3 * ci + field;
                for v in [0u8, 2, 4, 255] {
                    let mut bad = enc.clone();
                    bad[at] = v;
                    let r = timed(bad, &dts, w, h).expect("decode must terminate (N1)");
                    assert!(matches!(r, Err(DecodeError::Corrupt(m)) if m.contains("sub-sampled")), "comp {ci} field {field} = {v}: {r:?}");
                }
            }
        }
    }

    /// N2: COD/QCD/CAP/NLT fields are validated against what the encoder writes; one flipped byte of the
    /// code-block size used to make jph-rs allocate 358 MiB (and keep it in the thread's pool).
    #[test]
    fn cod_qcd_nlt_fields_are_pinned() {
        let (enc, dts, w, h) = mixed_tile();
        // (marker, byte offsets inside the body that must be exactly what the encoder wrote)
        let cod = find_marker(&enc, mk::COD) + 4;
        let qcd = find_marker(&enc, mk::QCD) + 4;
        let nlt = find_marker(&enc, mk::NLT) + 4;
        let cap = find_marker(&enc, mk::CAP) + 4;
        let must_reject: Vec<(&str, usize)> = vec![
            ("Scod (precincts/SOP/EPH)", cod),
            ("layers", cod + 3),
            ("mct", cod + 4),
            ("decompositions", cod + 5),
            ("xcb", cod + 6),
            ("ycb", cod + 7),
            ("code-block style", cod + 8),
            ("transform", cod + 9),
            ("Sqcd", qcd),
            ("QCD exponent low bits", qcd + 1),
            ("NLT component", nlt + 1),
            ("NLT type", nlt + 3),
            ("Pcap", cap),
        ];
        for (what, at) in must_reject {
            for v in [0x00u8, 0x01, 0x03, 0x7F, 0xFF] {
                if enc[at] == v {
                    continue;
                }
                let mut bad = enc.clone();
                bad[at] = v;
                let r = timed(bad, &dts, w, h).expect("decode must terminate");
                assert!(matches!(r, Err(DecodeError::Corrupt(_))), "{what} = {v:#x} must be rejected, got {:?}", r.map(|p| p.len()));
            }
        }
    }

    /// More than 256 components switch QCC's component index to two bytes (`Cqcc`/`Znlt` width rules): the header
    /// check must accept what the encoder writes at that size too.
    #[test]
    fn many_components_roundtrip() {
        let (w, h) = (8u32, 8u32);
        let dts: Vec<Dtype> = (0..300).map(|i| if i % 3 == 0 { Dtype::F16 } else if i % 3 == 1 { Dtype::U8 } else { Dtype::U32 }).collect();
        let chans: Vec<Vec<u8>> = dts.iter().enumerate().map(|(i, d)| pattern((w * h) as usize, d.size(), i)).collect();
        let refs: Vec<&[u8]> = chans.iter().map(Vec::as_slice).collect();
        let enc = encode_htj2k_multi(&refs, &dts, w, h).expect("encode");
        assert_eq!(decode_htj2k_multi(&enc, &dts, w, h, Wavelet::Rev53).as_ref(), Ok(&chans));
    }

    /// B4 (NLT case): an F16/F32 component whose binary-complement NLT segment is missing would decode to
    /// different numbers with no error; renaming the first NLT segment into a (valid, ignored) COM segment
    /// leaves a well-formed header whose NLT set no longer matches the dtypes.
    #[test]
    fn missing_nlt_on_a_float_component_is_rejected() {
        let (enc, dts, w, h) = mixed_tile();
        let mut bad = enc.clone();
        bad[find_marker(&enc, mk::NLT) + 1] = (mk::COM & 0xFF) as u8;
        let r = timed(bad, &dts, w, h).expect("terminates");
        assert!(matches!(r, Err(DecodeError::Corrupt(m)) if m.contains("NLT")), "{:?}", r.map(|p| p.len()));
    }

    /// Every single-byte mutation of the whole header region (8 values per offset) terminates in time and never
    /// panics; this is the sweep that found N1/N2 and now guards the class.
    #[test]
    fn header_byte_sweep_terminates() {
        let (enc, dts, w, h) = mixed_tile();
        let end = find_marker(&enc, mk::SOT) + 14;
        let mut rejected = 0usize;
        for at in 0..end {
            for v in [0x00u8, 0x01, 0x02, 0x7F, 0x80, 0xFF, enc[at] ^ 0xFF, enc[at].wrapping_add(1)] {
                let mut bad = enc.clone();
                bad[at] = v;
                // Ok is allowed (benign bytes: COM text, progression order, guard bits, Rsiz, Ccap);
                // a hang or panic is not.
                match timed(bad, &dts, w, h).expect("decode must terminate") {
                    Err(_) => rejected += 1,
                    Ok(planes) => assert_eq!(planes.len(), dts.len()),
                }
            }
        }
        assert!(rejected > end, "the sweep must exercise the validator ({rejected} rejections over {end} bytes)");
    }

    /// Tile-part headers: an extra marker before SOD (a tile-level COD would override the main header's code-block
    /// size) and a forged Psot are corruption, not parsed.
    #[test]
    fn tile_part_header_must_be_plain() {
        let (enc, dts, w, h) = mixed_tile();
        let sot = find_marker(&enc, mk::SOT);
        let mut bad = enc.clone();
        bad[sot + 12] = 0xFF;
        bad[sot + 13] = 0x52; // SOD -> COD
        assert!(matches!(timed(bad, &dts, w, h).expect("terminates"), Err(DecodeError::Corrupt(_))));
        let mut bad = enc.clone();
        bad[sot + 6..sot + 10].copy_from_slice(&u32::MAX.to_be_bytes()); // Psot past the end
        assert!(matches!(timed(bad, &dts, w, h).expect("terminates"), Err(DecodeError::Corrupt(_))));
        assert!(matches!(timed(Vec::new(), &dts, w, h).expect("terminates"), Err(DecodeError::Corrupt(_))));
    }

    /// The per-worker codec pool never retains the storage of a huge tile.
    #[test]
    fn huge_tiles_do_not_stay_pooled() {
        // The test thread is fresh: its pool starts empty.
        assert_eq!(with_cs(POOL_MAX_SAMPLES + 1, |_| ()), Some(()));
        assert!(CODESTREAM.with(|c| c.borrow().is_none()), "an over-threshold codec must be dropped");
        assert_eq!(with_cs(POOL_MAX_SAMPLES, |_| ()), Some(()));
        assert!(CODESTREAM.with(|c| c.borrow().is_some()), "a normal codec is pooled");
    }
    /// A smooth gradient plus deterministic noise as U8 or U16 samples (the LDR content the 9/7 profile targets).
    fn ldr_plane(w: u32, h: u32, dt: Dtype, noise: u32) -> Vec<u8> {
        let max = if dt == Dtype::U8 { 255u32 } else { 65535 };
        let mut s = 0x9E37_79B9u32;
        (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let base = (x * max / w.max(1) + y * max / h.max(1)) / 2;
                let v = (base + (s >> 16) % (noise + 1)).min(max);
                if dt == Dtype::U8 { vec![v as u8] } else { (v as u16).to_le_bytes().to_vec() }
            })
            .collect()
    }

    /// Largest absolute sample difference between two planes of `dt`.
    fn max_diff(a: &[u8], b: &[u8], dt: Dtype) -> u32 {
        let v = |p: &[u8], i: usize| if dt == Dtype::U8 { u32::from(p[i]) } else { u32::from(u16::from_le_bytes([p[2 * i], p[2 * i + 1]])) };
        (0..a.len() / dt.size()).map(|i| v(a, i).abs_diff(v(b, i))).max().unwrap_or(0)
    }

    /// The 9/7 profile: a mixed U8+U16 tile survives within a q-dependent error (the quantizer step is
    /// normalised to the sample range, so U8 and U16 see the same RELATIVE error), the stream carries the
    /// irreversible wavelet, and finer q is monotonically closer.
    #[test]
    fn irrev97_roundtrip_error_tracks_q() {
        let (w, h) = (96u32, 70u32);
        let dts = [Dtype::U8, Dtype::U16, Dtype::U8];
        let chans: Vec<Vec<u8>> = dts.iter().map(|&d| ldr_plane(w, h, d, 6)).collect();
        let refs: Vec<&[u8]> = chans.iter().map(Vec::as_slice).collect();
        let mut last = u32::MAX;
        for q in [8u8, 16, 24, 32] {
            let enc = encode_htj2k_multi_ldr(&refs, &dts, w, h, q).expect("encode");
            assert_eq!(num_components(&enc), Some(dts.len()));
            let dec = decode_htj2k_multi(&enc, &dts, w, h, Wavelet::Irrev97).expect("decode");
            assert!(decode_htj2k_multi(&enc, &dts, w, h, Wavelet::Rev53).is_err(), "a 9/7 stream is not a 5/3 one");
            let rel = |i: usize| max_diff(&chans[i], &dec[i], dts[i]) as f64 / if dts[i] == Dtype::U8 { 255.0 } else { 65535.0 };
            let worst = (0..dts.len()).map(rel).fold(0.0, f64::max);
            eprintln!("q={q}: {} B, worst relative error {worst:.5}", enc.len());
            assert!(worst <= 0.5f64.powf(f64::from(q) / 4.0) + 1e-3, "q={q}: error {worst}");
            let total: u32 = (0..dts.len()).map(|i| max_diff(&chans[i], &dec[i], dts[i])).sum();
            assert!(total <= last || last == u32::MAX, "error must not grow with q");
            last = total;
        }
    }

    /// Near-lossless q on 8-bit content: every sample within +-1 LSB (the documented 9/7 decode class).
    #[test]
    fn irrev97_fine_q_is_within_one_lsb() {
        let (w, h) = (64u32, 64u32);
        for (dt, q) in [(Dtype::U8, 24u8), (Dtype::U16, 40)] {
            let p = ldr_plane(w, h, dt, 3);
            let enc = encode_htj2k_multi_ldr(&[&p], &[dt], w, h, q).expect("encode");
            let dec = decode_htj2k_multi(&enc, &[dt], w, h, Wavelet::Irrev97).expect("decode").remove(0);
            assert!(max_diff(&p, &dec, dt) <= 1, "{dt:?} q={q}: {}", max_diff(&p, &dec, dt));
        }
    }

    /// A float / U32 channel in a 9/7 tile is a typed error on encode (never quantized, never a silent
    /// fallback) and `Unsupported` on decode.
    #[test]
    fn irrev97_rejects_non_ldr_dtypes() {
        for dt in [Dtype::F16, Dtype::Bf16, Dtype::F32, Dtype::U32] {
            let p = vec![0u8; 16 * 16 * dt.size()];
            let e = encode_htj2k_multi_ldr(&[&p], &[dt], 16, 16, 13).unwrap_err();
            assert_eq!(e, CodecError::Dtype { codec: "HTJ2K 9/7", dtype: dt });
            let ok = encode_htj2k_multi_ldr(&[&[0u8; 256]], &[Dtype::U8], 16, 16, 13).expect("u8");
            assert!(matches!(decode_htj2k_multi(&ok, &[dt], 16, 16, Wavelet::Irrev97), Err(DecodeError::Unsupported(_))));
        }
    }

    /// The 9/7 stream keeps resolution scalability: level 0 equals the full decode, level 1 halves the extent.
    #[test]
    fn irrev97_lod() {
        let (w, h) = (100u32, 70u32);
        let p = ldr_plane(w, h, Dtype::U8, 4);
        let enc = encode_htj2k_multi_ldr(&[&p], &[Dtype::U8], w, h, 20).expect("encode");
        let full = decode_htj2k_multi(&enc, &[Dtype::U8], w, h, Wavelet::Irrev97).expect("full");
        let (l0, rw, rh) = decode_htj2k_multi_level(&enc, &[Dtype::U8], w, h, 0, Wavelet::Irrev97).expect("l0");
        assert_eq!((l0, rw, rh), (full, w, h));
        let (l1, rw, rh) = decode_htj2k_multi_level(&enc, &[Dtype::U8], w, h, 1, Wavelet::Irrev97).expect("l1");
        assert_eq!((rw, rh, l1[0].len()), (50, 35, 50 * 35));
    }

    /// Hostile 9/7 header: the host-side check pins the wavelet, the quantization style and its length, so
    /// flipping COD's transform byte or the QCD style/length never reaches jph (typed `Corrupt`, no hang).
    #[test]
    fn irrev97_header_is_pinned() {
        let (w, h) = (32u32, 32u32);
        let p = ldr_plane(w, h, Dtype::U8, 2);
        let enc = encode_htj2k_multi_ldr(&[&p], &[Dtype::U8], w, h, 16).expect("encode");
        let at = |m: u16| enc.windows(2).position(|b| b == m.to_be_bytes()).expect("marker");
        let (cod, qcd) = (at(mk::COD), at(mk::QCD));
        for (what, pos) in [("COD transform", cod + 4 + 9), ("QCD style", qcd + 4)] {
            let mut bad = enc.clone();
            bad[pos] ^= 0x01;
            assert!(matches!(decode_htj2k_multi(&bad, &[Dtype::U8], w, h, Wavelet::Irrev97), Err(DecodeError::Corrupt(_))), "{what}");
        }
        let mut short = enc.clone();
        short[qcd + 3] -= 2; // QCD length: one subband entry fewer
        assert!(decode_htj2k_multi(&short, &[Dtype::U8], w, h, Wavelet::Irrev97).is_err());
    }
}
