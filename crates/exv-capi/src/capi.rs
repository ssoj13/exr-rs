//! Flat C ABI over `exv-codec`/`exv-video` — the FFI seam for native system hosts.
//!
//! This is the **only** crate in the project with `unsafe`/FFI: the core (`exv-codec`/`exv-format`/
//! `exv-video`) stays safe Rust. Native plugins (After Effects AEIO) link
//! `exv_capi.dll` and call these functions. The shape mirrors 1:1 the proven sibling
//! `apv-rs::apv-capi` (path-session: Rust owns the file I/O and decodes a frame on demand),
//! but **without** YCbCr/CICP/color management — EXV is already linear RGBA-float / arbitrary planar.
//!
//! Conventions:
//! - Every fallible call returns `ExvStatus` (`i32`); `0` = OK. The error text is
//!   via [`exv_last_error_message`] (thread-local — safe for the host's worker threads).
//! - Every status-returning export body runs under [`guard`], which catches a Rust panic and maps
//!   it to `ExvStatus::Internal` (a panic unwinding out of an `extern "C"` fn aborts the host).
//!   (Plain `#[unsafe(no_mangle)]` fns, not a macro: cbindgen must see every export to emit the header.)
//! - The encode output is an opaque, library-owned [`ExvBuffer`]: read via
//!   `exv_buffer_data`/`exv_buffer_len`, freed by `exv_buffer_free` (the length stays
//!   on the Rust side — no UB from a caller's length getting out of sync).
//! - Opaque handles (`ExvSession`/`ExvVideoEncoder`) are created/destroyed in
//!   open/close pairs. `ExvSession` is read-only after open → `exv_session_decode` is safe
//!   to call concurrently from many threads on the same handle.
//! - Handle/buffer out-pointers are NULLed on entry, so a caller's uninitialised handle is never left dangling on
//!   error; info/meta structs are written only on success.
//! - The canonical header `docs/exv-capi.h` is GENERATED from this file by cbindgen (`header_in_sync` guards drift).

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::sync::OnceLock;

use exv_codec::registry::{self, OpenError};
use exv_codec::{Htj2kMode, Profile};
use exv_format::fsio;
use exv_format::half::{f32_to_bf16, f32_to_half};
use exv_codec::image::{Image, RgbaPick, sample_f32};
use rayon::prelude::*;
use exv_format::{Channel, CodecId, DecodeLimits, Dtype, Header, Meta};
use exv_tile::BudgetExt;

/// ABI revision, bumped on ANY signature/struct/semantic change of the exports. The host checks it
/// (`exv_abi_version`) at load, so a stale `exv_capi.dll` next to a newer plugin is rejected up front
/// instead of crashing on a changed signature. Emitted as `#define EXV_ABI_VERSION` in `docs/exv-capi.h`.
pub const EXV_ABI_VERSION: u32 = 5;

/// C ABI status codes. Returned by every fallible export; `Ok` = 0.
#[repr(i32)]
pub enum ExvStatus {
    Ok = 0,
    InvalidArgument = 1,
    InvalidFormat = 3,
    Unsupported = 4,
    Internal = 255,
}

/// Pixel formats for `exv_session_decode` (passed as `u32`). EXV stores linear scene-referred RGBA — we
/// hand back exactly that (no tone-mapping); the host applies the display transform.
#[repr(u32)]
pub enum ExvPixelFormat {
    /// Interleaved RGBA `f32` (host-endian), linear scene-referred. The primary AE float path.
    RgbaF32 = 0,
    /// Interleaved RGBA `f16` (IEEE half, LE u16), linear scene-referred.
    RgbaF16 = 1,
}

/// Plane dtype codes for [`ExvEncParams::dtype_code`] (mirror of `exv_format::Dtype::code()`).
#[repr(u32)]
pub enum ExvDtype {
    F16 = 0,
    F32 = 1,
    Bf16 = 2,
    U8 = 3,
    U16 = 4,
    U32 = 5,
}

/// Codec ids for [`ExvEncParams::codec_id`] (mirror of `exv_format::CodecId`); any other id is
/// `UNSUPPORTED`. `q` rules: see [`ExvEncParams`].
#[repr(u32)]
pub enum ExvCodec {
    /// Reorder+delta -> rANS (GPU-decodable). `q` must be 0.
    LosslessFast = 1,
    /// Reversible 5/3 wavelet -> rANS (lossless). `q` must be 0.
    Wavelet53 = 2,
    /// Signed 5/3 wavelet + quantisation (lossy). `q` = quant bits, 1..=62.
    Wavelet53Lossy = 3,
    /// HTJ2K, one multi-component codestream per tile (project default; needs the library's htj2k backend, else `UNSUPPORTED`).
    /// `codec_mode` 0: `q` = 0 -> lossless, `q` = 1..=[`EXV_Q_HDR_MAX`] -> lossy hdr-codes quality knob (float
    /// channels only; ints stay exact). `codec_mode` 1 ([`ExvCodecMode::Htj2kIrreversible97`]): irreversible 9/7,
    /// `q` = 1..=[`EXV_Q_LDR_MAX`].
    Htj2k = 6,
    /// EXV-RT realtime block codec (lossy, GPU-parallel; 16x16 blocks): `q` = detail quantisation
    /// 0..=[`EXV_Q_RT_MAX`] (0 = lossless detail). Only F16 planes are block-coded; other dtypes ride raw.
    ExvRt = 4,
    /// Hardware `BC6H_SF16` 4x4 block codec: `q` = 0 -> lossless (`BC6H_SF16` base + rANS 16-bit residual),
    /// `q` = 1 -> lossy (`BC6H_SF16` blocks only). F16 planes are BC6H-coded; other dtypes ride raw.
    Bc6h = 7,
}

// NOTE: every `EXV_Q_*_MAX` below is a literal because cbindgen cannot follow another crate's const into the header;
// each mirrors its `exv_codec` owner (`WAVE_Q_MAX`/`HDR_Q_MAX`/`LDR_Q_MAX`/`RT_Q_MAX`) and is pinned by `q_limits_are_the_edges`.
/// Quality-knob limits per codec (the C side builds its option UI from these; a `q` outside is `INVALID_ARGUMENT`).
/// Wavelet-lossy quant bits.
pub const EXV_Q_WAVE_MAX: u32 = 62;
/// HTJ2K hdr-codes quality: `scale = clamp(16*q, 16, 256)`, so 16 is the finest step that exists.
pub const EXV_Q_HDR_MAX: u32 = 16;
/// HTJ2K irreversible 9/7 quality (`exv_codec::LDR_Q_MAX`, pinned by a test).
pub const EXV_Q_LDR_MAX: u32 = 40;
/// EXV-RT detail quantisation (F16 detail width).
pub const EXV_Q_RT_MAX: u32 = 15;
/// EXV-RT block edge the ABI fixes (the frozen GPU-friendly v0 block, same as the CLI default).
const RT_BLOCK: u8 = 16;

/// Values of [`ExvEncParams::codec_mode`]: a per-codec variant selector, so one `codec_id` can carry several
/// encode models without overloading `q`. `Default` (0) is what zero-initialised params mean for every codec.
#[repr(u32)]
pub enum ExvCodecMode {
    /// The codec's own default for its `q` (see [`ExvCodec`]).
    Default = 0,
    /// HTJ2K irreversible 9/7 (`ExvCodec::Htj2k` only): U8/U16 images, `q` = 1..=40 (larger = finer; q=16 is one
    /// 8-bit step). A float/U32 image is `UNSUPPORTED`-style encode failure (never quantized). Decode is exact
    /// to +-1 LSB across independent decoders (docs/FORMAT.md section 7).
    Htj2kIrreversible97 = 1,
}

/// Internal `u32` views of [`ExvPixelFormat`] (usable in `match` patterns; kept out of the header).
mod pixfmt {
    use super::ExvPixelFormat;
    pub(crate) const RGBA_F32: u32 = ExvPixelFormat::RgbaF32 as u32;
    pub(crate) const RGBA_F16: u32 = ExvPixelFormat::RgbaF16 as u32;
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn fail(status: ExvStatus, msg: impl Into<String>) -> i32 {
    let c = CString::new(msg.into()).unwrap_or_else(|_| CString::new("error").unwrap_or_default());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
    status as i32
}

fn okay() -> i32 {
    ExvStatus::Ok as i32
}

/// Run an export body, converting a Rust panic into `ExvStatus::Internal` + a last-error message.
/// `AssertUnwindSafe`: the bodies only touch caller-provided out-pointers and handles that the host
/// treats as poisoned after a non-OK status.
fn guard(name: &str, f: impl FnOnce() -> i32) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(p) => {
            let why = p
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".into());
            fail(ExvStatus::Internal, format!("panic in {name}: {why}"))
        }
    }
}

/// The last thread-local error message (NUL-terminated), or an empty string.
#[unsafe(no_mangle)]
pub extern "C" fn exv_last_error_message() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => c"".as_ptr(),
    })
}

/// ABI revision ([`EXV_ABI_VERSION`]); compare against the header's `EXV_ABI_VERSION` at load.
#[unsafe(no_mangle)]
pub extern "C" fn exv_abi_version() -> u32 {
    EXV_ABI_VERSION
}

/// Library version (semver), written into `out`.
#[repr(C)]
pub struct ExvVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

/// # Safety
/// `out` is a valid writable `*mut ExvVersion`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_version(out: *mut ExvVersion) -> i32 {
    guard("exv_version", || {
        if out.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_version: out is null");
        }
        unsafe {
            *out = ExvVersion {
                major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0),
                minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0),
                patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0),
                reserved: 0,
            };
        }
        okay()
    })
}

// ───────────────────────────── output buffer ─────────────────────────────

/// Opaque, library-owned byte buffer (encode output). Freed by [`exv_buffer_free`].
pub struct ExvBuffer {
    data: Vec<u8>,
}

/// Pointer to the buffer bytes (valid until [`exv_buffer_free`]).
/// # Safety
/// `buf` is a handle from an encode call (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_buffer_data(buf: *const ExvBuffer) -> *const u8 {
    match unsafe { buf.as_ref() } {
        Some(b) => b.data.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Buffer length in bytes.
/// # Safety
/// `buf` is a handle from an encode call (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_buffer_len(buf: *const ExvBuffer) -> usize {
    match unsafe { buf.as_ref() } {
        Some(b) => b.data.len(),
        None => 0,
    }
}

/// Free a library-allocated buffer.
/// # Safety
/// `buf` is a handle from an encode call (or null), freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_buffer_free(buf: *mut ExvBuffer) {
    if !buf.is_null() {
        drop(unsafe { Box::from_raw(buf) });
    }
}

/// Hand `data` to the caller as an [`ExvBuffer`] through `out`/`out_len` (`out_len` may be null).
///
/// # Safety
/// `out` is non-null and writable; `out_len` is null or writable.
unsafe fn emit(data: Vec<u8>, out: *mut *mut ExvBuffer, out_len: *mut usize) {
    let len = data.len();
    unsafe {
        *out = Box::into_raw(Box::new(ExvBuffer { data }));
        if !out_len.is_null() {
            *out_len = len;
        }
    }
}

/// Zero the encode out-pointers (`out_len` may be null) so an error leaves no dangling garbage.
///
/// # Safety
/// `out` is non-null and writable; `out_len` is null or writable.
unsafe fn clear_out(out: *mut *mut ExvBuffer, out_len: *mut usize) {
    unsafe {
        *out = std::ptr::null_mut();
        if !out_len.is_null() {
            *out_len = 0;
        }
    }
}

// ───────────────────────────── dtype <-> f32 ─────────────────────────────
// ONE implementation workspace-wide: widening is `exv_codec::image::sample_f32`, narrowing uses the
// `exv_codec::hdr` f16/bf16 converters (same NaN/rounding policy as the codecs and the viewer).

/// Pack one component (`comp` of 4) of interleaved RGBA-f32 into a planar plane of `dtype`.
/// Int types are clamped [0,1]→full range; f16/bf16 use the shared round-to-nearest-even narrowing.
fn pack_plane(rgba: &[f32], comp: usize, w: u32, h: u32, dtype: Dtype) -> Vec<u8> {
    let n = w as usize * h as usize;
    let mut out = Vec::with_capacity(n * dtype.size());
    for i in 0..n {
        let v = rgba.get(i * 4 + comp).copied().unwrap_or(0.0);
        match dtype {
            Dtype::F16 => out.extend_from_slice(&f32_to_half(v).to_le_bytes()),
            Dtype::Bf16 => out.extend_from_slice(&f32_to_bf16(v).to_le_bytes()),
            Dtype::F32 => out.extend_from_slice(&v.to_le_bytes()),
            Dtype::U8 => out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8),
            Dtype::U16 => out.extend_from_slice(&((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes()),
            Dtype::U32 => {
                out.extend_from_slice(&((v.clamp(0.0, 1.0) as f64 * 4_294_967_295.0 + 0.5) as u32).to_le_bytes())
            }
        }
    }
    out
}

// ───────────────────────────── file-URL → path ─────────────────────────────

/// Percent-decode `%XX` → UTF-8 bytes (RFC-3986). An invalid sequence
/// (`%` without two hex digits) is copied verbatim. `Err` if the decoded bytes are not valid UTF-8
/// (a silent U+FFFD substitution would open a DIFFERENT file than the host named).
fn percent_decode(s: &str) -> Result<String, String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(hi), Some(lo)) = ((b[i + 1] as char).to_digit(16), (b[i + 2] as char).to_digit(16))
        {
            out.push((hi * 16 + lo) as u8);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).map_err(|_| format!("percent-decoded path is not UTF-8: {s}"))
}

/// `X:` or `X:/…` — a Windows drive-absolute path.
fn is_drive(p: &str) -> bool {
    let b = p.as_bytes();
    b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' && b.get(2).is_none_or(|c| matches!(c, b'/' | b'\\'))
}

/// Normalize a path that a host (After Effects et al.) may pass as an RFC-8089 `file:` URL into a
/// plain filesystem path, percent-decoding it (`%20`/Cyrillic). Forms (scheme is case-insensitive):
/// `file:///C:/x` → `C:/x`; `file:///home/x` → `/home/x`; `file://localhost/x` → local;
/// `file://server/share/x` → UNC `//server/share/x`; `file://C:/x` (non-standard) → `C:/x`.
/// Non-URL input is returned as is (NOT percent-decoded — a path with a literal `%` is not mangled).
/// This is the ONE normalizer for every path entering the ABI (the C++ host must pass paths raw):
/// both I/O entry points ([`exv_session_open`], [`exv_write_file`]) go through it.
fn url_to_path(raw: &str) -> Result<String, String> {
    let Some(rest) = raw.get(..7).filter(|p| p.eq_ignore_ascii_case("file://")).map(|_| &raw[7..]) else {
        return Ok(raw.to_string());
    };
    let path = if let Some(p) = rest.strip_prefix('/') {
        // Empty authority: `/C:/x` is a drive path, anything else stays rooted (`/home/x`, `//unc`).
        if is_drive(p) { p.to_string() } else { rest.to_string() }
    } else {
        let (host, tail) = rest.split_once('/').unwrap_or((rest, ""));
        if host.eq_ignore_ascii_case("localhost") {
            if is_drive(tail) { tail.to_string() } else { format!("/{tail}") }
        } else if is_drive(rest) {
            rest.to_string()
        } else {
            format!("//{rest}")
        }
    };
    percent_decode(&path)
}

/// Borrow a NUL-terminated UTF-8 `path` argument and normalize it ([`url_to_path`]).
///
/// # Safety
/// `path` is a valid NUL-terminated C string.
unsafe fn path_arg(path: *const c_char, who: &str) -> Result<String, i32> {
    let raw = unsafe { CStr::from_ptr(path) }
        .to_str()
        .map_err(|_| fail(ExvStatus::InvalidArgument, format!("{who}: path is not UTF-8")))?;
    url_to_path(raw).map_err(|e| fail(ExvStatus::InvalidArgument, format!("{who}: {e}")))
}

/// Write `len` bytes from `data` to the file `utf8_path`. Unicode-correct: **Rust owns
/// the file I/O in both directions** (symmetric with [`exv_session_open`]). This works around the bug of the narrow
/// `fopen` on Windows, which interprets the path as ACP rather than UTF-8 → non-ASCII paths (Cyrillic/CJK
/// in the user/project name) would break when written from C++. A `file:` URL is normalized ([`url_to_path`]).
/// The write is atomic: a sibling temp file is renamed over the target, so a crash/ENOSPC mid-write
/// never destroys the previous file. (The rename fails on Windows while another process still has the
/// target memory-mapped — the host must release its session first.)
///
/// # Safety
/// `path` is NUL-terminated UTF-8; `data` points to `len` readable bytes (`len==0` → `data` may be null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_write_file(path: *const c_char, data: *const u8, len: usize) -> i32 {
    guard("exv_write_file", || {
        if path.is_null() || (data.is_null() && len != 0) {
            return fail(ExvStatus::InvalidArgument, "exv_write_file: null argument");
        }
        let path = match unsafe { path_arg(path, "exv_write_file") } {
            Ok(p) => p,
            Err(code) => return code,
        };
        let bytes = if len == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(data, len) } };
        match fsio::write_atomic(std::path::Path::new(&path), bytes) {
            Ok(()) => okay(),
            Err(e) => fail(ExvStatus::Internal, e.to_string()),
        }
    })
}

// ───────────────────────────── session (decode) ─────────────────────────────

/// Opaque decode session over one `.exv` (a still is just a 1-frame clip). **Lazy**: only the seek
/// index is parsed at open; each frame is decoded ON DEMAND in [`exv_session_decode`]. (Decoding every
/// frame at open is what froze After Effects on large clips: a blocking multi-minute decode + the whole
/// clip uncompressed in RAM.) The source is immutable after open; the only mutable state is the process
/// [`frame_cache`] (keyed by `id`), so concurrent [`exv_session_decode`] calls stay safe.
pub struct ExvSession {
    /// Compressed container: read-through below 64 MiB, mapped above (see `fsio::open_ro`).
    data: fsio::Src,
    index: exv_video::VideoIndex,
    /// Process-unique clip id ([`exv_video::GopCache::open_clip`]): the first half of this session's keys in [`frame_cache`].
    id: u64,
}

/// The ONE decoded-frame cache of the process ([`exv_video::GopCache`]), keyed `(session id, decode index, decoded
/// channels)`. The channel set is part of the key because a frame decodes ONLY the channels its request reads (the
/// fixed RGBA pick, or the host's channel matrix): a request for another set is another entry (decoded once, then
/// cached), never a frame with missing planes. AE keeps one routing per footage item, so in practice one set per session.
///
/// Why one global cache: After Effects re-requests the CURRENT frame on every redraw (pan/zoom/param tweak), so a
/// lazy per-call decode would re-decode it each time; a host opens many footage items, and per-session caps would SUM
/// past RAM; and it renders SEQUENTIALLY, where a stateless `decode_frame` re-decodes the whole P/B chain from the
/// GOP's I-frame per frame (`O(gop^2)`): the cache holds the references, so each frame is decoded once. One byte-LRU
/// over one budget bounds the whole process, loads are single-flight and reserve the frame's bytes BEFORE decoding.
/// The budget is `exv_tile::Budget::auto(None)` (half the RAM, capped by what is free, from the OS): no invented
/// constant; when the OS cannot answer the error is surfaced to the host, never replaced by a guess.
fn frame_cache() -> Result<&'static exv_video::GopCache, exv_tile::BudgetError> {
    Ok(&pools()?.0)
}

/// The ONE plane recycler of the process, shared by every session (`VideoIndex::with_pool`): a frame evicted from
/// [`frame_cache`] shelves its planes and the next decode of the same geometry takes them (no allocation, zero-fill or
/// page faults; the host process runs on the System allocator, where a fresh 4k frame costs ~30% of its decode).
/// Idle cap = the frame budget / [`exv_tile::budget::POOL_DIV`] (the decode-pool share the VRAM plan uses too): room for the
/// frames in flight, small beside the resident frames.
fn plane_pool() -> Result<&'static exv_tile::PlanePool, exv_tile::BudgetError> {
    Ok(&pools()?.1)
}

/// The decode ceiling of every session: the SAME process budget the frame cache is capped by, so an untrusted file
/// (a tiny index forging a huge frame) is refused by the budget, not by the 8 GiB `DecodeLimits` default.
fn limits() -> Result<DecodeLimits, exv_tile::BudgetError> {
    Ok(pools()?.2)
}

/// Parse an untrusted container index under `limits` and share `pool`: the ONE place a session index is built, so no
/// open path can forget the budget (`with_limits` first: it also re-syncs the index's default pool cap, then `with_pool`
/// replaces that pool by the shared one).
fn parse_idx(data: &[u8], limits: DecodeLimits, pool: exv_tile::PlanePool) -> Result<exv_video::VideoIndex, exv_format::DecodeError> {
    exv_video::VideoIndex::parse(data).map(|i| i.with_limits(limits).with_pool(pool))
}

/// The frame cache, the plane pool and the decode limits, built once from one OS-derived budget.
fn pools() -> Result<&'static (exv_video::GopCache, exv_tile::PlanePool, DecodeLimits), exv_tile::BudgetError> {
    static POOLS: OnceLock<Result<(exv_video::GopCache, exv_tile::PlanePool, DecodeLimits), exv_tile::BudgetError>> = OnceLock::new();
    POOLS
        .get_or_init(|| {
            let budget = exv_tile::Budget::auto(None)?;
            let idle = exv_tile::Budget::new(budget.cap() / exv_tile::budget::POOL_DIV);
            let limits = budget.limits();
            Ok((exv_video::GopCache::new(budget), exv_tile::PlanePool::new(idle), limits))
        })
        .as_ref()
        .map_err(|e| *e)
}

impl Drop for ExvSession {
    /// Free this session's cached frames now instead of letting them age out of the shared pool.
    fn drop(&mut self) {
        if let Ok(cache) = frame_cache() {
            cache.forget(self.id);
        }
    }
}

/// Stream info (mirror of `ExvStreamInfo`). Geometry is that of the first frame.
#[repr(C)]
pub struct ExvStreamInfo {
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
    pub num_channels: u32,
}

/// Geometry of frame `idx` (mirror of `ExvFrameInfo`).
#[repr(C)]
pub struct ExvFrameInfo {
    pub width: u32,
    pub height: u32,
    pub num_channels: u32,
}

/// Find a channel index by name (case-insensitive).
fn find_ch(header: &Header, name: &str) -> Option<usize> {
    header.channels.iter().position(|c| c.name.eq_ignore_ascii_case(name))
}


/// Open an `.exv` container by path (a still is a 1-frame `.exv`). The magic is verified; any other
/// file is `INVALID_FORMAT`. On success writes the handle into `*out`; free with [`exv_session_close`].
///
/// # Safety
/// `path` is a NUL-terminated UTF-8 path; `out` is a valid `*mut *mut`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_open(path: *const c_char, out: *mut *mut ExvSession) -> i32 {
    guard("exv_session_open", || {
        if path.is_null() || out.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_session_open: null argument");
        }
        unsafe { *out = std::ptr::null_mut() };
        // After Effects/hosts may pass a file:// URL (percent-encoded) — normalized here, nowhere else.
        let path = match unsafe { path_arg(path, "exv_session_open") } {
            Ok(p) => p,
            Err(code) => return code,
        };
        // LAZY open: read only the MAGIC + the seek index; frames decode on demand in
        // `exv_session_decode` (the After-Effects hang fix — see [`ExvSession`]).
        let data = match fsio::open_ro(std::path::Path::new(&path)) {
            Ok(m) => m,
            Err(e) => return fail(ExvStatus::Internal, e.to_string()),
        };
        let magic = match data.first_chunk::<4>() {
            Some(m) => u32::from_le_bytes(*m),
            None => return fail(ExvStatus::InvalidFormat, "file too short"),
        };
        if magic != exv_video::MAGIC {
            return fail(ExvStatus::InvalidFormat, format!("unknown magic {magic:#010x}"));
        }
        let (pool, limits, id) = match (plane_pool(), limits(), frame_cache()) {
            (Ok(p), Ok(l), Ok(c)) => (p.clone(), l, c.open_clip()),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return fail(ExvStatus::Internal, e.to_string()),
        };
        let index = match parse_idx(&data, limits, pool) {
            Ok(index) => index,
            // A well-formed file whose codec cannot be opened (unknown / vendor / not built) is UNSUPPORTED, not corrupt.
            Err(e @ exv_format::DecodeError::Codec(_)) => return fail(ExvStatus::Unsupported, format!("VideoIndex::parse: {e}")),
            Err(e) => return fail(ExvStatus::InvalidFormat, format!("VideoIndex::parse: {e}")),
        };
        let handle = Box::into_raw(Box::new(ExvSession { data, index, id }));
        unsafe { *out = handle };
        okay()
    })
}

/// Stream info (frame count + geometry of the first frame).
/// # Safety
/// `s` is a valid session handle; `out` is non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_info(s: *const ExvSession, out: *mut ExvStreamInfo) -> i32 {
    guard("exv_session_info", || {
        let Some(s) = (unsafe { s.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_info: null session");
        };
        if out.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_session_info: out is null");
        }
        // Geometry WITHOUT decoding: from the video index's header.
        unsafe {
            *out = ExvStreamInfo {
                frame_count: s.index.frame_count() as u32,
                width: s.index.header.width,
                height: s.index.header.height,
                num_channels: s.index.header.channels.len() as u32,
            };
        }
        okay()
    })
}

/// Geometry of frame `idx`.
/// # Safety
/// `s` is a valid session handle; `out` is non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_frame_info(s: *const ExvSession, idx: u32, out: *mut ExvFrameInfo) -> i32 {
    guard("exv_session_frame_info", || {
        let Some(s) = (unsafe { s.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_frame_info: null session");
        };
        if out.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_session_frame_info: out is null");
        }
        // All EXV frames in a container share geometry (v1) → geometry comes from the header/index
        // WITHOUT decoding. Bounds-check `idx` against the frame count.
        if idx as usize >= s.index.frame_count() {
            return fail(ExvStatus::InvalidArgument, format!("frame index {idx} out of range"));
        }
        unsafe {
            *out = ExvFrameInfo {
                width: s.index.header.width,
                height: s.index.header.height,
                num_channels: s.index.header.channels.len() as u32,
            };
        }
        okay()
    })
}

// ── Self-describing clip metadata (fps / alpha / colour / PAR / timecode) ─────────────────────

/// Flag bits of [`ExvMeta::flags`]: which optional fields are present.
#[repr(u32)]
pub enum ExvMetaFlag {
    Fps = 1,
    /// Pixel aspect ratio.
    Par = 2,
    /// Frames since 00:00:00:00.
    Timecode = 4,
}

/// Internal `u32` views of [`ExvMetaFlag`] (kept out of the header).
mod metaf {
    use super::ExvMetaFlag;
    pub(crate) const FPS: u32 = ExvMetaFlag::Fps as u32;
    pub(crate) const PAR: u32 = ExvMetaFlag::Par as u32;
    pub(crate) const TIMECODE: u32 = ExvMetaFlag::Timecode as u32;
    /// Every defined bit; a set bit outside this mask is rejected (struct from a newer host).
    pub(crate) const ALL: u32 = FPS | PAR | TIMECODE;
}

/// Versioned mirror of [`Meta`]. The caller sets `size = sizeof(ExvMeta)`; a mismatch is
/// `INVALID_ARGUMENT` (so a struct that grows later cannot be over-read/over-written by an older peer).
/// `primaries/transfer/range/alpha` are `0 = unspecified` (alpha: 1 straight, 2 premultiplied).
#[repr(C)]
pub struct ExvMeta {
    pub size: u32,
    pub flags: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub par_num: u32,
    pub par_den: u32,
    pub timecode: u32,
    pub primaries: u8,
    pub transfer: u8,
    pub range: u8,
    pub alpha: u8,
}

/// Validate a host [`ExvMeta`] and convert to [`Meta`].
fn meta_in(m: &ExvMeta) -> Result<Meta, String> {
    if m.size as usize != size_of::<ExvMeta>() {
        return Err(format!("ExvMeta.size {} != {}", m.size, size_of::<ExvMeta>()));
    }
    if m.flags & !metaf::ALL != 0 {
        return Err(format!("ExvMeta.flags {:#x} has unknown bits", m.flags));
    }
    if m.alpha > 2 {
        return Err(format!("ExvMeta.alpha {} (0..=2)", m.alpha));
    }
    let ratio = |on: bool, n: u32, d: u32, what: &str| match (on, n != 0 && d != 0) {
        (false, _) => Ok(None),
        (true, true) => Ok(Some((n, d))),
        (true, false) => Err(format!("ExvMeta.{what} {n}/{d} has a zero term")),
    };
    Ok(Meta {
        fps: ratio(m.flags & metaf::FPS != 0, m.fps_num, m.fps_den, "fps")?,
        pixel_aspect: ratio(m.flags & metaf::PAR != 0, m.par_num, m.par_den, "par")?,
        timecode: (m.flags & metaf::TIMECODE != 0).then_some(m.timecode),
        primaries: m.primaries,
        transfer: m.transfer,
        range: m.range,
        alpha: m.alpha,
        exr_headers: Vec::new(),
    })
}

/// Convert [`Meta`] to the ABI struct (absent = flag clear, zero fields).
fn meta_out(m: &Meta) -> ExvMeta {
    let (fps_num, fps_den) = m.fps.unwrap_or((0, 0));
    let (par_num, par_den) = m.pixel_aspect.unwrap_or((0, 0));
    ExvMeta {
        size: size_of::<ExvMeta>() as u32,
        flags: (u32::from(m.fps.is_some()) * metaf::FPS)
            | (u32::from(m.pixel_aspect.is_some()) * metaf::PAR)
            | (u32::from(m.timecode.is_some()) * metaf::TIMECODE),
        fps_num,
        fps_den,
        par_num,
        par_den,
        timecode: m.timecode.unwrap_or(0),
        primaries: m.primaries,
        transfer: m.transfer,
        range: m.range,
        alpha: m.alpha,
    }
}

/// The clip's authored metadata (fps / PAR / alpha interpretation / colour / timecode). A file
/// without metadata yields `flags == 0` and zeros (still `OK`). Caller sets `out->size`.
/// # Safety
/// `s` is a valid session handle; `out` points to a writable `ExvMeta` with `size` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_meta(s: *const ExvSession, out: *mut ExvMeta) -> i32 {
    guard("exv_session_meta", || {
        let Some(s) = (unsafe { s.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_meta: null session");
        };
        let Some(out) = (unsafe { out.as_mut() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_meta: out is null");
        };
        if out.size as usize != size_of::<ExvMeta>() {
            return fail(ExvStatus::InvalidArgument, format!("ExvMeta.size {} != {}", out.size, size_of::<ExvMeta>()));
        }
        *out = meta_out(&s.index.meta().cloned().unwrap_or_default());
        okay()
    })
}

/// Decode frame `idx` into `dst` with format `fmt` (RGBA_F32/RGBA_F16). `cap` is the capacity in
/// **bytes**, `stride` is the byte row step (0 = tightly packed). `*written` ← bytes written.
///
/// # Safety
/// `s` is valid; `dst` points to `cap` writable bytes; `written` is non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_decode(
    s: *const ExvSession,
    idx: u32,
    fmt: u32,
    dst: *mut u8,
    cap: usize,
    stride: u32,
    written: *mut usize,
) -> i32 {
    guard("exv_session_decode", || {
        let Some(s) = (unsafe { s.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_decode: null session");
        };
        if dst.is_null() || written.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_session_decode: null buffer");
        }
        // Fixed R/G/B/A routing: the channel pick is `Image::rgba_slots` (the SSOT shared with the viewer and the CPU
        // RGBA assembly: names, else Y / a lone channel as grey, else the first three); no alpha = opaque. Shares the `write_rgba` pixel path with the mapped channel-matrix
        // decode ([`exv_session_decode_mapped_rgbaf32`]); the frame decodes ON DEMAND via
        // `with_frame` (the AE-hang fix — O(one frame), not O(all frames)).
        let all: Vec<usize> = (0..s.channels().len()).collect();
        let slots = RgbaPick::pick(&s.index.header, &all).slots();
        let sel = |k: usize, absent: f32| slots[k].map_or(ChanSel::Fill(absent), ChanSel::Plane);
        let sels = [sel(0, 0.0), sel(1, 0.0), sel(2, 0.0), sel(3, 1.0)];
        s.with_frame(idx, &sels, |frame| unsafe { write_rgba(frame, &sels, fmt, dst, cap, stride, written) })
    })
}

// ── Channel enumeration + name-mapped RGBA decode (After-Effects channel matrix) ──────────────
//
// EXV is single-part in the AE path, so these mirror the proven `codec-ae` channel-router ABI
// (`codec_session_channel_count/_name/_decode_mapped_rgbaf32`) but WITHOUT `part_index`. They let a
// host present a channel MATRIX — route any named EXV channel (or the `<none>`/`<white>` sentinels)
// to each of R/G/B/A — instead of the fixed R/G/B/A-by-name mapping of `exv_session_decode`.

/// One RGBA-output source: a stored channel plane (by index), or a constant fill value.
enum ChanSel {
    Plane(usize),
    Fill(f32),
}

/// Resolve a channel name to a [`ChanSel`] against `header`. Sentinels: `<none>`/empty/absent → `0.0`,
/// `<white>` → `1.0`; a real name → its plane; an UNKNOWN name → `0.0` (treated as `<none>`, so a
/// stale map that names a now-absent channel degrades gracefully instead of erroring).
fn resolve_sel(header: &Header, name: Option<&str>) -> ChanSel {
    match name {
        Some(n) if n.eq_ignore_ascii_case("<white>") => ChanSel::Fill(1.0),
        Some(n) if !n.is_empty() && !n.eq_ignore_ascii_case("<none>") => {
            find_ch(header, n).map(ChanSel::Plane).unwrap_or(ChanSel::Fill(0.0))
        }
        _ => ChanSel::Fill(0.0),
    }
}

/// Write `frame` as interleaved RGBA (`fmt` = `RGBA_F32`/`RGBA_F16`) into `dst`, each output channel
/// (R,G,B,A = `sels[0..4]`) sourced from a plane sample or a constant fill. `cap`/`stride`/`written`
/// match [`exv_session_decode`]. Returns a `ExvStatus` i32. Every access is `checked_*`/`.get()` — no
/// panic across the FFI boundary (a panic there aborts the After Effects host). This is the shared
/// pixel writer for both the fixed-RGBA [`exv_session_decode`] and the mapped channel-matrix decode.
///
/// # Safety
/// `dst` points to `cap` writable bytes; `written` is non-null.
unsafe fn write_rgba(
    frame: &exv_video::Frame,
    sels: &[ChanSel; 4],
    fmt: u32,
    dst: *mut u8,
    cap: usize,
    stride: u32,
    written: *mut usize,
) -> i32 {
    let bytes_per_px = match fmt {
        pixfmt::RGBA_F32 => 16usize,
        pixfmt::RGBA_F16 => 8usize,
        other => return fail(ExvStatus::Unsupported, format!("pixel format {other}")),
    };
    let (w, h) = (frame.header.width as usize, frame.header.height as usize);
    // `checked_mul` everywhere — absurd (type-valid) geometry yields a status, not a wrap → OOB write.
    let Some(row_bytes) = w.checked_mul(bytes_per_px) else {
        return fail(ExvStatus::InvalidArgument, format!("row size overflow: {w} * {bytes_per_px}"));
    };
    let row_stride = if stride == 0 { row_bytes } else { stride as usize };
    if row_stride < row_bytes {
        return fail(ExvStatus::InvalidArgument, format!("stride {row_stride} < row {row_bytes}"));
    }
    let Some(need) = row_stride.checked_mul(h) else {
        return fail(ExvStatus::InvalidArgument, format!("buffer size overflow: {row_stride} * {h}"));
    };
    if cap < need {
        return fail(ExvStatus::InvalidArgument, format!("buffer too small: need {need}, got {cap}"));
    }
    // Resolve each selector ONCE (out of the pixel loop): a decoded plane + its dtype, or a fill. `with_frame` decodes
    // exactly the selected planes, so a selected plane that is absent is a decoder bug: a status, never a black guess.
    let mut resolved: [Result<(&[u8], Dtype), f32>; 4] = [Err(0.0); 4];
    for (r, sel) in resolved.iter_mut().zip(sels) {
        *r = match sel {
            ChanSel::Plane(ci) => match (frame.header.channels.get(*ci), frame.planes.get(*ci)) {
                (Some(c), Some(p)) => Ok((p, c.dtype)),
                _ => return fail(ExvStatus::Internal, format!("channel {ci} was not decoded")),
            },
            ChanSel::Fill(v) => Err(*v),
        };
    }
    let dstv = unsafe { std::slice::from_raw_parts_mut(dst, need) };
    // Output width of one component and its writer, chosen once (`fmt` is validated above).
    let (csz, put): (usize, fn(&mut [u8], f32)) = if fmt == pixfmt::RGBA_F32 {
        (4, |o, v| o.copy_from_slice(&v.to_le_bytes()))
    } else {
        (2, |o, v| o.copy_from_slice(&f32_to_half(v).to_le_bytes()))
    };
    // Rows are independent -> one rayon task per row (AE-8: the scalar loop dominated big-frame decode).
    // `need == row_stride * h`, so every chunk is a whole row; the stride padding is left untouched.
    dstv.par_chunks_mut(row_stride.max(1)).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            for (k, r) in resolved.iter().enumerate() {
                let v = match r {
                    Ok((p, dt)) => sample_f32(p, *dt, i),
                    Err(f) => *f,
                };
                let o = x * bytes_per_px + k * csz;
                if let Some(cell) = row.get_mut(o..o + csz) {
                    put(cell, v);
                }
            }
        }
    });
    unsafe { *written = need };
    okay()
}

impl ExvSession {
    /// The container's channel list (single part), read WITHOUT decoding a frame.
    fn channels(&self) -> &[Channel] {
        &self.index.header.channels
    }

    /// Run `f` with frame `idx`, decoded for exactly the planes `sels` read, and return its status. The frame comes from
    /// [`frame_cache`] (keyed by that channel set) and is decoded ON DEMAND on a miss (random access; unselected channels
    /// are never decoded: Z / ids are most of a multi-AOV frame); a range / budget / decode failure is reported through
    /// the error channel and its status returned, so callers just pass the code on. A checksum mismatch (corrupt
    /// storage) is `INVALID_FORMAT` with the message naming the frame / tile / channel.
    fn with_frame(&self, idx: u32, sels: &[ChanSel; 4], f: impl FnOnce(&exv_video::Frame) -> i32) -> i32 {
        if idx as usize >= self.index.frame_count() {
            return fail(ExvStatus::InvalidArgument, format!("frame index {idx} out of range"));
        }
        let mut chans: Vec<usize> = sels.iter().filter_map(|s| if let ChanSel::Plane(ci) = s { Some(*ci) } else { None }).collect();
        chans.sort_unstable();
        chans.dedup();
        if chans.is_empty() {
            // Only fills (`<none>` / `<white>`): nothing to decode.
            return f(&exv_video::Frame { header: self.index.header.clone(), planes: exv_video::Planes::default() });
        }
        let cache = match frame_cache() {
            Ok(c) => c,
            Err(e) => return fail(ExvStatus::Internal, e.to_string()),
        };
        // The lease pins the frame while `f` reads it.
        match cache.get(self.id, &self.data, &self.index, idx as usize, exv_video::Chans::Only(&chans), &exv_tile::Cancel::new()) {
            Ok(frame) => f(&frame),
            Err(e @ exv_video::CacheError::OverBudget { .. }) => fail(ExvStatus::Internal, format!("frame {idx}: {e}")),
            // Storage corruption: the backing store failed, not the request.
            Err(e @ exv_video::CacheError::Decode(exv_format::DecodeError::Checksum(_))) => fail(ExvStatus::InvalidFormat, format!("frame {idx}: {e}")),
            Err(e) => fail(ExvStatus::InvalidArgument, format!("frame {idx} decode failed: {e}")),
        }
    }
}

/// Number of channels in the container's single part (no decode).
/// # Safety
/// `s` is a valid session handle; `out_count` is non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_channel_count(s: *const ExvSession, out_count: *mut u32) -> i32 {
    guard("exv_session_channel_count", || {
        let Some(s) = (unsafe { s.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_channel_count: null session");
        };
        if out_count.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_session_channel_count: out is null");
        }
        unsafe { *out_count = s.channels().len() as u32 };
        okay()
    })
}

/// Name (UTF-8, NUL-terminated) of channel `chan_index` into `out_name[out_cap]`; `*out_written` =
/// bytes written INCLUDING the NUL. A name that does not fit is NOT truncated (a truncated name
/// would match no channel and silently render black): the call fails with `INVALID_ARGUMENT` and
/// `*out_written` = the required size including the NUL, so the host can retry with a bigger buffer.
/// # Safety
/// `s` is valid; `out_name` points to `out_cap` writable bytes; `out_written` is non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_channel_name(
    s: *const ExvSession,
    chan_index: u32,
    out_name: *mut c_char,
    out_cap: usize,
    out_written: *mut usize,
) -> i32 {
    guard("exv_session_channel_name", || {
        let Some(s) = (unsafe { s.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_channel_name: null session");
        };
        if out_name.is_null() || out_written.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_session_channel_name: null buffer");
        }
        let Some(ch) = s.channels().get(chan_index as usize) else {
            return fail(ExvStatus::InvalidArgument, format!("channel index {chan_index} out of range"));
        };
        let bytes = ch.name.as_bytes();
        let need = bytes.len() + 1; // + NUL
        unsafe { *out_written = need };
        if out_cap < need {
            return fail(
                ExvStatus::InvalidArgument,
                format!("channel {chan_index} name needs {need} bytes, buffer has {out_cap}"),
            );
        }
        let dstv = unsafe { std::slice::from_raw_parts_mut(out_name as *mut u8, need) };
        dstv[..bytes.len()].copy_from_slice(bytes);
        dstv[bytes.len()] = 0;
        okay()
    })
}

/// Decode frame `idx` routing NAMED channels into interleaved RGBA (`fmt`). Each of `map_r/g/b/a` is a
/// channel name, or a sentinel: `<none>`/empty/NULL → `0.0`, `<white>` → `1.0`; an unknown name → `0.0`.
/// This is the channel-matrix decode (the host picks the routing); [`exv_session_decode`] is the fixed
/// R/G/B/A shortcut. `cap`/`stride`/`written` semantics match [`exv_session_decode`]. The
/// frame is decoded on demand.
/// # Safety
/// `s` valid; `dst` points to `cap` writable bytes; `written` non-null; `map_*` are NUL-terminated or NULL.
#[allow(clippy::too_many_arguments)] // irreducible: 4 channel names + the standard decode-into args.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_decode_mapped_rgbaf32(
    s: *const ExvSession,
    idx: u32,
    fmt: u32,
    dst: *mut u8,
    cap: usize,
    stride: u32,
    map_r: *const c_char,
    map_g: *const c_char,
    map_b: *const c_char,
    map_a: *const c_char,
    written: *mut usize,
) -> i32 {
    guard("exv_session_decode_mapped_rgbaf32", || {
        let Some(s) = (unsafe { s.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_session_decode_mapped: null session");
        };
        if dst.is_null() || written.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_session_decode_mapped: null buffer");
        }
        // Borrow each map name (NULL → None → `<none>`; non-UTF-8 → None → `<none>`).
        let name = |p: *const c_char| -> Option<String> {
            if p.is_null() {
                None
            } else {
                unsafe { CStr::from_ptr(p) }.to_str().ok().map(str::to_owned)
            }
        };
        let (nr, ng, nb, na) = (name(map_r), name(map_g), name(map_b), name(map_a));
        let h = &s.index.header;
        let sels = [resolve_sel(h, nr.as_deref()), resolve_sel(h, ng.as_deref()), resolve_sel(h, nb.as_deref()), resolve_sel(h, na.as_deref())];
        s.with_frame(idx, &sels, |frame| unsafe { write_rgba(frame, &sels, fmt, dst, cap, stride, written) })
    })
}

/// Close a session from [`exv_session_open`].
/// # Safety
/// `s` is a handle from `exv_session_open` (or null), not used afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_session_close(s: *mut ExvSession) -> i32 {
    guard("exv_session_close", || {
        if !s.is_null() {
            drop(unsafe { Box::from_raw(s) });
        }
        okay()
    })
}

// ───────────────────────────── encode ─────────────────────────────

/// Encode parameters (mirror of `ExvEncParams`). The RGBA path always writes 4 channels R/G/B/A.
/// `dtype_code`: 0=F16,1=F32,2=Bf16,3=U8,4=U16,5=U32. `codec_id` + `codec_mode` + `q` (see [`profile_of`]; the
/// `EXV_Q_*_MAX` constants are the inclusive upper limits): 1=LOSSLESS_FAST and 2=WAVELET53 (`q` must be 0),
/// 3=WAVELET53_LOSSY (`q` 1..=`EXV_Q_WAVE_MAX`), 4=EXV_RT (`q` 0..=`EXV_Q_RT_MAX`), 6=HTJ2K (`codec_mode` 0: `q` = 0 ->
/// lossless, `q` 1..=`EXV_Q_HDR_MAX` -> lossy hdr codes; `codec_mode` 1: irreversible 9/7 for U8/U16, `q`
/// 1..=`EXV_Q_LDR_MAX`); any other id is `UNSUPPORTED`. `codec_mode` must be 0 for every other codec.
/// `tile_w`/`tile_h` = 0 -> the whole frame as a single tile.
#[repr(C)]
pub struct ExvEncParams {
    pub width: u32,
    pub height: u32,
    pub dtype_code: u32,
    pub codec_id: u32,
    pub q: u32,
    pub tile_w: u32,
    pub tile_h: u32,
    /// [`ExvCodecMode`] value; 0 = default.
    pub codec_mode: u32,
}

/// `q` as `u8` after the range check `lo..=hi` (a hard error naming the codec, never a clamp).
fn q_u8(q: u32, lo: u32, hi: u32, what: &str) -> Result<u8, i32> {
    match u8::try_from(q) {
        Ok(b) if (lo..=hi).contains(&q) => Ok(b),
        _ => Err(fail(ExvStatus::InvalidArgument, format!("{what} q={q} out of range {lo}..={hi}"))),
    }
}

/// Profile from the ABI `codec_id`/`codec_mode`/`q`. Hard errors, never a silent fallback: an unknown id is
/// `UNSUPPORTED`; a `q`/mode that the codec cannot honour (a lossless codec with `q != 0`, out-of-range
/// lossy `q`, a mode the codec does not have) is `INVALID_ARGUMENT`. HTJ2K `q == 0` is the lossless profile,
/// not "lossy at 0".
fn profile_of(codec_id: u32, codec_mode: u32, q: u32) -> Result<Profile, i32> {
    let id = CodecId(codec_id);
    let ldr = codec_mode == ExvCodecMode::Htj2kIrreversible97 as u32;
    if codec_mode != ExvCodecMode::Default as u32 && !(ldr && id == CodecId::HTJ2K) {
        return Err(fail(ExvStatus::InvalidArgument, format!("codec_mode {codec_mode} does not exist for codec {codec_id}")));
    }
    let params: Vec<u8> = match id {
        CodecId::LOSSLESS_FAST | CodecId::WAVELET53 if q != 0 => {
            return Err(fail(ExvStatus::InvalidArgument, format!("codec {codec_id} is lossless: q must be 0, got {q}")));
        }
        CodecId::LOSSLESS_FAST | CodecId::WAVELET53 => Vec::new(),
        CodecId::WAVELET53_LOSSY => vec![q_u8(q, 1, EXV_Q_WAVE_MAX, "lossy wavelet")?],
        CodecId::EXV_RT => Profile::exv_rt_dwt(RT_BLOCK, q_u8(q, 0, EXV_Q_RT_MAX, "EXV-RT")?).params(),
        CodecId::BC6H if q == 0 => Profile::bc6h_lossless().params(),
        CodecId::BC6H if q == 1 => Profile::bc6h_lossy().params(),
        CodecId::BC6H => return Err(fail(ExvStatus::InvalidArgument, format!("BC6H q={q} out of range 0..=1"))),
        CodecId::HTJ2K if ldr => Htj2kMode::Irrev97 { q: q_u8(q, 1, EXV_Q_LDR_MAX, "HTJ2K 9/7")? }.params(),
        CodecId::HTJ2K if q == 0 => Htj2kMode::Lossless.params(),
        CodecId::HTJ2K => Htj2kMode::HdrCodes { q: q_u8(q, 1, EXV_Q_HDR_MAX, "HTJ2K hdr")? }.params(),
        _ => return Err(fail(ExvStatus::Unsupported, format!("codec_id {codec_id}"))),
    };
    // A known id can still fail here: `NotBuilt` (HTJ2K without `htj2k-backend`) is `UNSUPPORTED`, params the
    // registry rejects (a bug in the q mapping above) are `INVALID_ARGUMENT`; the typed message names the id.
    registry::open(id, &params).map_err(|e| {
        let st = if matches!(e, OpenError::BadParams { .. }) { ExvStatus::InvalidArgument } else { ExvStatus::Unsupported };
        fail(st, e.to_string())
    })
}

/// Validated encode configuration derived from [`ExvEncParams`] (checked ONCE at open/encode time).
struct EncCfg {
    w: u32,
    h: u32,
    dtype: Dtype,
    prof: Profile,
    tile_w: u32,
    tile_h: u32,
    /// Required `rgba` length in `f32` (`w * h * 4`).
    need: usize,
}

/// Validate [`ExvEncParams`]: non-zero size, no size overflow, known dtype (checked BEFORE narrowing
/// to `u8` — 256 must not alias F16), valid codec/q.
fn cfg_of(p: &ExvEncParams) -> Result<EncCfg, i32> {
    if p.width == 0 || p.height == 0 {
        return Err(fail(ExvStatus::InvalidArgument, format!("empty frame {}x{}", p.width, p.height)));
    }
    let Some(need) = (p.width as usize).checked_mul(p.height as usize).and_then(|n| n.checked_mul(4)) else {
        return Err(fail(ExvStatus::InvalidArgument, format!("rgba size overflow: {}x{}", p.width, p.height)));
    };
    let Some(dtype) = u8::try_from(p.dtype_code).ok().and_then(Dtype::from_code) else {
        return Err(fail(ExvStatus::Unsupported, format!("dtype code {}", p.dtype_code)));
    };
    let prof = profile_of(p.codec_id, p.codec_mode, p.q)?;
    let tile_w = if p.tile_w == 0 { p.width } else { p.tile_w };
    let tile_h = if p.tile_h == 0 { p.height } else { p.tile_h };
    Ok(EncCfg { w: p.width, h: p.height, dtype, prof, tile_w, tile_h, need })
}

/// A host `float*` must be 4-byte aligned (documented in the header): forming a slice from a misaligned
/// pointer is undefined behaviour, so it is refused here as a normal argument error.
fn check_aligned(p: *const f32, who: &str) -> Result<(), i32> {
    if p.is_aligned() { Ok(()) } else { Err(fail(ExvStatus::InvalidArgument, format!("{who}: rgba is not 4-byte aligned"))) }
}

/// The 1-frame RGBA clip header for a validated config (the geometry/codec source of both the still
/// encoder and the streaming [`ExvVideoEncoder`]; `ClipWriter` re-stamps `frame_count`).
fn header_of(c: &EncCfg) -> Header {
    Header {
        width: c.w,
        height: c.h,
        frame_count: 1,
        tile_w: c.tile_w,
        tile_h: c.tile_h,
        codec_id: c.prof.codec_id().0,
        codec_params: c.prof.params(),
        channels: ["R", "G", "B", "A"].iter().map(|n| Channel { name: (*n).into(), dtype: c.dtype }).collect(),
    }
}

/// Build an `Image` (4 RGBA channels) from interleaved RGBA-f32. The length must match EXACTLY: an
/// oversize buffer means the host's frame differs from the declared geometry (wrong row pitch → garbage).
fn build_rgba_image(c: &EncCfg, rgba: &[f32]) -> Result<Image, i32> {
    if rgba.len() != c.need {
        return Err(fail(
            ExvStatus::InvalidArgument,
            format!("rgba length {} != {}x{}x4 = {}", rgba.len(), c.w, c.h, c.need),
        ));
    }
    let planes = (0..4).map(|k| pack_plane(rgba, k, c.w, c.h, c.dtype)).collect();
    Image::new(header_of(c), planes).map_err(|e| fail(ExvStatus::Internal, format!("Image::new: {e}")))
}

/// Encode a single interleaved RGBA-f32 frame (`[0,1]`+HDR) into a self-contained `.exv` (a 1-frame clip).
/// `*out` ← opaque [`ExvBuffer`] (null on error); free with [`exv_buffer_free`]. `rgba` must be
/// 4-byte aligned and exactly `w*h*4` floats.
///
/// # Safety
/// `params` is non-null; `rgba` points to `rgba_len` readable `f32`; `out` is non-null; `out_len` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_encode_rgba_f32(
    params: *const ExvEncParams,
    rgba: *const f32,
    rgba_len: usize,
    out: *mut *mut ExvBuffer,
    out_len: *mut usize,
) -> i32 {
    guard("exv_encode_rgba_f32", || {
        if out.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_encode_rgba_f32: out is null");
        }
        unsafe { clear_out(out, out_len) };
        let Some(p) = (unsafe { params.as_ref() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_encode_rgba_f32: null params");
        };
        if rgba.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_encode_rgba_f32: null rgba");
        }
        let cfg = match cfg_of(p) {
            Ok(c) => c,
            Err(code) => return code,
        };
        if let Err(code) = check_aligned(rgba, "exv_encode_rgba_f32") {
            return code;
        }
        let src = unsafe { std::slice::from_raw_parts(rgba, rgba_len) };
        let img = match build_rgba_image(&cfg, src) {
            Ok(i) => i,
            Err(code) => return code,
        };
        // A still IS a 1-frame `.exv` (the unified container).
        let container = match exv_video::write_still(&img, &cfg.prof) {
            Ok(c) => c,
            Err(e) => return fail(ExvStatus::Internal, format!("write_still: {e}")),
        };
        unsafe { emit(container, out, out_len) };
        okay()
    })
}

// ───────────────────────────── video encode (streaming) ─────────────────────────────
// open(path, params, meta) → add_rgba_f32(frame)… → finish(). Frames are encoded and appended to a
// temp file next to `path` AS THEY ARRIVE (memory = one frame, independent of the clip length); finish
// writes the index and atomically renames the temp over `path`. Freeing an unfinished encoder deletes
// the temp (abort): the destination is either the complete new clip or untouched, never partial.

/// Opaque streaming video encoder over `ClipWriter<OutFile>`.
pub struct ExvVideoEncoder {
    cfg: EncCfg,
    /// `None` once [`exv_venc_finish`] consumed it (a second finish is an error, not a double commit).
    w: Option<exv_video::ClipWriter<fsio::OutFile>>,
    /// Normalized destination, for error messages.
    dest: String,
}

/// Open a streaming video encoder that writes `path` (a `file:` URL is normalized) atomically: nothing
/// appears at `path` until [`exv_venc_finish`] succeeds. `gop` is the I-frame interval (>=1), `search`
/// the motion-search window (>=0, 0 = off). `meta` (fps / PAR / alpha / colour / timecode, `meta->size`
/// set) is recorded in the clip header; NULL = no metadata. Everything is validated here (path
/// writable, size, dtype, codec/q, meta), not at the first frame. B-frames are never produced.
/// # Safety
/// `path` is NUL-terminated UTF-8; `params` non-null; `meta` null or readable; `out` non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_venc_open(
    path: *const c_char,
    params: *const ExvEncParams,
    meta: *const ExvMeta,
    gop: u32,
    search: i32,
    out: *mut *mut ExvVideoEncoder,
) -> i32 {
    guard("exv_venc_open", || {
        if out.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_venc_open: out is null");
        }
        unsafe { *out = std::ptr::null_mut() };
        let Some(p) = (unsafe { params.as_ref() }).filter(|_| !path.is_null()) else {
            return fail(ExvStatus::InvalidArgument, "exv_venc_open: null path or params");
        };
        if gop == 0 || search < 0 {
            return fail(ExvStatus::InvalidArgument, format!("exv_venc_open: gop={gop} (>=1), search={search} (>=0)"));
        }
        let dest = match unsafe { path_arg(path, "exv_venc_open") } {
            Ok(p) => p,
            Err(code) => return code,
        };
        let cfg = match cfg_of(p) {
            Ok(c) => c,
            Err(code) => return code,
        };
        let meta = match unsafe { meta.as_ref() }.map(meta_in).transpose() {
            Ok(m) => m.unwrap_or_default(),
            Err(e) => return fail(ExvStatus::InvalidArgument, format!("exv_venc_open: {e}")),
        };
        let sink = match fsio::OutFile::create(std::path::Path::new(&dest)) {
            Ok(f) => f,
            Err(e) => return fail(ExvStatus::Internal, e.to_string()),
        };
        let opts = exv_video::WriteOpts { gop, search, b_frames: 0, meta };
        // On error `sink` drops inside `new` and deletes its temp file.
        let w = match exv_video::ClipWriter::new(sink, &header_of(&cfg), &cfg.prof, &opts) {
            Ok(w) => w,
            Err(e) => return fail(ExvStatus::Internal, format!("exv_venc_open '{dest}': {e}")),
        };
        unsafe { *out = Box::into_raw(Box::new(ExvVideoEncoder { cfg, w: Some(w), dest })) };
        okay()
    })
}

/// Encode and append one RGBA-f32 frame (order = the order of the calls). `rgba` must be 4-byte
/// aligned and exactly `w*h*4` floats of the geometry given at open. A failure poisons the encoder:
/// every later call fails and the host must [`exv_venc_free`] it (which deletes the partial output).
/// # Safety
/// `enc` is a valid handle; `rgba` points to `rgba_len` readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_venc_add_rgba_f32(enc: *mut ExvVideoEncoder, rgba: *const f32, rgba_len: usize) -> i32 {
    guard("exv_venc_add_rgba_f32", || {
        let Some(enc) = (unsafe { enc.as_mut() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_venc_add_rgba_f32: null encoder");
        };
        if rgba.is_null() {
            return fail(ExvStatus::InvalidArgument, "exv_venc_add_rgba_f32: null rgba");
        }
        let Some(w) = enc.w.as_mut() else {
            return fail(ExvStatus::InvalidArgument, "exv_venc_add_rgba_f32: encoder already finished");
        };
        if let Err(code) = check_aligned(rgba, "exv_venc_add_rgba_f32") {
            return code;
        }
        let src = unsafe { std::slice::from_raw_parts(rgba, rgba_len) };
        let img = match build_rgba_image(&enc.cfg, src) {
            Ok(i) => i,
            Err(code) => return code,
        };
        match w.push(img) {
            Ok(()) => okay(),
            Err(e) => fail(ExvStatus::Internal, format!("exv_venc_add_rgba_f32 '{}': {e}", enc.dest)),
        }
    })
}

/// Finish the clip: write the index, fsync and atomically rename the temp over the destination.
/// `*out_bytes` (null allowed) ← the file size. The encoder is NOT freed — call [`exv_venc_free`]
/// afterward (also after a failure: the temp is already gone, free only releases the handle). The host
/// must have released any session on the destination BEFORE this call (Windows cannot replace a
/// mapped file).
/// # Safety
/// `enc` is valid; `out_bytes` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_venc_finish(enc: *mut ExvVideoEncoder, out_bytes: *mut u64) -> i32 {
    guard("exv_venc_finish", || {
        if !out_bytes.is_null() {
            unsafe { *out_bytes = 0 };
        }
        let Some(enc) = (unsafe { enc.as_mut() }) else {
            return fail(ExvStatus::InvalidArgument, "exv_venc_finish: null encoder");
        };
        let Some(w) = enc.w.take() else {
            return fail(ExvStatus::InvalidArgument, "exv_venc_finish: encoder already finished");
        };
        let dest = &enc.dest;
        // Any error below drops the sink, which deletes the temp file.
        let (sink, total) = match w.finish() {
            Ok(r) => r,
            Err(e) => return fail(ExvStatus::Internal, format!("exv_venc_finish '{dest}': {e}")),
        };
        if let Err(e) = sink.commit() {
            return fail(ExvStatus::Internal, e.to_string());
        }
        if !out_bytes.is_null() {
            unsafe { *out_bytes = total };
        }
        okay()
    })
}

/// Free a video encoder from [`exv_venc_open`]. An encoder that was not successfully finished is
/// ABORTED: its temp file is deleted and the destination is left untouched.
/// # Safety
/// `enc` is a handle from `exv_venc_open` (or null), not used afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exv_venc_free(enc: *mut ExvVideoEncoder) {
    if !enc.is_null() {
        drop(unsafe { Box::from_raw(enc) });
    }
}

// ───────────────────────────── ABI layout pins ─────────────────────────────
// Compile-time guard for every `repr(C)` struct: a field added/reordered/resized fails the build here,
// and the same sizes are `static_assert`ed in the generated header (`cbindgen.toml` trailer), so the
// Rust and C++ sides cannot drift apart silently. Changing one means bumping `EXV_ABI_VERSION`.
const _: () = {
    assert!(size_of::<ExvVersion>() == 8);
    assert!(size_of::<ExvStreamInfo>() == 16);
    assert!(size_of::<ExvFrameInfo>() == 12);
    assert!(size_of::<ExvMeta>() == 32);
    assert!(std::mem::offset_of!(ExvMeta, primaries) == 28);
    assert!(size_of::<ExvEncParams>() == 32);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_params(w: u32, h: u32) -> ExvEncParams {
        ExvEncParams { width: w, height: h, dtype_code: 1, codec_id: 1, q: 0, tile_w: 0, tile_h: 0, codec_mode: 0 }
    }

    fn ramp_rgba(w: u32, h: u32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut v = vec![0f32; n * 4];
        for (i, px) in v.chunks_mut(4).enumerate() {
            px[0] = (i % 257) as f32 / 257.0;
            px[1] = (i % 131) as f32 / 131.0;
            px[2] = (i % 17) as f32 / 17.0;
            px[3] = 1.0;
        }
        v
    }

    /// exv_write_file writes bytes Unicode-correctly; they read back identically.
    #[test]
    fn capi_write_file_roundtrip() {
        let payload: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        let path = tmp("writefile_\u{43f}\u{440}\u{43e}\u{432}\u{435}\u{440}\u{43a}\u{430}.bin");
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let st = unsafe { exv_write_file(cpath.as_ptr(), payload.as_ptr(), payload.len()) };
        assert_eq!(st, ExvStatus::Ok as i32, "write: {:?}", unsafe { CStr::from_ptr(exv_last_error_message()) });
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        let _ = std::fs::remove_file(&path);
    }

    /// encode still (.exv) → open session → decode RGBA_F32 back (F32 dtype = lossless).
    #[test]
    fn capi_still_rgba_roundtrip() {
        let (w, h) = (40u32, 24u32);
        let rgba = ramp_rgba(w, h);
        let params = enc_params(w, h); // F32, LOSSLESS_FAST → exact roundtrip

        let mut buf: *mut ExvBuffer = std::ptr::null_mut();
        let mut blen = 0usize;
        let st = unsafe { exv_encode_rgba_f32(&params, rgba.as_ptr(), rgba.len(), &mut buf, &mut blen) };
        assert_eq!(st, ExvStatus::Ok as i32);
        let container = unsafe { std::slice::from_raw_parts(exv_buffer_data(buf), exv_buffer_len(buf)) }.to_vec();
        unsafe { exv_buffer_free(buf) };

        let path = tmp(&format!("still_{w}x{h}.exv"));
        std::fs::write(&path, &container).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();

        let mut sess: *mut ExvSession = std::ptr::null_mut();
        let st = unsafe { exv_session_open(cpath.as_ptr(), &mut sess) };
        assert_eq!(st, ExvStatus::Ok as i32, "open: {:?}", unsafe { CStr::from_ptr(exv_last_error_message()) });

        let mut info = ExvStreamInfo { frame_count: 0, width: 0, height: 0, num_channels: 0 };
        unsafe { exv_session_info(sess, &mut info) };
        assert_eq!((info.frame_count, info.width, info.height, info.num_channels), (1, w, h, 4));

        let mut back = vec![0u8; (w * h * 16) as usize];
        let mut written = 0usize;
        let st = unsafe { exv_session_decode(sess, 0, pixfmt::RGBA_F32, back.as_mut_ptr(), back.len(), 0, &mut written) };
        assert_eq!(st, ExvStatus::Ok as i32);
        assert_eq!(written, back.len());
        unsafe { exv_session_close(sess) };
        let _ = std::fs::remove_file(&path);

        let decoded = unsafe { std::slice::from_raw_parts(back.as_ptr() as *const f32, (w * h * 4) as usize) };
        for (a, b) in decoded.iter().zip(&rgba) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// The shared frame pool: 8 threads asking the same frame decode it ONCE (single-flight), a re-read is a hit,
    /// the cache is charged the frame's decoded bytes, and closing the session frees them.
    #[test]
    fn capi_frame_pool_single_flight_and_close_frees() {
        let (w, h) = (32u32, 16u32);
        let rgba = ramp_rgba(w, h);
        let params = enc_params(w, h);
        let mut buf: *mut ExvBuffer = std::ptr::null_mut();
        let mut blen = 0usize;
        assert_eq!(unsafe { exv_encode_rgba_f32(&params, rgba.as_ptr(), rgba.len(), &mut buf, &mut blen) }, ExvStatus::Ok as i32);
        let container = unsafe { std::slice::from_raw_parts(exv_buffer_data(buf), exv_buffer_len(buf)) }.to_vec();
        unsafe { exv_buffer_free(buf) };
        let path = tmp("pool.exv");
        std::fs::write(&path, &container).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, ExvStatus::Ok as i32);
        let s = unsafe { &*sess };
        let cache = frame_cache().expect("OS memory query");
        let id = s.id;
        // Every plane of the RGBA image, the channel set the fixed RGBA decode reads.
        let sels = [ChanSel::Plane(0), ChanSel::Plane(1), ChanSel::Plane(2), ChanSel::Plane(3)];
        let all = exv_video::Chans::Only(&[0, 1, 2, 3]);
        // A private index of the same bytes: `has` only reads geometry, and the session is gone at the end.
        let idx = exv_video::VideoIndex::parse(&container).expect("index");
        let mine = |c: &exv_video::GopCache| c.has(id, &idx, 0, all);
        assert!(!mine(cache));
        let hs: Vec<_> = (0..8)
            .map(|_| {
                let addr = sess as usize;
                std::thread::spawn(move || {
                    let s = unsafe { &*(addr as *const ExvSession) };
                    let sels = [ChanSel::Plane(0), ChanSel::Plane(1), ChanSel::Plane(2), ChanSel::Plane(3)];
                    s.with_frame(0, &sels, |_| ExvStatus::Ok as i32)
                })
            })
            .collect();
        for h in hs {
            assert_eq!(h.join().unwrap(), ExvStatus::Ok as i32);
        }
        assert!(mine(cache), "decoded frame is cached");
        assert_eq!(s.with_frame(9, &sels, |_| 0), ExvStatus::InvalidArgument as i32, "out of range");
        // A channel subset is its own entry and decodes ONLY its planes (the others are `None`, never zeros).
        let g = [ChanSel::Plane(1), ChanSel::Fill(0.0), ChanSel::Fill(0.0), ChanSel::Fill(1.0)];
        let got = s.with_frame(0, &g, |f| if f.planes.get(1).is_some() && f.planes.get(0).is_none() { 0 } else { 1 });
        assert_eq!(got, 0, "subset frame holds exactly the selected plane");
        assert!(cache.has(id, &idx, 0, exv_video::Chans::Only(&[1])));
        assert!(cache.stats().used >= (w * h * 16) as usize, "charged its decoded bytes");
        unsafe { exv_session_close(sess) };
        assert!(!mine(cache) && !cache.has(id, &idx, 0, exv_video::Chans::Only(&[1])), "close drops the session's frames");
        let _ = std::fs::remove_file(&path);
    }

    /// Strided decode: the first `row` bytes of each line match the tightly-packed decode.
    #[test]
    fn capi_decode_strided() {
        let (w, h) = (16u32, 8u32);
        let rgba = ramp_rgba(w, h);
        let params = enc_params(w, h);
        let mut buf: *mut ExvBuffer = std::ptr::null_mut();
        let mut blen = 0usize;
        assert_eq!(unsafe { exv_encode_rgba_f32(&params, rgba.as_ptr(), rgba.len(), &mut buf, &mut blen) }, ExvStatus::Ok as i32);
        let container = unsafe { std::slice::from_raw_parts(exv_buffer_data(buf), exv_buffer_len(buf)) }.to_vec();
        unsafe { exv_buffer_free(buf) };
        let path = tmp("strided.exv");
        std::fs::write(&path, &container).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, ExvStatus::Ok as i32);

        let row = (w * 16) as usize;
        let mut tight = vec![0u8; row * h as usize];
        let mut wr = 0usize;
        assert_eq!(unsafe { exv_session_decode(sess, 0, pixfmt::RGBA_F32, tight.as_mut_ptr(), tight.len(), 0, &mut wr) }, ExvStatus::Ok as i32);
        let stride = row + 64;
        let mut strided = vec![0u8; stride * h as usize];
        let mut wr2 = 0usize;
        assert_eq!(unsafe { exv_session_decode(sess, 0, pixfmt::RGBA_F32, strided.as_mut_ptr(), strided.len(), stride as u32, &mut wr2) }, ExvStatus::Ok as i32);
        assert_eq!(wr2, stride * h as usize);
        for r in 0..h as usize {
            assert_eq!(&strided[r * stride..r * stride + row], &tight[r * row..r * row + row], "row {r}");
        }
        unsafe { exv_session_close(sess) };
        let _ = std::fs::remove_file(&path);
    }

    /// Unique temp file path per process + test name: concurrent `cargo test` runs (and the parallel tests of one run)
    /// must never share a fixed file, or one run's cleanup deletes another's input.
    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("exv_capi_{}_{name}", std::process::id()))
    }

    /// Fresh empty scratch dir (so "no temp file left behind" can be asserted by listing it).
    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("exv_capi_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Open a streaming encoder on `path` (no meta); the tests below drive it by hand.
    fn venc_open(path: &std::path::Path, p: &ExvEncParams, meta: *const ExvMeta, gop: u32) -> (i32, *mut ExvVideoEncoder) {
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut enc: *mut ExvVideoEncoder = std::ptr::null_mut();
        (unsafe { exv_venc_open(cpath.as_ptr(), p, meta, gop, 0, &mut enc) }, enc)
    }

    /// video session: stream 3 frames straight to the file -> open -> 3 frames decode; the destination
    /// only appears at finish and no temp file survives.
    #[test]
    fn capi_video_roundtrip() {
        let (w, h) = (32u32, 16u32);
        let dir = scratch("video");
        let path = dir.join("clip.exv");
        let (st, enc) = venc_open(&path, &enc_params(w, h), std::ptr::null(), 2);
        assert_eq!(st, ExvStatus::Ok as i32, "{}", last_err());
        let rgba = ramp_rgba(w, h);
        for _ in 0..3 {
            assert_eq!(unsafe { exv_venc_add_rgba_f32(enc, rgba.as_ptr(), rgba.len()) }, ExvStatus::Ok as i32);
        }
        assert!(!path.exists(), "nothing at the destination before finish");
        let mut bytes = 0u64;
        assert_eq!(unsafe { exv_venc_finish(enc, &mut bytes) }, ExvStatus::Ok as i32, "{}", last_err());
        assert_eq!(bytes, std::fs::metadata(&path).unwrap().len());
        assert_eq!(unsafe { exv_venc_finish(enc, &mut bytes) }, ExvStatus::InvalidArgument as i32, "double finish");
        assert_eq!(unsafe { exv_venc_add_rgba_f32(enc, rgba.as_ptr(), rgba.len()) }, ExvStatus::InvalidArgument as i32);
        unsafe { exv_venc_free(enc) };
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "only the clip, no temp file");

        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, ExvStatus::Ok as i32);
        let mut info = ExvStreamInfo { frame_count: 0, width: 0, height: 0, num_channels: 0 };
        unsafe { exv_session_info(sess, &mut info) };
        assert_eq!(info.frame_count, 3);
        let mut back = vec![0u8; (w * h * 16) as usize];
        let mut wr = 0usize;
        assert_eq!(unsafe { exv_session_decode(sess, 2, pixfmt::RGBA_F32, back.as_mut_ptr(), back.len(), 0, &mut wr) }, ExvStatus::Ok as i32);
        unsafe { exv_session_close(sess) };
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Frames decoded (= `decode_one` spans) on THIS thread while `f` runs. A thread-local subscriber, so the other tests of
    /// the process (they share the global frame cache and run in parallel) cannot leak into the count; every decode of a
    /// session call runs on the calling thread (only its tiles fan out to rayon).
    fn decodes_here<R>(f: impl FnOnce() -> R) -> (R, usize) {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tracing_subscriber::layer::SubscriberExt;
        struct Count(Arc<AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Count {
            fn on_new_span(&self, a: &tracing::span::Attributes<'_>, _: &tracing::span::Id, _: tracing_subscriber::layer::Context<'_, S>) {
                if a.metadata().name() == "decode_one" {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let n = Arc::new(AtomicUsize::new(0));
        let r = tracing::subscriber::with_default(tracing_subscriber::registry().with(Count(n.clone())), f);
        (r, n.load(Ordering::Relaxed))
    }

    /// A sequential render of a long-GOP P clip (the After Effects shape) goes through the shared [`exv_video::GopCache`]:
    /// n frames cost n frame decodes (the stateless loop pays n (n + 1) / 2), every frame is resident after its decode, and
    /// the pixels equal the stateless chain decode of the same frame.
    #[test]
    fn capi_sequential_p_gop_reuses_references() {
        let (w, h, n) = (32u32, 16u32, 8u32);
        let dir = scratch("pgop");
        let path = dir.join("clip.exv");
        let (st, enc) = venc_open(&path, &enc_params(w, h), std::ptr::null(), n);
        assert_eq!(st, ExvStatus::Ok as i32, "{}", last_err());
        for k in 0..n {
            let rgba: Vec<f32> = ramp_rgba(w, h).iter().map(|v| v + k as f32 * 0.01).collect();
            assert_eq!(unsafe { exv_venc_add_rgba_f32(enc, rgba.as_ptr(), rgba.len()) }, ExvStatus::Ok as i32);
        }
        let mut bytes = 0u64;
        assert_eq!(unsafe { exv_venc_finish(enc, &mut bytes) }, ExvStatus::Ok as i32, "{}", last_err());
        unsafe { exv_venc_free(enc) };
        let data = std::fs::read(&path).unwrap();
        let idx = exv_video::VideoIndex::parse(&data).expect("index");
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, ExvStatus::Ok as i32);
        let (cache, id) = (frame_cache().expect("OS memory query"), unsafe { &*sess }.id);
        let all = exv_video::Chans::Only(&[0, 1, 2, 3]);
        let mut back = vec![0u8; (w * h * 16) as usize];
        let (_, stateless) = decodes_here(|| {
            for k in 0..n as usize {
                drop(exv_video::decode_frame(&data, &idx, k, all).expect("stateless"));
            }
        });
        assert_eq!(stateless, (n * (n + 1) / 2) as usize, "the stateless loop re-decodes the chain of every frame");
        let (_, cached) = decodes_here(|| {
            for k in 0..n {
                let mut wr = 0usize;
                assert_eq!(unsafe { exv_session_decode(sess, k, pixfmt::RGBA_F32, back.as_mut_ptr(), back.len(), 0, &mut wr) }, ExvStatus::Ok as i32, "{}", last_err());
                assert!(cache.has(id, &idx, k as usize, all), "frame {k} resident after its decode");
            }
        });
        assert_eq!(cached, n as usize, "each frame decoded once through the session");
        let want = exv_video::decode_frame(&data, &idx, (n - 1) as usize, all).expect("stateless");
        let last = unsafe { &*sess }.with_frame(n - 1, &[ChanSel::Plane(0), ChanSel::Plane(1), ChanSel::Plane(2), ChanSel::Plane(3)], |f| {
            i32::from((0..4).any(|c| f.planes.get(c) != want.planes.get(c)))
        });
        assert_eq!(last, 0, "cached frame == stateless chain decode");
        unsafe { exv_session_close(sess) };
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Abort paths: freeing an unfinished encoder, a failed add (poisoned writer), a finish with no frames and
    /// a bad open all leave the destination untouched (an existing file survives) and no temp file behind.
    #[test]
    fn capi_venc_abort_cleans_up() {
        let (w, h) = (16u32, 8u32);
        let dir = scratch("abort");
        let path = dir.join("keep.exv");
        std::fs::write(&path, b"previous").unwrap();
        let rgba = ramp_rgba(w, h);

        let (st, enc) = venc_open(&path, &enc_params(w, h), std::ptr::null(), 1);
        assert_eq!(st, 0, "{}", last_err());
        assert_eq!(unsafe { exv_venc_add_rgba_f32(enc, rgba.as_ptr(), rgba.len()) }, 0);
        unsafe { exv_venc_free(enc) }; // abort: never finished
        assert_eq!(std::fs::read(&path).unwrap(), b"previous");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "temp deleted on abort");

        let (_, enc) = venc_open(&path, &enc_params(w, h), std::ptr::null(), 1);
        let short = vec![0f32; rgba.len() - 4];
        assert_eq!(unsafe { exv_venc_add_rgba_f32(enc, short.as_ptr(), short.len()) }, ExvStatus::InvalidArgument as i32);
        assert_eq!(unsafe { exv_venc_finish(enc, std::ptr::null_mut()) }, ExvStatus::Internal as i32, "no frame was encoded");
        assert!(last_err().contains("exv_venc_finish"), "{}", last_err());
        unsafe { exv_venc_free(enc) };
        assert_eq!(std::fs::read(&path).unwrap(), b"previous");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "temp deleted after a failed finish");

        let (st, enc) = venc_open(&dir.join("no_such_dir").join("x.exv"), &enc_params(w, h), std::ptr::null(), 1);
        assert_eq!(st, ExvStatus::Internal as i32, "unwritable directory fails at open");
        assert!(enc.is_null());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The row-parallel pixel writer: F16 output with a padded stride equals the scalar reference
    /// (`f32_to_half` of each source sample), and the stride padding is left untouched.
    #[test]
    fn capi_decode_f16_strided_matches_reference() {
        let (w, h) = (37u32, 19u32); // odd sizes: no accidental alignment
        let rgba = ramp_rgba(w, h);
        let path = tmp("f16.exv");
        std::fs::write(&path, encode_still(&enc_params(w, h), &rgba)).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, 0);
        let (row, stride) = (w as usize * 8, w as usize * 8 + 24);
        let mut out = vec![0xABu8; stride * h as usize];
        let mut wr = 0usize;
        let fmt = ExvPixelFormat::RgbaF16 as u32;
        assert_eq!(unsafe { exv_session_decode(sess, 0, fmt, out.as_mut_ptr(), out.len(), stride as u32, &mut wr) }, 0, "{}", last_err());
        unsafe { exv_session_close(sess) };
        std::fs::remove_file(&path).unwrap();
        for y in 0..h as usize {
            for (i, v) in rgba[y * w as usize * 4..(y + 1) * w as usize * 4].iter().enumerate() {
                let o = y * stride + i * 2;
                assert_eq!(u16::from_le_bytes([out[o], out[o + 1]]), f32_to_half(*v), "y{y} i{i}");
            }
            assert!(out[y * stride + row..(y + 1) * stride].iter().all(|b| *b == 0xAB), "padding touched, row {y}");
        }
    }

    /// Session is read-only → concurrent decode from many threads is safe.
    #[test]
    fn capi_concurrent_decode() {
        let (w, h) = (48u32, 32u32);
        let rgba = ramp_rgba(w, h);
        let params = enc_params(w, h);
        let mut buf: *mut ExvBuffer = std::ptr::null_mut();
        let mut blen = 0usize;
        assert_eq!(unsafe { exv_encode_rgba_f32(&params, rgba.as_ptr(), rgba.len(), &mut buf, &mut blen) }, ExvStatus::Ok as i32);
        let container = unsafe { std::slice::from_raw_parts(exv_buffer_data(buf), exv_buffer_len(buf)) }.to_vec();
        unsafe { exv_buffer_free(buf) };
        let path = tmp("concurrent.exv");
        std::fs::write(&path, &container).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, ExvStatus::Ok as i32);

        let need = (w * h * 16) as usize;
        let addr = sess as usize; // raw ptr is not Send — pass the address
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(move || {
                        let s = addr as *const ExvSession;
                        let mut back = vec![0u8; need];
                        let mut written = 0usize;
                        let st = unsafe { exv_session_decode(s, 0, pixfmt::RGBA_F32, back.as_mut_ptr(), back.len(), 0, &mut written) };
                        (st, written)
                    })
                })
                .collect();
            for jh in handles {
                let (st, written) = jh.join().unwrap();
                assert_eq!(st, ExvStatus::Ok as i32);
                assert_eq!(written, need);
            }
        });
        unsafe { exv_session_close(sess) };
        let _ = std::fs::remove_file(&path);
    }

    /// Encode `rgba` as a still through the ABI and return the container bytes.
    fn encode_still(params: &ExvEncParams, rgba: &[f32]) -> Vec<u8> {
        let mut buf: *mut ExvBuffer = std::ptr::null_mut();
        let mut blen = 0usize;
        let st = unsafe { exv_encode_rgba_f32(params, rgba.as_ptr(), rgba.len(), &mut buf, &mut blen) };
        assert_eq!(st, ExvStatus::Ok as i32, "encode: {:?}", unsafe { CStr::from_ptr(exv_last_error_message()) });
        let v = unsafe { std::slice::from_raw_parts(exv_buffer_data(buf), exv_buffer_len(buf)) }.to_vec();
        unsafe { exv_buffer_free(buf) };
        v
    }

    fn last_err() -> String {
        unsafe { CStr::from_ptr(exv_last_error_message()) }.to_string_lossy().into_owned()
    }

    /// A panic inside an export body becomes `INTERNAL` + a message, never an unwind across FFI.
    #[test]
    fn guard_catches_panic() {
        let st = guard("exv_test", || panic!("boom"));
        assert_eq!(st, ExvStatus::Internal as i32);
        assert!(last_err().contains("panic in exv_test: boom"), "{}", last_err());
        assert_eq!(guard("exv_test", okay), 0);
    }

    /// `file:` URLs: ONE normalizer, percent-decoded, drive/UNC/POSIX aware; plain paths untouched.
    #[test]
    fn url_forms() {
        let ok = |s: &str| url_to_path(s).unwrap();
        assert_eq!(ok("file:///C:/My%20Dir/x.exv"), "C:/My Dir/x.exv");
        assert_eq!(ok("FILE:///c:/x.exv"), "c:/x.exv");
        assert_eq!(ok("file:///home/u/x.exv"), "/home/u/x.exv");
        assert_eq!(ok("file://localhost/C:/x.exv"), "C:/x.exv");
        assert_eq!(ok("file://localhost/home/x"), "/home/x");
        assert_eq!(ok("file://server/share/x.exv"), "//server/share/x.exv");
        assert_eq!(ok("file://C:/x.exv"), "C:/x.exv");
        assert_eq!(ok("file:///C:/%D0%BF%D1%80.exv"), "C:/\u{43f}\u{440}.exv");
        assert_eq!(ok("C:/100%real/x.exv"), "C:/100%real/x.exv");
        assert!(url_to_path("file:///C:/%FF.exv").is_err());
    }

    /// Codec/q validation: unknown id is UNSUPPORTED, bad q is INVALID_ARGUMENT, HTJ2K q=0 is LOSSLESS.
    #[test]
    fn codec_validation() {
        assert_eq!(profile_of(99, 0, 0).err(), Some(ExvStatus::Unsupported as i32));
        assert!(profile_of(4, 0, 4).is_ok(), "EXV_RT is exposed");
        assert_eq!(profile_of(1, 0, 3).err(), Some(ExvStatus::InvalidArgument as i32));
        assert_eq!(profile_of(3, 0, 0).err(), Some(ExvStatus::InvalidArgument as i32));
        assert_eq!(profile_of(3, 0, 63).err(), Some(ExvStatus::InvalidArgument as i32));
        assert_eq!(profile_of(3, 0, 256 + 8).err(), Some(ExvStatus::InvalidArgument as i32));
        assert!(profile_of(3, 0, 8).is_ok());
        if registry::open(CodecId::HTJ2K, &[]).is_ok() {
            assert!(profile_of(6, 0, 0).unwrap().is_lossless(), "HTJ2K q=0 must be lossless");
            assert_eq!(profile_of(6, 0, 12).unwrap().params(), vec![1, 12], "hdr codes");
            assert_eq!(profile_of(6, 1, 16).unwrap().params(), vec![2, 16], "irreversible 9/7");
            assert_eq!(profile_of(6, 0, EXV_Q_HDR_MAX + 1).err(), Some(ExvStatus::InvalidArgument as i32), "hdr q saturates at 16");
            // 9/7 q range is the registry's (1..=40); a mode on another codec / an unknown mode is InvalidArgument.
            for (id, mode, q) in [(6u32, 1u32, 0u32), (6, 1, 41), (6, 1, 300), (1, 1, 0), (3, 1, 4), (6, 2, 8)] {
                assert_eq!(profile_of(id, mode, q).err(), Some(ExvStatus::InvalidArgument as i32), "{id}/{mode}/{q}");
            }
        }
    }

    /// The published `EXV_Q_*_MAX` limits are the exact edges `profile_of` accepts (the C UI is built from them).
    #[test]
    fn q_limits_are_the_edges() {
        let bad = Some(ExvStatus::InvalidArgument as i32);
        // cbindgen needs literal values to emit the header consts, so these mirror `exv_codec`'s (the SSOT) and are pinned here.
        assert_eq!(EXV_Q_LDR_MAX, u32::from(exv_codec::LDR_Q_MAX), "pinned to the codec's constant");
        assert_eq!(EXV_Q_WAVE_MAX, u32::from(exv_codec::WAVE_Q_MAX), "pinned to the codec's constant");
        assert_eq!(EXV_Q_HDR_MAX, u32::from(exv_codec::HDR_Q_MAX), "pinned to the codec's constant");
        assert_eq!(EXV_Q_RT_MAX, u32::from(exv_codec::RT_Q_MAX), "pinned to the codec's constant");
        assert!(profile_of(3, 0, EXV_Q_WAVE_MAX).is_ok());
        assert_eq!(profile_of(3, 0, EXV_Q_WAVE_MAX + 1).err(), bad);
        for q in [0, EXV_Q_RT_MAX] {
            assert!(profile_of(4, 0, q).is_ok(), "rt q={q}");
        }
        assert_eq!(profile_of(4, 0, EXV_Q_RT_MAX + 1).err(), bad);
        assert_eq!(profile_of(4, 1, 0).err(), bad, "no codec_mode for EXV_RT");
        if registry::open(CodecId::HTJ2K, &[]).is_ok() {
            assert!(profile_of(6, 0, EXV_Q_HDR_MAX).is_ok());
            assert!(profile_of(6, 1, EXV_Q_LDR_MAX).is_ok());
            assert!(profile_of(6, 1, 1).is_ok());
            assert_eq!(profile_of(6, 1, EXV_Q_LDR_MAX + 1).err(), bad);
        }
    }

    /// Every (dtype, codec, mode, q) the AE export dialog offers by default (cpp/exv-ae `kModes` rows) really
    /// encodes through the ABI; the float-only/int-only pairings it forbids are the ones the library refuses.
    #[test]
    fn capi_every_ae_mode_encodes() {
        let rgba = ramp_rgba(32, 16);
        let htj2k = registry::open(CodecId::HTJ2K, &[]).is_ok();
        let (f16, f32_, u8_, u16_) = (ExvDtype::F16 as u32, ExvDtype::F32 as u32, ExvDtype::U8 as u32, ExvDtype::U16 as u32);
        let rows: [(u32, u32, u32, u32, bool); 11] = [
            (f16, 6, 0, 0, htj2k),
            (f16, 6, 0, 8, htj2k),
            (u16_, 6, 1, 16, htj2k),
            (u8_, 6, 1, EXV_Q_LDR_MAX, htj2k),
            (f16, 1, 0, 0, true),
            (f32_, 2, 0, 0, true),
            (f16, 3, 0, 8, true),
            (f16, 4, 0, 4, true),
            (f16, 7, 0, 0, true),
            (f16, 7, 0, 1, true),
            (u16_, 1, 0, 0, true),
        ];
        for (dtype, id, mode, q, available) in rows {
            if !available {
                continue;
            }
            let p = ExvEncParams { dtype_code: dtype, codec_id: id, codec_mode: mode, q, ..enc_params(32, 16) };
            assert!(!encode_still(&p, &rgba).is_empty(), "dtype {dtype} codec {id} mode {mode} q {q}");
        }
    }

    /// EXV-RT through the ABI: an F16 clip encodes, opens and decodes back to F16-level accuracy at q=0.
    #[test]
    fn capi_rt_roundtrip() {
        let (w, h) = (40u32, 24u32);
        let rgba = ramp_rgba(w, h);
        let params = ExvEncParams { dtype_code: ExvDtype::F16 as u32, codec_id: ExvCodec::ExvRt as u32, ..enc_params(w, h) };
        let path = tmp("rt.exv");
        std::fs::write(&path, encode_still(&params, &rgba)).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, 0, "{}", last_err());
        let mut back = vec![0u8; (w * h * 16) as usize];
        let mut wr = 0usize;
        assert_eq!(unsafe { exv_session_decode(sess, 0, pixfmt::RGBA_F32, back.as_mut_ptr(), back.len(), 0, &mut wr) }, 0, "{}", last_err());
        unsafe { exv_session_close(sess) };
        std::fs::remove_file(&path).unwrap();
        let decoded: Vec<f32> = back.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
        let worst = decoded.iter().zip(&rgba).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0f32, f32::max);
        assert!(worst < 2e-3, "q=0 RT is F16-accurate, worst rel err {worst}");
    }

    /// Params are validated up front: 256 must not alias F16, zero size and wrong buffer lengths fail.
    #[test]
    fn enc_param_validation() {
        let rgba = ramp_rgba(8, 4);
        let mut p = enc_params(8, 4);
        p.dtype_code = 256; // would narrow to 0 = F16
        let mut buf: *mut ExvBuffer = std::ptr::null_mut();
        let mut n = 0usize;
        assert_eq!(unsafe { exv_encode_rgba_f32(&p, rgba.as_ptr(), rgba.len(), &mut buf, &mut n) }, ExvStatus::Unsupported as i32);
        assert!(buf.is_null());
        let p = enc_params(0, 4);
        assert_eq!(unsafe { exv_encode_rgba_f32(&p, rgba.as_ptr(), rgba.len(), &mut buf, &mut n) }, ExvStatus::InvalidArgument as i32);
        let p = enc_params(8, 4);
        let big = vec![0f32; rgba.len() + 4]; // oversize buffer = geometry mismatch
        assert_eq!(unsafe { exv_encode_rgba_f32(&p, big.as_ptr(), big.len(), &mut buf, &mut n) }, ExvStatus::InvalidArgument as i32);
        let bad = ExvEncParams { codec_id: 99, ..enc_params(8, 4) };
        let (st, enc) = venc_open(&tmp("bad.exv"), &bad, std::ptr::null(), 1);
        assert_eq!(st, ExvStatus::Unsupported as i32, "validated at open");
        assert!(enc.is_null());
        let (st, enc2) = venc_open(&tmp("bad.exv"), &p, std::ptr::null(), 0);
        assert_eq!(st, ExvStatus::InvalidArgument as i32, "gop 0");
        assert!(enc2.is_null());
    }

    /// HTJ2K lossless through the ABI roundtrips bit-exactly (the project default codec is reachable).
    #[test]
    fn capi_htj2k_lossless_roundtrip() {
        if registry::open(CodecId::HTJ2K, &[]).is_err() {
            return; // built without the htj2k backend
        }
        let (w, h) = (40u32, 24u32);
        let rgba = ramp_rgba(w, h);
        let params = ExvEncParams { codec_id: 6, ..enc_params(w, h) };
        let container = encode_still(&params, &rgba);
        let path = tmp("htj2k.exv");
        std::fs::write(&path, &container).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, 0, "{}", last_err());
        let mut back = vec![0u8; (w * h * 16) as usize];
        let mut wr = 0usize;
        assert_eq!(unsafe { exv_session_decode(sess, 0, pixfmt::RGBA_F32, back.as_mut_ptr(), back.len(), 0, &mut wr) }, 0);
        unsafe { exv_session_close(sess) };
        std::fs::remove_file(&path).unwrap();
        let decoded = unsafe { std::slice::from_raw_parts(back.as_ptr() as *const f32, rgba.len()) };
        assert_eq!(decoded, &rgba[..], "lossless must be exact");
    }

    /// fps/PAR/alpha metadata passed to open comes back from the session; a bad meta fails at open.
    #[test]
    fn capi_meta_roundtrip() {
        let (w, h) = (16u32, 8u32);
        let dir = scratch("meta");
        let path = dir.join("meta.exv");
        let rgba = ramp_rgba(w, h);
        let params = enc_params(w, h);
        let m = ExvMeta {
            size: size_of::<ExvMeta>() as u32,
            flags: metaf::FPS | metaf::PAR,
            fps_num: 30000,
            fps_den: 1001,
            par_num: 2,
            par_den: 1,
            timecode: 0,
            primaries: 0,
            transfer: 0,
            range: 0,
            alpha: 2,
        };
        let bad = ExvMeta { size: 1, ..meta_out(&Meta::default()) };
        let zero_den = ExvMeta { fps_den: 0, ..meta_out(&Meta { fps: Some((24, 1)), ..Meta::default() }) };
        for b in [&bad, &zero_den] {
            let (st, enc) = venc_open(&path, &params, b, 1);
            assert_eq!(st, ExvStatus::InvalidArgument as i32);
            assert!(enc.is_null());
        }
        let (st, enc) = venc_open(&path, &params, &m, 1);
        assert_eq!(st, 0, "{}", last_err());
        assert_eq!(unsafe { exv_venc_add_rgba_f32(enc, rgba.as_ptr(), rgba.len()) }, 0);
        assert_eq!(unsafe { exv_venc_finish(enc, std::ptr::null_mut()) }, 0, "{}", last_err());
        unsafe { exv_venc_free(enc) };

        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, 0);
        let mut out = ExvMeta { size: size_of::<ExvMeta>() as u32, ..meta_out(&Meta::default()) };
        assert_eq!(unsafe { exv_session_meta(sess, &mut out) }, 0);
        assert_eq!((out.flags, out.fps_num, out.fps_den, out.par_num, out.par_den, out.alpha), (3, 30000, 1001, 2, 1, 2));
        unsafe { exv_session_close(sess) };
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A channel name that does not fit is an error carrying the required size (never truncated).
    #[test]
    fn channel_name_too_small() {
        let (w, h) = (8u32, 4u32);
        let container = encode_still(&enc_params(w, h), &ramp_rgba(w, h));
        let path = tmp("chname.exv");
        std::fs::write(&path, &container).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, 0);
        let mut small = [0 as c_char; 1];
        let mut wr = 0usize;
        assert_eq!(
            unsafe { exv_session_channel_name(sess, 0, small.as_mut_ptr(), small.len(), &mut wr) },
            ExvStatus::InvalidArgument as i32
        );
        assert_eq!(wr, 2, "'R' + NUL");
        let mut ok = [0 as c_char; 8];
        assert_eq!(unsafe { exv_session_channel_name(sess, 0, ok.as_mut_ptr(), ok.len(), &mut wr) }, 0);
        assert_eq!(unsafe { CStr::from_ptr(ok.as_ptr()) }.to_str().unwrap(), "R");
        unsafe { exv_session_close(sess) };
        std::fs::remove_file(&path).unwrap();
    }

    /// The write is atomic (temp + rename): the target is replaced whole, no temp file is left behind.
    #[test]
    fn write_file_replaces_atomically() {
        let dir = std::env::temp_dir().join(format!("exv_capi_atomic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.bin");
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        for payload in [&b"first-and-longer"[..], &b"2nd"[..]] {
            assert_eq!(unsafe { exv_write_file(cpath.as_ptr(), payload.as_ptr(), payload.len()) }, 0);
            assert_eq!(std::fs::read(&path).unwrap(), payload);
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "no temp file left next to the target");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Opening a non-container file is `INVALID_FORMAT`; the out handle stays null.
    #[test]
    fn open_rejects_garbage() {
        let path = tmp("garbage.exv");
        std::fs::write(&path, b"not an exv file at all").unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::dangling_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, ExvStatus::InvalidFormat as i32);
        assert!(sess.is_null());
        std::fs::remove_file(&path).unwrap();
    }

    /// A forged seek index (huge frame count, wild trailer offset) is `INVALID_FORMAT` — never an
    /// allocation abort or a panic (the index parser lives in exv-video; this pins the ABI path).
    #[test]
    fn open_rejects_forged_index() {
        let (w, h) = (8u32, 4u32);
        let good = encode_still(&enc_params(w, h), &ramp_rgba(w, h));
        let trailer = good.len() - 8;
        let index_off = u64::from_le_bytes(good[trailer..].try_into().unwrap()) as usize;
        let mut huge_count = good.clone();
        huge_count[index_off..index_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut wild_trailer = good.clone();
        wild_trailer[trailer..].copy_from_slice(&u64::MAX.to_le_bytes());
        let truncated = good[..good.len() / 2].to_vec();
        for (name, bytes) in [("count", huge_count), ("trailer", wild_trailer), ("trunc", truncated)] {
            let path = tmp(&format!("forged_{name}.exv"));
            std::fs::write(&path, &bytes).unwrap();
            let cpath = CString::new(path.to_str().unwrap()).unwrap();
            let mut sess: *mut ExvSession = std::ptr::null_mut();
            let st = unsafe { exv_session_open(cpath.as_ptr(), &mut sess) };
            assert_eq!(st, ExvStatus::InvalidFormat as i32, "{name}: {}", last_err());
            assert!(sess.is_null());
            std::fs::remove_file(&path).unwrap();
        }
    }

    /// P0-1: a tiny file whose header FORGES a huge frame is refused by the budget the session index is built with
    /// (typed `Limit`, before any plane is allocated), and an opened session carries the process budget as its limits.
    #[test]
    fn forged_huge_frame_is_refused_by_the_session_budget() {
        let good = encode_still(&enc_params(8, 4), &ramp_rgba(8, 4));
        assert!(good.len() < 3 << 10, "the forged file stays tiny");
        let mut forged = good.clone();
        // prelude = 13 bytes, then the Header: width@2 height@6 tile_w@14 tile_h@18 -> one 32768-square tile.
        for off in [2usize, 6, 14, 18] {
            forged[13 + off..13 + off + 4].copy_from_slice(&32768u32.to_le_bytes());
        }
        let small = DecodeLimits::new(64 << 20);
        let pool = exv_tile::PlanePool::new(exv_tile::Budget::new(1 << 20));
        let idx = parse_idx(&forged, small, pool).expect("parse never judges the declared size");
        assert_eq!(idx.limits(), small, "the index carries the budget it was built with");
        let res = exv_video::decode_frame(&forged, &idx, 0, exv_video::Chans::All);
        assert!(matches!(res, Err(exv_format::DecodeError::Limit { max, .. }) if max == 64 << 20), "{:?}", res.map(|f| f.planes.len()));

        // The real open path wires the same helper to the process budget.
        let path = tmp("budget_limits.exv");
        std::fs::write(&path, &good).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut sess: *mut ExvSession = std::ptr::null_mut();
        assert_eq!(unsafe { exv_session_open(cpath.as_ptr(), &mut sess) }, ExvStatus::Ok as i32, "{}", last_err());
        assert_eq!(unsafe { &*sess }.index.limits(), limits().expect("OS memory query"));
        unsafe { exv_session_close(sess) };
        std::fs::remove_file(&path).unwrap();
    }

    /// A well-formed container whose codec cannot be opened (a vendor id, an unassigned id) is `UNSUPPORTED`
    /// with a message naming the id — not `INVALID_FORMAT` (the file is fine, this build cannot decode it).
    #[test]
    fn open_unknown_codec_is_unsupported() {
        let (w, h) = (8u32, 4u32);
        let good = encode_still(&enc_params(w, h), &ramp_rgba(w, h));
        // Container prelude is 13 bytes; in the header width/height/frame_count/tile_w/tile_h precede codec_id.
        let at = 13 + exv_format::header::FRAME_COUNT_OFF + 12;
        assert_eq!(good[at..at + 4], 1u32.to_le_bytes(), "codec_id offset moved: fix `at`");
        for id in [0x8000_0001u32, 5] {
            let mut bytes = good.clone();
            bytes[at..at + 4].copy_from_slice(&id.to_le_bytes());
            let path = tmp(&format!("codec_{id:x}.exv"));
            std::fs::write(&path, &bytes).unwrap();
            let cpath = CString::new(path.to_str().unwrap()).unwrap();
            let mut sess: *mut ExvSession = std::ptr::null_mut();
            let st = unsafe { exv_session_open(cpath.as_ptr(), &mut sess) };
            let msg = last_err();
            assert_eq!(st, ExvStatus::Unsupported as i32, "{id:#x}: {msg}");
            assert!(sess.is_null());
            assert!(msg.contains(&id.to_string()) || msg.to_lowercase().contains(&format!("{id:x}")), "{id:#x}: {msg}");
            std::fs::remove_file(&path).unwrap();
        }
    }

    /// `docs/exv-capi.h` is GENERATED from this file by cbindgen: regenerate in memory and fail on any
    /// drift (signature, struct layout, enum value, ABI version). `EXV_BLESS_HEADER=1` rewrites the file.
    #[test]
    fn header_in_sync() {
        let dir = env!("CARGO_MANIFEST_DIR");
        let cfg = cbindgen::Config::from_file(format!("{dir}/cbindgen.toml")).expect("cbindgen.toml");
        let bindings = cbindgen::Builder::new()
            .with_crate(dir)
            .with_config(cfg)
            .generate()
            .expect("cbindgen failed to parse capi.rs");
        let mut buf = Vec::new();
        bindings.write(&mut buf);
        let fresh = String::from_utf8(buf).expect("utf8 header");
        let path = format!("{dir}/../../docs/exv-capi.h");
        if std::env::var_os("EXV_BLESS_HEADER").is_some() {
            std::fs::write(&path, &fresh).expect("write header");
            return;
        }
        let have = std::fs::read_to_string(&path).expect("read header");
        // Compare line by line: the checkout may be CRLF (git autocrlf) while cbindgen emits LF.
        assert!(have.lines().eq(fresh.lines()), "docs/exv-capi.h is stale: run `EXV_BLESS_HEADER=1 cargo test -p exv-capi header_in_sync`");
        assert!(fresh.contains(&format!("#define EXV_ABI_VERSION {EXV_ABI_VERSION}")));
        let n = fresh.matches("exv_").count();
        assert!(n >= 22, "only {n} exv_ symbols in the generated header");
    }
}
