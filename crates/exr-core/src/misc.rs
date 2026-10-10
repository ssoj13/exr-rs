//! Port of OpenEXR `ImfMisc.h/.cpp` — miscellaneous layout helpers for EXR I/O.
//!
//! This module contains the shared low-level helpers that scanline and
//! tiled I/O use to compute per-line byte sizes, line-buffer offsets, pixel
//! type sizes, and to scatter/gather data between compressed line buffers and
//! frame-buffer slices.
//!
//! # Functions that live elsewhere
//!
//! The following C++ functions from `ImfMisc.cpp` are **not** in this module
//! because they depend on types defined in other layers:
//!
//! * `bytesPerDeepLineTable` (two overloads), `copyIntoDeepFrameBuffer`,
//!   `copyFromDeepFrameBuffer` and `sampleCount` (inline in header) — deep-image
//!   sizing, scatter/gather and sample-count access; implemented inside the
//!   deep readers/writers (`crate::deep`).
//! * `usesLongNames` — depends on `Header`; lives in `header.rs` as a method.
//! * `getChunkOffsetTableSize` / `getScanlineChunkOffsetTableSize` — depends on
//!   `Header` and `ImfPartType`; lives in the part layer (`multipart/output.rs`).
//! * `defaultFormat` / `numLinesInBuffer` — depend on `Compressor`; live in
//!   `compressor/mod.rs`.
//! * `WidenFilename` / `getLibraryVersion` — platform / metadata utilities;
//!   see `container/version.rs` (`WidenFilename` needs no port there).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMisc.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMisc.cpp>

use imath_rs::modp;

use crate::attr::{ChannelList, PixelType};
use crate::error::{ExrError, Result};

/// Fallibly allocate a `Vec<T>` of `n` elements, each a clone of `fill` —
/// the checked replacement for `vec![fill; n]` wherever `n` is derived from
/// header fields an attacker controls.
///
/// `vec![fill; n]` (and `Vec::resize`, `Vec::with_capacity`) call the global
/// allocator through `RawVec::allocate_in`, which on failure calls
/// `handle_alloc_error` — this **aborts the process**, it is not a `Result`
/// a caller can catch, unlike C++'s `operator new` throwing a catchable
/// `std::bad_alloc`. `Vec::try_reserve_exact` is the one Vec API that reports
/// an allocation failure as a `Result` instead of aborting. Once it succeeds
/// the capacity is guaranteed, so the follow-up `resize` to fill the vector
/// cannot itself trigger a second (potentially failing) allocation.
///
/// This does not change what C++ *tries* to do for a crafted, absurdly large
/// header field — both languages attempt the same allocation — it changes
/// what happens when that attempt fails: C++ throws `std::bad_alloc` (or
/// `OverflowExc` for the length checks it does first) and the caller can
/// recover; unchecked Rust allocation aborts the whole process. This makes
/// Rust's failure mode match C++'s (a catchable error), not merely "attempt
/// a smaller allocation."
pub(crate) fn try_filled_vec<T: Clone>(n: usize, fill: T) -> Result<Vec<T>> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(n).map_err(|e| {
        ExrError::Arg(format!(
            "Cannot allocate a buffer of {n} elements ({e}); the file's declared \
             dimensions are too large for available memory."
        ))
    })?;
    v.resize(n, fill);
    Ok(v)
}

// ---------------------------------------------------------------------------
// pixelTypeSize — port of `int pixelTypeSize(PixelType type)`
// ---------------------------------------------------------------------------

/// Return the on-disk size in bytes of one sample of `type_` — port of
/// `pixelTypeSize`.
///
/// C++ uses `Xdr::size<T>()` which equals `sizeof(T)` for these three types:
/// `UINT → 4`, `HALF → 2`, `FLOAT → 4`.
///
/// # Errors
/// Returns [`ExrError::Arg`] for [`PixelType::NumPixelTypes`] (unknown type),
/// matching the C++ `throw IEX_NAMESPACE::ArgExc("Unknown pixel type.")`.
pub fn pixel_type_size(type_: PixelType) -> Result<usize> {
    match type_ {
        PixelType::Uint => Ok(4),
        PixelType::Half => Ok(2),
        PixelType::Float => Ok(4),
        PixelType::NumPixelTypes => Err(ExrError::Arg("Unknown pixel type.".into())),
    }
}

// ---------------------------------------------------------------------------
// numSamples — port of `int numSamples(int s, int a, int b)`
// ---------------------------------------------------------------------------

/// Return the number of samples a channel with sub-sampling rate `s` has in
/// the closed interval `[a, b]` — port of `numSamples`.
///
/// Uses Imath's `divp` (floored division) so the result is always positive
/// regardless of whether `a` or `b` are negative. Computed in `i64` (the C++ `int`
/// version overflows for `a`/`b` within `s` of the `i32` limits, which a public,
/// user-built window can reach); `s` must be >= 1 (`divp` by 0 panics, as in C++ it
/// is undefined) — `api::store::sample_dims` checks that before calling.
///
/// # Examples
/// A channel with `s=2` (samples at 0, 2, 4, 6, 8, …) has 2 samples in
/// `[1, 5]` and 3 samples in `[2, 6]`.
pub fn num_samples(s: i32, a: i32, b: i32) -> i64 {
    let (s, a, b) = (i64::from(s), i64::from(a), i64::from(b));
    let (a1, b1) = (a.div_euclid(s), b.div_euclid(s));
    b1 - a1 + i64::from(a1 * s >= a)
}

// ---------------------------------------------------------------------------
// bytesPerLineTable — port of `size_t bytesPerLineTable(const Header&, vector<size_t>&)`
// ---------------------------------------------------------------------------

/// Build a table giving, for each scanline in `data_window`, how many bytes
/// are required to store all pixels of all channels (tightly packed) —
/// port of `bytesPerLineTable`.
///
/// Returns the maximum bytes-per-line value across all scanlines (used by the
/// line-buffer allocator).
///
/// `min_y`/`max_y` define the inclusive data-window y range; `width` is
/// `data_window.max.x - data_window.min.x + 1`.
///
/// Unlike the C++ version which takes a `Header`, this function accepts the
/// decomposed fields so it can be tested without a full `Header`.
///
/// # Errors
/// Propagates [`ExrError::Arg`] from [`pixel_type_size`] for invalid types.
pub fn bytes_per_line_table(
    min_y: i32,
    max_y: i32,
    width: i32,
    channels: &ChannelList,
    bytes_per_line: &mut Vec<usize>,
) -> Result<usize> {
    let nlines = (max_y - min_y + 1) as usize;
    // Fallible, not `resize` -- `nlines` is
    // driven directly by the header's `dataWindow` height, unbounded by any
    // chunk-count cap for a multi-line codec's `bytesPerLine` table itself
    // (see `try_filled_vec`'s docs for why an infallible `resize` is the
    // wrong tool here). C++ has the identical eager, full-height allocation
    // (`ImfMisc.cpp::bytesPerLineTable`) but a failed `new` there is a
    // catchable `std::bad_alloc`, not a process abort.
    *bytes_per_line = try_filled_vec(nlines, 0usize)?;

    for (_, ch) in channels.iter() {
        // Bytes contributed by this channel to every y that satisfies sampling.
        let n_bytes = pixel_type_size(ch.type_)?
            * (width as usize / ch.x_sampling as usize);

        for (i, y) in (min_y..=max_y).enumerate() {
            if modp(y, ch.y_sampling) == 0 {
                // `ImfMisc.cpp:160-166` (verified): a scanline whose
                // accumulated byte count would overflow is a corrupt/hostile
                // channel list, not a Rust-level panic (debug) or silent
                // wraparound (release, before this fix).
                bytes_per_line[i] =
                    bytes_per_line[i]
                        .checked_add(n_bytes)
                        .ok_or_else(|| ExrError::Io("Scanline size too large.".into()))?;
            }
        }
    }

    let max_bytes = bytes_per_line.iter().copied().max().unwrap_or(0);
    Ok(max_bytes)
}

/// How many of `[min_y, max_y]` (inclusive, absolute pixel `y`) satisfy
/// `modp(y, y_sampling) == 0` — the row test [`bytes_per_line_table`] uses to
/// decide whether scanline `y` carries any bytes for a channel with this
/// `y_sampling`. Reduces to the full row count when `y_sampling <= 1`.
///
/// The single home for this row-inclusion convention (`compute_sampled_height`,
/// `internal_util.h:12-47`): DWA's `channel_dims` and HTJ2K's
/// `decode_chunk` (y-subsampling fix) both
/// need a per-channel row count for a sub-sampled channel's OWN reduced grid,
/// and previously cross-imported it from `compressor::dwa` into
/// `compressor::htj2k` — moved here instead so neither codec depends on the
/// other's internals for a convention that belongs to the generic packed-
/// buffer layout, not to DWA.
pub fn active_row_count(min_y: i32, max_y: i32, y_sampling: i32) -> usize {
    if y_sampling <= 1 {
        return (max_y - min_y + 1).max(0) as usize;
    }
    (min_y..=max_y).filter(|&y| modp(y, y_sampling) == 0).count()
}

// ---------------------------------------------------------------------------
// offsetInLineBufferTable — port of both C++ overloads
// ---------------------------------------------------------------------------

/// Build a table of byte offsets within a line buffer for scanlines
/// `scanline1..=scanline2` (indices into `bytes_per_line`) — port of the
/// range-based `offsetInLineBufferTable` overload.
///
/// `scanline1 = 0` represents the first line in the data window.
/// Every time `i % lines_in_line_buffer == 0` the running offset resets to 0,
/// because a new line-buffer block starts.
pub fn offset_in_line_buffer_table_range(
    bytes_per_line: &[usize],
    scanline1: usize,
    scanline2: usize,
    lines_in_line_buffer: usize,
    offset_in_line_buffer: &mut Vec<usize>,
) {
    offset_in_line_buffer.resize(bytes_per_line.len(), 0);

    let mut offset: usize = 0;
    for i in scanline1..=scanline2 {
        if i % lines_in_line_buffer == 0 {
            offset = 0;
        }
        offset_in_line_buffer[i] = offset;
        offset += bytes_per_line[i];
    }
}

/// Build a line-buffer offset table for the full range `0..bytes_per_line.len()`
/// — port of the single-argument `offsetInLineBufferTable` overload.
pub fn offset_in_line_buffer_table(
    bytes_per_line: &[usize],
    lines_in_line_buffer: usize,
    offset_in_line_buffer: &mut Vec<usize>,
) {
    let len = bytes_per_line.len();
    if len == 0 {
        offset_in_line_buffer.clear();
        return;
    }
    offset_in_line_buffer_table_range(
        bytes_per_line,
        0,
        len - 1,
        lines_in_line_buffer,
        offset_in_line_buffer,
    );
}

// ---------------------------------------------------------------------------
// lineBufferMinY / lineBufferMaxY
// ---------------------------------------------------------------------------

/// First scanline of the line-buffer block that contains scanline `y` —
/// port of `lineBufferMinY`.
///
/// `min_y` is the data-window minimum y; `lines_in_line_buffer` is the codec's
/// block height.
#[inline]
pub fn line_buffer_min_y(y: i32, min_y: i32, lines_in_line_buffer: i32) -> i32 {
    ((y - min_y) / lines_in_line_buffer) * lines_in_line_buffer + min_y
}

/// Last scanline of the line-buffer block that contains scanline `y` —
/// port of `lineBufferMaxY`.
#[inline]
pub fn line_buffer_max_y(y: i32, min_y: i32, lines_in_line_buffer: i32) -> i32 {
    line_buffer_min_y(y, min_y, lines_in_line_buffer) + lines_in_line_buffer - 1
}

// ---------------------------------------------------------------------------
// skipChannel — port of `void skipChannel(const char*& readPtr, PixelType, size_t xSize)`
// ---------------------------------------------------------------------------

/// Advance a read cursor past `x_size` pixels of `type_in_file` — port of
/// `skipChannel`.
///
/// In C++ this calls `Xdr::skip`, which advances the `const char*&` pointer
/// by `pixelTypeSize(type) * xSize`.  Here the equivalent is to return the
/// number of bytes to skip, which the caller adds to its cursor.
///
/// # Errors
/// Returns [`ExrError::Arg`] for unknown pixel types.
pub fn skip_channel(type_in_file: PixelType, x_size: usize) -> Result<usize> {
    Ok(pixel_type_size(type_in_file)? * x_size)
}

// ---------------------------------------------------------------------------
// fillChannelWithZeroes — port of `void fillChannelWithZeroes(...)`
// ---------------------------------------------------------------------------

/// Fill `x_size` pixels of `type_` with zero bytes into `buf` starting at
/// `offset` — port of `fillChannelWithZeroes`.
///
/// C++ writes to a `char*& writePtr` that advances past the written data; here
/// we write into a `&mut [u8]` slice starting at `offset` and return the new
/// offset (past the written region).
///
/// The C++ function branches on `Compressor::Format` (`XDR` vs `NATIVE`).
/// For all three pixel types, zero in both XDR and native representation is
/// all-zero bytes (IEEE 754 `0.0f` / `0.0h` and integer `0` are all-zero), so
/// a single `write_bytes(0)` path is correct for both formats.
///
/// # Errors
/// Returns [`ExrError::Arg`] for unknown pixel types.
pub fn fill_channel_with_zeroes(
    buf: &mut [u8],
    offset: usize,
    type_: PixelType,
    x_size: usize,
) -> Result<usize> {
    let sz = pixel_type_size(type_)?;
    let n = sz * x_size;
    buf[offset..offset + n].fill(0);
    Ok(offset + n)
}

// ---------------------------------------------------------------------------
// copyIntoFrameBuffer — port of `void copyIntoFrameBuffer(...)`
// ---------------------------------------------------------------------------

/// Data format of the compressed line/tile buffer — mirrors
/// `Compressor::Format` from `ImfCompressor.h`.
///
/// `Xdr` means machine-independent (little-endian) wire format; `Native` means
/// the host's own byte order.  The codec layer sets this when it decompresses
/// a block.  Scanline I/O with `NO_COMPRESSION` always uses `Xdr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferFormat {
    /// XDR / little-endian (machine-independent) — `Compressor::XDR`.
    Xdr,
    /// Host-native byte order — `Compressor::NATIVE`.
    Native,
}

/// Reinterpret `bytes` (little-endian pairs) as `u16` values into `out` —
/// shared SSOT for every codec that gathers a packed little-endian `u16`
/// buffer (audit block C; previously PIZ's gather loop and several DWA
/// sub-stream decodes each had their own inline `chunks_exact(2)`/
/// `from_le_bytes` loop). Hard error on a length mismatch rather than a
/// silent truncation or panic.
pub(crate) fn le_bytes_to_u16s(bytes: &[u8], out: &mut [u16]) -> Result<()> {
    if bytes.len() != out.len() * 2 {
        return Err(ExrError::Logic(format!(
            "le_bytes_to_u16s: length mismatch ({} bytes for {} u16 slots)",
            bytes.len(),
            out.len()
        )));
    }
    for (dst, pair) in out.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
        *dst = u16::from_le_bytes([pair[0], pair[1]]);
    }
    Ok(())
}

/// Reinterpret `values` as little-endian byte pairs into `out` — the inverse
/// of [`le_bytes_to_u16s`], shared the same way for every codec's scatter/
/// pack loop. Hard error on a length mismatch rather than a silent
/// truncation or panic.
pub(crate) fn u16s_to_le_bytes(values: &[u16], out: &mut [u8]) -> Result<()> {
    if out.len() != values.len() * 2 {
        return Err(ExrError::Logic(format!(
            "u16s_to_le_bytes: length mismatch ({} u16 values for {} bytes)",
            values.len(),
            out.len()
        )));
    }
    for (dst, &v) in out.as_chunks_mut::<2>().0.iter_mut().zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

/// Convert `count` packed `FS`-byte source samples (`src`, exactly
/// `count * FS` bytes) with `conv` and store each `W`-byte result into
/// `write_buf`, `x_stride` bytes apart starting at `write_ptr` -- the
/// shared per-arm loop body for [`copy_into_frame_buffer`]. Returns the
/// write pointer after the last sample.
///
/// # Why const-generic sizes, and why this is fast with no `unsafe`
/// `FS`/`W` are compile-time sizes and `conv` works on fixed arrays, so
/// every per-sample copy is a fixed-width load/store. The source is one
/// pre-sliced range walked as fixed `[u8; FS]` arrays, and the destination is
/// the exact byte range this call touches, `(count-1)*x_stride + W` bytes,
/// sliced once and walked with `chunks_mut(x_stride)`, so no index is
/// bounds-checked per sample. `chunks_mut` (not `chunks_exact_mut`) because
/// the LAST pixel needs only `W` bytes, not a full `x_stride` slot. The
/// earlier shape took the size as a runtime argument and read the source by
/// index through a closure; it was not inlined, and a profile of a
/// single-threaded 16k NONE read put 52% of all samples in it.
///
/// The fallback loop serves what the chunked walk cannot:
/// * `x_stride == 0` -- the documented broadcast/fill pattern (every sample
///   lands on the SAME address); incompatible with disjoint chunking.
/// * `x_stride < W` -- an overlapping-write layout (representable via the
///   public `Slice` API even though no real file produces it).
///
/// Both paths return an error rather than panic if a range turns out
/// short; the caller's up-front whole-range check makes that unreachable
/// in practice.
#[inline(always)]
fn scatter_convert<const FS: usize, const W: usize>(
    write_buf: &mut [u8],
    write_ptr: usize,
    x_stride: usize,
    count: usize,
    src: &[u8],
    conv: impl Fn([u8; FS]) -> [u8; W],
) -> Result<usize> {
    if count == 0 {
        return Ok(write_ptr);
    }
    let short = || ExrError::Logic("scatter: a pre-checked range came up short".into());
    if src.len() != count * FS {
        return Err(short());
    }
    if x_stride >= W {
        let total = (count - 1) * x_stride + W;
        let range = write_buf.get_mut(write_ptr..write_ptr + total).ok_or_else(short)?;
        crate::probe::hit(crate::probe::Probe::ScatterTypedFast);
        if x_stride == W {
            for (d, s) in range.as_chunks_mut::<W>().0.iter_mut().zip(src.as_chunks::<FS>().0.iter()) {
                let (Some(d), Some(s)) = (d.first_chunk_mut::<W>(), s.first_chunk::<FS>()) else {
                    return Err(short());
                };
                *d = conv(*s);
            }
        } else {
            for (d, s) in range.chunks_mut(x_stride).zip(src.as_chunks::<FS>().0.iter()) {
                let (Some(d), Some(s)) = (d.first_chunk_mut::<W>(), s.first_chunk::<FS>()) else {
                    return Err(short());
                };
                *d = conv(*s);
            }
        }
        return Ok(write_ptr + count * x_stride);
    }
    let mut wp = write_ptr;
    for s in src.as_chunks::<FS>().0.iter() {
        let s = s.first_chunk::<FS>().ok_or_else(short)?;
        let d = write_buf.get_mut(wp..).and_then(|d| d.first_chunk_mut::<W>()).ok_or_else(short)?;
        *d = conv(*s);
        wp += x_stride;
    }
    Ok(wp)
}

/// Store the same `W`-byte `value` `count` times, `x_stride` bytes apart --
/// the fill-channel counterpart of [`scatter_convert`] (no source).
#[inline(always)]
fn scatter_fill<const W: usize>(
    write_buf: &mut [u8],
    write_ptr: usize,
    x_stride: usize,
    count: usize,
    value: [u8; W],
) -> Result<usize> {
    if count == 0 {
        return Ok(write_ptr);
    }
    let short = || ExrError::Logic("scatter fill: a pre-checked range came up short".into());
    if x_stride >= W {
        let total = (count - 1) * x_stride + W;
        let range = write_buf.get_mut(write_ptr..write_ptr + total).ok_or_else(short)?;
        crate::probe::hit(crate::probe::Probe::ScatterTypedFast);
        for d in range.chunks_mut(x_stride) {
            *d.first_chunk_mut::<W>().ok_or_else(short)? = value;
        }
        return Ok(write_ptr + count * x_stride);
    }
    let mut wp = write_ptr;
    for _ in 0..count {
        *write_buf.get_mut(wp..).and_then(|d| d.first_chunk_mut::<W>()).ok_or_else(short)? = value;
        wp += x_stride;
    }
    Ok(wp)
}

/// Copy one channel's worth of pixels from a compressed line/tile buffer into
/// a frame-buffer slice, performing on-the-fly type conversion —
/// port of `copyIntoFrameBuffer`.
///
/// # Parameters
/// * `read_buf` — the compressed (or uncompressed) pixel data for this channel.
/// * `read_offset` — byte position in `read_buf` where this channel starts;
///   advanced by the number of bytes consumed and returned.
/// * `write_buf` — the caller's frame buffer (the backing store for the slice).
/// * `write_ptr` — byte offset of the leftmost pixel to write (must equal
///   `slice.base_offset + y/ySampling * yStride` for the current row).
/// * `end_ptr` — byte offset of the rightmost pixel to write (inclusive).
/// * `x_stride` — `slice.x_stride`.
/// * `fill` — if `true`, ignore `read_buf` and fill with `fill_value`.
/// * `fill_value` — value to use when `fill` is `true`.
/// * `format` — buffer encoding.
/// * `type_fb` — pixel type in the frame buffer slice.
/// * `type_file` — pixel type stored in the file channel.
///
/// Returns `(new_read_offset, new_write_ptr)`.
///
/// # Errors
/// Returns [`ExrError::Arg`] for unknown pixel types.
#[allow(clippy::too_many_arguments)]
pub fn copy_into_frame_buffer(
    read_buf: &[u8],
    read_offset: usize,
    write_buf: &mut [u8],
    mut write_ptr: usize,
    count: usize,
    x_stride: usize,
    fill: bool,
    fill_value: f64,
    format: BufferFormat,
    type_fb: PixelType,
    type_file: PixelType,
) -> Result<(usize, usize)> {
    // Every branch below indexes
    // `write_buf[write_ptr..write_ptr+size]` `count` times, advancing
    // `write_ptr` by `x_stride` after each one. This is `count`-DRIVEN (a
    // `for _ in 0..count` loop), matching C++ 3.4's `unpack.c` (`for s in
    // 0..w`, i.e. driven by the pixel count, not by comparing addresses) —
    // NOT the withdrawn `while write_ptr <= end_ptr { ..; write_ptr +=
    // x_stride }` shape, which hung forever whenever `x_stride == 0` and more
    // than one pixel was requested (a legal broadcast/fill pattern: every
    // pixel lands at the SAME address, so the address-comparison loop guard
    // never advances past `end_ptr`). `x_stride == 0` is otherwise ordinary
    // here: every iteration just re-addresses the same byte range.
    //
    // The write-bounds check below still only needs the highest address ever
    // touched, which for `x_stride >= 0` and `count > 0` is always
    // `write_ptr + (count - 1) * x_stride` (the LAST iteration's address —
    // earlier ones are never farther out since the walk is monotonic).
    if count > 0 {
        let size_fb = pixel_type_size(type_fb)?;
        let last_offset = (count - 1)
            .checked_mul(x_stride)
            .ok_or_else(|| ExrError::Arg("Frame buffer slice address overflow.".into()))?;
        let last_ptr = write_ptr
            .checked_add(last_offset)
            .ok_or_else(|| ExrError::Arg("Frame buffer slice address overflow.".into()))?;
        let required = last_ptr
            .checked_add(size_fb)
            .ok_or_else(|| ExrError::Arg("Frame buffer slice address overflow.".into()))?;
        if required > write_buf.len() {
            return Err(ExrError::Arg(format!(
                "Frame buffer slice write range [{write_ptr}, {required}) exceeds the \
                 destination buffer's length ({})." ,
                write_buf.len()
            )));
        }
    }

    if fill {
        // File has no data for this channel — write fill_value into every pixel.
        write_ptr = match type_fb {
            PixelType::Uint => {
                scatter_fill(write_buf, write_ptr, x_stride, count, (fill_value as u32).to_ne_bytes())?
            }
            PixelType::Half => {
                // Convert f64 fill to half via f32 path.
                let h = imath_rs::Half::from(fill_value as f32);
                scatter_fill(write_buf, write_ptr, x_stride, count, h.bits().to_ne_bytes())?
            }
            PixelType::Float => {
                scatter_fill(write_buf, write_ptr, x_stride, count, (fill_value as f32).to_ne_bytes())?
            }
            PixelType::NumPixelTypes => {
                return Err(ExrError::Arg("Unknown pixel data type.".into()));
            }
        };
        return Ok((read_offset, write_ptr));
    }

    use imath_rs::Half;

    // These five conversions are the single source of truth: see
    // `crate::convert` (a port of `ImfConvert.{h,cpp}` + the live 3.4
    // `internal_coding.h` primitives) for the UINT<->HALF overflow-rule
    // rationale. The closures below are thin bit-level adapters over it.
    let half_to_f32 = |bits: u16| -> f32 { f32::from(Half::from_bits(bits)) };
    let half_to_uint = |bits: u16| -> u32 { crate::convert::half_to_uint(Half::from_bits(bits)) };
    let float_to_uint = crate::convert::float_to_uint;
    let uint_to_half = |u: u32| -> u16 { crate::convert::uint_to_half(u).bits() };
    let float_to_half = |f: f32| -> u16 { crate::convert::float_to_half(f).bits() };

    // The channel's packed source run: `count` file-typed samples.
    let file_size = pixel_type_size(type_file)?;
    let src_len = count
        .checked_mul(file_size)
        .ok_or_else(|| ExrError::Arg("Frame buffer slice address overflow.".into()))?;
    let src = read_offset
        .checked_add(src_len)
        .and_then(|end| read_buf.get(read_offset..end))
        .ok_or_else(|| {
            ExrError::Input(format!(
                "Pixel data for {count} samples at offset {read_offset} exceeds the decoded \
                 buffer's length ({}).",
                read_buf.len()
            ))
        })?;

    // One arm per (frame-buffer type, file type) pair. `$from` decodes the
    // source bytes: `from_le_bytes` for XDR (EXR's machine-independent wire
    // format is LITTLE-endian, `ImfXdr.h`; this MUST match
    // `container::xdr`, the single tested source of truth) or
    // `from_ne_bytes` for NATIVE. The destination frame buffer is native
    // order, so every store is `to_ne_bytes`.
    macro_rules! convert_arms {
        ($from:ident) => {
            match (type_fb, type_file) {
                // --- UINT fb ---
                (PixelType::Uint, PixelType::Uint) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 4]| u32::$from(b).to_ne_bytes(),
                ),
                (PixelType::Uint, PixelType::Half) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 2]| half_to_uint(u16::$from(b)).to_ne_bytes(),
                ),
                (PixelType::Uint, PixelType::Float) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 4]| float_to_uint(f32::from_bits(u32::$from(b))).to_ne_bytes(),
                ),
                // --- HALF fb ---
                (PixelType::Half, PixelType::Uint) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 4]| uint_to_half(u32::$from(b)).to_ne_bytes(),
                ),
                (PixelType::Half, PixelType::Half) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 2]| u16::$from(b).to_ne_bytes(),
                ),
                (PixelType::Half, PixelType::Float) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 4]| float_to_half(f32::from_bits(u32::$from(b))).to_ne_bytes(),
                ),
                // --- FLOAT fb ---
                (PixelType::Float, PixelType::Uint) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 4]| (u32::$from(b) as f32).to_ne_bytes(),
                ),
                (PixelType::Float, PixelType::Half) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 2]| half_to_f32(u16::$from(b)).to_ne_bytes(),
                ),
                (PixelType::Float, PixelType::Float) => scatter_convert(
                    write_buf, write_ptr, x_stride, count, src,
                    |b: [u8; 4]| u32::$from(b).to_ne_bytes(),
                ),
                _ => return Err(ExrError::Arg("Unknown pixel data type.".into())),
            }
        };
    }

    write_ptr = if format == BufferFormat::Xdr {
        convert_arms!(from_le_bytes)?
    } else {
        convert_arms!(from_ne_bytes)?
    };
    Ok((read_offset + src_len, write_ptr))
}

/// Load `count` `S`-byte samples from `read_buf`, `x_stride` bytes apart
/// starting at `read_ptr`, convert each with `conv` and store them packed
/// into `write_buf` from `write_offset` -- the GATHER mirror of
/// [`scatter_convert`], and the loop body of [`copy_from_frame_buffer`].
/// Returns `(new_write_offset, new_read_ptr)`.
///
/// Same design as `scatter_convert`: const-generic `S`, the packed
/// destination sliced once and walked as fixed `[u8; S]` arrays, the
/// strided source sliced once to its exact extent and walked with
/// `chunks(x_stride)` (the last pixel needs only `S` bytes). A profile of a
/// single-threaded 16k NONE write had 18% of all samples in the earlier
/// per-sample indexed loop. `x_stride < S` (incl. the `x_stride == 0`
/// broadcast) takes the indexed fallback.
#[inline(always)]
fn gather_convert<const S: usize>(
    write_buf: &mut [u8],
    write_offset: usize,
    read_buf: &[u8],
    read_ptr: usize,
    count: usize,
    x_stride: usize,
    conv: impl Fn([u8; S]) -> [u8; S],
) -> Result<(usize, usize)> {
    if count == 0 {
        return Ok((write_offset, read_ptr));
    }
    let short = || ExrError::Logic("gather: a pre-checked range came up short".into());
    let dst_len = count * S;
    let dst = write_buf.get_mut(write_offset..write_offset + dst_len).ok_or_else(|| {
        ExrError::Logic(format!(
            "gather: {dst_len} bytes at offset {write_offset} exceed the line buffer"
        ))
    })?;
    if x_stride >= S {
        let total = (count - 1) * x_stride + S;
        let src = read_buf.get(read_ptr..read_ptr + total).ok_or_else(short)?;
        crate::probe::hit(crate::probe::Probe::GatherTypedFast);
        for (d, s) in dst.as_chunks_mut::<S>().0.iter_mut().zip(src.chunks(x_stride)) {
            let (Some(d), Some(s)) = (d.first_chunk_mut::<S>(), s.first_chunk::<S>()) else {
                return Err(short());
            };
            *d = conv(*s);
        }
    } else {
        let mut rp = read_ptr;
        for d in dst.as_chunks_mut::<S>().0.iter_mut() {
            let s = read_buf.get(rp..).and_then(|s| s.first_chunk::<S>()).ok_or_else(short)?;
            let d = d.first_chunk_mut::<S>().ok_or_else(short)?;
            *d = conv(*s);
            rp += x_stride;
        }
    }
    Ok((write_offset + dst_len, read_ptr + count * x_stride))
}

// ---------------------------------------------------------------------------
// copyFromFrameBuffer — port of `void copyFromFrameBuffer(...)`
// ---------------------------------------------------------------------------

/// Copy one channel's worth of pixels from a frame-buffer slice into a
/// compressed line/tile buffer — port of `copyFromFrameBuffer`.
///
/// This is the write-path counterpart of [`copy_into_frame_buffer`].
/// No type conversion is performed (C++ `copyFromFrameBuffer` does not convert
/// — the `type` in frame buffer and file must match).
///
/// # Parameters
/// * `write_buf` — the output line/tile buffer to write into.
/// * `write_offset` — byte position in `write_buf` where this channel starts;
///   advanced and returned.
/// * `read_buf` — the caller's frame buffer.
/// * `read_ptr` — byte offset of the leftmost pixel.
/// * `end_ptr` — byte offset of the rightmost pixel (inclusive).
/// * `x_stride` — frame-buffer x stride in bytes.
/// * `format` — buffer encoding (`Xdr` or `Native`).
/// * `type_` — pixel type (same in frame buffer and file).
///
/// Returns `(new_write_offset, new_read_ptr)`.
///
/// # Errors
/// Returns [`ExrError::Arg`] for unknown pixel types.
// C++ copyFromFrameBuffer has 8 parameters; #[allow] keeps the 1:1 port intact.
#[allow(clippy::too_many_arguments)]
pub fn copy_from_frame_buffer(
    write_buf: &mut [u8],
    write_offset: usize,
    read_buf: &[u8],
    read_ptr: usize,
    count: usize,
    x_stride: usize,
    format: BufferFormat,
    type_: PixelType,
) -> Result<(usize, usize)> {
    let sz = pixel_type_size(type_)?;

    // The mirror-image of
    // `copy_into_frame_buffer` above, for the GATHER direction — see that
    // function's comment for why this is `count`-driven (matches C++ 3.4's
    // `unpack.c` pixel-count loop) rather than an address-comparison `while`
    // loop, which hung forever whenever `x_stride == 0` and `count > 1` (a
    // legal "every pixel reads the same source address" pattern).
    if count > 0 {
        let last_offset = (count - 1)
            .checked_mul(x_stride)
            .ok_or_else(|| ExrError::Arg("Frame buffer slice address overflow.".into()))?;
        let last_ptr = read_ptr
            .checked_add(last_offset)
            .ok_or_else(|| ExrError::Arg("Frame buffer slice address overflow.".into()))?;
        let required = last_ptr
            .checked_add(sz)
            .ok_or_else(|| ExrError::Arg("Frame buffer slice address overflow.".into()))?;
        if required > read_buf.len() {
            return Err(ExrError::Arg(format!(
                "Frame buffer slice read range [{read_ptr}, {required}) exceeds the \
                 source buffer's length ({})." ,
                read_buf.len()
            )));
        }
    }

    // XDR stores little-endian (EXR's machine-independent wire format; this
    // MUST match `container::xdr`, the single tested source of truth):
    // decode the native frame-buffer value, store `to_le_bytes`. NATIVE
    // copies the bytes verbatim.
    match (format, type_) {
        (BufferFormat::Xdr, PixelType::Half) => gather_convert(
            write_buf, write_offset, read_buf, read_ptr, count, x_stride,
            |b: [u8; 2]| u16::from_ne_bytes(b).to_le_bytes(),
        ),
        (BufferFormat::Xdr, PixelType::Uint | PixelType::Float) => gather_convert(
            write_buf, write_offset, read_buf, read_ptr, count, x_stride,
            |b: [u8; 4]| u32::from_ne_bytes(b).to_le_bytes(),
        ),
        (BufferFormat::Native, PixelType::Half) => gather_convert(
            write_buf, write_offset, read_buf, read_ptr, count, x_stride,
            |b: [u8; 2]| b,
        ),
        (BufferFormat::Native, PixelType::Uint | PixelType::Float) => gather_convert(
            write_buf, write_offset, read_buf, read_ptr, count, x_stride,
            |b: [u8; 4]| b,
        ),
        (_, PixelType::NumPixelTypes) => Err(ExrError::Arg("Unknown pixel data type.".into())),
    }
}

// ---------------------------------------------------------------------------
// convertInPlace — port of `void convertInPlace(char*&, const char*&, PixelType, size_t)`
// ---------------------------------------------------------------------------

/// Convert `num_pixels` samples from native layout to XDR in-place (or between
/// two byte slices) — port of `convertInPlace`.
///
/// The C++ version reads from `fromPtr` (which may alias `toPtr`) and writes
/// XDR to `toPtr`.  Here `src` and `dst` are separate slices for safety; both
/// must be at least `pixelTypeSize(type_) * num_pixels` bytes long.
///
/// # Errors
/// Returns [`ExrError::Arg`] for unknown pixel types.
pub fn convert_in_place(
    dst: &mut [u8],
    src: &[u8],
    type_: PixelType,
    num_pixels: usize,
) -> Result<()> {
    let sz = pixel_type_size(type_)?;
    // XDR is little-endian — this MUST match `container::xdr` (the tested source
    // of truth for EXR LE byte order).
    match type_ {
        PixelType::Uint | PixelType::Float => {
            for i in 0..num_pixels {
                let off = i * sz;
                let v = u32::from_ne_bytes([
                    src[off], src[off + 1], src[off + 2], src[off + 3],
                ]);
                dst[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        PixelType::Half => {
            for i in 0..num_pixels {
                let off = i * sz;
                let v = u16::from_ne_bytes([src[off], src[off + 1]]);
                dst[off..off + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        PixelType::NumPixelTypes => {
            return Err(ExrError::Arg("Unknown pixel data type.".into()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Safe parallel-scatter destination planner
// ---------------------------------------------------------------------------
//
// NOT a C++ port (`ImfMisc.cpp` has no equivalent — C++ scatters from worker
// threads by raw pointer arithmetic, `ImfThreading`/`LineBufferTask`, which
// has no safe-Rust analogue). This is the "destination planner" from
// the parallel read design: given each chunk's bounding write interval
// into the caller's frame-buffer slab, either prove the chunks are pairwise
// disjoint and hand each one its own exclusive `&mut [u8]` sub-slice (no
// `unsafe`, via `split_at_mut`), or report `None` so the caller falls back to
// a fully serial scatter — the same bytes either way.

/// Move the front `n` bytes out of `*buf`, replacing `*buf` with the
/// remainder. Needed because `<&mut [u8]>::split_at_mut` re-borrows through
/// `&mut self` and would otherwise tie both halves to a short-lived borrow of
/// the local variable holding `*buf`, rather than to `buf`'s own lifetime.
/// `mem::take` (via `&mut [u8]: Default = &mut []`) lets the split's lifetime
/// be exactly `buf`'s.
fn split_off_front<'a>(buf: &mut &'a mut [u8], n: usize) -> &'a mut [u8] {
    let taken = std::mem::take(buf);
    let (front, rest) = taken.split_at_mut(n);
    *buf = rest;
    front
}

/// Per-chunk parallel-scatter destination: an exclusive sub-slice of the
/// caller's `out` buffer plus the absolute byte offset (into the ORIGINAL
/// `out`) that `piece[0]` corresponds to. A scatter function that computed an
/// absolute `write_ptr` into `out` writes to `piece[write_ptr - base]`
/// instead.
pub(crate) struct ChunkDest<'a> {
    pub piece: &'a mut [u8],
    pub base: usize,
}

/// One chunk's write footprint as a SET of half-open byte ranges (alloc
/// audit follow-up, 2026-09-24) -- generalizes the original "one range per
/// chunk" [`plan_disjoint`] to chunks whose write footprint is genuinely
/// several DISJOINT pieces of `out`, not one contiguous span. The flat
/// scanline/tiled readers pass exactly one range per chunk (their address
/// model has one contiguous destination per chunk, so nothing changes for
/// them); deep passes one range PER ACTIVE SLICE, because deep's
/// `sample_data` heap is per-channel-contiguous (all of channel A's
/// samples, then all of channel B's, ...) -- a single merged range per
/// line/tile would necessarily span through every OTHER line's/tile's data
/// sitting between two channels' regions, which is exactly the bug this
/// generalization fixes. `SmallVec<[_; 1]>` keeps the overwhelmingly common single-range
/// case allocation-free.
pub(crate) type RangeSet = smallvec::SmallVec<[std::ops::Range<usize>; 1]>;

/// One chunk's set of destinations, aligned 1:1 (same length, same order)
/// with the [`RangeSet`] that produced it.
pub(crate) type DestSet<'a> = smallvec::SmallVec<[ChunkDest<'a>; 1]>;

/// Prove every chunk's [`RangeSet`] (into `out`, in the caller's original
/// chunk order) is pairwise disjoint from every OTHER chunk's AND from its
/// own sibling ranges, and if so, split `out` into one exclusive
/// [`DestSet`] per chunk (same order as `chunks`, and each `DestSet`'s own
/// entries in the same order as that chunk's `RangeSet`); otherwise return
/// `None`.
///
/// An empty range (`start == end`) trivially never overlaps anything and
/// gets an empty piece — this covers a slice/chunk that (for this call's
/// window) contributes no bytes, e.g. it was `skip`/out of range for this
/// row. A chunk may also contribute a `RangeSet` with FEWER entries than
/// another chunk (e.g. a channel absent for this particular row); ranges
/// are matched to destinations purely by flattened sort order, not by
/// per-chunk position, so this needs no padding.
///
/// # Errors
/// `None` when any two non-empty ranges (from any two chunks, or from the
/// same chunk) overlap, or when the globally last range's end exceeds
/// `out.len()` (a caller bug: the planner is only ever fed ranges the
/// serial scatter path would itself have written inside `out`, so this
/// should not happen, but the check is unconditional rather than a
/// `debug_assert!`, matching the no-panic rule).
pub(crate) fn plan_disjoint<'a>(out: &'a mut [u8], chunks: &[RangeSet]) -> Option<Vec<DestSet<'a>>> {
    // One destination buffer is the one-plane case of THE planner below.
    plan_disjoint_planes(&mut [out], chunks)
}

/// THE planner ([`plan_disjoint`] is its one-buffer form): [`plan_disjoint`]'s contract over
/// SEVERAL destination buffers (the planar read: one allocation per output plane). The ranges live
/// in one VIRTUAL address space in which plane `i` starts at `plane_starts(planes)[i]` (the
/// cumulative length of the planes before it), so ranges of different planes can never overlap,
/// and every non-empty range must lie wholly inside ONE plane. Returns one [`DestSet`] per chunk
/// (each entry's `base` is a virtual address), or `None` on any overlap / straddle / range outside
/// every plane. Never panics, whatever the ranges (inverted, empty, overlapping, huge).
///
/// Empty ranges are set aside BEFORE the overlap check: an empty range sorted between two
/// overlapping ones would otherwise hide the overlap (the check compares neighbours) and make the
/// split arithmetic underflow.
pub(crate) fn plan_disjoint_planes<'a>(
    planes: &mut [&'a mut [u8]],
    chunks: &[RangeSet],
) -> Option<Vec<DestSet<'a>>> {
    let starts = plane_starts(planes);
    let mut flat: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
    for (ci, ranges) in chunks.iter().enumerate() {
        for r in ranges {
            flat.push((ci, r.clone()));
        }
    }
    let n = flat.len();
    // Non-empty ranges only, in address order: neighbours then decide disjointness completely.
    let mut order: Vec<usize> = (0..n).filter(|&i| flat[i].1.start < flat[i].1.end).collect();
    order.sort_by_key(|&i| (flat[i].1.start, flat[i].1.end));
    if order.windows(2).any(|w| flat[w[0]].1.end > flat[w[1]].1.start) {
        return None;
    }
    // The plane a non-empty range lies in (it must fit entirely inside one).
    let mut plane_idx: Vec<Option<usize>> = vec![None; n];
    for &i in &order {
        let r = &flat[i].1;
        plane_idx[i] = Some((0..planes.len()).find(|&p| {
            !planes[p].is_empty()
                && starts[p] <= r.start
                && starts[p].checked_add(planes[p].len()).is_some_and(|e| r.end <= e)
        })?);
    }
    crate::probe::hit(crate::probe::Probe::PlanDisjointAccepted);

    // Everything is validated: take the planes apart.
    let mut rests: Vec<&'a mut [u8]> = planes.iter_mut().map(std::mem::take).collect();
    let mut consumed: Vec<usize> = starts;
    let mut pieces: Vec<Option<ChunkDest<'a>>> = (0..n).map(|_| None).collect();
    for &i in &order {
        let iv = flat[i].1.clone();
        let p = plane_idx[i]?;
        // `iv.start >= consumed[p]` holds (sorted, disjoint, inside the plane); checked anyway.
        let gap = iv.start.checked_sub(consumed[p])?;
        let _ = split_off_front(&mut rests[p], gap);
        let piece = split_off_front(&mut rests[p], iv.end - iv.start);
        pieces[i] = Some(ChunkDest { piece, base: iv.start });
        consumed[p] = iv.end;
    }
    // Empty ranges (and inverted ones) write nothing: an empty piece anchored at their start.
    for (i, slot) in pieces.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(ChunkDest { piece: &mut [], base: flat[i].1.start });
        }
    }
    let pieces: Vec<ChunkDest<'a>> = pieces.into_iter().collect::<Option<Vec<_>>>()?;
    let mut result: Vec<DestSet<'a>> = (0..chunks.len()).map(|_| DestSet::new()).collect();
    for (piece, (ci, _)) in pieces.into_iter().zip(flat) {
        result[ci].push(piece);
    }
    Some(result)
}

/// Start of each plane in the virtual address space of [`plan_disjoint_planes`]: the cumulative
/// length of the planes before it.
pub(crate) fn plane_starts(planes: &[&mut [u8]]) -> Vec<usize> {
    let mut at = 0usize;
    planes
        .iter()
        .map(|p| {
            let s = at;
            at += p.len();
            s
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// One range per chunk (single-element `RangeSet`s) -- the flat
    /// scanline/tiled shape, unchanged semantics from before this function
    /// generalized to per-chunk range SETS.
    fn one(r: std::ops::Range<usize>) -> RangeSet {
        smallvec::smallvec![r]
    }

    #[test]
    fn plan_disjoint_splits_non_overlapping_ranges_in_original_order() {
        let mut buf = vec![0u8; 30];
        // Out of order and with a gap, on purpose.
        let chunks = vec![one(20..25), one(0..10), one(10..15)];
        let mut dests = plan_disjoint(&mut buf, &chunks).expect("disjoint");
        assert_eq!(dests.len(), 3);
        dests[0][0].piece.fill(1); // 20..25
        dests[1][0].piece.fill(2); // 0..10
        dests[2][0].piece.fill(3); // 10..15
        assert_eq!(dests[0][0].base, 20);
        assert_eq!(dests[1][0].base, 0);
        assert_eq!(dests[2][0].base, 10);
        drop(dests);
        let expected: Vec<u8> =
            (0..30).map(|i| if i < 10 { 2 } else if i < 15 { 3 } else if (20..25).contains(&i) { 1 } else { 0 }).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn plan_disjoint_rejects_overlapping_ranges() {
        let mut buf = vec![0u8; 30];
        let chunks = vec![one(0..10), one(5..15)];
        assert!(plan_disjoint(&mut buf, &chunks).is_none());
    }

    #[test]
    fn plan_disjoint_allows_adjacent_and_empty_ranges() {
        let mut buf = vec![0u8; 30];
        let chunks = vec![one(0..10), one(10..10), one(10..20)];
        let dests = plan_disjoint(&mut buf, &chunks).expect("disjoint");
        assert_eq!(dests[1][0].piece.len(), 0);
    }

    #[test]
    fn plan_disjoint_rejects_out_of_bounds() {
        let mut buf = vec![0u8; 10];
        let chunks = vec![one(0..5), one(5..20)];
        assert!(plan_disjoint(&mut buf, &chunks).is_none());
    }

    /// The whole point of the generalization: a chunk whose write footprint
    /// is genuinely TWO disjoint pieces of `out` (e.g. deep's per-channel-
    /// contiguous heap) is accepted and split correctly, and a range
    /// belonging to ONE chunk still correctly rejects overlap against a
    /// range belonging to a DIFFERENT chunk regardless of which chunk
    /// "owns" which piece.
    #[test]
    fn plan_disjoint_handles_multi_range_chunks() {
        let mut buf = vec![0u8; 40];
        // Chunk 0 (e.g. "line 0"): writes into [0,5) of channel A's region
        // AND [20,25) of channel B's region -- two disjoint pieces.
        // Chunk 1 ("line 1"): [5,10) of A and [25,30) of B -- byte-adjacent
        // to chunk 0's pieces on both sides, still disjoint.
        let chunks = vec![
            smallvec::smallvec![0..5, 20..25],
            smallvec::smallvec![5..10, 25..30],
        ];
        let mut dests = plan_disjoint(&mut buf, &chunks).expect("disjoint");
        assert_eq!(dests.len(), 2);
        assert_eq!(dests[0].len(), 2, "chunk 0 must get one destination per range it contributed");
        assert_eq!(dests[1].len(), 2);
        dests[0][0].piece.fill(1); // A: 0..5
        dests[0][1].piece.fill(2); // B: 20..25
        dests[1][0].piece.fill(3); // A: 5..10
        dests[1][1].piece.fill(4); // B: 25..30
        drop(dests);
        let expected: Vec<u8> = (0..40)
            .map(|i| match i {
                0..=4 => 1,
                5..=9 => 3,
                20..=24 => 2,
                25..=29 => 4,
                _ => 0,
            })
            .collect();
        assert_eq!(buf, expected);
    }

    /// Negative test: two ranges that GENUINELY alias (from different
    /// chunks, or from the same chunk's own two pieces) must still be
    /// rejected -- the multi-range generalization must not accidentally
    /// weaken the overlap check to "only checked within a chunk" or "only
    /// checked between chunks".
    #[test]
    fn plan_disjoint_rejects_aliasing_across_and_within_chunks() {
        let mut buf = vec![0u8; 30];
        // Cross-chunk aliasing: chunk 0's second range overlaps chunk 1's
        // first range.
        let cross = vec![smallvec::smallvec![0..5, 10..20], smallvec::smallvec![15..25]];
        assert!(plan_disjoint(&mut buf, &cross).is_none(), "cross-chunk aliasing must be rejected");

        // Within-chunk aliasing: a single chunk's own two ranges overlap
        // each other (e.g. two slices of the SAME channel double-mapped).
        let within: Vec<RangeSet> = vec![smallvec::smallvec![0..10, 5..15]];
        assert!(plan_disjoint(&mut buf, &within).is_none(), "within-chunk (same chunk) aliasing must be rejected");
    }

    use crate::attr::{Channel, ChannelList, PixelType};

    // ---- try_filled_vec ----

    /// A count whose element size overflows `isize::MAX` bytes hits
    /// `Vec::try_reserve_exact`'s CAPACITY-OVERFLOW error path -- computed
    /// from the count and element size alone, before the allocator is ever
    /// called. This is deterministic on any platform and allocates nothing
    /// (no real multi-GB request), unlike testing an actual OOM would be.
    #[test]
    fn try_filled_vec_errors_on_capacity_overflow_not_abort() {
        // `usize::MAX` elements of `u64` (8 bytes each): the byte-size
        // computation alone overflows `usize`/`isize::MAX`, so this must
        // come back as a clean `Err`, never touch the real allocator, and
        // certainly never abort the test process.
        let result = try_filled_vec(usize::MAX, 0u64);
        assert!(
            result.is_err(),
            "an allocation whose size overflows isize::MAX must error, not abort"
        );
    }

    #[test]
    fn try_filled_vec_succeeds_for_a_normal_size() {
        let v = try_filled_vec(5usize, 7u8).unwrap();
        assert_eq!(v, vec![7u8; 5]);
    }

    // ---- pixelTypeSize ----

    #[test]
    fn pixel_type_size_uint() {
        assert_eq!(pixel_type_size(PixelType::Uint).unwrap(), 4);
    }

    #[test]
    fn pixel_type_size_half() {
        assert_eq!(pixel_type_size(PixelType::Half).unwrap(), 2);
    }

    #[test]
    fn pixel_type_size_float() {
        assert_eq!(pixel_type_size(PixelType::Float).unwrap(), 4);
    }

    #[test]
    fn pixel_type_size_invalid_errors() {
        assert!(pixel_type_size(PixelType::NumPixelTypes).is_err());
    }

    // ---- numSamples ----

    #[test]
    fn num_samples_s2_a1_b5() {
        // C++ comment: channel with s=2 has 2 samples in [1, 5]
        assert_eq!(num_samples(2, 1, 5), 2);
    }

    #[test]
    fn num_samples_s2_a2_b6() {
        // C++ comment: channel with s=2 has 3 samples in [2, 6]
        assert_eq!(num_samples(2, 2, 6), 3);
    }

    /// `i32` arithmetic would overflow at the limits; the `i64` version is exact.
    #[test]
    fn num_samples_extreme_window_does_not_overflow() {
        assert_eq!(num_samples(2, i32::MIN, i32::MAX), 1i64 << 31);
        assert_eq!(num_samples(3, i32::MIN + 1, i32::MAX), 1431655765);
    }

    #[test]
    fn num_samples_s1_full() {
        // s=1: every scanline is sampled; count == b - a + 1
        assert_eq!(num_samples(1, 0, 9), 10);
    }

    // ---- bytesPerLineTable ----

    /// Build a 2-channel (HALF + FLOAT) channel list over a data window of
    /// width 4 (x: 0..=3), height 3 (y: 0..=2), no sub-sampling.
    ///
    /// Per line: HALF=2*4=8 bytes, FLOAT=4*4=16 bytes → 24 bytes per line.
    fn two_channel_list() -> ChannelList {
        let mut cl = ChannelList::new();
        cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        cl
    }

    #[test]
    fn bytes_per_line_table_two_channels() {
        let cl = two_channel_list();
        let mut bpl = Vec::new();
        let max = bytes_per_line_table(0, 2, 4, &cl, &mut bpl).unwrap();
        // Width=4, HALF→2 bytes, FLOAT→4 bytes: 8 + 16 = 24 per line
        assert_eq!(bpl.len(), 3);
        assert!(bpl.iter().all(|&b| b == 24));
        assert_eq!(max, 24);
    }

    #[test]
    fn bytes_per_line_table_with_y_subsampling() {
        // channel "A": UINT, xSampling=1, ySampling=2, width=4
        // Only even y rows contribute: y=0 → 16, y=1 → 0, y=2 → 16
        let mut cl = ChannelList::new();
        cl.insert("A", Channel::new(PixelType::Uint, 1, 2, false)).unwrap();
        let mut bpl = Vec::new();
        let max = bytes_per_line_table(0, 2, 4, &cl, &mut bpl).unwrap();
        assert_eq!(bpl[0], 16); // y=0 (even) → 4*4=16
        assert_eq!(bpl[1],  0); // y=1 (odd)  → 0
        assert_eq!(bpl[2], 16); // y=2 (even) → 16
        assert_eq!(max, 16);
    }

    // ---- activeRowCount ----

    /// Mutation-testing gap fix (moved here from `compressor::dwa`, which
    /// used to own this function):
    /// hand-derived boundary values (not a round trip) against
    /// `active_row_count`'s own documented contract (`modp(y, y_sampling) ==
    /// 0` for `y` in `[min_y, max_y]`). A symmetric range like `(0, 9, 2)` has
    /// an equal split between matching/non-matching rows, which cannot
    /// distinguish `==` from `!=` in the filter predicate - these ranges are
    /// chosen asymmetric on purpose.
    #[test]
    fn active_row_count_matches_hand_derived_counts() {
        // y in 0..=10 (11 values): modp(y,2)==0 for {0,2,4,6,8,10} -> 6.
        assert_eq!(active_row_count(0, 10, 2), 6);
        // y in 3..=12 (10 values): modp(y,3)==0 for {3,6,9,12} -> 4.
        assert_eq!(active_row_count(3, 12, 3), 4);
        // y_sampling <= 1 takes the early-return branch: every row counts.
        assert_eq!(active_row_count(5, 9, 1), 5);
        assert_eq!(active_row_count(5, 9, 0), 5);
    }

    // ---- offsetInLineBufferTable ----

    #[test]
    fn offset_in_line_buffer_table_basic() {
        // bytesPerLine = [24, 24, 24], linesInBuffer = 2
        // i=0: 0 % 2 == 0 → offset=0, advance by 24
        // i=1: 1 % 2 != 0 → offset=24, advance by 24
        // i=2: 2 % 2 == 0 → offset=0, advance by 24
        let bpl = vec![24usize, 24, 24];
        let mut offsets = Vec::new();
        offset_in_line_buffer_table(&bpl, 2, &mut offsets);
        assert_eq!(offsets, vec![0, 24, 0]);
    }

    #[test]
    fn offset_in_line_buffer_table_single_line_buffer() {
        let bpl = vec![10usize, 20, 30];
        let mut offsets = Vec::new();
        offset_in_line_buffer_table(&bpl, 1, &mut offsets);
        // Every line starts a new buffer → all offsets are 0.
        assert_eq!(offsets, vec![0, 0, 0]);
    }

    // ---- lineBufferMinY / lineBufferMaxY ----

    #[test]
    fn line_buffer_min_max_y() {
        // minY=0, linesInBuffer=16
        // y=5  → min=0, max=15
        // y=16 → min=16, max=31
        // y=17 → min=16, max=31
        assert_eq!(line_buffer_min_y(5,  0, 16),  0);
        assert_eq!(line_buffer_max_y(5,  0, 16), 15);
        assert_eq!(line_buffer_min_y(16, 0, 16), 16);
        assert_eq!(line_buffer_max_y(16, 0, 16), 31);
        assert_eq!(line_buffer_min_y(17, 0, 16), 16);
    }

    #[test]
    fn line_buffer_min_y_with_nonzero_miny() {
        // minY=100, linesInBuffer=32
        // y=115 → (115-100)/32=0 → min=100, max=131
        // y=132 → (132-100)/32=1 → min=132, max=163
        assert_eq!(line_buffer_min_y(115, 100, 32), 100);
        assert_eq!(line_buffer_max_y(115, 100, 32), 131);
        assert_eq!(line_buffer_min_y(132, 100, 32), 132);
    }

    // ---- skipChannel ----

    #[test]
    fn skip_channel_half() {
        assert_eq!(skip_channel(PixelType::Half, 5).unwrap(), 10);
    }

    #[test]
    fn skip_channel_float() {
        assert_eq!(skip_channel(PixelType::Float, 3).unwrap(), 12);
    }

    #[test]
    fn skip_channel_uint() {
        assert_eq!(skip_channel(PixelType::Uint, 2).unwrap(), 8);
    }

    // ---- fillChannelWithZeroes ----

    #[test]
    fn fill_channel_with_zeroes_float() {
        let mut buf = vec![0xFFu8; 32];
        let end = fill_channel_with_zeroes(&mut buf, 4, PixelType::Float, 3).unwrap();
        assert_eq!(end, 16); // 4 + 3*4
        assert!(buf[4..16].iter().all(|&b| b == 0));
        // Bytes before and after are untouched.
        assert!(buf[0..4].iter().all(|&b| b == 0xFF));
        assert!(buf[16..].iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn fill_channel_with_zeroes_half() {
        let mut buf = vec![0xAAu8; 16];
        fill_channel_with_zeroes(&mut buf, 0, PixelType::Half, 4).unwrap();
        assert!(buf[0..8].iter().all(|&b| b == 0));
    }

    // ---- copyIntoFrameBuffer ----

    #[test]
    fn copy_into_fb_half_to_half_native() {
        use imath_rs::Half;
        // Build a src with 3 HALF values: 1.0, 2.0, 3.0 (native layout)
        let h1 = Half::from(1.0f32).bits().to_ne_bytes();
        let h2 = Half::from(2.0f32).bits().to_ne_bytes();
        let h3 = Half::from(3.0f32).bits().to_ne_bytes();
        let mut read_buf = vec![0u8; 6];
        read_buf[0..2].copy_from_slice(&h1);
        read_buf[2..4].copy_from_slice(&h2);
        read_buf[4..6].copy_from_slice(&h3);

        let mut write_buf = vec![0u8; 6];
        let (new_ro, _new_wp) = copy_into_frame_buffer(
            &read_buf, 0,
            &mut write_buf, 0, 3, // write_ptr=0, count=3 (stride 2)
            2, false, 0.0,
            BufferFormat::Native,
            PixelType::Half, PixelType::Half,
        ).unwrap();
        assert_eq!(new_ro, 6);
        // Values should be identical
        assert_eq!(write_buf[0..2], h1);
        assert_eq!(write_buf[2..4], h2);
        assert_eq!(write_buf[4..6], h3);
    }

    /// `ImfConvert.cpp` `uintToHalf` / `floatToHalf`: a value above HALF_MAX is
    /// `+inf`, NOT the largest finite half.  The plain IEEE conversion rounds
    /// 65505..65519 down to 65504 (0x7bff) — verified divergence vs C++ 3.4.15,
    /// which returns 0x7c00 for the same UINT/FLOAT inputs.
    #[test]
    fn copy_into_fb_over_half_max_becomes_inf() {
        const INF: u16 = 0x7c00;
        const MAX_FINITE: u16 = 0x7bff;
        // UINT -> HALF genuinely clamps at HALF_MAX (`internal_coding.h:275
        // uint_to_half`: `ui > 65504 -> 0x7c00`), verified against the oracle.
        for (uint_val, expect) in [(65504u32, MAX_FINITE), (65505, INF), (65519, INF), (70000, INF)]
        {
            let read_buf = uint_val.to_ne_bytes().to_vec();
            let mut write_buf = vec![0u8; 2];
            copy_into_frame_buffer(
                &read_buf, 0, &mut write_buf, 0, 1, 2, false, 0.0,
                BufferFormat::Native, PixelType::Half, PixelType::Uint,
            )
            .unwrap();
            assert_eq!(
                u16::from_ne_bytes([write_buf[0], write_buf[1]]),
                expect,
                "uint {uint_val} -> half"
            );
        }
    }

    /// A FLOAT source (confirmed against C++ 3.4's live read path
    /// `OpenEXRCore/unpack.c` -> `internal_coding.h::float_to_half` ->
    /// Imath's round-to-nearest `imath_float_to_half`)
    /// overflows to +/-inf only at `|bits| > 0x477fefff` (|f| >= 65520), NOT at
    /// HALF_MAX (65504). 65504 < |f| < 65520 rounds to the largest finite half.
    /// This supersedes an earlier claim that FLOAT overflows at
    /// HALF_MAX like UINT does — that claim was never checked against the
    /// oracle and was wrong.
    #[test]
    fn copy_into_fb_float_overflow_matches_oracle_threshold() {
        const INF: u16 = 0x7c00;
        const NEG_INF: u16 = 0xfc00;
        const MAX_FINITE: u16 = 0x7bff;
        const NEG_MAX_FINITE: u16 = 0xfbff;
        for (f, expect) in [
            (65504.0f32, MAX_FINITE),
            (65505.0, MAX_FINITE),
            (65519.0, MAX_FINITE),
            (65520.0, INF),
            (70000.0, INF),
            (-65505.0, NEG_MAX_FINITE),
        ] {
            let read_buf = f.to_ne_bytes().to_vec();
            let mut write_buf = vec![0u8; 2];
            copy_into_frame_buffer(
                &read_buf, 0, &mut write_buf, 0, 1, 2, false, 0.0,
                BufferFormat::Native, PixelType::Half, PixelType::Float,
            )
            .unwrap();
            assert_eq!(
                u16::from_ne_bytes([write_buf[0], write_buf[1]]),
                expect,
                "float {f} -> half"
            );
        }
        // Far below -HALF_MAX is still -inf.
        let read_buf = (-70000.0f32).to_ne_bytes().to_vec();
        let mut write_buf = vec![0u8; 2];
        copy_into_frame_buffer(
            &read_buf, 0, &mut write_buf, 0, 1, 2, false, 0.0,
            BufferFormat::Native, PixelType::Half, PixelType::Float,
        )
        .unwrap();
        assert_eq!(u16::from_ne_bytes([write_buf[0], write_buf[1]]), NEG_INF);
    }

    #[test]
    fn copy_into_fb_fill_float() {
        let read_buf = vec![0u8; 0];
        let mut write_buf = vec![0u8; 12];
        let fill_val: f64 = 2.5;
        let (_, _) = copy_into_frame_buffer(
            &read_buf, 0,
            &mut write_buf, 0, 3, // count=3 (stride 4)
            4, true, fill_val,
            BufferFormat::Native,
            PixelType::Float, PixelType::Float,
        ).unwrap();
        let expected = (fill_val as f32).to_ne_bytes();
        assert_eq!(write_buf[0..4], expected);
        assert_eq!(write_buf[4..8], expected);
        assert_eq!(write_buf[8..12], expected);
    }

    // ---- copyFromFrameBuffer ----

    #[test]
    fn copy_from_fb_half_xdr_roundtrip() {
        use imath_rs::Half;
        // Write 2 halves in native to write_buf via XDR path, then verify
        // the XDR bytes (little-endian — EXR's machine-independent format).
        let h1bits = Half::from(1.0f32).bits();
        let h2bits = Half::from(2.0f32).bits();
        let mut read_buf = vec![0u8; 4];
        read_buf[0..2].copy_from_slice(&h1bits.to_ne_bytes());
        read_buf[2..4].copy_from_slice(&h2bits.to_ne_bytes());

        let mut write_buf = vec![0u8; 4];
        let (new_wo, _) = copy_from_frame_buffer(
            &mut write_buf, 0,
            &read_buf, 0, 2,
            2, BufferFormat::Xdr, PixelType::Half,
        ).unwrap();
        assert_eq!(new_wo, 4);
        // XDR = little-endian
        assert_eq!(write_buf[0..2], h1bits.to_le_bytes());
        assert_eq!(write_buf[2..4], h2bits.to_le_bytes());
    }

    /// Lock test: the per-pixel XDR write path must produce byte-identical
    /// output to `container::xdr` (the single tested source of truth for EXR's
    /// little-endian wire format) and to the known-good C++/EXR layout.
    ///
    /// For FLOAT 1.0 the on-disk bytes are `00 00 80 3F` (little-endian IEEE
    /// 754) — matching C++ OpenEXR.  A regression to big-endian here would
    /// produce `3F 80 00 00` and fail this assertion.
    #[test]
    fn copy_from_fb_float_xdr_matches_container_xdr() {
        use crate::container::xdr;
        use crate::io::MemOStream;

        // --- Pixel path: one FLOAT 1.0 through copy_from_frame_buffer (Xdr). ---
        let src = 1.0f32.to_ne_bytes(); // native frame-buffer layout
        let mut pixel_path = vec![0u8; 4];
        let (new_wo, _) = copy_from_frame_buffer(
            &mut pixel_path, 0,
            &src, 0, 1, // single pixel: read_ptr=0, count=1
            4, BufferFormat::Xdr, PixelType::Float,
        ).unwrap();
        assert_eq!(new_wo, 4);

        // --- 1. Byte-exact known-good EXR/C++ layout for FLOAT 1.0. ---
        assert_eq!(pixel_path, [0x00, 0x00, 0x80, 0x3F]);

        // --- 2. Identical to the tested container::xdr writer. ---
        let mut xdr_path = MemOStream::new();
        xdr::write_f32(&mut xdr_path, 1.0f32).unwrap();
        assert_eq!(pixel_path.as_slice(), xdr_path.data());
    }

    // ---- convertInPlace ----

    #[test]
    fn convert_in_place_uint() {
        let src: [u8; 4] = 42u32.to_ne_bytes();
        let mut dst = [0u8; 4];
        convert_in_place(&mut dst, &src, PixelType::Uint, 1).unwrap();
        assert_eq!(dst, 42u32.to_le_bytes());
    }

    /// A destination buffer shorter than the slice's own
    /// addressing (e.g. a caller passing a 2x2-tile-sized buffer for a HALF
    /// slice that actually needs 8 bytes, not 4) must error cleanly, never
    /// panic with an out-of-bounds slice index.
    #[test]
    fn copy_into_fb_short_write_buffer_errors_not_panics() {
        let read_buf = [0u8; 8]; // 4 HALF samples worth of XDR bytes.
        let mut write_buf = vec![0u8; 4]; // only room for 2 HALF samples.
        let err = copy_into_frame_buffer(
            &read_buf, 0, &mut write_buf, 0, 4, // count=4 (needs 8 bytes, buffer has 4)
            2, false, 0.0, BufferFormat::Xdr, PixelType::Half, PixelType::Half,
        )
        .expect_err("a too-short destination buffer must error, not panic");
        let _ = format!("{err}");
    }

    /// The same guard applies to a FILL slice (no matching file
    /// channel) — a short destination must still error, not panic.
    #[test]
    fn copy_into_fb_short_write_buffer_fill_errors_not_panics() {
        let read_buf: [u8; 0] = [];
        let mut write_buf = vec![0u8; 4];
        let err = copy_into_frame_buffer(
            &read_buf, 0, &mut write_buf, 0, 4, 2, true, 7.0, BufferFormat::Xdr,
            PixelType::Half, PixelType::Half,
        )
        .expect_err("a too-short fill destination must error, not panic");
        let _ = format!("{err}");
    }

    /// The mirror on the gather side (`copy_from_frame_buffer`, used by
    /// `OutputFile::write_pixels`): a caller's live pixel buffer shorter than
    /// the slice's own addressing must error, not panic.
    #[test]
    fn copy_from_fb_short_read_buffer_errors_not_panics() {
        let mut write_buf = vec![0u8; 8];
        let read_buf = [0u8; 4]; // only 2 HALF samples, slice wants 4.
        let err = copy_from_frame_buffer(
            &mut write_buf, 0, &read_buf, 0, 4, 2, BufferFormat::Xdr, PixelType::Half,
        )
        .expect_err("a too-short source buffer must error, not panic");
        let _ = format!("{err}");
    }

    // -----------------------------------------------------------------------
    // `scatter_convert` / `scatter_fill` / `gather_convert`: the chunked fast
    // path must be byte-identical to an independent per-sample reference
    // across every x_stride shape (packed == W, padded 10/16, the
    // overlapping 1 < W, and the x_stride == 0 broadcast), sample counts
    // {0, 1, 2, 17} and destination offsets, and the probe must record
    // exactly when the chunked path ran.
    // -----------------------------------------------------------------------

    const STRIDES: [usize; 5] = [0, 1, 10, 16, usize::MAX]; // MAX = "packed" (W)
    const COUNTS: [usize; 4] = [0, 1, 2, 17];
    const OFFSETS: [usize; 3] = [0, 3, 7];

    /// A cheap, deterministic, non-degenerate byte sequence: sample `i`
    /// (1-based) maps to `[i, i*3+1, i*7+2, i*11+3]` truncated as needed.
    fn sample_bytes(i: usize) -> [u8; 4] {
        [
            (i & 0xFF) as u8,
            ((i * 3 + 1) & 0xFF) as u8,
            ((i * 7 + 2) & 0xFF) as u8,
            ((i * 11 + 3) & 0xFF) as u8,
        ]
    }

    /// Bytes one strided walk touches: `(count-1)*x_stride + w`.
    fn touched(count: usize, x_stride: usize, w: usize) -> usize {
        if count == 0 { 0 } else { (count - 1) * x_stride + w }
    }

    /// A test conversion that changes every byte and depends on its
    /// position, so a swapped, shifted or skipped sample shows up.
    fn conv_ref<const FS: usize, const W: usize>(b: [u8; FS]) -> [u8; W] {
        std::array::from_fn(|k| b[k % FS].wrapping_mul(3).wrapping_add(k as u8 + 1))
    }

    fn check_scatter<const FS: usize, const W: usize>() {
        for &xs in &STRIDES {
            let x_stride = if xs == usize::MAX { W } else { xs };
            for &count in &COUNTS {
                for &offset in &OFFSETS {
                    let src: Vec<u8> =
                        (1..=count).flat_map(|i| sample_bytes(i)[..FS].to_vec()).collect();
                    let len = offset + touched(count, x_stride, W) + 16; // untouched slack
                    let mut got = vec![0xAAu8; len];
                    let mut want = vec![0xAAu8; len];
                    // Independent reference: plain per-sample indexing.
                    let mut wp = offset;
                    for i in 0..count {
                        let mut b = [0u8; FS];
                        b.copy_from_slice(&src[i * FS..i * FS + FS]);
                        want[wp..wp + W].copy_from_slice(&conv_ref::<FS, W>(b));
                        wp += x_stride;
                    }

                    crate::probe::reset(crate::probe::Probe::ScatterTypedFast);
                    let end = scatter_convert(&mut got, offset, x_stride, count, &src, conv_ref::<FS, W>)
                        .unwrap();
                    let ctx = format!("FS={FS} W={W} x_stride={x_stride} count={count} offset={offset}");
                    assert_eq!(got, want, "{ctx}: bytes");
                    assert_eq!(end, wp, "{ctx}: returned write_ptr");
                    let fast = count > 0 && x_stride >= W;
                    assert_eq!(
                        crate::probe::get(crate::probe::Probe::ScatterTypedFast),
                        fast as u64,
                        "{ctx}: wrong path"
                    );
                }
            }
        }
    }

    #[test]
    fn scatter_convert_matches_reference_across_matrix() {
        check_scatter::<1, 1>();
        check_scatter::<2, 2>();
        check_scatter::<4, 4>();
        // Widening and narrowing, as the HALF<->FLOAT/UINT arms do.
        check_scatter::<2, 4>();
        check_scatter::<4, 2>();
    }

    fn check_fill<const W: usize>() {
        let value: [u8; W] = std::array::from_fn(|k| 0x10 + k as u8);
        for &xs in &STRIDES {
            let x_stride = if xs == usize::MAX { W } else { xs };
            for &count in &COUNTS {
                for &offset in &OFFSETS {
                    let len = offset + touched(count, x_stride, W) + 16;
                    let mut got = vec![0xAAu8; len];
                    let mut want = vec![0xAAu8; len];
                    let mut wp = offset;
                    for _ in 0..count {
                        want[wp..wp + W].copy_from_slice(&value);
                        wp += x_stride;
                    }
                    crate::probe::reset(crate::probe::Probe::ScatterTypedFast);
                    let end = scatter_fill(&mut got, offset, x_stride, count, value).unwrap();
                    let ctx = format!("W={W} x_stride={x_stride} count={count} offset={offset}");
                    assert_eq!(got, want, "{ctx}: bytes");
                    assert_eq!(end, wp, "{ctx}: returned write_ptr");
                    let fast = count > 0 && x_stride >= W;
                    assert_eq!(
                        crate::probe::get(crate::probe::Probe::ScatterTypedFast),
                        fast as u64,
                        "{ctx}: wrong path"
                    );
                }
            }
        }
    }

    #[test]
    fn scatter_fill_matches_reference_across_matrix() {
        check_fill::<2>();
        check_fill::<4>();
    }

    fn check_gather<const S: usize>() {
        for &xs in &STRIDES {
            let x_stride = if xs == usize::MAX { S } else { xs };
            for &count in &COUNTS {
                for &offset in &OFFSETS {
                    // Strided source: sample i at offset + i*x_stride.
                    let src_len = offset + touched(count, x_stride, S) + 16;
                    let src: Vec<u8> = (0..src_len).map(|i| sample_bytes(i + 1)[i % 4]).collect();
                    let dst_off = 5;
                    let len = dst_off + count * S + 16;
                    let mut got = vec![0xAAu8; len];
                    let mut want = vec![0xAAu8; len];
                    let mut rp = offset;
                    for i in 0..count {
                        let mut b = [0u8; S];
                        b.copy_from_slice(&src[rp..rp + S]);
                        want[dst_off + i * S..dst_off + (i + 1) * S].copy_from_slice(&conv_ref::<S, S>(b));
                        rp += x_stride;
                    }
                    crate::probe::reset(crate::probe::Probe::GatherTypedFast);
                    let (wo, rpe) =
                        gather_convert(&mut got, dst_off, &src, offset, count, x_stride, conv_ref::<S, S>)
                            .unwrap();
                    let ctx = format!("S={S} x_stride={x_stride} count={count} offset={offset}");
                    assert_eq!(got, want, "{ctx}: bytes");
                    assert_eq!((wo, rpe), (dst_off + count * S, rp), "{ctx}: returned offsets");
                    let fast = count > 0 && x_stride >= S;
                    assert_eq!(
                        crate::probe::get(crate::probe::Probe::GatherTypedFast),
                        fast as u64,
                        "{ctx}: wrong path"
                    );
                }
            }
        }
    }

    #[test]
    fn gather_convert_matches_reference_across_matrix() {
        check_gather::<2>();
        check_gather::<4>();
    }

    /// A decoded buffer shorter than the requested samples is an error, not
    /// a panic (the old closure indexed `read_buf` per sample and panicked).
    #[test]
    fn copy_into_frame_buffer_rejects_a_short_decoded_buffer() {
        let read_buf = [0u8; 7]; // 4 HALF samples need 8 bytes
        let mut write_buf = vec![0u8; 64];
        let err = copy_into_frame_buffer(
            &read_buf, 0, &mut write_buf, 0, 4, 2, false, 0.0,
            BufferFormat::Xdr, PixelType::Half, PixelType::Half,
        )
        .unwrap_err();
        assert!(matches!(err, ExrError::Input(_)), "{err:?}");
        assert!(write_buf.iter().all(|&b| b == 0), "nothing may be written before the check");
    }

    /// A line buffer too short for the gathered samples is an error, not a
    /// panic.
    #[test]
    fn copy_from_frame_buffer_rejects_a_short_line_buffer() {
        let read_buf = [1u8; 64];
        let mut write_buf = vec![0u8; 7]; // 4 HALF samples need 8 bytes
        let err = copy_from_frame_buffer(
            &mut write_buf, 0, &read_buf, 0, 4, 10, BufferFormat::Xdr, PixelType::Half,
        )
        .unwrap_err();
        assert!(matches!(err, ExrError::Logic(_)), "{err:?}");
    }
}

#[cfg(test)]
mod planner_fuzz {
    use super::{plane_starts, plan_disjoint_planes, RangeSet};
    use std::ops::Range;

    /// xorshift64: deterministic, no dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// RV_X3c E-2 / S-1: the ONE planner never panics on arbitrary intervals (empty, inverted,
    /// overlapping, beyond the buffers), accepts exactly the layouts that are pairwise disjoint and
    /// inside a plane, and hands out pieces that really are disjoint, of the right length and base.
    #[test]
    fn planner_never_panics_and_is_sound_and_complete() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut accepted, mut refused) = (0, 0);
        for _ in 0..30_000 {
            let nplanes = 1 + rng.next(4);
            let lens: Vec<usize> = (0..nplanes).map(|_| rng.next(41)).collect();
            let total: usize = lens.iter().sum();
            let nchunks = rng.next(7);
            let chunks: Vec<RangeSet> = (0..nchunks)
                .map(|_| {
                    (0..1 + rng.next(3))
                        .map(|_| {
                            let (a, b) = (rng.next(total + 6), rng.next(total + 6));
                            // Mostly well-formed non-empty, sometimes empty or inverted.
                            match rng.next(8) {
                                0 => a..a,
                                1 => a.max(b)..a.min(b),
                                _ => a.min(b)..a.max(b),
                            }
                        })
                        .collect()
                })
                .collect();

            let mut bufs: Vec<Vec<u8>> = lens.iter().map(|&l| vec![0u8; l]).collect();
            let mut planes: Vec<&mut [u8]> = bufs.iter_mut().map(|b| &mut b[..]).collect();
            let starts = plane_starts(&planes);

            // Oracle: pairwise disjoint non-empty ranges, each inside one plane.
            let flat: Vec<Range<usize>> =
                chunks.iter().flat_map(|c| c.iter().cloned()).filter(|r| r.start < r.end).collect();
            let inside = |r: &Range<usize>| {
                (0..nplanes).any(|p| lens[p] > 0 && starts[p] <= r.start && r.end <= starts[p] + lens[p])
            };
            let disjoint = flat.iter().enumerate().all(|(i, a)| {
                flat.iter().skip(i + 1).all(|b| a.end <= b.start || b.end <= a.start)
            });
            let valid = disjoint && flat.iter().all(inside);

            // `tags`: the pieces of an accepted plan, each filled with its own tag; the borrow of
            // `bufs` ends with this block, so the bytes can be inspected afterwards.
            let accepted_now = {
                let planned = plan_disjoint_planes(&mut planes, &chunks);
                assert_eq!(planned.is_some(), valid, "lens={lens:?} chunks={chunks:?}");
                match planned {
                    None => false,
                    Some(mut sets) => {
                        let mut tag = 0u8;
                        for (set, ranges) in sets.iter_mut().zip(&chunks) {
                            assert_eq!(set.len(), ranges.len());
                            for (piece, r) in set.iter_mut().zip(ranges) {
                                if r.start < r.end {
                                    assert_eq!((piece.piece.len(), piece.base), (r.end - r.start, r.start));
                                    tag = tag.wrapping_add(1).max(1);
                                    piece.piece.fill(tag);
                                } else {
                                    assert!(piece.piece.is_empty());
                                }
                            }
                        }
                        true
                    }
                }
            };
            drop(planes);
            if accepted_now {
                accepted += 1;
                // Distinct tags per piece and no shared byte: exactly the wanted number of bytes written.
                let written: usize = bufs.iter().map(|b| b.iter().filter(|&&x| x != 0).count()).sum();
                let wanted: usize = flat.iter().map(|r| r.end - r.start).sum();
                assert_eq!(written, wanted, "pieces overlapped: lens={lens:?} chunks={chunks:?}");
            } else {
                refused += 1;
            }
        }
        assert!(accepted > 1000 && refused > 1000, "fuzz did not exercise both outcomes ({accepted}/{refused})");
    }
}
