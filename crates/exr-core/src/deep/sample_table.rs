//! Shared deep sample-count addressing helpers used by the deep writers.
//!
//! These factor the small, bug-prone pieces of the deep "count then data"
//! model that are identical between the deep scanline writer
//! ([`crate::deep::scanline_output`]) and the deep tiled writer
//! ([`crate::deep::tiled_output`]) so the cumulative-table logic and the
//! flat-heap pixel addressing live in exactly one place.
//!
//! # The flat per-channel sample heap (safe-Rust adaptation of C++ `char**`)
//!
//! In C++ a deep channel's `base` is a `char**` — a per-pixel pointer into a
//! sample heap.  This port (see [`crate::deep::scanline_output`] module docs)
//! cannot model raw pointers (zero `unsafe`, no FFI), so the caller instead
//! supplies a **flat per-channel sample heap** in which each pixel's sample
//! array is laid out contiguously in **whole-data-window `(x, y)` raster
//! order**.  The byte offset at which pixel `(x, y)`'s samples begin within a
//! channel's heap is therefore `samples_before_pixel(x, y) * sample_stride`,
//! where `samples_before_pixel` is the running total of every sample of every
//! pixel that precedes `(x, y)` in raster order.  This is the same model the
//! deep scanline writer/reader use (`samples_before_line`), generalised to an
//! arbitrary pixel so the deep *tiled* gather can index a tile's pixels —
//! which are an interior rectangle of that same global raster order.

use crate::attr::{ChannelList, Compression};
use crate::error::{ExrError, Result};
use crate::io::IStream;
use crate::misc::pixel_type_size;

/// Byte width of one Xdr `int` cumulative sample-count table entry.
const SIZE_INT: usize = 4;

/// Byte offset of pixel (x,y)'s count entry: base_offset + y*y_stride + x*x_stride.
/// None if negative (out of buffer). The one address formula for every deep
/// count read/write/verify site.
///
/// `base_offset` is the sample-count slice's `base_offset` (C++'s base pointer):
/// for the historical whole-window layout it is `0` (absolute data-window
/// coordinates); a local `w*h` raster instead sets it to
/// `-(min_y*yStride + min_x*xStride)` so the low corner maps to offset `0`.
pub fn count_byte_offset(
    base_offset: i64,
    x_stride: i32,
    y_stride: i32,
    x: i32,
    y: i32,
) -> Option<usize> {
    let off = base_offset + y as i64 * y_stride as i64 + x as i64 * x_stride as i64;
    usize::try_from(off).ok()
}

/// Read one pixel's sample count from the caller's flat sample-count buffer —
/// port of the C++ inline `sampleCount(base, xStride, yStride, x, y)`
/// (ImfMisc.h), which reads the `int` at `base + y*yStride + x*xStride`.
///
/// The sample-count buffer holds one `u32` (Xdr LE) per pixel, addressed via the
/// shared [`count_byte_offset`] formula (`base_offset + y*yStride + x*xStride`) —
/// the sample-count slice's `base_offset` is the safe-Rust equivalent of C++'s
/// base pointer.  A negative computed offset or an out-of-buffer range is a
/// **hard error** ([`ExrError::Arg`]) — the caller-supplied buffer is untrusted,
/// and a zero fallback here would conflate "invalid memory" with "a legitimate
/// zero-sample pixel", silently producing a structurally-valid but
/// semantically-corrupt deep image.
pub fn sample_count_at(
    buf: &[u8],
    base_offset: i64,
    x_stride: i32,
    y_stride: i32,
    x: i32,
    y: i32,
) -> Result<u32> {
    crate::probe::hit(crate::probe::Probe::SampleCountAt);
    let off = count_byte_offset(base_offset, x_stride, y_stride, x, y).ok_or_else(|| {
        ExrError::Arg("Deep sample-count address is out of range (negative offset).".into())
    })?;
    match buf.get(off..off + 4) {
        Some(b) => Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        None => Err(ExrError::Arg(format!(
            "Deep sample-count address {off} is out of range of the sample-count \
             buffer (len {}).",
            buf.len()
        ))),
    }
}

/// Number of samples that precede pixel `(x, y)` in **whole-data-window raster
/// order** — i.e. Σ of the sample counts of every pixel on lines `[min_y, y)`
/// (all x in `[min_x, max_x]`) plus every pixel on line `y` strictly left of
/// `x` (`[min_x, x)`).
///
/// Multiplying by a channel's `sample_stride` gives that channel's byte offset
/// for pixel `(x, y)` in the flat per-channel heap, keeping the heap globally
/// contiguous per channel (see the module docs).  Used by the deep tiled gather
/// to address the pixels of a tile rectangle, which are an interior sub-window
/// of the same global raster order the caller laid the heap out in.
// Mirrors the C++ per-pixel sample-offset arithmetic; the window bounds + pixel
// are passed explicitly (8 params) to keep the helper pure and stateless.
#[allow(clippy::too_many_arguments)]
pub fn samples_before_pixel(
    buf: &[u8],
    base_offset: i64,
    x_stride: i32,
    y_stride: i32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    x: i32,
    y: i32,
) -> Result<u64> {
    let mut total: u64 = 0;
    // Full lines before y.
    let mut yy = min_y;
    while yy < y {
        for xx in min_x..=max_x {
            total += sample_count_at(buf, base_offset, x_stride, y_stride, xx, yy)? as u64;
        }
        yy += 1;
    }
    // Partial line y, pixels strictly left of x.
    let mut xx = min_x;
    while xx < x {
        total += sample_count_at(buf, base_offset, x_stride, y_stride, xx, y)? as u64;
        xx += 1;
    }
    Ok(total)
}

/// A **growable** per-row sample-count prefix cache, giving
/// [`samples_before_pixel`]-equivalent addressing in **O(1) amortized** (the
/// readers' sourcing rule is described below).
///
/// Before this, every deep tiled gather/scatter (and the deep scanline
/// writer's per-channel `samples_before_line`) recomputed the running total
/// from `(min_x, min_y)` on *every single pixel/line* lookup, making a whole
/// tiled read/write **O((W·H)²·channels)** — a real hang at production
/// resolution (a 1920x1080 4-channel tiled deep image is ~8×10¹² count
/// reads). This cache computes each row's total **at most once** (extending
/// forward as new rows are queried, regardless of query order — both
/// `INCREASING_Y` and `DECREASING_Y` writes/reads land on the same cached
/// values) for a total **O(width*height)** cost across a whole file, with
/// O(1) lookups thereafter.
///
/// Unlike a fixed eager whole-window table, this never reads a row that was
/// never queried — required because the deep tiled **writer** intentionally
/// allows a per-level sample-count buffer that only covers the rows a given
/// mip level needs (a file is written one level at a time, and
/// `sample_count_at` is the single authoritative bound); eagerly
/// scanning up to the file's global `max_y` would spuriously fail a
/// legitimately short per-level buffer.
///
/// Both deep **readers** source this cache from the caller-supplied
/// `sample_counts` buffer (already fully populated, independently per chunk,
/// by the mandatory first `read_pixel_sample_counts` pass) rather than by
/// re-decoding each earlier chunk's own count-table bytes from the stream —
/// so a damaged/missing earlier chunk no longer blocks reading a later line
/// during the second (`read_pixels`) pass, matching C++'s per-pointer
/// independence.
pub struct RowPrefixCache {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    /// `cache[i]` = total samples in rows `[min_y, min_y + i)` (`cache[0] ==
    /// 0` always). Extended lazily, one not-yet-summed row at a time.
    cache: Vec<u64>,
    /// The single most recently built row's full within-row cumulative array
    /// (`arr[i]` = Σ counts for `x` in `[min_x, min_x + i)`), and which `y`
    /// it is for. Without this, `before_pixel` walking `[min_x, x)` from
    /// scratch on every single pixel made a tile gather/scatter O(W) per
    /// pixel — for a tile-by-tile access pattern (all channels of one row,
    /// then the next row) this single-row cache turns repeat queries of the
    /// *same* row (across channels, and across horizontally adjacent tiles
    /// once their column reaches the cached row again) into O(1) array
    /// lookups, at the cost of one O(width) rebuild per distinct row visited.
    current_y: Option<i32>,
    current_row: Vec<u64>,
}

impl RowPrefixCache {
    /// Create an empty cache for one logical heap domain.
    pub fn new(min_x: i32, max_x: i32, min_y: i32) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            cache: vec![0],
            current_y: None,
            current_row: Vec::new(),
        }
    }

    /// Begin another domain/count buffer while retaining the prefix allocations.
    pub fn reset(&mut self, min_x: i32, max_x: i32, min_y: i32) {
        self.min_x = min_x;
        self.max_x = max_x;
        self.min_y = min_y;
        self.cache.clear();
        self.cache.push(0);
        self.current_y = None;
        self.current_row.clear();
    }

    /// Reuse the within-row array; count lookup may map the heap domain to storage.
    fn row_array<F>(&mut self, y: i32, count: &mut F) -> Result<&[u64]>
    where
        F: FnMut(i32, i32) -> Result<u32>,
    {
        if self.current_y != Some(y) {
            self.current_y = None;
            self.current_row.clear();
            self.current_row.push(0);
            let mut total = 0u64;
            for x in self.min_x..=self.max_x {
                total = total.checked_add(u64::from(count(x, y)?)).ok_or_else(|| {
                    ExrError::Arg("Deep row sample-count total overflows u64.".into())
                })?;
                self.current_row.push(total);
            }
            self.current_y = Some(y);
        }
        Ok(&self.current_row)
    }

    fn before_line_mapped<F>(&mut self, y: i32, count: &mut F) -> Result<u64>
    where
        F: FnMut(i32, i32) -> Result<u32>,
    {
        if y < self.min_y {
            return Err(ExrError::Arg(format!(
                "Deep sample heap row {y} lies before the first row {} of the range it is addressed from.",
                self.min_y
            )));
        }
        let want = (i64::from(y) - i64::from(self.min_y)) as usize;
        while self.cache.len() <= want {
            let row_y = (i64::from(self.min_y) + self.cache.len() as i64 - 1) as i32;
            let row_total = self.row_array(row_y, count)?.last().copied().unwrap_or(0);
            let total = self
                .cache
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(row_total)
                .ok_or_else(|| {
                    ExrError::Arg("Deep heap sample-count total overflows u64.".into())
                })?;
            self.cache.push(total);
        }
        Ok(self.cache[want])
    }

    /// Samples before a row using the count slice's ordinary coordinates.
    pub fn before_line(
        &mut self,
        buf: &[u8],
        base_offset: i64,
        x_stride: i32,
        y_stride: i32,
        y: i32,
    ) -> Result<u64> {
        self.before_line_mapped(y, &mut |x, y| {
            sample_count_at(buf, base_offset, x_stride, y_stride, x, y)
        })
    }

    /// Samples before a pixel using the count slice's ordinary coordinates.
    #[allow(clippy::too_many_arguments)]
    pub fn before_pixel(
        &mut self,
        buf: &[u8],
        base_offset: i64,
        x_stride: i32,
        y_stride: i32,
        x: i32,
        y: i32,
    ) -> Result<u64> {
        self.before_pixel_mapped(x, y, |x, y| {
            sample_count_at(buf, base_offset, x_stride, y_stride, x, y)
        })
    }

    /// Samples before a pixel in this heap domain, with independently mapped counts.
    pub fn before_pixel_mapped<F>(&mut self, x: i32, y: i32, mut count: F) -> Result<u64>
    where
        F: FnMut(i32, i32) -> Result<u32>,
    {
        if x < self.min_x || x > self.max_x {
            return Err(ExrError::Arg(format!(
                "Deep sample heap column {x} is outside {}..={}.",
                self.min_x, self.max_x
            )));
        }
        let line_total = self.before_line_mapped(y, &mut count)?;
        let idx = (i64::from(x) - i64::from(self.min_x)) as usize;
        let partial = self.row_array(y, &mut count)?[idx];
        line_total
            .checked_add(partial)
            .ok_or_else(|| ExrError::Arg("Deep heap sample-count total overflows u64.".into()))
    }
}
/// Encode one row's per-pixel sample counts as the OpenEXR **cumulative** count
/// table — the running total after each pixel, written as an Xdr LE `int` — and
/// append it to `out`.  This is the shared encode half of the deep "count then
/// data" model, used per-scanline by the deep scanline writer and per-tile-row
/// by the deep tiled writer; the running total resets to 0 at the start of each
/// call (i.e. each row).
///
/// The on-wire cumulative field is a **signed** `int`.  A row whose total would
/// exceed `i32::MAX` is not representable; C++ silently `wrapping_add`ed it into
/// a negative running total, writing a corrupt, non-monotonic table.  We reject
/// it with [`ExrError::Input`] instead, so a corrupt table is never emitted.  For
/// every valid row (total `<= i32::MAX`) the bytes are identical to the C++
/// output, preserving byte-exact parity.
pub fn encode_cumulative_row<I>(counts: I, out: &mut Vec<u8>) -> Result<()>
where
    I: IntoIterator<Item = u32>,
{
    // `cumulative` cannot overflow: the moment it passes `i32::MAX` we return,
    // so before each add it is `<= i32::MAX` and after `< 2^33` — far below u64.
    let mut cumulative: u64 = 0;
    for n in counts {
        cumulative += u64::from(n);
        if cumulative > i32::MAX as u64 {
            return Err(ExrError::Input(format!(
                "Deep row sample-count total {cumulative} exceeds the on-wire i32 limit."
            )));
        }
        out.extend_from_slice(&(cumulative as i32).to_le_bytes());
    }
    Ok(())
}

/// Decode `n` consecutive entries of a deep cumulative sample-count table (Xdr
/// LE `int` running totals) starting at entry index `base`, writing the
/// per-pixel counts (first differences) into `out[base .. base + n]`.  Inverse
/// of [`encode_cumulative_row`]; the running total resets to 0 at the start of
/// each call (one call per scanline / per tile row).
///
/// The table must be **non-negative and non-decreasing**: each cumulative value
/// has to be `>= 0` and `>= the previous one`.  C++ recovered the delta with
/// `wrapping_sub`, which turns a corrupt signed wrap (e.g. a jump to `i32::MIN`)
/// into a huge positive count that slips past a naive `delta < 0` check;
/// validating the cumulative value directly rejects such corruption up front.
/// For a valid table the recovered deltas are identical to the C++ result.
///
/// Caller contract: `out.len() >= base + n` (the callers size `out` to the exact
/// pixel count from already-validated window/tile geometry).
pub fn decode_cumulative_row(bytes: &[u8], base: usize, n: usize, out: &mut [u32]) -> Result<()> {
    let mut prev: i64 = 0;
    for k in 0..n {
        let idx = base + k;
        let off = idx * SIZE_INT;
        let b = bytes
            .get(off..off + SIZE_INT)
            .ok_or_else(|| ExrError::Input("Truncated deep count table.".into()))?;
        let cumulative = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64;
        if cumulative < 0 || cumulative < prev {
            return Err(ExrError::Input(format!(
                "Deep count table is not monotonically increasing at entry {idx} (corrupt chunk)."
            )));
        }
        out[idx] = (cumulative - prev) as u32;
        prev = cumulative;
    }
    Ok(())
}

/// Validate a deep chunk's declared **sample-data** sizes BEFORE any buffer is
/// allocated from the (untrusted) on-wire `packed_data` / `unpacked` u64 fields —
/// the guard against memory exhaustion via an attacker-declared chunk
/// size.
///
/// `per_pixel` are the chunk's already-decoded per-pixel sample counts (one entry
/// per chunk pixel — a whole scanline for the deep scanline reader, a tile
/// rectangle for the deep tiled reader), already validated non-negative and
/// monotonic by [`decode_cumulative_row`].  `channels` is the header channel list.
///
/// The deep writers emit exactly `unpackedDataSize = (Σ per_pixel) *
/// (Σ_channels pixel_type_size(channel))` bytes of sample data
/// (`gather_scanline` / `gather_tile` write `total_samples * sample_size` per
/// channel, in header-channel order), and store the sample data either compressed
/// (strictly smaller than unpacked) or raw (`packed == unpacked`), so a valid
/// chunk always has `unpacked == expected` and `packed_data <= unpacked`.  A
/// malformed file that declares either field near `u64::MAX` therefore fails this
/// check with [`ExrError::Input`] *before* a giant `vec![0u8; …]` is requested.
///
/// The expected-size computation itself uses checked u64 arithmetic so it can
/// never overflow/panic on adversarial input.  On success returns the validated
/// `(packed_data, unpacked)` as `usize`, ready for a now-bounded allocation.
pub fn check_sample_data_sizes(
    per_pixel: &[u32],
    channels: &ChannelList,
    packed_data: u64,
    unpacked: u64,
) -> Result<(usize, usize)> {
    // Σ of the chunk's per-pixel sample counts (checked; counts are u32).
    let mut total_samples: u64 = 0;
    for &c in per_pixel {
        total_samples = total_samples
            .checked_add(u64::from(c))
            .ok_or_else(|| ExrError::Input("Deep chunk sample-count total overflows u64.".into()))?;
    }
    // Σ over header channels of the on-disk per-sample byte size.
    let mut bytes_per_sample: u64 = 0;
    for (_name, ch) in channels.iter() {
        bytes_per_sample = bytes_per_sample
            .checked_add(pixel_type_size(ch.type_)? as u64)
            .ok_or_else(|| {
                ExrError::Input("Deep channel byte-size total overflows u64.".into())
            })?;
    }
    let expected = total_samples.checked_mul(bytes_per_sample).ok_or_else(|| {
        ExrError::Input("Deep chunk unpacked-size computation overflows u64.".into())
    })?;

    // C++ `OpenEXRCore/decoding.c:231-247` (`unpack_sample_table`):
    // `totsamp >= INT32_MAX` is rejected outright (our per-pixel counts are
    // already individually validated non-negative by `decode_cumulative_row`,
    // but the SUM across the whole chunk needs its own cap — the writer never
    // emits this, but a crafted file could combine many valid-looking rows).
    if total_samples >= i32::MAX as u64 {
        return Err(ExrError::Input(format!(
            "Deep chunk total sample count {total_samples} is at or above the \
             on-wire i32 limit (corrupt chunk)."
        )));
    }
    // C++ rejects only `totsamp * combSampSize > unpacked_size`
    // (`decoding.c:270`) — an `unpacked` LARGER than `expected` (e.g. a
    // writer that pads) is accepted, not just an exact match. We used to
    // require exact equality, which rejected valid third-party files.
    if unpacked < expected {
        return Err(ExrError::Input(format!(
            "Deep chunk unpackedDataSize {unpacked} is smaller than the size \
             {expected} implied by the decoded sample counts (corrupt chunk)."
        )));
    }
    // C++ enforces `packed == unpacked` only for `EXR_COMPRESSION_NONE`
    // (`OpenEXRCore/chunk.c` cross-check after the deep/non-deep leader
    // parse, shared by scanline and tiled); for a compressed chunk there is
    // no leader-level relationship between `packed` and `unpacked` at all
    // (the codec's own decompressor is the source of truth). The caller
    // (which knows whether this chunk used NONE) enforces the exact-match
    // case; this shared helper keeps the general `packed <= unpacked` guard
    // for every writer we know of (ours included) as an extra allocation
    // bound, since no real writer ever emits `packed > unpacked`.
    if packed_data > unpacked {
        return Err(ExrError::Input(format!(
            "Deep chunk packedDataSize {packed_data} exceeds unpackedDataSize \
             {unpacked} (corrupt chunk)."
        )));
    }
    // C++ `OpenEXRCore/chunk.c:1019-1037,1397-1420`: both size fields
    // must fit an `int` (`<= INT_MAX`). Without this cap, an internally
    // *consistent* pair of huge sizes (declared `unpacked` == the huge
    // `expected` computed from a crafted count table) would otherwise sail
    // through the equality check above and reach the caller's allocation.
    if unpacked > i32::MAX as u64 {
        return Err(ExrError::Input(format!(
            "Deep chunk unpackedDataSize {unpacked} exceeds INT_MAX (corrupt chunk)."
        )));
    }
    if packed_data > i32::MAX as u64 {
        return Err(ExrError::Input(format!(
            "Deep chunk packedDataSize {packed_data} exceeds INT_MAX (corrupt chunk)."
        )));
    }

    let packed_usize = usize::try_from(packed_data)
        .map_err(|_| ExrError::Input(format!("Packed data size {packed_data} too large.")))?;
    let unpacked_usize = usize::try_from(unpacked)
        .map_err(|_| ExrError::Input(format!("Unpacked data size {unpacked} too large.")))?;
    Ok((packed_usize, unpacked_usize))
}

/// Every check a deep chunk's declared data sizes get once its count table is
/// decoded, with `stream` positioned at the start of the packed sample data.
/// The ONE validation point for both deep readers (scanline and tiled), run on
/// every count-table read -- `readPixelSampleCounts` included, as in C++, where
/// the chunk-leader checks (`OpenEXRCore/chunk.c`) and `unpack_sample_table`'s
/// "Corrupt sample count table" (`decoding.c:270-275`) run whenever counts are
/// decoded, not only when sample data is.
///
/// Adds to [`check_sample_data_sizes`]: `packed == unpacked` under
/// `NoCompression` (`chunk.c` cross-check), and the packed data must fit in
/// the rest of the file (`chunk.c:1062-1066` / `:1397-1420`).
pub(crate) fn check_chunk_sizes(
    per_pixel: &[u32],
    channels: &ChannelList,
    compression: Compression,
    packed_data: u64,
    unpacked: u64,
    stream: &mut dyn IStream,
) -> Result<(usize, usize)> {
    let (packed, unpacked) = check_sample_data_sizes(per_pixel, channels, packed_data, unpacked)?;
    if compression == Compression::NoCompression && packed != unpacked {
        return Err(ExrError::Input(format!(
            "Mismatch between unpacked and packed size with uncompressed \
             data: packed is {packed}; unpacked is {unpacked}."
        )));
    }
    if let Some(file_size) = stream.size() {
        let remaining = file_size.saturating_sub(stream.tellg()?);
        if packed as u64 > remaining {
            return Err(ExrError::Input(format!(
                "Deep chunk packedDataSize {packed} exceeds the \
                 {remaining} bytes remaining in the file (corrupt chunk)."
            )));
        }
    }
    Ok((packed, unpacked))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack cumulative `int` values as the on-wire LE byte table.
    fn cum(vals: &[i32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn encode_then_decode_roundtrips() {
        let counts = [3u32, 0, 5, 2];
        let mut enc = Vec::new();
        encode_cumulative_row(counts.iter().copied(), &mut enc).unwrap();
        assert_eq!(enc, cum(&[3, 3, 8, 10])); // running totals
        let mut out = vec![0u32; counts.len()];
        decode_cumulative_row(&enc, 0, counts.len(), &mut out).unwrap();
        assert_eq!(out, counts);
    }

    #[test]
    fn encode_accepts_total_exactly_i32_max() {
        let mut enc = Vec::new();
        encode_cumulative_row([i32::MAX as u32], &mut enc).unwrap();
        assert_eq!(enc, cum(&[i32::MAX]));
    }

    #[test]
    fn encode_rejects_total_one_over_i32_max() {
        let mut enc = Vec::new();
        assert!(encode_cumulative_row([i32::MAX as u32, 1], &mut enc).is_err());
    }

    #[test]
    fn decode_rejects_negative_cumulative() {
        let mut out = vec![0u32; 1];
        assert!(decode_cumulative_row(&cum(&[i32::MIN]), 0, 1, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_signed_wrap_that_fooled_wrapping_sub() {
        // prev=1, cumulative=i32::MIN: old `wrapping_sub` yielded +i32::MAX and
        // passed the naive `delta < 0` guard; the checked decoder must reject it.
        let mut out = vec![0u32; 2];
        assert!(decode_cumulative_row(&cum(&[1, i32::MIN]), 0, 2, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_decreasing() {
        let mut out = vec![0u32; 2];
        assert!(decode_cumulative_row(&cum(&[5, 3]), 0, 2, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_truncated() {
        let mut out = vec![0u32; 2];
        assert!(decode_cumulative_row(&cum(&[5]), 0, 2, &mut out).is_err());
    }

    #[test]
    fn sample_count_at_ok_and_out_of_range() {
        // 2x1 buffer: counts 7, 9 (u32 LE), base_offset=0, x_stride=4, y_stride=8.
        let buf: Vec<u8> = [7u32, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(sample_count_at(&buf, 0, 4, 8, 0, 0).unwrap(), 7);
        assert_eq!(sample_count_at(&buf, 0, 4, 8, 1, 0).unwrap(), 9);
        // Past the end of the buffer -> hard error (not a zero fallback).
        assert!(sample_count_at(&buf, 0, 4, 8, 2, 0).is_err());
        // Negative computed offset -> hard error.
        assert!(sample_count_at(&buf, 0, 4, 8, -1, 0).is_err());
    }

    #[test]
    fn sample_count_at_applies_base_offset() {
        // A local raster whose logical origin is shifted: the two real count
        // entries (7, 9) live at bytes [8..16), preceded by two padding entries.
        // base_offset=8 addresses the local origin so pixel (0,0) -> byte 8.
        let buf: Vec<u8> = [0u32, 0, 7, 9]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(sample_count_at(&buf, 8, 4, 8, 0, 0).unwrap(), 7);
        assert_eq!(sample_count_at(&buf, 8, 4, 8, 1, 0).unwrap(), 9);
        assert_eq!(count_byte_offset(8, 4, 8, 0, 0), Some(8));
        // A base_offset that drives the resulting address negative -> Err / None.
        assert!(sample_count_at(&buf, -4, 4, 8, 0, 0).is_err());
        assert_eq!(count_byte_offset(-4, 4, 8, 0, 0), None);
    }

    #[test]
    fn decode_resets_per_row() {
        // Two 2-pixel rows; each call resets the running total to 0.
        // row0 counts 3,1 -> cum 3,4 ; row1 counts 2,2 -> cum 2,4
        let bytes = cum(&[3, 4, 2, 4]);
        let mut out = vec![0u32; 4];
        decode_cumulative_row(&bytes, 0, 2, &mut out).unwrap();
        decode_cumulative_row(&bytes, 2, 2, &mut out).unwrap();
        assert_eq!(out, [3, 1, 2, 2]);
    }

    /// Flat `u32` LE count buffer for a `w`x`h` window, `base_offset=0`,
    /// `x_stride=4`, `y_stride=4*w` — matches every call site's convention.
    fn count_buf(counts: &[u32], w: i32) -> (Vec<u8>, i32, i32) {
        let x_stride = 4;
        let y_stride = 4 * w;
        let buf: Vec<u8> = counts.iter().flat_map(|c| c.to_le_bytes()).collect();
        (buf, x_stride, y_stride)
    }

    #[test]
    fn row_prefix_before_line_matches_hand_sum() {
        // 3x2 window: row0 = [1,2,3] (total 6), row1 = [4,0,5] (total 9).
        let counts = [1u32, 2, 3, 4, 0, 5];
        let (buf, xs, ys) = count_buf(&counts, 3);
        let mut cache = RowPrefixCache::new(0, 2, 0);
        assert_eq!(cache.before_line(&buf, 0, xs, ys, 0).unwrap(), 0);
        assert_eq!(cache.before_line(&buf, 0, xs, ys, 1).unwrap(), 6);
        assert_eq!(cache.before_line(&buf, 0, xs, ys, 2).unwrap(), 15);
    }

    #[test]
    fn row_prefix_before_pixel_matches_hand_sum() {
        let counts = [1u32, 2, 3, 4, 0, 5];
        let (buf, xs, ys) = count_buf(&counts, 3);
        let mut cache = RowPrefixCache::new(0, 2, 0);
        // Row 0: before (0,0)=0, before (1,0)=1, before (2,0)=3.
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 0, 0).unwrap(), 0);
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 1, 0).unwrap(), 1);
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 2, 0).unwrap(), 3);
        // Row 1: before (0,1) = row0 total (6); before (2,1) = 6+4+0=10.
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 0, 1).unwrap(), 6);
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 2, 1).unwrap(), 10);
    }

    #[test]
    fn row_prefix_query_order_independent() {
        // Querying rows out of order (as a DECREASING_Y write/read would)
        // must give the same answers as querying in order.
        let counts = [2u32, 1, 0, 3, 2, 1];
        let (buf, xs, ys) = count_buf(&counts, 3);
        let mut forward = RowPrefixCache::new(0, 2, 0);
        let f0 = forward.before_line(&buf, 0, xs, ys, 0).unwrap();
        let f1 = forward.before_line(&buf, 0, xs, ys, 1).unwrap();

        let mut backward = RowPrefixCache::new(0, 2, 0);
        let b1 = backward.before_line(&buf, 0, xs, ys, 1).unwrap();
        let b0 = backward.before_line(&buf, 0, xs, ys, 0).unwrap();

        assert_eq!((f0, f1), (b0, b1));
    }

    #[test]
    fn row_prefix_revisiting_a_row_reuses_cache_correctly() {
        // Re-querying the SAME row after visiting others must still return
        // the correct value (proves the single-row `current_row` cache
        // invalidates correctly on eviction rather than returning stale data).
        let counts = [1u32, 1, 5, 5];
        let (buf, xs, ys) = count_buf(&counts, 2);
        let mut cache = RowPrefixCache::new(0, 1, 0);
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 1, 0).unwrap(), 1);
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 1, 1).unwrap(), 2 + 5); // evicts row 0
        assert_eq!(cache.before_pixel(&buf, 0, xs, ys, 1, 0).unwrap(), 1); // re-visit row 0
    }

    #[test]
    fn row_prefix_out_of_range_address_is_hard_error() {
        // A cache scoped wider than the buffer must error, not silently
        // treat the missing rows as zero (hard-error policy).
        let counts = [1u32, 1];
        let (buf, xs, ys) = count_buf(&counts, 2);
        let mut cache = RowPrefixCache::new(0, 1, 0);
        assert!(cache.before_line(&buf, 0, xs, ys, 5).is_err());
    }
}
