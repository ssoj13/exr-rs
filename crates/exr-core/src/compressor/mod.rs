//! Port of OpenEXR `ImfCompressor.h` / `ImfCompressor.cpp` — the abstract
//! compressor interface, codec factory, and NONE (uncompressed) path.
//!
//! This module defines the seam that the codecs (RLE, ZIP, PIZ, …) plug into.
//! Every codec in [`Compression::ALL`] has a factory branch; only
//! `Compression::Invalid` returns an `Err(ExrError::Arg(..))` — **not** a
//! panic — so the rest of the crate can call the factory and handle the error
//! cleanly.
//!
//! # Design notes vs C++
//!
//! The C++ `Compressor` class uses an internal `char*` output buffer that both
//! `compress` and `uncompress` point `outPtr` at.  Rust has no stable way to do
//! this with borrowed slices across trait objects, so we own `Vec<u8>` scratch
//! buffers inside each compressor and return `&[u8]` slices into them.  The
//! semantics are identical: the slice is valid until the next call.
//!
//! `newCompressor` returns `Ok(None)` for `NoCompression` — exactly as the C++
//! factory returns `nullptr` — because the NONE path means the caller writes raw
//! line data with no compressor object at all.
//!
//! `new_tile_compressor` mirrors `newTileCompressor`; likewise returns `Ok(None)`
//! for `NoCompression`.
//!
//! `num_lines_in_buffer` mirrors the free function `numLinesInBuffer(Compression)`
//! in `ImfCompressor.cpp` lines 352–360 (which delegates to
//! `getCompressionNumScanlines` already ported as `Compression::num_scanlines`).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfCompressor.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfCompressor.cpp>

use crate::attr::Compression;
use crate::error::{ExrError, Result};
use crate::header::Header;
use imath_rs::Box2i;

pub mod b44;
pub mod dwa;
pub mod htj2k;
pub mod piz;
pub mod predictor;
pub mod pxr24;
pub mod rle;
pub mod zip;
pub(crate) mod zlib;

pub use b44::B44Compressor;
pub use dwa::DwaCompressor;
pub use htj2k::Htj2kCompressor;
pub use piz::PizCompressor;
pub use pxr24::Pxr24Compressor;
pub use rle::RleCompressor;
pub use zip::ZipCompressor;

// ---------------------------------------------------------------------------
// Format enum — mirrors `Compressor::Format` (ImfCompressor.h lines 71-75)
// ---------------------------------------------------------------------------

/// Pixel-data format used by [`Compressor::compress`] and
/// [`Compressor::uncompress`].
///
/// Mirrors `Compressor::Format` in `ImfCompressor.h`.
///
/// * `Xdr` — the machine-independent "Xdr" on-disk format (little-endian,
///   half/float/int encoded byte-by-byte).  The default for all compressors
///   that do not override `format()`.
/// * `Native` — the host machine's native binary layout.  Used by codecs
///   (e.g. PIZ) that shuffle bytes themselves and therefore hand back data
///   already in the host byte order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Host-native binary format (`NATIVE` in C++).
    Native,
    /// Machine-independent Xdr format (`XDR` in C++, the default).
    Xdr,
}

// ---------------------------------------------------------------------------
// Compressor trait — object-safe abstract interface
// ---------------------------------------------------------------------------

/// Abstract compressor interface — port of the C++ `class Compressor` vtable.
///
/// Each codec implements this trait.  The factory [`new_compressor`] and
/// [`new_tile_compressor`] return `Box<dyn Compressor>`.
///
/// # Buffer ownership
/// Both `compress` and `uncompress` return a `&[u8]` that borrows from an
/// **internal** scratch buffer owned by the compressor.  The slice is valid
/// until the next call to either method.  This mirrors the C++ pattern of
/// returning `(outPtr, size)` pointing into a member `_memory_buffer`.
///
/// # Error handling
/// All methods return `Result`.  No method may panic, call `unwrap`, or call
/// `todo!`/`unimplemented!` — failures are reported as `Err(ExrError::..)`.
///
/// # `Send`
/// A `Box<dyn Compressor>` crosses thread boundaries in the four parallel
/// read/write call sites (`scanline::input`/`output`, `tiled::input`/
/// `output`), so the trait must be `Send`. Every codec struct holds only
/// owned `Vec`s/tables/plain fields (no thread-local or raw pointer state),
/// so this is a free upcast — the compiler verifies it, no `unsafe impl`
/// needed anywhere.
///
/// Those four call sites USED TO build the codec via rayon's `map_init`,
/// whose own docs are explicit that `init` runs "as needed" for a job split
/// — **not** once per worker thread and not once per item, only
/// APPROXIMATING (never guaranteeing) C++'s `ScanLineProcessGroup`/
/// per-thread `lineBuffers` pool. The measurement this doc used to say
/// would justify a real `CodecPool` happened: a `#[cfg(test)]`
/// construction counter showed `map_init` constructing a FRESH codec for
/// essentially EVERY chunk/tile at every thread count (256/256 scanline
/// chunks, ~8192/8192 tiles) — rayon's adaptive splitting was fine-grained
/// enough that "once per job split" was, in practice, indistinguishable
/// from the pre-`map_init` "once per chunk" bug it was meant to fix.
/// [`CodecPool`] is that real pool: checkout/checkin
/// around EACH item (not tied to rayon's split granularity at all), used by
/// all four flat call sites plus deep scanline/tiled read and write.
pub trait Compressor: Send {
    /// Maximum number of scan lines processed by a single call to
    /// `compress` / `uncompress`.
    ///
    /// Mirrors `virtual int numScanLines() const` (ImfCompressor.h line 62).
    fn num_scan_lines(&self) -> i32;

    /// Pixel-data format expected / produced by this compressor.
    ///
    /// The default C++ implementation returns `XDR`; every codec that handles
    /// data in native byte order overrides this to return `NATIVE`.
    ///
    /// Mirrors `virtual Format format() const` (ImfCompressor.h line 78).
    fn format(&self) -> Format;

    /// Compress `data` representing scan lines starting at `min_y`.
    ///
    /// Returns a slice into an internal buffer; valid until the next call.
    ///
    /// Mirrors `virtual int compress(inPtr, inSize, minY, outPtr&)`.
    fn compress(&mut self, data: &[u8], min_y: i32) -> Result<&[u8]>;

    /// Uncompress `data` representing compressed scan lines starting at
    /// `min_y`, expected to decode to exactly `expected_len` bytes.
    ///
    /// Returns a slice into an internal buffer; valid until the next call.
    ///
    /// `expected_len` mirrors the size C's `internal_exr_undo_*` entry
    /// points always take explicitly (e.g. `undo_zip_impl(..., uncompressed_size)`,
    /// `internal_exr_undo_rle(..., uncompressed_size)`) — every codec knows the
    /// caller's exact target length per call, not just an upper bound baked
    /// in at construction. Before this parameter existed, a codec that
    /// needed its OUTPUT capacity sized correctly (ZIP/ZIPS/PXR24: zlib does
    /// not record the uncompressed length, so the decode buffer must already
    /// be large enough) could only get that from a construction-time
    /// `max_scan_line_size` (`new_compressor`'s own parameter) — fine for
    /// scanline/tiled, where every chunk in a file shares one worst-case
    /// size, but wrong for deep, where the sample-count/data-table sizes
    /// genuinely vary chunk to chunk, forcing a FRESH compressor per chunk
    /// just to get the right buffer size. Every codec now
    /// derives its output capacity from `expected_len` per call instead, so
    /// ONE persistent compressor can serve every chunk regardless of size.
    /// A self-describing format (RLE: decode runs until the input is
    /// exhausted) does not NEED `expected_len` to know where to stop, but
    /// still receives it, matching C's uniform `undo_*` signature, and MAY
    /// use it to validate the decoded length against the caller's
    /// expectation rather than trusting the stream alone.
    ///
    /// Mirrors `virtual int uncompress(inPtr, inSize, minY, outPtr&)` plus
    /// the `uncompressed_size` every OpenEXRCore `internal_exr_undo_*` takes.
    fn uncompress(&mut self, data: &[u8], min_y: i32, expected_len: usize) -> Result<&[u8]>;

    /// Hand the codec's output buffer to the caller without a copy, then
    /// refill it from `pooled` — a caller with its own
    /// buffer pool (the parallel encode paths in `scanline::output`/
    /// `tiled::output`) uses this to avoid the `pool.take() + .to_vec()`
    /// copy after `compress()`/`compress_tile()`: `pooled` becomes the
    /// codec's NEW `out` (already correctly capacitied from the pool, so the
    /// next `compress()` call has no cold-start allocation either), and the
    /// codec's just-produced compressed bytes move out into `*pooled`
    /// (formerly the caller's pooled buffer) with no element copy — just a
    /// `mem::swap`.
    ///
    /// `produced_len` is the length of the slice the preceding
    /// `compress()`/`compress_tile()` returned. The default implementation
    /// swaps `self.out` out whole, which is right only when that slice WAS
    /// all of `self.out` (true for every codec in this crate). A mismatch
    /// would silently write the wrong bytes to the file, so it is a hard
    /// [`ExrError::Logic`] instead, with nothing swapped.
    fn swap_out(&mut self, pooled: &mut Vec<u8>, produced_len: usize) -> Result<()> {
        let out = self.out_buf_mut();
        if out.len() != produced_len {
            return Err(ExrError::Logic(format!(
                "swap_out: codec produced {produced_len} bytes but its out buffer holds {}",
                out.len()
            )));
        }
        std::mem::swap(out, pooled);
        // `pooled` now holds the produced bytes; `self.out` holds the pool
        // buffer's stale content, cleared so the codec starts clean.
        self.out_buf_mut().clear();
        Ok(())
    }

    /// The codec's own output buffer (backs the default [`Self::swap_out`]).
    /// Every codec in this crate names this field `out`; this accessor is
    /// the one place that convention is load-bearing rather than incidental.
    fn out_buf_mut(&mut self) -> &mut Vec<u8>;

    /// Compress one tile's pixel data, encoded against the tile's own pixel
    /// rectangle `range` (not the scanline rectangle derived from the data
    /// window's X extent).
    ///
    /// # Why this exists
    /// In OpenEXR v3.3.11 the base `Compressor::compressTile` simply forwards
    /// the tile's `Box2i range` straight into `runEncodeStep` (ImfCompressor.cpp
    /// lines 118-124), whereas `compress(minY)` rebuilds the range from
    /// `header.dataWindow()` and only overrides `range.min/max.y` (lines 92-103).
    /// For rectangle-insensitive codecs (NONE/RLE/ZIP/ZIPS) the two produce the
    /// same chunk, so the **default impl mirrors the byte-identical scanline
    /// path** by calling `compress(data, range.min.y)`.  Rectangle-sensitive
    /// codecs (PIZ/B44/B44A/PXR24/DWAA/DWAB) override this to encode against the
    /// tile's X *and* Y extent — see each codec's override.
    ///
    /// Port target: `Compressor::compressTile` (v3.3.11 ImfCompressor.cpp:118).
    fn compress_tile(&mut self, data: &[u8], range: Box2i) -> Result<&[u8]> {
        self.compress(data, range.min.y)
    }

    /// Uncompress one tile's compressed data, decoded against the tile's own
    /// pixel rectangle `range`.
    ///
    /// The default impl mirrors `Compressor::uncompressTile`
    /// (v3.3.11 ImfCompressor.cpp:126-132), which forwards `range` into
    /// `runDecodeStep`; for rectangle-insensitive codecs that is byte-identical
    /// to `uncompress(data, range.min.y, expected_len)`.  Rectangle-sensitive
    /// codecs override.
    fn uncompress_tile(&mut self, data: &[u8], range: Box2i, expected_len: usize) -> Result<&[u8]> {
        self.uncompress(data, range.min.y, expected_len)
    }
}

// ---------------------------------------------------------------------------
// NoneCompressor — identity codec for NoCompression (not used by factory;
// factory returns Ok(None) instead, but this type is available for tests)
// ---------------------------------------------------------------------------

/// Identity compressor for `Compression::NoCompression`.
///
/// The C++ factory returns `nullptr` for NONE, meaning the caller writes/reads
/// raw scan line data directly.  `new_compressor` therefore returns `Ok(None)`.
///
/// This struct exists for completeness and testing: it simply passes bytes
/// through unchanged, returns `Format::Xdr`, and reports 1 scan line per chunk
/// — exactly the values `ImfCompressor.cpp` would produce if a NONE compressor
/// object existed.  Scanline I/O uses `Ok(None)` from the factory and
/// never instantiates this directly.
pub struct NoneCompressor {
    /// Internal scratch buffer; compress/uncompress store the input slice copy
    /// here so they can return a `&[u8]` with the correct lifetime.
    buf: Vec<u8>,
}

impl NoneCompressor {
    /// Create a new identity compressor.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl Default for NoneCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for NoneCompressor {
    /// NONE stores one scan line per chunk.
    ///
    /// `Compression::NoCompression.num_scanlines()` returns 1; this mirrors that.
    fn num_scan_lines(&self) -> i32 {
        1
    }

    /// NONE uses XDR format — data is written in the machine-independent
    /// byte order that all non-NATIVE codecs use.
    ///
    /// C++: `Compressor::format()` default returns `XDR`.
    fn format(&self) -> Format {
        Format::Xdr
    }

    /// Pass-through: copies `data` into the internal buffer and returns it.
    fn compress(&mut self, data: &[u8], _min_y: i32) -> Result<&[u8]> {
        self.buf.clear();
        self.buf.extend_from_slice(data);
        Ok(&self.buf)
    }

    /// Pass-through: copies `data` into the internal buffer and returns it.
    fn uncompress(&mut self, data: &[u8], _min_y: i32, _expected_len: usize) -> Result<&[u8]> {
        self.buf.clear();
        self.buf.extend_from_slice(data);
        Ok(&self.buf)
    }

    fn out_buf_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }
}

// ---------------------------------------------------------------------------
// CodecPool — checkout/checkin reuse for Box<dyn Compressor>
// ---------------------------------------------------------------------------

/// A thread-safe pool of reusable `Box<dyn Compressor>` instances -- the
/// [`crate::buf::BufPool`] idea applied to codecs instead of byte buffers.
///
/// # Why this exists
/// See [`Compressor`]'s trait doc for the measurement that justified
/// building this: `map_init`'s "once per rayon job split" reuse turned out,
/// in practice, to be "once per item" at every thread count (adaptive
/// splitting is fine-grained enough that almost every chunk/tile IS its own
/// job split). `CodecPool` decouples reuse from rayon's split granularity
/// entirely: [`Self::checkout`]/[`Self::checkin`] happen once per item, and
/// the free list is simply whatever codecs are momentarily idle across ALL
/// worker threads -- the pool grows to at most the number of codecs
/// simultaneously in use (bounded by the thread count), then every later
/// item reuses one instead of constructing fresh.
///
/// # Soundness
/// Every codec in this crate is safe to reuse across successive, unrelated
/// items within the SAME `(compression, header)` configuration:
/// `compress`/`uncompress` never carry state across calls beyond their own
/// scratch buffers, which each call fully rewrites before returning (this
/// is the exact assumption `map_init`'s "one codec reused across every item
/// in a job split" design already relied on -- `CodecPool` only widens that
/// reuse scope, it does not introduce a new assumption). A caller must only
/// ever check out codecs built by the SAME `make` closure (same compression
/// type, same header, same size bound) into one pool -- this crate's own
/// call sites each own exactly one `CodecPool` per read/write call, so that
/// invariant holds by construction.
pub struct CodecPool {
    free: std::sync::Mutex<Vec<Box<dyn Compressor>>>,
    /// How many codecs `checkout` constructed rather than reused
    /// (`Probe::CodecCreated`; test-only, zero-sized otherwise). Object-owned
    /// because checkout runs on many rayon workers for one read/write; the
    /// owning pipeline [`absorb`](crate::probe::absorb)s it after joining.
    probes: crate::probe::SharedProbes,
}

impl CodecPool {
    /// A new, empty pool. The first `checkout` on every worker thread will
    /// construct fresh; the pool fills up as codecs are `checkin`'d.
    pub fn new() -> Self {
        CodecPool { free: std::sync::Mutex::new(Vec::new()), probes: Default::default() }
    }

    /// Check out a compressor: pop one from the free list if available,
    /// else call `make` to construct a fresh one (the same
    /// `new_compressor`/`new_tile_compressor` factory call every call site
    /// used to make per item). `Ok(None)` (`Compression::NoCompression`) is
    /// never pooled: there is no codec object to reuse, and `make` returns
    /// `Ok(None)` immediately. `Err` propagates a construction failure
    /// unchanged.
    pub fn checkout<F>(&self, make: F) -> Result<Option<Box<dyn Compressor>>>
    where
        F: FnOnce() -> Result<Option<Box<dyn Compressor>>>,
    {
        if let Some(c) = self.free_list().pop() {
            return Ok(Some(c));
        }
        self.probes.hit(crate::probe::Probe::CodecCreated);
        make()
    }

    /// Return a compressor to the pool for a future `checkout` to reuse.
    /// `None` (the `NoCompression` case, or nothing checked out) is a
    /// no-op.
    pub fn checkin(&self, compressor: Option<Box<dyn Compressor>>) {
        if let Some(c) = compressor {
            self.free_list().push(c);
        }
    }

    /// [`Self::checkout`] as a guard: the codec returns to the pool when the
    /// [`Lease`] drops, on EVERY exit path (an early `?` included), so a
    /// failing item cannot shrink the pool. Not returned while the thread is
    /// unwinding from a panic - a codec that was mid-call is discarded.
    pub fn lease<F>(&self, make: F) -> Result<Lease<'_>>
    where
        F: FnOnce() -> Result<Option<Box<dyn Compressor>>>,
    {
        Ok(Lease { pool: self, codec: self.checkout(make)? })
    }

    /// The free list. A poisoned lock is recovered, not re-raised: the list
    /// only holds whole, idle codecs (a codec in use during the panic is not
    /// in it), so it stays consistent, and the panic already propagates on
    /// its own thread. Every codec fully resets its scratch per call.
    fn free_list(&self) -> std::sync::MutexGuard<'_, Vec<Box<dyn Compressor>>> {
        self.free.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Codecs currently idle in the pool.
    #[cfg(test)]
    pub(crate) fn free_count(&self) -> usize {
        self.free_list().len()
    }

    /// This pool's probes (`Probe::CodecCreated`).
    #[cfg(test)]
    pub(crate) fn probes(&self) -> &crate::probe::SharedProbes {
        &self.probes
    }
}

/// A codec on loan from a [`CodecPool`] (see [`CodecPool::lease`]).
pub struct Lease<'a> {
    pool: &'a CodecPool,
    codec: Option<Box<dyn Compressor>>,
}

impl Lease<'_> {
    /// The codec, or `None` for `Compression::NoCompression` (raw path).
    pub fn as_mut(&mut self) -> Option<&mut Box<dyn Compressor>> {
        self.codec.as_mut()
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.pool.checkin(self.codec.take());
        }
    }
}

/// Hands this pool's probe counts to the thread that drops it: the owning
/// read/write call, which creates the pool as a local and only lends it to
/// its workers. A test on that thread then reads worker hits with
/// `probe::get` like any caller-thread probe.
#[cfg(test)]
impl Drop for CodecPool {
    fn drop(&mut self) {
        crate::probe::absorb(&self.probes);
    }
}

impl Default for CodecPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod codec_pool_tests {
    use super::*;

    fn make_header() -> Header {
        Header::new()
    }

    /// A checked-out-then-checked-in codec is reused by the next checkout
    /// instead of a fresh construction -- the pool's whole reason to exist.
    /// Proven via `created()` (not pointer identity, which `Box<dyn Trait>`
    /// doesn't expose conveniently): a SECOND checkout after a checkin must
    /// NOT bump the construction count.
    ///
    /// Mutation check: made `checkin` a no-op (never pushing to the free
    /// list) -- `created()` after the second checkout became 2 instead of
    /// 1 (a fresh construction every time, since the free list stayed
    /// empty). Reverted after confirming red.
    #[test]
    fn checked_out_then_checked_in_is_reused_not_reconstructed() {
        let pool = CodecPool::new();
        let hdr = make_header();
        let make = || new_compressor(Compression::Rle, 1024, &hdr);

        let c1 = pool.checkout(make).unwrap();
        assert_eq!(pool.probes().get(crate::probe::Probe::CodecCreated), 1, "first checkout on an empty pool must construct");
        pool.checkin(c1);

        let c2 = pool.checkout(make).unwrap();
        assert_eq!(pool.probes().get(crate::probe::Probe::CodecCreated), 1, "checkout after checkin must reuse, not construct fresh");
        pool.checkin(c2);

        let c3 = pool.checkout(make).unwrap();
        assert_eq!(pool.probes().get(crate::probe::Probe::CodecCreated), 1, "still reusing the same one codec on a third checkout");
        pool.checkin(c3);
    }

    /// Two SIMULTANEOUSLY checked-out codecs (never checked back in before
    /// the next checkout) force a second real construction -- the pool
    /// only ever hands out as many codecs as are actually available.
    #[test]
    fn two_concurrently_outstanding_checkouts_construct_two_codecs() {
        let pool = CodecPool::new();
        let hdr = make_header();
        let make = || new_compressor(Compression::Rle, 1024, &hdr);

        let c1 = pool.checkout(make).unwrap();
        let c2 = pool.checkout(make).unwrap();
        assert_eq!(pool.probes().get(crate::probe::Probe::CodecCreated), 2, "both outstanding at once -- the free list was empty both times");
        pool.checkin(c1);
        pool.checkin(c2);
        assert_eq!(pool.probes().get(crate::probe::Probe::CodecCreated), 2, "checking both back in must not construct anything new");

        // Now the pool has 2 idle codecs -- two MORE checkouts must reuse
        // both, not construct a third.
        let c4 = pool.checkout(make).unwrap();
        let c5 = pool.checkout(make).unwrap();
        assert_eq!(pool.probes().get(crate::probe::Probe::CodecCreated), 2, "two idle codecs already available -- must reuse both");
        pool.checkin(c4);
        pool.checkin(c5);
    }

    /// `NoCompression` never pools anything -- every checkout goes straight
    /// through to `make` (which is free: `Ok(None)`, no allocation), and
    /// `checkin(None)` is a documented no-op.
    #[test]
    fn no_compression_is_never_pooled() {
        let pool = CodecPool::new();
        let hdr = make_header();
        let make = || new_compressor(Compression::NoCompression, 1024, &hdr);

        let c1 = pool.checkout(make).unwrap();
        assert!(c1.is_none());
        pool.checkin(c1);

        let c2 = pool.checkout(make).unwrap();
        assert!(c2.is_none());
        pool.checkin(c2);
        // `created()` counts every `make` fallback call, including the
        // `Ok(None)` ones -- this asserts the SHAPE (both went through
        // `make`, neither came from a nonexistent free-list entry), not a
        // specific count.
        assert_eq!(pool.probes().get(crate::probe::Probe::CodecCreated), 2);
    }
}

// ---------------------------------------------------------------------------
// Factory — new_compressor / new_tile_compressor
// ---------------------------------------------------------------------------

/// Construct a compressor for `compression` type for a **scanline** image.
///
/// Returns `Ok(None)` when `compression` is `NoCompression`, because the caller
/// writes raw scan-line bytes directly — there is no compressor object.
///
/// Returns `Err(ExrError::Arg(..))` for `Compression::Invalid` and propagates
/// codec-constructor errors.  **Never panics.**
///
/// Mirrors `Compressor* newCompressor(Compression, size_t, const Header&)`
/// (ImfCompressor.cpp lines 278–346).
///
/// # Parameters
/// * `compression` — codec selector from the file header.
/// * `max_scan_line_size` — maximum uncompressed bytes per scan line; passed
///   to the codec constructor so it can pre-allocate output buffers.
/// * `_header` — file header, passed to the codec constructors that need it
///   (e.g. PIZ needs the channel list).
pub fn new_compressor(
    compression: Compression,
    max_scan_line_size: usize,
    header: &Header,
) -> Result<Option<Box<dyn Compressor>>> {
    // Structural detail: which scanline codec was selected (debug).
    log::debug!(
        "new_compressor: codec={compression:?} max_scan_line_size={max_scan_line_size}"
    );
    crate::probe::hit(crate::probe::Probe::NewCompressor);
    match compression {
        // C++: `case NO_COMPRESSION: default: break;` → returns nullptr.
        Compression::NoCompression => Ok(None),

        // C++: `case RLE_COMPRESSION: return new RleCompressor(hdr, maxScanLineSize);`
        Compression::Rle => Ok(Some(Box::new(rle::RleCompressor::new(
            max_scan_line_size,
        )))),

        // C++: `case ZIPS_COMPRESSION: return new ZipCompressor(hdr, maxScanLineSize, 1);`
        Compression::Zips => Ok(Some(Box::new(zip::ZipCompressor::new(
            header,
            max_scan_line_size,
            1,
        )?))),
        // C++: `case ZIP_COMPRESSION: return new ZipCompressor(hdr, maxScanLineSize, 16);`
        Compression::Zip => Ok(Some(Box::new(zip::ZipCompressor::new(
            header,
            max_scan_line_size,
            16,
        )?))),

        // C++: `case PIZ_COMPRESSION: return new PizCompressor(hdr, maxScanLineSize, 32);`
        // PIZ batches 32 scanlines per chunk (getCompressionNumScanlines(PIZ) == 32)
        // and is channel-aware, so the codec captures the channel layout from the header.
        Compression::Piz => Ok(Some(Box::new(piz::PizCompressor::new(
            header,
            max_scan_line_size,
            32,
        )?))),
        // C++: `case PXR24_COMPRESSION: return new Pxr24Compressor(hdr, maxScanLineSize, 16);`
        // PXR24 batches 16 scanlines per chunk (getCompressionNumScanlines(PXR24) == 16)
        // and is channel-aware, so the codec captures the channel layout from the header.
        Compression::Pxr24 => Ok(Some(Box::new(pxr24::Pxr24Compressor::new(
            header,
            max_scan_line_size,
            16,
        )?))),
        // C++: `case B44_COMPRESSION: return new B44Compressor(hdr, maxScanLineSize, 32, false);`
        // B44 batches 32 scanlines per chunk (getCompressionNumScanlines(B44) == 32)
        // and is channel-aware (HALF blocks vs FLOAT/UINT passthrough), so the codec
        // captures the channel layout from the header.  `optimize_flat=false` → plain B44.
        Compression::B44 => Ok(Some(Box::new(b44::B44Compressor::new(
            header,
            max_scan_line_size,
            32,
            false,
        )?))),
        // C++: `case B44A_COMPRESSION: return new B44Compressor(hdr, maxScanLineSize, 32, true);`
        // B44A is identical to B44 but enables the 3-byte flat-block encoding
        // (`optFlatFields == true`).
        Compression::B44a => Ok(Some(Box::new(b44::B44Compressor::new(
            header,
            max_scan_line_size,
            32,
            true,
        )?))),
        // C++: `case DWAA_COMPRESSION: return new DwaCompressor(hdr, DWAA, ..., 32, ...);`
        // DWAA batches 32 scanlines per chunk; scanline files use static Huffman
        // for AC (DEFLATE is only used for tiled DWAA), so `use_deflate_for_ac=false`.
        Compression::Dwaa => Ok(Some(Box::new(dwa::DwaCompressor::new(
            header,
            dwa::DWAA_SCANLINES,
            false,
        )?))),
        // C++: `case DWAB_COMPRESSION: return new DwaCompressor(hdr, DWAB, ..., 256, ...);`
        // DWAB batches 256 scanlines per chunk and always uses static Huffman.
        Compression::Dwab => Ok(Some(Box::new(dwa::DwaCompressor::new(
            header,
            dwa::DWAB_SCANLINES,
            false,
        )?))),
        // OpenEXR 3.4 HTJ2K extension: `new HTCompressor(hdr, maxScanLineSize, n)`
        // with n = 256 (`HTJ2K256`) or 32 (`HTJ2K32`).  The codec is
        // channel-aware (it builds a JPEG 2000 codestream per chunk), so it
        // captures the channel layout + data window from the header.
        Compression::Htj2k256 => Ok(Some(Box::new(htj2k::Htj2kCompressor::new(
            header,
            max_scan_line_size,
            256,
        )?))),
        Compression::Htj2k32 => Ok(Some(Box::new(htj2k::Htj2kCompressor::new(
            header,
            max_scan_line_size,
            32,
        )?))),
        Compression::Invalid => Err(ExrError::Arg(
            "Invalid/unknown compression type".into(),
        )),
    }
}

/// Clamp the buffer-sizing inputs to `new_tile_compressor` to what any single
/// tile of `header`'s image could ever actually need.
///
/// `tile_line_size`/`num_tile_lines` are the RAW nominal `tileDesc.xSize` /
/// `ySize` from the header, before clipping to any real tile's range. Every
/// per-codec constructor below sizes internal scratch / decode-capacity
/// buffers from these two numbers up front (`ZipCompressor::new`,
/// `Pxr24Compressor::new`, `PizCompressor::new`, `B44Compressor::new`,
/// `Htj2kCompressor::new` all compute `max_chunk_size = tile_line_size *
/// num_tile_lines` at construction), so a crafted header whose `tiles`
/// attribute declares a nominal tile far larger than the actual data window
/// (e.g. `32768x32767` over a 1x1 image) committed a multi-hundred-MB to
/// multi-GB allocation before a single real tile was ever read or written.
///
/// No tile chunk — nominal or clipped — can ever legitimately need more bytes
/// per line, or more lines, than the image's own data window has, so clamping
/// against the data window is always safe: it can only shrink an
/// attacker-inflated nominal tile size, never a real one (an interior tile
/// exactly `tileDesc.xSize` wide only exists when the image is at least that
/// wide to begin with).
fn clamp_tile_buffer_dims(
    header: &Header,
    tile_line_size: usize,
    num_tile_lines: usize,
) -> Result<(usize, usize)> {
    let bytes_per_pixel = crate::tiled::misc::calculate_bytes_per_pixel(header)?;
    let dw = header.data_window()?;
    let image_width = dw.max.x.saturating_sub(dw.min.x).saturating_add(1).max(0) as usize;
    let image_height = dw.max.y.saturating_sub(dw.min.y).saturating_add(1).max(0) as usize;

    let max_line_bytes = bytes_per_pixel.saturating_mul(image_width);
    Ok((
        tile_line_size.min(max_line_bytes),
        num_tile_lines.min(image_height),
    ))
}

/// Construct a compressor for `compression` type for a **tiled** image.
///
/// Returns `Ok(None)` for `NoCompression`; `Err` for `Compression::Invalid`.
///
/// Mirrors `Compressor* newTileCompressor(Compression, size_t, size_t,
/// const Header&)` (ImfCompressor.cpp lines 363–427).
///
/// # Parameters
/// * `compression` — codec selector.
/// * `_tile_line_size` — max uncompressed bytes per line within a tile.
/// * `_num_tile_lines` — max lines per tile.
/// * `_header` — file header, passed to the codec constructors that need it.
pub fn new_tile_compressor(
    compression: Compression,
    tile_line_size: usize,
    num_tile_lines: usize,
    header: &Header,
) -> Result<Option<Box<dyn Compressor>>> {
    // Clamp before any per-codec constructor runs — see
    // `clamp_tile_buffer_dims`'s doc comment.
    let (tile_line_size, num_tile_lines) =
        clamp_tile_buffer_dims(header, tile_line_size, num_tile_lines)?;
    // Structural detail: which tile codec was selected (debug).
    log::debug!(
        "new_tile_compressor: codec={compression:?} tile_line_size={tile_line_size} \
         num_tile_lines={num_tile_lines}"
    );
    match compression {
        Compression::NoCompression => Ok(None),

        // C++: `new RleCompressor(hdr, uiMult(tileLineSize, numTileLines))`.
        Compression::Rle => Ok(Some(Box::new(rle::RleCompressor::new(
            crate::checked_arithmetic::ui_mult(tile_line_size, num_tile_lines)?,
        )))),
        // C++ `newTileCompressor` (ImfCompressor.cpp:375-378): BOTH ZIPS and ZIP
        // build `new ZipCompressor(hdr, tileLineSize, numTileLines)` — the
        // per-tile scanline count is the TILE height (`numTileLines`), NOT the
        // scanline-path 1/16.  The ZipCompressor sizes its chunk buffer as
        // `maxScanLineSize * numScanLines`, so passing `num_tile_lines` makes a
        // whole tile fit (the earlier hardcoded 1/16 under-sized the ZIPS tile
        // buffer to a single line → `InsufficientSpace` on tiles taller than 1).
        Compression::Zips | Compression::Zip => Ok(Some(Box::new(zip::ZipCompressor::new(
            header,
            tile_line_size,
            num_tile_lines.max(1) as i32,
        )?))),
        // C++ `newTileCompressor`: `new PizCompressor(hdr, tileLineSize * numTileLines, numTileLines)`.
        Compression::Piz => Ok(Some(Box::new(piz::PizCompressor::new(
            header,
            tile_line_size,
            num_tile_lines.max(1) as i32,
        )?))),
        // C++ `newTileCompressor`: `new Pxr24Compressor(hdr, tileLineSize, numTileLines)`.
        Compression::Pxr24 => Ok(Some(Box::new(pxr24::Pxr24Compressor::new(
            header,
            tile_line_size,
            num_tile_lines.max(1) as i32,
        )?))),
        // C++ `newTileCompressor`: `new B44Compressor(hdr, tileLineSize * numTileLines, numTileLines, false)`.
        Compression::B44 => Ok(Some(Box::new(b44::B44Compressor::new(
            header,
            tile_line_size,
            num_tile_lines.max(1) as i32,
            false,
        )?))),
        // C++ `newTileCompressor`: `new B44Compressor(hdr, ..., numTileLines, true)` (B44A).
        Compression::B44a => Ok(Some(Box::new(b44::B44Compressor::new(
            header,
            tile_line_size,
            num_tile_lines.max(1) as i32,
            true,
        )?))),
        // C++ `newTileCompressor` (ImfCompressor.cpp:403-416): BOTH DWAA and DWAB
        // build `new DwaCompressor(hdr, tileLineSize, numTileLines, ac)` — the
        // per-tile scanline count is the TILE height (`numTileLines`), NOT the
        // scanline-path 32/256.  (OpenEXRCore equivalently sets
        // `_numScanLines = chunk.height` per chunk, internal_dwa_compressor.h:141/181.)
        // Passing a hardcoded 32/256 capped `rectangle()` at 32 rows, so a tile
        // taller than the DWA block (e.g. 64×64) decoded only 32 of its rows
        // ("size 20480 ≠ 40960").  DWAA uses DEFLATE for AC, DWAB STATIC_HUFFMAN —
        // exactly the `true`/`false` here.
        Compression::Dwaa => Ok(Some(Box::new(dwa::DwaCompressor::new(
            header,
            num_tile_lines.max(1) as i32,
            true,
        )?))),
        Compression::Dwab => Ok(Some(Box::new(dwa::DwaCompressor::new(
            header,
            num_tile_lines.max(1) as i32,
            false,
        )?))),
        // OpenEXR 3.4 HTJ2K extension (tiled): like the C++ `newTileCompressor`
        // for the other channel-aware codecs, the per-tile codec processes the
        // tile's lines as one chunk; pass `numTileLines` as the scanline count.
        Compression::Htj2k256 | Compression::Htj2k32 => {
            Ok(Some(Box::new(htj2k::Htj2kCompressor::new(
                header,
                tile_line_size,
                num_tile_lines.max(1) as i32,
            )?)))
        }
        Compression::Invalid => Err(ExrError::Arg(
            "Invalid/unknown compression type".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// num_lines_in_buffer — free function
// ---------------------------------------------------------------------------

/// Return the number of scan lines compressed into a single chunk for `comp`.
///
/// Returns `Err(ExrError::Arg)` for `Compression::Invalid` (unknown type),
/// mirroring the C++ throw in `numLinesInBuffer` when `numScanlines < 1`.
///
/// Mirrors the free function `int numLinesInBuffer(Compression comp)` in
/// `ImfCompressor.cpp` lines 352–360, which delegates to
/// `getCompressionNumScanlines` (already ported as `Compression::num_scanlines`).
///
/// Also mirrors `defaultFormat(Compressor*)` in `ImfMisc.cpp` line 251:
/// when there is no compressor object (NONE path, `Ok(None)` from factory),
/// the caller should use `Format::Xdr` — just as `defaultFormat(nullptr)`
/// returns `Compressor::XDR`.
pub fn num_lines_in_buffer(comp: Compression) -> Result<i32> {
    let n = comp.num_scanlines();
    if n < 1 {
        // C++: throw IEX_NAMESPACE::ArgExc("Unknown compression type")
        return Err(ExrError::Arg(format!(
            "Unknown compression type (num_scanlines={n}) in num_lines_in_buffer"
        )));
    }
    Ok(n)
}

/// Decide which bytes a writer actually stores for one chunk: the compressed
/// encoding, or the raw (uncompressed) block.
///
/// **This is the single source of truth for the "store raw" decision.**  It
/// belongs to the *writer*, not to the codec: a codec compresses, the writer
/// decides.  Every chunk-writing seam (scanline, tiled, deep scanline, deep
/// tiled — single-part and multipart alike) must route its chunk through this
/// function before recording `dataSize` and writing the bytes.
///
/// # The rule
/// C++ keeps the compressed chunk only when it is **strictly smaller** than the
/// raw block; at equality (and of course when the codec expanded) it stores the
/// raw bytes:
///
/// * `ImfOutputFile.cpp:573` — `if (compSize < _lineBuffer->dataSize) {…}`
/// * `ImfTiledOutputFile.cpp:783` — `if (compSize < _tileBuffer->dataSize) {…}`
/// * `ImfDeepScanLineOutputFile.cpp:751` — same test for the sample data
///   (the sample-count table uses the mirror-image `>=` test at
///   `ImfDeepScanLineOutputFile.cpp:723`)
/// * `ImfDeepTiledOutputFile.cpp:1010` — same test for the sample data
///   (count table: `ImfDeepTiledOutputFile.cpp:973`, against
///   `maxSampleCountTableSize`)
///
/// # Why equality must store raw
/// Readers detect a raw passthrough purely by size: a chunk whose stored
/// `dataSize` equals the block's uncompressed size is taken verbatim, with no
/// codec invoked.  That is how both C++ (the core's `decompress_data`) and
/// every reader in this crate work — the complete list of the reader-side
/// mirrors of this rule, so the two sides can be checked against each other:
///
/// | Reader | Raw test |
/// |---|---|
/// | `scanline/input.rs` `read_chunk` | `data_size == expected_uncompressed` |
/// | `tiled/input.rs` `read_tile_data` | `data_size == expected_uncompressed` |
/// | `deep/scanline_input.rs` `read_count_table` | `packed_count == raw_table_size` |
/// | `deep/scanline_input.rs` sample data | `packed_data == unpacked_size` |
/// | `deep/tiled_input.rs` count table | `packed_count == max_sample_count_table_size` (or `== table_data_size`) |
/// | `deep/tiled_input.rs` sample data | `packed_data == unpacked_size` |
///
/// Writing a *compressed* chunk that happens to be exactly the
/// uncompressed size therefore produces a file that every implementation —
/// ours and C++ — reads back scrambled.  Some codecs' encodings hit that size
/// exactly on small images (repro: a 32x32 two-FLOAT B44 file).
///
/// # NATIVE format
/// C++ has an `else if (format == Compressor::NATIVE) convertToXdr(…)` branch
/// on the raw path, because a NATIVE codec's raw block is still in host byte
/// order.  Every codec ported here reports [`Format::Xdr`] (verified: the only
/// `fn format` impls in `compressor/` all return `Format::Xdr`), so that branch
/// is unreachable and no conversion is needed.  The writer seams assert this
/// with `debug_assert_eq!(self.format, Format::Xdr)`.  **If a codec ever
/// reports [`Format::Native`], `convertToXdr` must be ported before this
/// helper's raw path is correct for it.**
#[inline]
pub fn stored_chunk<'a>(compressed: &'a [u8], raw: &'a [u8]) -> &'a [u8] {
    if compressed.len() < raw.len() {
        compressed
    } else {
        raw
    }
}

/// The owned-buffer form of [`stored_chunk`], for the writers: turns one
/// gathered chunk (`raw[..data_size]`, usually a `BufPool` buffer) into
/// its on-disk bytes without copying. NO_COMPRESSION and the raw-store
/// case return `raw` itself, truncated. A codec that wins (strictly
/// smaller, as in C++) moves its output out through
/// [`Compressor::swap_out`] and takes `raw`'s allocation as its next `out`.
/// `compress` runs the codec (scanline `compress` or `compress_tile`) and
/// returns the produced length.
pub(crate) fn finish_chunk(
    compressor: Option<&mut Box<dyn Compressor>>,
    mut raw: Vec<u8>,
    data_size: usize,
    compress: impl FnOnce(&mut dyn Compressor, &[u8]) -> Result<usize>,
) -> Result<Vec<u8>> {
    raw.truncate(data_size);
    let Some(c) = compressor else {
        return Ok(raw);
    };
    if raw.is_empty() {
        return Ok(raw);
    }
    let produced = compress(c.as_mut(), &raw)?;
    if produced < raw.len() {
        c.swap_out(&mut raw, produced)?;
    }
    Ok(raw)
}

/// The grow-or-truncate scratch-buffer policy
/// moved to [`crate::buf::ensure_len`] -- a neutral module, since
/// this helper is used far outside the compressor (scanline/tiled read/
/// write, `exr-gpu`'s upload/readback). Re-exported here `pub(crate)` for
/// the many existing internal call sites inside this module tree that still
/// say `compressor::ensure_len`; new code should use `crate::buf::ensure_len`
/// directly. (V3 had widened this to a public item living here; moved to
/// `buf` as part of the BufPool work so the public API lives in one place.)
pub(crate) use crate::buf::ensure_len;

/// Return the pixel-data [`Format`] for an optional compressor.
///
/// When no compressor is present (NONE path — `Ok(None)` from the factory),
/// returns `Format::Xdr`, mirroring C++ `defaultFormat(nullptr) == Compressor::XDR`
/// (`ImfMisc.cpp` line 251–254).
///
/// When a compressor is present, delegates to `compressor.format()`.
pub fn default_format(compressor: Option<&dyn Compressor>) -> Format {
    match compressor {
        Some(c) => c.format(),
        // C++ defaultFormat(nullptr) returns Compressor::XDR
        None => Format::Xdr,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;

    fn make_header() -> Header {
        Header::new()
    }

    /// `Compressor::swap_out`'s default impl swaps the
    /// codec's `out` buffer with the caller's pooled one with NO element
    /// copy, then clears whichever buffer is now `self.out` (the pooled
    /// buffer's OLD, stale content) -- NOT the buffer handed back to the
    /// caller, which must keep the codec's just-produced bytes intact.
    /// Poisons the pooled buffer with implausible bytes before the swap (so
    /// a "clear the wrong side" bug would leave them behind in the RETURNED
    /// buffer) and checks BOTH sides: the returned buffer holds exactly what
    /// `compress()` produced, and the codec's own `out` (probed via a
    /// second `compress()` call reusing the now-swapped-in buffer) is not
    /// contaminated by the pooled buffer's prior content.
    /// `swap_out` hands over the codec's whole `out` buffer, so it must refuse
    /// when the length `compress()` reported differs from it. Otherwise a
    /// codec returning a sub-slice would silently write extra bytes.
    #[test]
    fn swap_out_rejects_a_length_mismatch_and_swaps_nothing() {
        let mut c = NoneCompressor::new();
        let produced = c.compress(b"hello world", 0).unwrap().len();
        let mut pooled = vec![0xAAu8; 4];
        match c.swap_out(&mut pooled, produced - 1) {
            Err(ExrError::Logic(_)) => {}
            other => panic!("expected ExrError::Logic, got {other:?}"),
        }
        assert_eq!(pooled, vec![0xAAu8; 4], "nothing may be swapped on a mismatch");
        assert_eq!(c.buf.len(), produced, "the codec keeps its output on a mismatch");
    }

    #[test]
    fn swap_out_moves_bytes_without_copying_and_clears_the_recipient_not_the_payload() {
        let mut c = NoneCompressor::new();
        let produced = c.compress(b"hello world", 0).unwrap().to_vec();

        // Poison the "pooled" buffer the caller hands in -- implausible bytes
        // at a length that does NOT match `produced`, so any leftover would
        // be visually obvious in a failed assertion.
        let mut pooled = vec![0xAAu8; 64];
        let pooled_capacity_before = pooled.capacity();

        c.swap_out(&mut pooled, produced.len()).unwrap();

        // The caller's buffer must now hold EXACTLY the compressed bytes --
        // no poison bytes left over, no truncation/extension bug.
        assert_eq!(pooled, produced, "swap_out must hand back the codec's exact output, byte for byte");

        // Reusing the pool buffer's allocation (not reallocating) is the
        // whole point of swap_out -- the codec's `out` should now be the
        // (cleared) former pooled buffer, so a FRESH compress() reuses that
        // capacity rather than growing from zero.
        let next = c.compress(b"a different, unrelated payload", 0).unwrap().to_vec();
        assert_eq!(next, b"a different, unrelated payload", "the codec's out buffer must be clean after swap_out, not contaminated by the pooled buffer's prior 0xAA content");
        // (best-effort capacity check -- NoneCompressor's buf started life at
        // whatever `pooled`'s allocation was, so it should be at least that
        // big without a fresh grow from empty)
        assert!(
            c.buf.capacity() >= pooled_capacity_before || next.len() <= pooled_capacity_before,
            "expected the swapped-in buffer's capacity to be reused"
        );
    }

    /// A crafted header whose `tiles` attribute declares a
    /// nominal tile far larger than the actual data window must have its
    /// buffer-sizing inputs clamped down to the image's own extent, not the
    /// raw nominal tile size — see `clamp_tile_buffer_dims`'s doc comment.
    /// Repro shape:
    /// `tiles = 32768x32767` over a 4x4 image.
    #[test]
    fn clamp_tile_buffer_dims_bounds_a_crafted_oversized_nominal_tile() {
        use crate::attr::{
            Channel, ChannelList, ChannelListAttribute, LevelMode, LevelRoundingMode, LineOrder,
            PixelType, TileDescription, TileDescriptionAttribute,
        };
        use imath_rs::V2f;

        let mut hdr = Header::with_dimensions(
            4,
            4,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Y", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 32768,
                y_size: 32767,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .unwrap();

        // Nominal (unclipped) inputs, as `new_tile_compressor`'s callers
        // compute them from the header alone — bytes_per_pixel(FLOAT=4) *
        // 32768 per line, 32767 lines: ~4.3 GB unclamped.
        let bpp = 4usize;
        let nominal_line_bytes = bpp * 32768;
        let nominal_lines = 32767usize;

        let (clamped_line_bytes, clamped_lines) =
            clamp_tile_buffer_dims(&hdr, nominal_line_bytes, nominal_lines).unwrap();

        // The 4x4 image can never need more than 4 pixels (16 bytes) per line
        // or more than 4 lines, however large the nominal tile claims to be.
        assert_eq!(clamped_line_bytes, bpp * 4, "line-bytes must clamp to the image width");
        assert_eq!(clamped_lines, 4, "line count must clamp to the image height");
        assert!(
            clamped_line_bytes.saturating_mul(clamped_lines) < 1_000_000,
            "clamped buffer size must stay tiny, not the ~4.3 GB nominal size"
        );
    }

    /// A real tiled file whose data window is genuinely large must NOT be
    /// clamped below what a real interior tile needs (no false-positive
    /// shrink of legitimate content).
    #[test]
    fn clamp_tile_buffer_dims_keeps_legitimate_large_tiles() {
        use crate::attr::{
            Channel, ChannelList, ChannelListAttribute, LevelMode, LevelRoundingMode, LineOrder,
            PixelType, TileDescription, TileDescriptionAttribute,
        };
        use imath_rs::V2f;

        let mut hdr = Header::with_dimensions(
            4096,
            4096,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Y", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr.insert(
            "tiles",
            TileDescriptionAttribute::new(TileDescription {
                x_size: 64,
                y_size: 64,
                mode: LevelMode::OneLevel,
                rounding_mode: LevelRoundingMode::RoundDown,
            }),
        )
        .unwrap();

        let bpp = 2usize; // HALF
        let nominal_line_bytes = bpp * 64;
        let nominal_lines = 64usize;
        let (clamped_line_bytes, clamped_lines) =
            clamp_tile_buffer_dims(&hdr, nominal_line_bytes, nominal_lines).unwrap();
        assert_eq!(clamped_line_bytes, nominal_line_bytes, "a real 64-wide tile in a 4096-wide image must not shrink");
        assert_eq!(clamped_lines, nominal_lines, "a real 64-tall tile in a 4096-tall image must not shrink");
    }

    // ---- new_compressor ----

    #[test]
    fn none_returns_ok_none() {
        let hdr = make_header();
        let result = new_compressor(Compression::NoCompression, 1024, &hdr);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn zip_returns_compressor_16_lines() {
        // ZIP: 16 scan lines per chunk, Xdr.
        let hdr = make_header();
        let c = new_compressor(Compression::Zip, 1024, &hdr).unwrap();
        let c = c.expect("ZIP factory must return a compressor");
        assert_eq!(c.num_scan_lines(), 16);
        assert_eq!(c.format(), Format::Xdr);
    }

    #[test]
    fn zips_returns_compressor_1_line() {
        // ZIPS is implemented: 1 scan line per chunk, Xdr.
        let hdr = make_header();
        let c = new_compressor(Compression::Zips, 1024, &hdr).unwrap();
        let c = c.expect("ZIPS factory must return a compressor");
        assert_eq!(c.num_scan_lines(), 1);
        assert_eq!(c.format(), Format::Xdr);
    }

    #[test]
    fn rle_returns_compressor() {
        // RLE: the factory yields a codec.
        let hdr = make_header();
        let c = new_compressor(Compression::Rle, 1024, &hdr).unwrap();
        let c = c.expect("RLE factory must return a compressor");
        assert_eq!(c.num_scan_lines(), 1);
        assert_eq!(c.format(), Format::Xdr);
    }

    #[test]
    fn tile_rle_returns_compressor() {
        let hdr = make_header();
        let c = new_tile_compressor(Compression::Rle, 64, 32, &hdr).unwrap();
        assert!(c.is_some(), "tile RLE factory must return a compressor");
    }

    #[test]
    fn pxr24_returns_compressor_16_lines() {
        // PXR24: 16 scan lines per chunk, Xdr.
        let hdr = make_header();
        let c = new_compressor(Compression::Pxr24, 1024, &hdr).unwrap();
        let c = c.expect("PXR24 factory must return a compressor");
        assert_eq!(c.num_scan_lines(), 16);
        assert_eq!(c.format(), Format::Xdr);
    }

    #[test]
    fn piz_returns_compressor_32_lines() {
        // PIZ: 32 scan lines per chunk, Xdr.
        let hdr = make_header();
        let c = new_compressor(Compression::Piz, 1024, &hdr).unwrap();
        let c = c.expect("PIZ factory must return a compressor");
        assert_eq!(c.num_scan_lines(), 32);
        assert_eq!(c.format(), Format::Xdr);
    }

    #[test]
    fn b44_and_b44a_return_compressor_32_lines() {
        // B44/B44A: 32 scan lines per chunk, Xdr.
        let hdr = make_header();
        for codec in [Compression::B44, Compression::B44a] {
            let c = new_compressor(codec, 1024, &hdr).unwrap();
            let c = c.expect("B44/B44A factory must return a compressor");
            assert_eq!(c.num_scan_lines(), 32, "{codec:?}");
            assert_eq!(c.format(), Format::Xdr, "{codec:?}");
        }
    }

    #[test]
    fn dwaa_dwab_return_compressor_with_correct_block_height() {
        // DWAA/DWAB: 32 / 256 scan lines, Xdr.
        let hdr = make_header();
        for (codec, lines) in [(Compression::Dwaa, 32), (Compression::Dwab, 256)] {
            let c = new_compressor(codec, 1024, &hdr).unwrap();
            let c = c.expect("DWAA/DWAB factory must return a compressor");
            assert_eq!(c.num_scan_lines(), lines, "{codec:?}");
            assert_eq!(c.format(), Format::Xdr, "{codec:?}");
        }
    }

    #[test]
    fn invalid_returns_err() {
        let hdr = make_header();
        let result = new_compressor(Compression::Invalid, 1024, &hdr);
        assert!(result.is_err());
    }

    // ---- new_tile_compressor ----

    #[test]
    fn tile_none_returns_ok_none() {
        let hdr = make_header();
        let result = new_tile_compressor(Compression::NoCompression, 64, 32, &hdr);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn tile_dwaa_returns_compressor() {
        let hdr = make_header();
        let c = new_tile_compressor(Compression::Dwaa, 64, 32, &hdr).unwrap();
        assert!(c.is_some(), "tile DWAA factory must return a compressor");
    }

    #[test]
    fn tile_invalid_returns_err() {
        let hdr = make_header();
        let result = new_tile_compressor(Compression::Invalid, 64, 32, &hdr);
        assert!(result.is_err());
    }

    #[test]
    fn tile_piz_returns_compressor() {
        let hdr = make_header();
        let c = new_tile_compressor(Compression::Piz, 64, 32, &hdr).unwrap();
        assert!(c.is_some(), "tile PIZ factory must return a compressor");
    }

    // ---- NoneCompressor trait ----

    #[test]
    fn none_compressor_num_scan_lines_is_1() {
        let c = NoneCompressor::new();
        assert_eq!(c.num_scan_lines(), 1);
    }

    #[test]
    fn none_compressor_format_is_xdr() {
        let c = NoneCompressor::new();
        assert_eq!(c.format(), Format::Xdr);
    }

    #[test]
    fn none_compressor_compress_passthrough() {
        let mut c = NoneCompressor::new();
        let data = b"hello scanline";
        let out = c.compress(data, 0).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn none_compressor_uncompress_passthrough() {
        let mut c = NoneCompressor::new();
        let data = b"hello scanline";
        let out = c.uncompress(data, 0, 0).unwrap();
        assert_eq!(out, data);
    }

    // ---- num_lines_in_buffer ----

    #[test]
    fn num_lines_none_is_1() {
        assert_eq!(num_lines_in_buffer(Compression::NoCompression).unwrap(), 1);
    }

    #[test]
    fn num_lines_zip_is_16() {
        assert_eq!(num_lines_in_buffer(Compression::Zip).unwrap(), 16);
    }

    #[test]
    fn num_lines_piz_is_32() {
        assert_eq!(num_lines_in_buffer(Compression::Piz).unwrap(), 32);
    }

    #[test]
    fn num_lines_dwab_is_256() {
        assert_eq!(num_lines_in_buffer(Compression::Dwab).unwrap(), 256);
    }

    #[test]
    fn num_lines_invalid_is_err() {
        let result = num_lines_in_buffer(Compression::Invalid);
        assert!(result.is_err());
    }

    // ---- stored_chunk ----

    #[test]
    fn stored_chunk_keeps_smaller_compressed() {
        let raw = [1u8, 2, 3, 4];
        let compressed = [9u8, 9, 9];
        assert_eq!(stored_chunk(&compressed, &raw), &compressed[..]);
    }

    #[test]
    fn stored_chunk_stores_raw_at_equal_size() {
        // The encoding is exactly the uncompressed size, so it is
        // indistinguishable from a raw chunk on disk — the raw bytes must win.
        let raw = [1u8, 2, 3, 4];
        let compressed = [9u8, 9, 9, 9];
        assert_eq!(stored_chunk(&compressed, &raw), &raw[..]);
    }

    #[test]
    fn stored_chunk_stores_raw_when_encoding_expanded() {
        let raw = [1u8, 2, 3];
        let compressed = [9u8, 9, 9, 9];
        assert_eq!(stored_chunk(&compressed, &raw), &raw[..]);
    }

    /// Every ported codec must report `Xdr`; the C++ `convertToXdr` branch on
    /// the writer's raw path is only reachable for a `Native` codec and is not
    /// ported.  If this ever fails, port `convertToXdr` before changing it.
    #[test]
    fn every_codec_reports_xdr_format() {
        let hdr = make_header();
        for codec in Compression::ALL {
            if let Some(c) = new_compressor(codec, 1024, &hdr).expect("factory") {
                assert_eq!(c.format(), Format::Xdr, "scanline {codec:?}");
            }
            if let Some(c) = new_tile_compressor(codec, 64, 32, &hdr).expect("tile factory") {
                assert_eq!(c.format(), Format::Xdr, "tiled {codec:?}");
            }
        }
    }

    // ---- default_format ----

    #[test]
    fn default_format_none_is_xdr() {
        // mirrors C++ defaultFormat(nullptr) == Compressor::XDR
        assert_eq!(default_format(None), Format::Xdr);
    }

    #[test]
    fn default_format_with_compressor_delegates() {
        let c = NoneCompressor::new();
        // NoneCompressor reports Xdr
        assert_eq!(default_format(Some(&c as &dyn Compressor)), Format::Xdr);
    }

    /// `CodecPool::lease` returns its codec on every normal exit - an early
    /// `?` error included - and discards it while unwinding from a panic (a
    /// codec that was mid-call is not trusted).
    #[test]
    fn lease_returns_the_codec_on_every_path_but_a_panic() {
        let pool = CodecPool::new();
        let header = make_header();
        let make = || new_compressor(crate::attr::Compression::Zip, 64, &header);

        let ok: Result<()> = (|| {
            let _lease = pool.lease(make)?;
            Ok(())
        })();
        ok.unwrap();
        assert_eq!(pool.free_count(), 1, "returned after a normal exit");

        let failing = || -> Result<()> {
            let mut lease = pool.lease(make)?;
            assert!(lease.as_mut().is_some());
            Err(crate::error::ExrError::Arg("early exit".into()))
        };
        assert!(failing().is_err());
        assert_eq!(pool.free_count(), 1, "returned (and reused) after an early `?`");

        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = pool.lease(make).unwrap();
            panic!("boom");
        }));
        assert!(r.is_err());
        assert_eq!(pool.free_count(), 0, "the mid-call codec is dropped, not returned");
    }
}
