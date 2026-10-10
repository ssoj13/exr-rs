//! Process-global ZIP/DWA default compression-level state
//! (`Header::{set_,}default_{zip,dwa}_compression_level`) — tests
//! that MUTATE this state live in their OWN process, here, not in the `-p
//! exr-core --lib` unit-test binary.
//!
//! # Why this file exists
//! `Header::default_zip_compression_level`/`default_dwa_compression_level`
//! are genuine process-global state (matching C++
//! `Imf::setDefaultZipCompressionLevel`/`setDefaultDwaCompressionLevel` --
//! not per-test, not per-thread). `cargo test` runs unit tests in the SAME
//! process on parallel threads, so a lock around every MUTATOR only
//! serializes mutators against each other -- it does nothing for the many
//! OTHER lib tests (DWA/ZIP/PXR24 codec tests, parity-shaped round trips)
//! that READ the effective compression level (`Header::zip_level`/
//! `dwa_level`, which fall back to the process default when no per-header
//! override is set) without taking any lock at all, because there is no
//! reasonable way to make every such test lock-aware. That produced a real,
//! observed flake: `compressor::dwa::tests::
//! dwa_compression_level_comes_from_the_header` (which never touches the
//! global itself -- every header it builds has an explicit
//! `add_dwa_compression_level` override) intermittently saw a DWA chunk size
//! off by a couple of bytes (1668 vs 1670) because `compressor::pxr24::
//! tests::pxr24_follows_process_global_default_zip_level` was
//! concurrently flipping the process-global default ZIP level on another
//! thread of the SAME process, and DWA's internal AC/RLE zlib sub-streams
//! resolve their own zip level through that same global when unset.
//!
//! A lock cannot fix readers you don't control, so the systematic fix is
//! process isolation instead: every test that WRITES the process-global
//! default lives here, in its own `cargo test` binary / OS process, which
//! shares no memory with the `--lib` binary at all -- no reader anywhere
//! else can observe a value this file's tests set, structurally, not just
//! by convention. Within this single process the 4 tests below still touch
//! one shared global, so they are serialized by a plain local `Mutex` (no
//! need to reuse `header::test_support`'s lock, which is `pub(crate)` and
//! therefore invisible to an integration test crate anyway).
//!
//! These are the SAME 4 tests that used to live under
//! `header::test_support::COMPRESSION_LEVEL_TEST_LOCK` in `header.rs` (2) and
//! `compressor/pxr24.rs` (2), ported to use only the PUBLIC API: the two
//! pure `Header` accessor tests move verbatim (they never touched compressor
//! internals), and the two PXR24 zip-level tests -- which used to reach into the
//! crate-private `Pxr24Compressor` directly -- now write real files through
//! the public `scanline::OutputFile`/`InputFile` and compare bytes/sizes.

use std::sync::{Mutex, OnceLock};

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::{InputFile, OutputFile};
use exr_core::FloatAttribute;

use imath_rs::{Half, V2f};

/// Serializes every test in this file against the ONE process-global
/// zip/dwa-compression-level state (see the module doc). A plain `Mutex`,
/// not `header::test_support`'s (that one is `pub(crate)`, invisible here).
fn lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// RAII guard: resets both process-global defaults to "unset" on drop, so a
/// failed `assert!` mid-test still restores the shared state for whichever
/// test in this file runs next (locals drop in reverse declaration order, so
/// this runs BEFORE the lock guard releases when both are held).
struct ResetGuard;
impl Drop for ResetGuard {
    fn drop(&mut self) {
        Header::set_default_zip_compression_level(-1);
        Header::set_default_dwa_compression_level(45.0);
    }
}

/// A deterministic per-pixel value for channel index `ci` (0=H, 1=U, 2=Z) at
/// `(x, y)`, shared by the planar buffer builder and the PLANAR `FrameBuffer`
/// this test drives `OutputFile`/`InputFile` through (three separate,
/// contiguous per-channel planes in ONE caller-owned buffer -- the same
/// convention `half_conversion_boundary.rs` and other integration tests in this
/// crate use for their own frame buffers).
fn pixel_value(ci: usize, x: i32, y: i32) -> f32 {
    (ci as f32) * 1000.0 + (y as f32) * 10.0 + x as f32 + 0.5
}

/// Build the H (HALF) plane: `w*h` samples, row-major, `pixel_value(0, x, y)`.
fn half_plane(w: i32, h: i32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h) as usize * 2);
    for y in 0..h {
        for x in 0..w {
            buf.extend_from_slice(&Half::from(pixel_value(0, x, y)).bits().to_le_bytes());
        }
    }
    buf
}
/// Build the U (UINT) plane: `w*h` samples, row-major, `pixel_value(1, x, y)`.
fn uint_plane(w: i32, h: i32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h) as usize * 4);
    for y in 0..h {
        for x in 0..w {
            buf.extend_from_slice(&(pixel_value(1, x, y) as u32).to_le_bytes());
        }
    }
    buf
}
/// Build the Z (FLOAT) plane: `w*h` samples, row-major, `pixel_value(2, x, y)`.
fn float_plane(w: i32, h: i32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h) as usize * 4);
    for y in 0..h {
        for x in 0..w {
            buf.extend_from_slice(&pixel_value(2, x, y).to_bits().to_le_bytes());
        }
    }
    buf
}

/// The single combined planar buffer (H-plane, then U-plane, then Z-plane)
/// used both to WRITE (as the `pixels` bytes handed to `write_pixels`) and
/// to compare a READ-back result against.
fn planar_buffer(w: i32, h: i32) -> Vec<u8> {
    let mut buf = half_plane(w, h);
    buf.extend_from_slice(&uint_plane(w, h));
    buf.extend_from_slice(&float_plane(w, h));
    buf
}

/// The `FrameBuffer` describing `planar_buffer`'s layout: H at byte 0,
/// U right after the H-plane, Z right after the U-plane, each its own
/// contiguous `w*h`-element plane (`x_stride` = element size, `y_stride` =
/// `w` * element size, `base_offset` = the plane's start).
fn planar_frame_buffer(w: i32, h: i32) -> FrameBuffer {
    let n = (w * h) as i64;
    let h_base = 0i64;
    let u_base = h_base + n * 2;
    let z_base = u_base + n * 4;
    let mut fb = FrameBuffer::new();
    fb.insert(
        "H",
        Slice::new(PixelType::Half, h_base, 2, (w as usize) * 2, 1, 1, 0.0, false, false).unwrap(),
    )
    .unwrap();
    fb.insert(
        "U",
        Slice::new(PixelType::Uint, u_base, 4, (w as usize) * 4, 1, 1, 0.0, false, false).unwrap(),
    )
    .unwrap();
    fb.insert(
        "Z",
        Slice::new(PixelType::Float, z_base, 4, (w as usize) * 4, 1, 1, 0.0, false, false).unwrap(),
    )
    .unwrap();
    fb
}

/// Write a small PXR24 file (channels H:HALF, U:UINT, Z:FLOAT) through the
/// PUBLIC `scanline::OutputFile`, with an optional per-header
/// `set_zip_level` override, and return the full file bytes.
fn write_pxr24(w: i32, h: i32, zip_level_override: Option<i32>) -> Vec<u8> {
    let mut hdr = Header::with_dimensions(
        w,
        h,
        1.0,
        V2f { x: 0.0, y: 0.0 },
        1.0,
        LineOrder::IncreasingY,
        Compression::Pxr24,
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("H", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("U", Channel::new(PixelType::Uint, 1, 1, false)).unwrap();
    cl.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    if let Some(level) = zip_level_override {
        hdr.set_zip_level(level);
    }

    let pixels = planar_buffer(w, h);
    let fb = planar_frame_buffer(w, h);

    let mut stream = MemOStream::new();
    {
        let mut of = OutputFile::new(&mut stream, &hdr).unwrap();
        of.set_frame_buffer(&fb).unwrap();
        of.write_pixels(&pixels, h).unwrap();
    }
    stream.into_data()
}

/// Read back a `write_pxr24`-shaped file's `H`/`U`/`Z` channels into the SAME
/// planar layout `planar_buffer` produces, for a direct round-trip
/// comparison against the original pixel bytes.
fn read_pxr24(bytes: Vec<u8>, w: i32, h: i32) -> Vec<u8> {
    let mut is = MemIStream::new(bytes);
    let mut inf = InputFile::new(&mut is).unwrap();
    let fb = planar_frame_buffer(w, h);
    inf.set_frame_buffer(&fb).unwrap();
    let mut out = vec![0u8; planar_buffer(w, h).len()];
    inf.read_pixels(0, h - 1, &mut out).unwrap();
    out
}

/// (Ported from `header::tests::
/// zip_and_dwa_compression_level_override_and_default`) — a per-header
/// override wins over the process-global default; with no override, the
/// header falls back to the (possibly just changed) process-global default.
/// Pure `Header` accessor state -- no file I/O, no compressor internals.
#[test]
fn per_header_override_wins_over_process_default_and_reset_falls_back() {
    let _lock = lock().lock().unwrap();
    let _reset = ResetGuard;

    assert_eq!(Header::default_zip_compression_level(), -1);
    assert_eq!(Header::default_dwa_compression_level(), 45.0);

    let mut h = Header::new();
    assert_eq!(h.zip_level(), -1);
    assert_eq!(h.dwa_level(), 45.0);

    h.set_zip_level(9);
    h.set_dwa_level(90.0);
    assert_eq!(h.zip_level(), 9);
    assert_eq!(h.dwa_level(), 90.0);

    // Changing the process default must not affect an already-overridden header.
    Header::set_default_zip_compression_level(2);
    assert_eq!(h.zip_level(), 9);

    h.reset_default_compression_levels();
    assert_eq!(h.zip_level(), 2);
    assert_eq!(h.dwa_level(), 45.0);
}

/// (Ported from `header::tests::
/// insert_dwa_compression_level_attribute_updates_record`) — inserting a
/// `dwaCompressionLevel` float attribute also updates the header's
/// compression-level record. Pure `Header` accessor state.
#[test]
fn insert_dwa_compression_level_attribute_updates_record() {
    let _lock = lock().lock().unwrap();
    let _reset = ResetGuard;

    let mut h = Header::new();
    assert_eq!(h.dwa_level(), 45.0);
    h.insert("dwaCompressionLevel", FloatAttribute::new(200.0)).unwrap();
    assert_eq!(h.dwa_level(), 200.0);
}

/// (Ported from `compressor::pxr24::tests::
/// pxr24_ignores_per_header_zip_level_override`) — a per-header
/// `set_zip_level` override must be silently IGNORED by PXR24 (C++'s
/// `apply_pxr24_impl` calls `exr_compress_buffer(ctx, -1, ...)` with a
/// literal `-1`, never `header->zipCompressionLevel()`,
/// `internal_pxr24.c:219-226`). With the process-global default left at its
/// startup value (`-1`), the WHOLE FILE must be byte-identical whether or not
/// a per-header override is set.
#[test]
fn pxr24_ignores_per_header_zip_level_override() {
    let _lock = lock().lock().unwrap();
    let _reset = ResetGuard;
    assert_eq!(Header::default_zip_compression_level(), -1, "test isolation invariant");

    let (w, h) = (9i32, 5i32);
    let plain = write_pxr24(w, h, None);
    let overridden = write_pxr24(w, h, Some(9)); // PXR24 must not see this at all.

    assert_eq!(
        plain, overridden,
        "a per-header zip_level override must not change PXR24's file bytes"
    );
}

/// (Ported from `compressor::pxr24::tests::
/// pxr24_follows_process_global_default_zip_level`) — PXR24 DOES follow
/// the process-global default zip level (C++'s hardcoded `-1` resolves
/// through `exr_get_default_zip_compression_level` before falling back to
/// `EXR_DEFAULT_ZLIB_COMPRESS_LEVEL == 4`, `compression.c:89-117`), even
/// though a *per-header* override does not (previous test). Changing the
/// process default must change PXR24's file bytes, and the result must still
/// round-trip.
#[test]
fn pxr24_follows_process_global_default_zip_level() {
    let _lock = lock().lock().unwrap();
    let _reset = ResetGuard;
    assert_eq!(Header::default_zip_compression_level(), -1, "test isolation invariant");

    let (w, h) = (9i32, 5i32);
    let at_default = write_pxr24(w, h, None);

    Header::set_default_zip_compression_level(1); // fastest, != the 4 fallback
    let at_level1 = write_pxr24(w, h, None);

    assert_ne!(
        at_default, at_level1,
        "changing the process-global default zip level must change PXR24's file bytes"
    );

    let restored = read_pxr24(at_level1, w, h);
    assert_eq!(restored, planar_buffer(w, h), "level-1 PXR24 output must still round-trip");
}
