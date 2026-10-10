//! Process-global maximum image/tile size state
//! (`Header::{set_,}max_{image,tile}_size`) — tests that MUTATE this
//! state live in their OWN process, here, not in the `-p exr-core --lib`
//! unit-test binary.
//!
//! # Why this file exists
//! Same class of hazard as `tests/global_compression_defaults.rs` (read that
//! file's module doc for the full derivation): `Header::max_image_size`/
//! `max_tile_size` are genuine process-global state (matching C++
//! `Imf::setMaxImageSize`/`setMaxTileSize`), and `Header::sanity_check`
//! reads them (with `(0, 0)` meaning "no limit") on EVERY header any codec/
//! writer/reader test in the `--lib` binary builds -- unlocked, because
//! there is no reasonable way to make every such test lock-aware. A lock
//! around only the MUTATING tests (the old `header::tests::
//! MAX_SIZE_TEST_LOCK`) serializes those mutators against each other, but
//! does nothing for the many other tests that read the effective limit
//! without taking it: if one of the 3 tests below happened to run
//! concurrently with, say, a codec test building a 64x64 header while the
//! process-global max image width was temporarily set to 10, that unrelated
//! test's `sanity_check` would spuriously fail. This had not been observed
//! to flake yet (unlike the compression-level case), but it is structurally
//! the identical hazard, so it gets the identical fix pre-emptively: process
//! isolation, not more locking. Ported using only the PUBLIC API (this one
//! needed no rewrite -- `Header::set_max_image_size`/`set_max_tile_size`/
//! `max_image_size`/`max_tile_size`/`sanity_check`/`channels_mut`/`insert`
//! are already all public, unlike PXR24's internals in the compression-level
//! file).

use std::sync::{Mutex, OnceLock};

use exr_core::attr::{Channel, PixelType, TileDescription, TileDescriptionAttribute};
use exr_core::header::Header;

/// Serializes every test in this file against the ONE process-global
/// max-image/max-tile-size state.
fn lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// RAII guard: resets both process-global maxima to "unlimited" (`(0, 0)`)
/// on drop, so a failed `assert!` mid-test still restores the shared state
/// for whichever test in this file runs next.
struct ResetGuard;
impl Drop for ResetGuard {
    fn drop(&mut self) {
        Header::set_max_image_size(0, 0);
        Header::set_max_tile_size(0, 0);
    }
}

/// Ported verbatim from `header::tests::max_image_and_tile_size_round_trip`.
#[test]
fn max_image_and_tile_size_round_trip() {
    let _lock = lock().lock().unwrap();
    let _reset = ResetGuard;
    Header::set_max_image_size(1000, 2000);
    assert_eq!(Header::max_image_size(), (1000, 2000));
    Header::set_max_tile_size(64, 128);
    assert_eq!(Header::max_tile_size(), (64, 128));
}

/// (Ported verbatim from `header::tests::sanity_check_enforces_max_image_width`)
/// — `sanity_check` rejects a data window wider than the process-global
/// maximum image width.
#[test]
fn sanity_check_enforces_max_image_width() {
    let _lock = lock().lock().unwrap();
    let _reset = ResetGuard;
    Header::set_max_image_size(10, 0);

    let mut h = Header::new();
    h.channels_mut().unwrap().insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    // Header::new()'s default data window is 64x64 -- wider than 10.
    assert!(h.sanity_check(false, false).is_err());
}

/// (Ported verbatim from `header::tests::sanity_check_enforces_max_tile_width`)
/// — `sanity_check` rejects a tile wider than the process-global maximum
/// tile width.
#[test]
fn sanity_check_enforces_max_tile_width() {
    let _lock = lock().lock().unwrap();
    let _reset = ResetGuard;
    Header::set_max_tile_size(16, 0);

    let mut h = Header::new();
    h.channels_mut().unwrap().insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    h.insert(
        "tiles",
        TileDescriptionAttribute::new(TileDescription {
            x_size: 32,
            y_size: 32,
            ..Default::default()
        }),
    )
    .unwrap();
    assert!(h.sanity_check(true, false).is_err());
}
