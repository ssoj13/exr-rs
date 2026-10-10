//! Deep-image I/O and compositing, matching OpenEXR v3.4.15.
//!
//! A deep pixel contains a variable number of samples per channel, typically
//! one per surface or volume fragment along the camera ray.
//!
//! # Count then sample data
//!
//! Read the per-pixel counts first, size the caller-owned sample heaps, then
//! read the samples. [`DeepFrameBuffer`] describes the count raster separately
//! from the channel samples; [`DeepSlice`] supplies each channel's sample stride.
//!
//! C++ deep slices use a table of per-pixel pointers (`T**`). This safe Rust
//! API uses flat sample heaps and checked integer offsets instead. See
//! [`frame_buffer`] for the shared addressing contract, including independent
//! count-slice and channel tile-coordinate flags.
//!
//! Reader heaps start at the requested scanline range or tile rectangle.
//! When filling a larger heap incrementally, adjust each channel's
//! `base_offset` for the new request and install the updated frame buffer.
//! Writers retain their documented window or level heap domain.
//!
//! # Compression
//!
//! Deep data supports **NONE**, **RLE**, and **ZIPS**;
//! [`crate::attr::Compression::is_valid_deep`] defines this set. The count table
//! is stored separately from sample data. Sixteen-line ZIP is not deep-valid.
//!
//! # Modules
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`frame_buffer`] | Count-raster and flat sample-heap descriptors |
//! | [`sample_table`] | Checked cumulative-count encoding and decoding |
//! | [`scanline_input`] / [`scanline_output`] | Deep scanline readers and writers |
//! | [`tiled_input`] / [`tiled_output`] | Deep tiled readers and writers, including levels |
//! | [`compositing`] | Per-pixel deep compositing |
//! | [`composite_scanline`] | Compositing across deep scanline sources |
//!
//! Reference implementation: OpenEXR v3.4.15, `ImfDeep*` in IlmImf.
//! <https://github.com/AcademySoftwareFoundation/openexr/tree/v3.4.15/src/lib/OpenEXR/>

pub mod composite_scanline;
pub mod compositing;
pub mod frame_buffer;
pub mod sample_table;
pub mod scanline_input;
pub mod scanline_output;
pub mod tiled_input;
pub mod tiled_output;

pub use composite_scanline::{
    get_maximum_sample_count, set_maximum_sample_count, CompositeDeepScanLine,
    DeepScanLineCompositeSource,
};
pub use compositing::{DeepCompositing, DefaultDeepCompositing};
pub use frame_buffer::{DeepFrameBuffer, DeepSlice};
pub use scanline_input::DeepScanLineInputFile;
pub use scanline_output::{
    DeepScanLineCopySource, DeepScanLineOutputFile, DeepScanLineOutputPartWriter,
};
pub(crate) use scanline_output::DeepScanLineOutputFileState;
pub use tiled_input::DeepTiledInputFile;
pub use tiled_output::{DeepTiledCopySource, DeepTiledOutputFile, DeepTiledOutputPartWriter};
pub(crate) use tiled_output::DeepTiledOutputFileState;

/// The `(x_tile_coords, y_tile_coords)` flags every slice shares, or `None`
/// when there are no slices or they disagree. The deep tiled parallel
/// reader and writer precompute one per-tile sample prefix shared by all
/// channels, which is only valid when every slice addresses pixels the
/// same way; on `None` they decline to the serial path.
pub(crate) fn uniform_tile_coords(mut flags: impl Iterator<Item = (bool, bool)>) -> Option<(bool, bool)> {
    let first = flags.next()?;
    flags.all(|f| f == first).then_some(first)
}

#[cfg(test)]
mod uniform_tile_coords_tests {
    use super::uniform_tile_coords;

    #[test]
    fn agrees_only_when_every_slice_matches() {
        assert_eq!(uniform_tile_coords(std::iter::empty()), None);
        assert_eq!(uniform_tile_coords([(true, false)].into_iter()), Some((true, false)));
        assert_eq!(uniform_tile_coords([(false, true), (false, true)].into_iter()), Some((false, true)));
        assert_eq!(uniform_tile_coords([(false, false), (true, false)].into_iter()), None, "mixed x");
        assert_eq!(uniform_tile_coords([(false, false), (false, true)].into_iter()), None, "mixed y");
    }
}
