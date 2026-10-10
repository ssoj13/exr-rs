//! Per-type **Part wrappers** — ports of OpenEXR's `ImfOutputPart` /
//! `ImfInputPart` / `ImfTiled{Output,Input}Part` /
//! `ImfDeepScanLine{Output,Input}Part` / `ImfDeepTiled{Output,Input}Part`.
//!
//! In C++ each `*Part` class is a **thin adapter**: it holds a
//! `MultiPart{Output,Input}File` plus a part index, obtains the underlying
//! single-part file object for that part, and forwards `setFrameBuffer` /
//! `writeTile(s)` / `readTile(s)` / `writePixels` / `readPixels` /
//! `readPixelSampleCounts` to it.  The actual chunks are written/read through
//! the SAME single-part logic (Phases 4–7) but with the part's `partNumber`
//! prefixed — the unified `part_number: Option<i32>` seam (see
//! [`crate::scanline::OutputFile::from_part`] and the matching `from_part`
//! constructors on the tiled / deep files).
//!
//! Here, mirroring that design with safe Rust lifetimes, each `*Part` is a
//! newtype that **owns** the underlying single-part file for the part's
//! lifetime (the file already borrows the multipart file's stream).  It is
//! obtained from the container's typed accessor —
//! [`MultiPartOutputFile::tiled_output_part`](crate::multipart::MultiPartOutputFile::tiled_output_part)
//! etc. — and exposes the underlying file's full API through `Deref` /
//! `DerefMut`, so `set_frame_buffer` / `write_tile` / `read_pixels` /
//! `read_pixel_sample_counts` / `header` are all forwarded with no duplicated
//! logic (single point of responsibility).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/>
//! (`ImfOutputPart.cpp`, `ImfInputPart.cpp`, `ImfTiledOutputPart.cpp`,
//! `ImfTiledInputPart.cpp`, `ImfDeepScanLineOutputPart.cpp`,
//! `ImfDeepScanLineInputPart.cpp`, `ImfDeepTiledOutputPart.cpp`,
//! `ImfDeepTiledInputPart.cpp`).

use std::ops::{Deref, DerefMut};

use crate::deep::{
    DeepScanLineCompositeSource, DeepScanLineCopySource, DeepScanLineOutputPartWriter,
    DeepTiledCopySource, DeepTiledOutputPartWriter,
};
use crate::error::Result;
use crate::header::Header;
use crate::deep::frame_buffer::DeepFrameBuffer;
use crate::multipart::input::{DeepScanLineInputPartReader, DeepTiledInputPartReader};
use crate::scanline::{InputFilePartReader, OutputPartWriter, ScanLineCopySource};
use crate::tiled::output::TiledCopySource;
use crate::tiled::{TiledInputPartReader, TiledOutputPartWriter};

/// Macro that defines a thin `*Part` newtype wrapping an owned single-part
/// file and forwarding its whole API via `Deref`/`DerefMut`.  This is the
/// faithful Rust analogue of the C++ `*Part` adapter: the wrapper adds no
/// behaviour of its own — every call lands on the underlying file (which
/// already carries the part's `partNumber` seam).
macro_rules! part_wrapper {
    ($(#[$meta:meta])* $name:ident, $inner:ident) => {
        $(#[$meta])*
        pub struct $name<'a>($inner<'a>);

        impl<'a> $name<'a> {
            /// Wrap an underlying single-part file as this part's adapter.
            /// Obtained from the container's typed part accessor.
            pub fn new(inner: $inner<'a>) -> Self {
                Self(inner)
            }

            /// Borrow the underlying single-part file.
            pub fn inner(&self) -> &$inner<'a> {
                &self.0
            }

            /// Mutably borrow the underlying single-part file.
            pub fn inner_mut(&mut self) -> &mut $inner<'a> {
                &mut self.0
            }

            /// Consume the wrapper, returning the underlying file.
            pub fn into_inner(self) -> $inner<'a> {
                self.0
            }
        }

        impl<'a> Deref for $name<'a> {
            type Target = $inner<'a>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<'a> DerefMut for $name<'a> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
}

part_wrapper! {
    /// Scanline output part — port of `Imf::OutputPart`.  Forwards
    /// `set_frame_buffer` / `write_pixels` / `header` to the underlying
    /// [`OutputPartWriter`] (a borrowing view over this part's
    /// persisted writer state, not a fresh `OutputFile` per call) for this
    /// part (chunks carry the leading `partNumber`).
    OutputPart, OutputPartWriter
}

part_wrapper! {
    /// Scanline input part — port of `Imf::InputPart`.  Forwards
    /// `set_frame_buffer` / `read_pixels` / `header` to the underlying
    /// [`InputFilePartReader`] (a borrowing view
    /// over this part's persisted reader state, not a fresh `InputFile`
    /// built from scratch each call) for this part (the leading `partNumber`
    /// is verified).
    InputPart, InputFilePartReader
}

part_wrapper! {
    /// Tiled output part — port of `Imf::TiledOutputPart`.  Forwards
    /// `set_frame_buffer` / `write_tile(s)` / `header` to the underlying
    /// [`TiledOutputPartWriter`] (a borrowing
    /// view over this part's persisted writer state, not a fresh
    /// `TiledOutputFile` per call) for this part.
    TiledOutputPart, TiledOutputPartWriter
}

part_wrapper! {
    /// Tiled input part — port of `Imf::TiledInputPart`.  Forwards
    /// `set_frame_buffer` / `read_tile(s)` / `header` to the underlying
    /// [`TiledInputPartReader`] (a borrowing view
    /// over this part's persisted reader state, not a fresh `TiledInputFile`
    /// built from scratch each call) for this part.
    TiledInputPart, TiledInputPartReader
}

// `TiledOutputFile::copy_pixels` (trait redesigned
// for the borrowing part views) is generic over `TiledCopySource`;
// trait bounds on a generic parameter are not satisfied via `Deref` coercion
// the way a concrete-type call site is, so `TiledInputPart`/`InputPart` —
// C++'s `TiledInputPart&`/`InputPart&` `copyPixels` overloads — each need
// their own thin forwarding impl here, alongside the wrapper they forward.
impl<'a> TiledCopySource for TiledInputPart<'a> {
    fn ensure_tiled(&mut self) -> Result<()> {
        self.inner_mut().ensure_tiled()
    }
    fn header(&self) -> &crate::header::Header {
        self.inner().header()
    }
    fn tile_order(&mut self) -> Result<crate::tiled::input::TileOrder> {
        self.inner_mut().tile_order()
    }
    fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        self.inner_mut().raw_tile_data(dx, dy, lx, ly)
    }
}

impl<'a> TiledCopySource for InputPart<'a> {
    fn ensure_tiled(&mut self) -> Result<()> {
        self.inner_mut().ensure_tiled()
    }
    fn header(&self) -> &crate::header::Header {
        self.inner().header()
    }
    fn tile_order(&mut self) -> Result<crate::tiled::input::TileOrder> {
        self.inner_mut().tile_order()
    }
    fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<(i32, i32, i32, i32, Vec<u8>)> {
        self.inner_mut().raw_tile_data(dx, dy, lx, ly)
    }
}

// `OutputFile::copy_pixels` is generic over `ScanLineCopySource`
// (`scanline/output.rs`); same reasoning as the `TiledCopySource` impls
// above — `InputPart` (C++'s `InputPart&` `copyPixels` overload) needs its
// own thin forwarding impl.
impl<'a> ScanLineCopySource for InputPart<'a> {
    fn header(&self) -> &crate::header::Header {
        self.inner().header()
    }
    fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>)> {
        self.inner_mut().raw_pixel_data(first_scan_line)
    }
}

part_wrapper! {
    /// Deep scanline output part — port of `Imf::DeepScanLineOutputPart`.
    /// Forwards `set_frame_buffer` / `write_pixels` / `header` to the
    /// underlying [`DeepScanLineOutputPartWriter`] (a
    /// borrowing view over this part's persisted writer
    /// state, not a fresh `DeepScanLineOutputFile` per call) for this part.
    DeepScanLineOutputPart, DeepScanLineOutputPartWriter
}

part_wrapper! {
    /// Deep scanline input part — port of `Imf::DeepScanLineInputPart`.
    /// Forwards `set_frame_buffer` / `read_pixel_sample_counts` / `read_pixels`
    /// / `header` to the underlying [`DeepScanLineInputPartReader`] (a
    /// borrowing view over this part's persisted
    /// reader state, not a fresh `DeepScanLineInputFile` built from scratch
    /// each call) for this part.
    DeepScanLineInputPart, DeepScanLineInputPartReader
}

part_wrapper! {
    /// Deep tiled output part — port of `Imf::DeepTiledOutputPart`.  Forwards
    /// `set_frame_buffer` / `write_tile(s)` / `header` to the underlying
    /// [`DeepTiledOutputPartWriter`] (a borrowing
    /// view over this part's persisted writer state, not a fresh
    /// `DeepTiledOutputFile` per call) for this part.
    DeepTiledOutputPart, DeepTiledOutputPartWriter
}

// `DeepScanLineOutputFile::copy_pixels` / `DeepTiledOutputFile::copy_pixels`
// are generic over `DeepScanLineCopySource` /
// `DeepTiledCopySource` (`deep/scanline_output.rs` / `deep/tiled_output.rs`).
// Both traits have the same duck-typed,
// no-lifetime shape as `TiledCopySource` (`header` +
// `raw_pixel_data` / `total_tiles`+`tile_order`+`raw_tile_data`), specifically
// so a borrowing cache view (which never owns a concrete file to hand back)
// can implement them.
// `DeepScanLineInputPart`/`DeepTiledInputPart` wrap the borrowing views
// ([`DeepScanLineInputPartReader`] / [`DeepTiledInputPartReader`]), which now
// carry the matching forwarding methods (`multipart/input.rs`), so these
// impls delegate to them directly.
impl<'a> DeepScanLineCopySource for DeepScanLineInputPart<'a> {
    fn header(&self) -> &Header {
        self.inner().header()
    }
    fn raw_pixel_data(&mut self, first_scan_line: i32) -> Result<(i32, Vec<u8>, Vec<u8>, u64)> {
        self.inner_mut().raw_pixel_data(first_scan_line)
    }
}

impl<'a> DeepTiledCopySource for DeepTiledInputPart<'a> {
    fn header(&self) -> &Header {
        self.inner().header()
    }
    fn total_tiles(&self) -> Result<i64> {
        self.inner().total_tiles()
    }
    fn tile_order(&self) -> Result<crate::tiled::input::TileOrder> {
        self.inner().tile_order()
    }
    fn raw_tile_data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
    ) -> Result<crate::deep::tiled_input::RawTileData> {
        self.inner_mut().raw_tile_data(dx, dy, lx, ly)
    }
}

// `CompositeDeepScanLine::add_source` is generic over
// `DeepScanLineCompositeSource`; same reasoning as the `*CopySource` impls
// above -- `DeepScanLineInputPart` (C++'s `addSource(DeepScanLineInputPart*)`
// overload) needs its own thin forwarding impl.
impl<'a> DeepScanLineCompositeSource for DeepScanLineInputPart<'a> {
    fn header(&self) -> &Header {
        self.inner().header()
    }
    fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        self.inner_mut().set_frame_buffer(frame_buffer)
    }
    fn read_pixel_sample_counts(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        self.inner_mut().read_pixel_sample_counts(scan_line1, scan_line2, sample_counts)
    }
    fn read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        self.inner_mut().read_pixels(scan_line1, scan_line2, sample_counts, sample_data)
    }
}

part_wrapper! {
    /// Deep tiled input part — port of `Imf::DeepTiledInputPart`.  Forwards
    /// `set_frame_buffer` / `read_pixel_sample_counts` / `read_tile(s)` /
    /// `header` to the underlying [`DeepTiledInputPartReader`] (a
    /// borrowing view over this part's persisted reader
    /// state, not a fresh `DeepTiledInputFile` built from scratch each call)
    /// for this part.
    DeepTiledInputPart, DeepTiledInputPartReader
}
