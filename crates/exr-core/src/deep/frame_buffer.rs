//! Port of OpenEXR `ImfDeepFrameBuffer.h/.cpp` — `DeepSlice` and `DeepFrameBuffer`.
//!
//! C++ deep slices address a table of per-pixel pointers. This safe Rust API
//! instead stores each channel's samples in a flat caller-owned heap:
//!
//! ```text
//! sample_address = base_offset + (samples_before_pixel + sample_index) * sample_stride
//! ```
//!
//! A separate UINT count slice describes the count raster with its own base,
//! x/y byte strides and tile-coordinate flags. Deep-channel x/y strides retain
//! the C++ descriptor shape; the flat-heap readers do not dereference a pointer
//! table or use those strides to locate samples. Channel base offsets and
//! sample strides select planar or interleaved sample heaps.
//!
//! Readers address the heap from the requested scanline/tile rectangle, so a
//! caller can allocate counts and samples for one row or tile. For tiled reads,
//! each channel's tile-coordinate flags independently choose whether its heap
//! restarts at each tile along that axis. Absolute axes span the requested
//! rectangle; local axes span the actual, clipped tile. The count slice's flags
//! select count storage coordinates independently of these heap domains.
//!
//! First read counts, allocate the sample heap, then read data. Writers retain
//! their documented window/level heap domain, allowing incremental writes.
//! All addressing uses checked integer offsets; there are no raw pointers.
//!
//! Port target: OpenEXR v3.4.15 `ImfDeepFrameBuffer.h/.cpp`.

use std::collections::BTreeMap;

use crate::attr::name::Name;
use crate::attr::PixelType;
use crate::error::{ExrError, Result};
use crate::frame_buffer::Slice;

// ---------------------------------------------------------------------------
// DeepSlice — port of `struct DeepSlice : public Slice`
// ---------------------------------------------------------------------------

/// Describes a single *deep* channel inside a caller-owned buffer — port of
/// `struct DeepSlice : public Slice`.
///
/// A `DeepSlice` is a [`Slice`] **plus** `sample_stride`, the byte distance
/// between two consecutive samples of the *same* pixel.  See the module-level
/// docs for the flat sample-heap addressing model.
///
/// C++ uses public inheritance (`DeepSlice : public Slice`); the Rust port
/// composes a [`Slice`] field and re-exposes the common geometry through it.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepSlice {
    /// The common [`Slice`] geometry (type, base offset, x/y strides, sampling,
    /// fill value, tile-coord flags).  Mirrors the `Slice` base subobject.
    pub slice: Slice,

    /// Byte distance between consecutive samples of one pixel.
    ///
    /// In C++ this is `int sampleStride` (the constructor takes a `size_t spst`
    /// and stores `static_cast<int>(spst)`).  We keep the public field an `i32`
    /// to match the C++ type exactly.
    pub sample_stride: i32,
}

impl DeepSlice {
    /// Construct a `DeepSlice` — port of the C++ `DeepSlice::DeepSlice`
    /// constructor.
    ///
    /// The C++ constructor signature orders the parameters as
    /// `(type, base, xStride, yStride, sampleStride, xSampling, ySampling,
    /// fillValue, xTileCoords, yTileCoords)` and forwards all but `sampleStride`
    /// to the [`Slice`] base.  We mirror that order precisely.
    ///
    /// `base_offset` is the signed byte offset that replaces the C++ `char*
    /// base` pointer (see [`crate::frame_buffer`] module docs).  `sample_stride`
    /// takes a `usize` here to mirror the C++ `size_t spst`, stored back as an
    /// `i32` (matching the C++ `static_cast<int>(spst)`).
    // C++ DeepSlice() has 10 parameters; #[allow] keeps the 1:1 port intact.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        type_: PixelType,
        base_offset: i64,
        x_stride: usize,
        y_stride: usize,
        sample_stride: usize,
        x_sampling: i32,
        y_sampling: i32,
        fill_value: f64,
        x_tile_coords: bool,
        y_tile_coords: bool,
    ) -> Result<Self> {
        Ok(Self {
            slice: Slice::new(
                type_,
                base_offset,
                x_stride,
                y_stride,
                x_sampling,
                y_sampling,
                fill_value,
                x_tile_coords,
                y_tile_coords,
            )?,
            sample_stride: sample_stride as i32,
        })
    }
}

impl Default for DeepSlice {
    /// C++ default `DeepSlice()`: a default [`Slice`] (`type=HALF`, sampling
    /// 1×1, all strides/offsets `0`, fill `0.0`, tile-coord flags false) plus
    /// `sampleStride == 0`.
    fn default() -> Self {
        Self {
            slice: Slice::default(),
            sample_stride: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DeepFrameBuffer — port of `class DeepFrameBuffer`
// ---------------------------------------------------------------------------

/// A map from channel name to [`DeepSlice`], plus a dedicated sample-count
/// [`Slice`] — port of `class DeepFrameBuffer`.
///
/// This mirrors [`crate::frame_buffer::FrameBuffer`] one-to-one (a
/// [`BTreeMap`] sorted by name, identical to the C++ `std::map<Name, ...>`),
/// and additionally holds `_sampleCounts`, the per-pixel sample-count slice.
///
/// The `[]` operator in C++ throws on a missing key; the Rust equivalents
/// [`DeepFrameBuffer::get`] / [`DeepFrameBuffer::get_mut`] return `Result`.
#[derive(Clone, Debug, Default)]
pub struct DeepFrameBuffer {
    /// Sorted name → deep slice map (mirrors C++ `SliceMap _map`).
    map: BTreeMap<Name, DeepSlice>,

    /// Per-pixel sample-count slice (mirrors C++ `Slice _sampleCounts`).
    ///
    /// Its `type_` must be `UINT` (enforced by
    /// [`DeepFrameBuffer::insert_sample_count_slice`]).  Defaults to a
    /// default-constructed [`Slice`] until explicitly set, exactly as the C++
    /// member does.
    sample_counts: Slice,
}

impl DeepFrameBuffer {
    /// Create an empty `DeepFrameBuffer`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a deep slice — port of `DeepFrameBuffer::insert`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if `name` is empty, matching the C++ `THROW`
    /// for empty-string names.
    pub fn insert(&mut self, name: impl Into<Name>, slice: DeepSlice) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(ExrError::Arg(
                "Frame buffer slice name cannot be an empty string.".into(),
            ));
        }
        // Enforce the layout invariant at the deep consumption seam (symmetric
        // with the flat `FrameBuffer::insert`): a `DeepSlice` built via a struct
        // literal or mutated after construction bypasses the fallible factory.
        slice.slice.validate_layout()?;
        self.map.insert(name, slice);
        Ok(())
    }

    /// Borrow an existing deep slice by name — port of
    /// `DeepFrameBuffer::operator[]`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if no slice with that name exists.
    pub fn get<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Result<&DeepSlice> {
        let bytes = Name::normalize(name.as_ref());
        self.map.get(bytes).ok_or_else(|| {
            ExrError::Arg(format!(
                "Cannot find frame buffer slice \"{}\".",
                String::from_utf8_lossy(bytes)
            ))
        })
    }

    /// Mutably borrow an existing deep slice by name — port of
    /// `DeepFrameBuffer::operator[]`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if no slice with that name exists.
    pub fn get_mut<K: AsRef<[u8]> + ?Sized>(&mut self, name: &K) -> Result<&mut DeepSlice> {
        let bytes = Name::normalize(name.as_ref());
        match self.map.get_mut(bytes) {
            Some(slice) => Ok(slice),
            None => Err(ExrError::Arg(format!(
                "Cannot find frame buffer slice \"{}\".",
                String::from_utf8_lossy(bytes)
            ))),
        }
    }

    /// Find a deep slice by name without error — port of
    /// `DeepFrameBuffer::findSlice`.
    ///
    /// Returns `None` if no slice with that name exists.
    pub fn find_slice<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Option<&DeepSlice> {
        self.map.get(Name::normalize(name.as_ref()))
    }

    /// Mutably find a deep slice by name — port of `DeepFrameBuffer::findSlice`
    /// (non-const).
    pub fn find_slice_mut<K: AsRef<[u8]> + ?Sized>(&mut self, name: &K) -> Option<&mut DeepSlice> {
        self.map.get_mut(Name::normalize(name.as_ref()))
    }

    /// Iterate over `(name, slice)` pairs in sorted-name order — port of
    /// `DeepFrameBuffer::begin/end`.
    pub fn iter(&self) -> impl Iterator<Item = (&Name, &DeepSlice)> {
        self.map.iter()
    }

    /// Mutably iterate over `(name, slice)` pairs in sorted-name order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Name, &mut DeepSlice)> {
        self.map.iter_mut()
    }

    /// Find an iterator position by name — port of `DeepFrameBuffer::find`.
    ///
    /// Returns `None` if not found.
    pub fn find<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Option<(&Name, &DeepSlice)> {
        self.map.get_key_value(Name::normalize(name.as_ref()))
    }

    /// Register the per-pixel sample-count slice — port of
    /// `DeepFrameBuffer::insertSampleCountSlice`.
    ///
    /// The sample-count slice gives `N(x, y)`, the number of samples at each
    /// pixel.  C++ rejects any type other than `UINT`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if `slice.type_ != PixelType::Uint`, mirroring
    /// the C++ `throw ArgExc`.
    pub fn insert_sample_count_slice(&mut self, slice: Slice) -> Result<()> {
        if slice.type_ != PixelType::Uint {
            return Err(ExrError::Arg(
                "The type of sample count slice should be UINT.".into(),
            ));
        }
        // Enforce the layout invariant on the count slice too (positive sampling,
        // representable strides) so the deep count-address math never receives an
        // invalid struct-literal/mutated slice.
        slice.validate_layout()?;
        self.sample_counts = slice;
        Ok(())
    }

    /// Borrow the per-pixel sample-count slice — port of
    /// `DeepFrameBuffer::getSampleCountSlice`.
    pub fn get_sample_count_slice(&self) -> &Slice {
        &self.sample_counts
    }

    /// Number of deep slices in this frame buffer.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the frame buffer has no deep slices.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::PixelType;

    // ---- DeepSlice: defaults match C++ ----

    #[test]
    fn deep_slice_default_matches_cpp() {
        // C++ default DeepSlice(): Slice default (HALF, 1x1 sampling, all
        // strides/offsets 0, fill 0.0, flags false) + sampleStride == 0.
        let d = DeepSlice::default();
        assert_eq!(d.slice.type_, PixelType::Half);
        assert_eq!(d.slice.base_offset, 0);
        assert_eq!(d.slice.x_stride, 0);
        assert_eq!(d.slice.y_stride, 0);
        assert_eq!(d.slice.x_sampling, 1);
        assert_eq!(d.slice.y_sampling, 1);
        assert_eq!(d.slice.fill_value, 0.0);
        assert!(!d.slice.x_tile_coords);
        assert!(!d.slice.y_tile_coords);
        assert_eq!(d.sample_stride, 0);
    }

    #[test]
    fn deep_slice_new_stores_sample_stride() {
        // sampleStride passed as size_t, stored as int — verify the cast.
        let d = DeepSlice::new(
            PixelType::Float,
            0,
            8,   // x_stride
            800, // y_stride
            4,   // sample_stride (sizeof FLOAT)
            1,
            1,
            0.0,
            false,
            false,
        )
        .unwrap();
        assert_eq!(d.slice.type_, PixelType::Float);
        assert_eq!(d.slice.x_stride, 8);
        assert_eq!(d.slice.y_stride, 800);
        assert_eq!(d.sample_stride, 4);
    }

    // ---- DeepFrameBuffer: insert / find round-trip ----

    #[test]
    fn dfb_insert_and_get() {
        let mut fb = DeepFrameBuffer::new();
        let s = DeepSlice::new(PixelType::Half, 0, 8, 80, 2, 1, 1, 0.0, false, false).unwrap();
        fb.insert("R", s.clone()).unwrap();
        let got = fb.get("R").unwrap();
        assert_eq!(*got, s);
        assert_eq!(got.sample_stride, 2);
    }

    #[test]
    fn dfb_insert_empty_name_errors() {
        let mut fb = DeepFrameBuffer::new();
        let err = fb.insert("", DeepSlice::default()).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn dfb_get_missing_errors() {
        let fb = DeepFrameBuffer::new();
        let err = fb.get("R").unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn dfb_find_slice_missing_is_none() {
        let fb = DeepFrameBuffer::new();
        assert!(fb.find_slice("R").is_none());
    }

    #[test]
    fn dfb_find_slice_present() {
        let mut fb = DeepFrameBuffer::new();
        fb.insert("G", DeepSlice::default()).unwrap();
        assert!(fb.find_slice("G").is_some());
    }

    #[test]
    fn dfb_iteration_order_is_sorted() {
        // Insert in reverse; iteration must come out sorted (matching C++ map).
        let mut fb = DeepFrameBuffer::new();
        fb.insert("Z", DeepSlice::default()).unwrap();
        fb.insert("A", DeepSlice::default()).unwrap();
        fb.insert("R", DeepSlice::default()).unwrap();
        let names: Vec<&str> = fb.iter().map(|(n, _)| n.to_str().unwrap_or("")).collect();
        assert_eq!(names, vec!["A", "R", "Z"]);
    }

    #[test]
    fn dfb_replace_existing() {
        let mut fb = DeepFrameBuffer::new();
        let s1 = DeepSlice::default();
        let mut s2 = DeepSlice::default();
        s2.slice.type_ = PixelType::Float;
        fb.insert("R", s1).unwrap();
        fb.insert("R", s2).unwrap();
        assert_eq!(fb.get("R").unwrap().slice.type_, PixelType::Float);
        assert_eq!(fb.len(), 1);
    }

    // ---- sample-count slice ----

    #[test]
    fn dfb_sample_count_slice_round_trip() {
        let mut fb = DeepFrameBuffer::new();
        // Sample-count slice must be UINT.
        let sc = Slice {
            type_: PixelType::Uint,
            x_stride: 4,
            y_stride: 40,
            ..Slice::default()
        };
        fb.insert_sample_count_slice(sc.clone()).unwrap();
        assert_eq!(*fb.get_sample_count_slice(), sc);
    }

    #[test]
    fn dfb_sample_count_slice_rejects_non_uint() {
        let mut fb = DeepFrameBuffer::new();
        let sc = Slice {
            type_: PixelType::Half,
            ..Slice::default()
        };
        let err = fb.insert_sample_count_slice(sc).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn dfb_default_sample_count_slice_is_default() {
        // Before being set, the sample-count slice is a default Slice (HALF),
        // exactly as the C++ default-constructed `_sampleCounts` member.
        let fb = DeepFrameBuffer::new();
        assert_eq!(*fb.get_sample_count_slice(), Slice::default());
    }

    // ---- deep addressing math (real production code, independently checked) ----
    //
    // These tests deliberately do NOT re-implement the address formula
    // inline (`base_offset + (x/xSampling)*xStride + (y/ySampling)*yStride`)
    // and compare it to a literal computed with that same formula -- a wrong
    // formula would never be caught, since the test and the "expected value"
    // would be the same code. That formula (`DeepSlice::x_stride`/`y_stride`)
    // is also not the one any real deep I/O path in this crate uses --
    // `DeepSlice`'s pointer-slot addressing is a documented, intentional
    // deviation that is structurally ported but functionally UNUSED (see
    // the module docs: the flat-heap readers do not dereference it); every
    // actual reader/writer instead calls `sample_table::count_byte_offset`/
    // `sample_count_at` on the SAMPLE-COUNT slice. These tests now call
    // THAT real production function, with the expected byte offset computed
    // independently (by hand, in the comment, not by calling the function
    // under test) and cross-checked by planting a known value at that exact
    // byte position and reading it back through the SAME production
    // function -- a wrong formula (swapped x/y, wrong sign, off-by-one
    // stride) fails both the offset assertion and the round-trip read.

    #[test]
    fn count_byte_offset_matches_independently_computed_address() {
        use crate::deep::sample_table::{count_byte_offset, sample_count_at};

        // A 10-wide grid of u32 (Xdr LE) sample counts, base_offset = 0,
        // x_stride = 4 (sizeof u32), y_stride = 40 (10 * 4).
        //
        // Pixel (3, 2)'s byte offset, computed BY HAND (not via the
        // function under test): row 2 starts at byte 2*40 = 80; column 3 is
        // a further 3*4 = 12 bytes in -> 80 + 12 = 92.
        let base_offset = 0i64;
        let x_stride = 4i32;
        let y_stride = 40i32;

        let off = count_byte_offset(base_offset, x_stride, y_stride, 3, 2)
            .expect("offset must be non-negative");
        assert_eq!(off, 92, "count_byte_offset diverged from the independently hand-derived address");

        // Round-trip: plant a known value at the hand-derived byte position
        // and confirm the SAME production function (`sample_count_at`)
        // reads it back -- proves the formula doesn't just happen to match
        // arithmetically but actually locates the real data.
        let mut buf = vec![0u8; 400];
        let known_value: u32 = 0xDEAD_BEEF;
        buf[92..96].copy_from_slice(&known_value.to_le_bytes());
        let got = sample_count_at(&buf, base_offset, x_stride, y_stride, 3, 2)
            .expect("sample_count_at must succeed for an in-range address");
        assert_eq!(got, known_value);
    }

    #[test]
    fn count_byte_offset_with_negative_base_offset_for_a_local_raster() {
        use crate::deep::sample_table::{count_byte_offset, sample_count_at};

        // A LOCAL count raster whose low corner is absolute pixel
        // (minX=5, minY=10) must map to byte 0 -- the sample-count slice
        // convention for a non-origin window:
        // base_offset = -(minY*yStride + minX*xStride).
        let x_stride = 4i32;
        let y_stride = 40i32;
        let base_offset = -(10i64 * y_stride as i64 + 5i64 * x_stride as i64); // -420

        // Low corner (5, 10) -> byte 0, hand-derived: -420 + 10*40 + 5*4
        // = -420 + 400 + 20 = 0.
        let off_corner = count_byte_offset(base_offset, x_stride, y_stride, 5, 10)
            .expect("the raster's own low corner must be in-range");
        assert_eq!(off_corner, 0);

        // A different pixel (7, 11), hand-derived: -420 + 11*40 + 7*4
        // = -420 + 440 + 28 = 48.
        let off_other = count_byte_offset(base_offset, x_stride, y_stride, 7, 11)
            .expect("offset must be non-negative");
        assert_eq!(off_other, 48, "count_byte_offset diverged from the independently hand-derived address");

        // Round-trip both through sample_count_at with distinct known values.
        let mut buf = vec![0u8; 200];
        buf[0..4].copy_from_slice(&111u32.to_le_bytes());
        buf[48..52].copy_from_slice(&222u32.to_le_bytes());
        assert_eq!(sample_count_at(&buf, base_offset, x_stride, y_stride, 5, 10).unwrap(), 111);
        assert_eq!(sample_count_at(&buf, base_offset, x_stride, y_stride, 7, 11).unwrap(), 222);

        // A pixel BEFORE the raster's low corner (4, 10) must be a hard
        // negative-address error (hard-error policy), not a panic or a
        // silent wrap: -420 + 10*40 + 4*4 = -420 + 400 + 16 = -4.
        assert!(count_byte_offset(base_offset, x_stride, y_stride, 4, 10).is_none());
        assert!(sample_count_at(&buf, base_offset, x_stride, y_stride, 4, 10).is_err());
    }
}
