//! Port of OpenEXR `ImfFrameBuffer.h/.cpp` — `Slice` and `FrameBuffer`.
//!
//! A [`FrameBuffer`] is a map from channel name to [`Slice`], describing where
//! and how each image channel lives in the caller's memory.  The I/O layer
//! (scanline/tiled read/write, added in later waves) uses these descriptors to
//! scatter/gather pixels between the compressed file and the application buffer.
//!
//! # Safe-Rust model of `char* base`
//!
//! In C++ a `Slice` stores a raw `char* base` whose address already has the
//! origin offset baked in (computed by `Slice::Make` as `ptr - offx - offy`).
//! The pixel-address formula is then just `base + xp/xSampling * xStride +
//! yp/ySampling * yStride` with no further subtraction.
//!
//! In this safe-Rust port there is **no raw pointer** in `Slice`.  Instead:
//!
//! * [`Slice::base_offset`] is an `i64` byte offset that encodes the same
//!   baked-in adjustment.  For a slice constructed directly (not via `Make`) it
//!   is `0` (the caller's pointer is conceptually the start of the buffer).
//!   For a slice produced by [`Slice::make`] it equals
//!   `-(offx + offy)` in signed bytes, exactly mirroring the C++ subtraction.
//!
//! * The actual buffer reference (`&[u8]` / `&mut [u8]`) is **not** stored in
//!   `Slice`.  It is passed at the call site when the I/O layer reads or writes
//!   pixels.  The I/O layer resolves the pixel address as:
//!
//!   ```text
//!   let idx = base_offset
//!       + (xp / x_sampling) as i64 * x_stride as i64
//!       + (yp / y_sampling) as i64 * y_stride as i64;
//!   buf[idx as usize]   // after bounds-checked cast
//!   ```
//!
//! This gives byte-identical address arithmetic to the C++ without any `unsafe`.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfFrameBuffer.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfFrameBuffer.cpp>

use std::collections::BTreeMap;

use imath_rs::{Box2i, V2i};

use crate::attr::name::Name;
use crate::attr::PixelType;
use crate::error::{ExrError, Result};

// ---------------------------------------------------------------------------
// Slice — port of `struct Slice`
// ---------------------------------------------------------------------------

/// Describes a single image channel inside a caller-owned pixel buffer.
///
/// Port of `struct Slice` in `ImfFrameBuffer.h`.
///
/// See the module-level docs for the safe-Rust treatment of `char* base`.
#[derive(Clone, Debug, PartialEq)]
pub struct Slice {
    /// Sample data type (UINT / HALF / FLOAT).
    pub type_: PixelType,

    /// Signed byte offset into the caller's buffer that mirrors the C++
    /// `char* base` pointer with the origin adjustment baked in.
    ///
    /// For a slice built with [`Slice::make`] this equals
    /// `-(origin.x / x_sampling * x_stride + origin.y / y_sampling * y_stride)`.
    /// For a manually constructed `Slice` it is `0` (the caller's buffer
    /// pointer is the conceptual origin).
    ///
    /// The I/O layer computes pixel byte addresses as:
    /// `base_offset + (xp / x_sampling) * x_stride + (yp / y_sampling) * y_stride`.
    pub base_offset: i64,

    /// Distance in bytes between horizontally adjacent pixels.
    ///
    /// Defaults (in [`Slice::make`]) to `sizeof(type_)`.
    pub x_stride: usize,

    /// Distance in bytes between vertically adjacent rows.
    ///
    /// Defaults (in [`Slice::make`]) to `(w / x_sampling) * x_stride`.
    pub y_stride: usize,

    /// Horizontal sub-sampling: pixel (x, _) is present only when
    /// `x % x_sampling == 0`.
    pub x_sampling: i32,

    /// Vertical sub-sampling: pixel (_, y) is present only when
    /// `y % y_sampling == 0`.
    pub y_sampling: i32,

    /// Fill value written into the slice when the file has no corresponding
    /// channel.
    pub fill_value: f64,

    /// If `true`, pixel x-coordinates are relative to the tile's upper-left
    /// corner (tiled files only; has no effect for scanline files).
    pub x_tile_coords: bool,

    /// If `true`, pixel y-coordinates are relative to the tile's upper-left
    /// corner (tiled files only; has no effect for scanline files).
    pub y_tile_coords: bool,
}

impl Slice {
    /// Construct a `Slice` with explicit layout — port of the C++ constructor.
    ///
    /// `base_offset` is the signed byte offset that replaces the C++ `char*
    /// base` pointer (see module-level docs).  For callers who are not using
    /// [`Slice::make`] the sensible value is `0` (the provided buffer starts at
    /// the image origin).
    // C++ Slice() has 9 parameters; #[allow] keeps the 1:1 port intact.
    //
    // # Errors
    // Returns [`ExrError::Arg`] if the layout invariant fails
    // ([`Slice::validate_layout`]): invalid pixel type, non-positive sampling, or
    // a stride outside the `i64` range.  C++ `Slice()` trusts the caller; we
    // reject an invalid layout at construction so it can never reach the I/O math.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        type_: PixelType,
        base_offset: i64,
        x_stride: usize,
        y_stride: usize,
        x_sampling: i32,
        y_sampling: i32,
        fill_value: f64,
        x_tile_coords: bool,
        y_tile_coords: bool,
    ) -> Result<Self> {
        let slice = Self {
            type_,
            base_offset,
            x_stride,
            y_stride,
            x_sampling,
            y_sampling,
            fill_value,
            x_tile_coords,
            y_tile_coords,
        };
        slice.validate_layout()?;
        Ok(slice)
    }

    /// Validate the slice's self-contained layout invariants — the single source
    /// of truth for a `Slice`'s constructible invariant.
    ///
    /// Checks: a valid pixel type, **positive** sampling (the I/O address math
    /// divides by `x_sampling`/`y_sampling`, so a zero would divide-by-zero and a
    /// negative would produce a nonsensical stride), and strides representable as
    /// `i64` (the offset math widens strides to `i64`).  A zero stride is allowed
    /// — it is the C++ default and the encoding used for zero-filled absent
    /// channels, and is degenerate but not unsafe.
    ///
    /// This does NOT check buffer extent (the maximum addressed byte), which
    /// needs the consuming buffer's length and is verified at the I/O seam.
    /// Because a `Slice`'s fields are public (and could be set by a struct
    /// literal that bypasses the factories, or mutated afterwards), the invariant
    /// is enforced both here and at [`FrameBuffer::insert`], so no unvalidated
    /// slice can reach the I/O layer.
    pub fn validate_layout(&self) -> Result<()> {
        if self.type_ == PixelType::NumPixelTypes {
            return Err(ExrError::Arg("Slice has an invalid pixel type.".into()));
        }
        if self.x_sampling <= 0 || self.y_sampling <= 0 {
            return Err(ExrError::Arg(format!(
                "Slice sampling must be positive (got x_sampling={}, y_sampling={}).",
                self.x_sampling, self.y_sampling
            )));
        }
        if i64::try_from(self.x_stride).is_err() || i64::try_from(self.y_stride).is_err() {
            return Err(ExrError::Arg(
                "Slice x/y stride exceeds the i64 range used by the address math.".into(),
            ));
        }
        Ok(())
    }

    /// Compute a fully-described `Slice` from a pointer represented as a
    /// `usize` address and an image geometry — port of `Slice::Make(type, ptr,
    /// origin, w, h, ...)`.
    ///
    /// This is the primary factory.  It mirrors the C++ overload that takes an
    /// explicit `origin: V2i` and dimensions `w`/`h`.
    ///
    /// # Defaults
    /// * `x_stride == 0` → `sizeof(type_)`.
    /// * `y_stride == 0` → `(w / x_sampling) * x_stride`.
    ///
    /// # Safe-Rust adaptation
    /// The C++ version computes `base = reinterpret_cast<intptr_t>(ptr)` then
    /// returns `reinterpret_cast<char*>(base - offx - offy)`.  Here `ptr` is
    /// passed as a `usize` (the address the caller obtained from their buffer),
    /// and `base_offset` is stored as `(ptr as i64) - offx - offy`.
    ///
    /// When the I/O layer resolves a pixel it adds `base_offset + coord_offset`
    /// and uses the result as a byte index into the caller's buffer — identical
    /// arithmetic to C++.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] for an invalid pixel type, non-positive
    /// sampling, a negative width, arithmetic that overflows the address math, or
    /// a layout that fails [`Slice::validate_layout`].
    // C++ Slice::Make has 11 parameters; #[allow] keeps the 1:1 port intact.
    #[allow(clippy::too_many_arguments)]
    pub fn make(
        type_: PixelType,
        ptr: usize,
        origin: V2i,
        w: i64,
        mut x_stride: usize,
        mut y_stride: usize,
        x_sampling: i32,
        y_sampling: i32,
        fill_value: f64,
        x_tile_coords: bool,
        y_tile_coords: bool,
    ) -> Result<Self> {
        // Sampling counts are positive by contract (channel xSampling/ySampling
        // are >= 1; header-derived channels are validated in `Header`). A public
        // caller passing 0 would divide-by-zero in the stride/offset math below
        // — and `make` returns `Result`, so it must reject rather than panic;
        // negative sampling yields a nonsensical negative stride/offset. This
        // guard sits at function entry so it also covers the `y_stride == 0`
        // derivation (`w / x_sampling`), not just the unconditional offx/offy
        // divisions. C++ `Slice::Make` trusts the caller here (UB on 0); we
        // surface a clean error instead, keeping the no-panic invariant.
        if x_sampling <= 0 || y_sampling <= 0 {
            return Err(ExrError::Arg(format!(
                "Slice::make: x_sampling and y_sampling must be positive \
                 (got x_sampling={x_sampling}, y_sampling={y_sampling})."
            )));
        }
        if x_stride == 0 {
            x_stride = match type_ {
                PixelType::Uint => 4,
                PixelType::Half => 2,
                PixelType::Float => 4,
                PixelType::NumPixelTypes => {
                    return Err(ExrError::Arg("Invalid pixel type.".into()));
                }
            };
        }
        // A negative width would make `w / xSampling` negative and wrap when cast
        // to `usize` below; reject it (C++ trusts a non-negative width here).
        if w < 0 {
            return Err(ExrError::Arg(format!(
                "Slice::make: width must be non-negative (got w={w})."
            )));
        }
        if y_stride == 0 {
            // C++: yStride = size_t(w / xSampling) * xStride.  Checked so an
            // extreme width/stride returns an error instead of wrapping usize.
            let pixels = (w / x_sampling as i64) as usize;
            y_stride = pixels.checked_mul(x_stride).ok_or_else(|| {
                ExrError::Arg("Slice::make: derived y_stride overflows usize.".into())
            })?;
        }

        // Mirror C++: offx = (origin.x / xSampling) * xStride; base = ptr - offx
        // - offy.  Widened to i64 (as C++) and checked so extreme strides/origins
        // return an error rather than overflowing the signed address math.
        let x_stride_i64 = i64::try_from(x_stride)
            .map_err(|_| ExrError::Arg("Slice::make: x_stride exceeds the i64 range.".into()))?;
        let y_stride_i64 = i64::try_from(y_stride)
            .map_err(|_| ExrError::Arg("Slice::make: y_stride exceeds the i64 range.".into()))?;
        let offx = (origin.x as i64 / x_sampling as i64)
            .checked_mul(x_stride_i64)
            .ok_or_else(|| ExrError::Arg("Slice::make: x offset overflows i64.".into()))?;
        let offy = (origin.y as i64 / y_sampling as i64)
            .checked_mul(y_stride_i64)
            .ok_or_else(|| ExrError::Arg("Slice::make: y offset overflows i64.".into()))?;
        let base_offset = (ptr as i64)
            .checked_sub(offx)
            .and_then(|v| v.checked_sub(offy))
            .ok_or_else(|| ExrError::Arg("Slice::make: base offset overflows i64.".into()))?;

        let slice = Self {
            type_,
            base_offset,
            x_stride,
            y_stride,
            x_sampling,
            y_sampling,
            fill_value,
            x_tile_coords,
            y_tile_coords,
        };
        slice.validate_layout()?;
        Ok(slice)
    }

    /// Convenience overload: derive `w`/`h` from a `Box2i` data window — port
    /// of `Slice::Make(type, ptr, dataWindow, ...)`.
    // C++ Make-from-Box2i overload has 10 parameters; #[allow] keeps the 1:1 port.
    #[allow(clippy::too_many_arguments)]
    pub fn make_from_window(
        type_: PixelType,
        ptr: usize,
        data_window: Box2i,
        x_stride: usize,
        y_stride: usize,
        x_sampling: i32,
        y_sampling: i32,
        fill_value: f64,
        x_tile_coords: bool,
        y_tile_coords: bool,
    ) -> Result<Self> {
        let w = data_window.max.x as i64 - data_window.min.x as i64 + 1;
        Self::make(
            type_,
            ptr,
            data_window.min,
            w,
            x_stride,
            y_stride,
            x_sampling,
            y_sampling,
            fill_value,
            x_tile_coords,
            y_tile_coords,
        )
    }
}

impl Default for Slice {
    /// C++ default `Slice()`: `type=HALF`, all strides/offsets 0, sampling 1×1,
    /// fill 0.0, tile-coord flags false.
    fn default() -> Self {
        Self {
            type_: PixelType::Half,
            base_offset: 0,
            x_stride: 0,
            y_stride: 0,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        }
    }
}

// ---------------------------------------------------------------------------
// FrameBuffer — port of `class FrameBuffer`
// ---------------------------------------------------------------------------

/// A map from channel name to [`Slice`] — port of `class FrameBuffer`.
///
/// C++ uses `std::map<Name, Slice>` (sorted by name).  We mirror that with a
/// [`BTreeMap`] so iteration order is identical.
///
/// The `[]` operator in C++ throws on missing key; the Rust equivalent
/// [`FrameBuffer::get`] / [`FrameBuffer::get_mut`] return `Result`.
#[derive(Clone, Debug, Default)]
pub struct FrameBuffer {
    map: BTreeMap<Name, Slice>,
    /// Which destination buffer each slice lives in (absent = buffer 0): the planar-read extension,
    /// see [`FrameBuffer::insert_in`]. OpenEXR has one `char*` per slice, so this is how the safe
    /// port expresses "this channel is in ANOTHER caller buffer" without a pointer.
    buffers: BTreeMap<Name, usize>,
}

impl FrameBuffer {
    /// Create an empty `FrameBuffer`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a slice — port of `FrameBuffer::insert`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if `name` is empty, matching the C++ `THROW`
    /// for empty-string names.
    ///
    /// `name` accepts anything convertible to a byte-string [`Name`].
    pub fn insert(&mut self, name: impl Into<Name>, slice: Slice) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(ExrError::Arg(
                "Frame buffer slice name cannot be an empty string.".into(),
            ));
        }
        // Enforce the layout invariant at the consumption seam: a `Slice` built by
        // a struct literal (fields are public) or mutated after construction
        // bypasses the fallible factories, so this is the robust gate that keeps
        // an invalid slice from reaching the I/O address math.
        slice.validate_layout()?;
        self.buffers.remove(&name);
        self.map.insert(name, slice);
        Ok(())
    }

    /// [`insert`](Self::insert) a slice whose pixels live in destination buffer `buffer`
    /// (an index into the `planes` slice later given to the planar read, e.g.
    /// `ScanLineInputFile::read_pixels_planes`). The slice's `base_offset` and strides are
    /// relative to THAT buffer. Extension beyond OpenEXR (one pointer per slice there): the
    /// caller owns one allocation per output plane, as an `AVFrame` or any planar image does.
    ///
    /// # Errors
    /// As [`insert`](Self::insert).
    pub fn insert_in(&mut self, name: impl Into<Name>, slice: Slice, buffer: usize) -> Result<()> {
        let name = name.into();
        self.insert(name.clone(), slice)?;
        if buffer != 0 {
            self.buffers.insert(name, buffer);
        }
        Ok(())
    }

    /// The destination buffer index of the slice named `name` (0 unless set by
    /// [`insert_in`](Self::insert_in)).
    pub fn buffer_of<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> usize {
        self.buffers.get(Name::normalize(name.as_ref())).copied().unwrap_or(0)
    }

    /// Borrow an existing slice by name — port of `FrameBuffer::operator[]`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if no slice with that name exists.
    pub fn get<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Result<&Slice> {
        let bytes = Name::normalize(name.as_ref());
        self.map.get(bytes).ok_or_else(|| {
            ExrError::Arg(format!(
                "Cannot find frame buffer slice \"{}\".",
                String::from_utf8_lossy(bytes)
            ))
        })
    }

    /// Mutably borrow an existing slice by name — port of `FrameBuffer::operator[]`.
    ///
    /// # Errors
    /// Returns [`ExrError::Arg`] if no slice with that name exists.
    pub fn get_mut<K: AsRef<[u8]> + ?Sized>(&mut self, name: &K) -> Result<&mut Slice> {
        let bytes = Name::normalize(name.as_ref());
        match self.map.get_mut(bytes) {
            Some(slice) => Ok(slice),
            None => Err(ExrError::Arg(format!(
                "Cannot find frame buffer slice \"{}\".",
                String::from_utf8_lossy(bytes)
            ))),
        }
    }

    /// Find a slice by name without error — port of `FrameBuffer::findSlice`.
    ///
    /// Returns `None` if no slice with that name exists.
    pub fn find_slice<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Option<&Slice> {
        self.map.get(Name::normalize(name.as_ref()))
    }

    /// Mutably find a slice by name — port of `FrameBuffer::findSlice` (non-const).
    pub fn find_slice_mut<K: AsRef<[u8]> + ?Sized>(&mut self, name: &K) -> Option<&mut Slice> {
        self.map.get_mut(Name::normalize(name.as_ref()))
    }

    /// Iterate over `(name, slice)` pairs in sorted-name order — port of
    /// `FrameBuffer::begin/end`.
    pub fn iter(&self) -> impl Iterator<Item = (&Name, &Slice)> {
        self.map.iter()
    }

    /// Mutably iterate over `(name, slice)` pairs in sorted-name order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Name, &mut Slice)> {
        self.map.iter_mut()
    }

    /// Find an iterator position by name — port of `FrameBuffer::find`.
    ///
    /// Returns `None` if not found.
    pub fn find<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Option<(&Name, &Slice)> {
        self.map.get_key_value(Name::normalize(name.as_ref()))
    }

    /// Number of slices in this frame buffer.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the frame buffer has no slices.
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
    use imath_rs::{Box2i, V2i};
    use crate::attr::PixelType;

    // ---- FrameBuffer: insert / find / iterate ----

    #[test]
    fn fb_insert_and_get() {
        let mut fb = FrameBuffer::new();
        let s = Slice::default();
        fb.insert("R", s.clone()).unwrap();
        let got = fb.get("R").unwrap();
        assert_eq!(got.type_, PixelType::Half);
    }

    #[test]
    fn fb_insert_empty_name_errors() {
        let mut fb = FrameBuffer::new();
        let err = fb.insert("", Slice::default()).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn fb_get_missing_errors() {
        let fb = FrameBuffer::new();
        let err = fb.get("R").unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn fb_find_slice_missing_is_none() {
        let fb = FrameBuffer::new();
        assert!(fb.find_slice("R").is_none());
    }

    #[test]
    fn fb_find_slice_present() {
        let mut fb = FrameBuffer::new();
        fb.insert("G", Slice::default()).unwrap();
        assert!(fb.find_slice("G").is_some());
    }

    #[test]
    fn fb_iteration_order_is_sorted() {
        // Insert in reverse order; iteration must come out sorted (matching C++ map).
        let mut fb = FrameBuffer::new();
        fb.insert("Z", Slice::default()).unwrap();
        fb.insert("A", Slice::default()).unwrap();
        fb.insert("R", Slice::default()).unwrap();
        let names: Vec<&str> = fb.iter().map(|(n, _)| n.to_str().unwrap_or("")).collect();
        assert_eq!(names, vec!["A", "R", "Z"]);
    }

    #[test]
    fn fb_replace_existing() {
        let mut fb = FrameBuffer::new();
        let s1 = Slice { type_: PixelType::Half, ..Slice::default() };
        let s2 = Slice { type_: PixelType::Float, ..Slice::default() };
        fb.insert("R", s1).unwrap();
        fb.insert("R", s2).unwrap();
        assert_eq!(fb.get("R").unwrap().type_, PixelType::Float);
        assert_eq!(fb.len(), 1);
    }

    // ---- Slice::make derives strides correctly ----

    #[test]
    fn slice_make_default_strides_half() {
        // HALF, width=10, origin=(0,0), xSampling=1, ySampling=1
        // xStride = 2 (sizeof half), yStride = 10 * 2 = 20
        let s = Slice::make(
            PixelType::Half,
            0,
            V2i::new(0, 0),
            10,
            0, 0,
            1, 1,
            0.0, false, false,
        ).unwrap();
        assert_eq!(s.x_stride, 2);
        assert_eq!(s.y_stride, 20);
    }

    #[test]
    fn slice_make_default_strides_float() {
        // FLOAT, width=8, origin=(0,0), xSampling=1, ySampling=1
        // xStride = 4, yStride = 8 * 4 = 32
        let s = Slice::make(
            PixelType::Float,
            0,
            V2i::new(0, 0),
            8,
            0, 0,
            1, 1,
            0.0, false, false,
        ).unwrap();
        assert_eq!(s.x_stride, 4);
        assert_eq!(s.y_stride, 32);
    }

    #[test]
    fn slice_make_default_strides_uint() {
        // UINT, width=4, origin=(0,0), xSampling=1, ySampling=1
        // xStride = 4, yStride = 4 * 4 = 16
        let s = Slice::make(
            PixelType::Uint,
            0,
            V2i::new(0, 0),
            4,
            0, 0,
            1, 1,
            0.0, false, false,
        ).unwrap();
        assert_eq!(s.x_stride, 4);
        assert_eq!(s.y_stride, 16);
    }

    #[test]
    fn slice_make_rejects_zero_and_negative_sampling() {
        // Zero x_sampling: must error, NOT panic (the offx/offy and y_stride
        // derivations divide by sampling). Use y_stride=0 to also exercise the
        // `w / x_sampling` derivation path.
        let err = Slice::make(
            PixelType::Half,
            0,
            V2i::new(0, 0),
            10,
            0, 0,
            0, 1,
            0.0, false, false,
        )
        .unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));

        // Zero y_sampling.
        let err = Slice::make(
            PixelType::Half,
            0,
            V2i::new(0, 0),
            10,
            0, 0,
            1, 0,
            0.0, false, false,
        )
        .unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));

        // Negative sampling yields nonsensical strides/offsets — reject too.
        let err = Slice::make(
            PixelType::Half,
            0,
            V2i::new(0, 0),
            10,
            0, 0,
            -1, 1,
            0.0, false, false,
        )
        .unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn slice_make_from_window_inherits_sampling_guard() {
        // make_from_window forwards to make(), so the guard applies here too.
        let dw = Box2i { min: V2i::new(0, 0), max: V2i::new(9, 9) };
        let err = Slice::make_from_window(
            PixelType::Half,
            0,
            dw,
            0, 0,
            0, 0,
            0.0, false, false,
        )
        .unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn slice_make_base_offset_with_origin() {
        // HALF, width=10, origin=(2, 3), xSampling=1, ySampling=1
        // xStride = 2, yStride = 10 * 2 = 20
        // offx = (2/1) * 2 = 4,  offy = (3/1) * 20 = 60
        // base_offset = 0 - 4 - 60 = -64
        let ptr: usize = 0;
        let s = Slice::make(
            PixelType::Half,
            ptr,
            V2i::new(2, 3),
            10,
            0, 0,
            1, 1,
            0.0, false, false,
        ).unwrap();
        assert_eq!(s.x_stride, 2);
        assert_eq!(s.y_stride, 20);
        assert_eq!(s.base_offset, -64_i64);
    }

    #[test]
    fn slice_make_from_window() {
        // dataWindow = [(0,0)..(9,4)] → w=10, h=5
        // HALF, ptr=0, origin=(0,0)
        // xStride=2, yStride=20
        let dw = Box2i { min: V2i::new(0, 0), max: V2i::new(9, 4) };
        let s = Slice::make_from_window(
            PixelType::Half,
            0,
            dw,
            0, 0,
            1, 1,
            0.0, false, false,
        ).unwrap();
        assert_eq!(s.x_stride, 2);
        assert_eq!(s.y_stride, 20);
        assert_eq!(s.base_offset, 0);
    }

    #[test]
    fn slice_make_invalid_type_errors() {
        let err = Slice::make(
            PixelType::NumPixelTypes,
            0,
            V2i::new(0, 0),
            10,
            0, 0,
            1, 1,
            0.0, false, false,
        ).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }

    #[test]
    fn slice_make_explicit_strides_not_overridden() {
        // Explicit xStride=8, yStride=128 must NOT be overridden by defaults.
        let s = Slice::make(
            PixelType::Float,
            0,
            V2i::new(0, 0),
            10,
            8, 128,
            1, 1,
            0.0, false, false,
        ).unwrap();
        assert_eq!(s.x_stride, 8);
        assert_eq!(s.y_stride, 128);
    }

    // ---- Slice layout invariant ----

    #[test]
    fn slice_new_valid_ok() {
        let s = Slice::new(PixelType::Half, 0, 2, 20, 1, 1, 0.0, false, false).unwrap();
        assert_eq!(s.x_stride, 2);
        assert_eq!(s.y_stride, 20);
    }

    #[test]
    fn slice_new_rejects_bad_layout() {
        // Non-positive sampling (would divide-by-zero / negate the strides) and an
        // invalid pixel type are rejected at construction.
        assert!(Slice::new(PixelType::Half, 0, 2, 20, 0, 1, 0.0, false, false).is_err());
        assert!(Slice::new(PixelType::Half, 0, 2, 20, 1, -1, 0.0, false, false).is_err());
        assert!(Slice::new(PixelType::NumPixelTypes, 0, 2, 20, 1, 1, 0.0, false, false).is_err());
    }

    #[test]
    fn slice_new_rejects_oversized_stride() {
        // A stride outside the i64 range used by the address math is rejected.
        assert!(Slice::new(PixelType::Half, 0, usize::MAX, 20, 1, 1, 0.0, false, false).is_err());
    }

    #[test]
    fn fb_insert_rejects_invalid_slice() {
        // A struct-literal slice bypasses the fallible factory; insert is the
        // robust seam that still rejects an invalid (zero-sampling) layout.
        let bad = Slice {
            type_: PixelType::Half,
            base_offset: 0,
            x_stride: 2,
            y_stride: 20,
            x_sampling: 0,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        };
        let mut fb = FrameBuffer::new();
        assert!(fb.insert("R", bad).is_err());
    }

    #[test]
    fn slice_make_rejects_negative_width() {
        let err = Slice::make(
            PixelType::Float,
            0,
            V2i::new(0, 0),
            -5,
            0, 0,
            1, 1,
            0.0, false, false,
        ).unwrap_err();
        assert!(matches!(err, ExrError::Arg(_)));
    }
}
