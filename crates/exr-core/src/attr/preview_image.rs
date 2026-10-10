//! Port of OpenEXR `ImfPreviewImage.h/.cpp` and `ImfPreviewImageAttribute.h/.cpp`.
//!
//! A [`PreviewImage`] is a small, low-dynamic-range RGBA thumbnail stored in an
//! EXR header.  Each pixel is a [`PreviewRgba`] (4 × `u8`).  The `"preview"`
//! attribute is laid out on disk as `width` (`u32`), `height` (`u32`), then
//! `4 * width * height` bytes (r, g, b, a per pixel, row-major).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPreviewImage.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPreviewImage.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPreviewImageAttribute.cpp>

use crate::attr::attribute::{AttrValue, TypedAttribute};
use crate::checked_arithmetic;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// PreviewRgba — mirrors `struct PreviewRgba` in ImfPreviewImage.h
// ---------------------------------------------------------------------------

/// One preview-image pixel — port of `struct PreviewRgba`.
///
/// Intensity is proportional to `pow(x/255, 2.2)` for `x` in `{r,g,b}`; `a` is
/// the pixel's alpha (`0 == transparent`, `255 == opaque`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreviewRgba {
    /// Red component.
    pub r: u8,
    /// Green component.
    pub g: u8,
    /// Blue component.
    pub b: u8,
    /// Alpha component (`0` transparent … `255` opaque).
    pub a: u8,
}

impl PreviewRgba {
    /// Construct a pixel — mirrors `PreviewRgba(r, g, b, a)`.
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

impl Default for PreviewRgba {
    /// C++ default `PreviewRgba()`: `r=0, g=0, b=0, a=255`.
    fn default() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }
    }
}

// ---------------------------------------------------------------------------
// PreviewImage — mirrors `class PreviewImage` in ImfPreviewImage.h/.cpp
// ---------------------------------------------------------------------------

/// A small RGBA preview/thumbnail image — port of `class PreviewImage`.
///
/// Pixel `(x, y)` is at index `y * width + x`; pixel `(0, 0)` is the upper-left
/// corner.  The `pixels` vector always holds exactly `width * height` entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviewImage {
    /// Width in pixels (`PreviewImage::_width`).
    width: u32,
    /// Height in pixels (`PreviewImage::_height`).
    height: u32,
    /// Row-major pixel array (`PreviewImage::_pixels`), length `width * height`.
    pixels: Vec<PreviewRgba>,
}

// Two u32 dimensions, followed by four wire bytes per pixel.
fn encoded_size(pixel_count: u64) -> Option<u64> {
    pixel_count.checked_mul(4)?.checked_add(8)
}

impl PreviewImage {
    /// Check the pixel count, allocation size and signed wire size without allocating.
    ///
    /// Use this before generating a preview whose dimensions come from a header.
    /// The constructor uses the same checks before accepting or allocating pixels.
    pub fn checked_pixel_count(width: u32, height: u32) -> Result<usize> {
        let count = checked_arithmetic::ui_mult_u32(width, height)? as usize;
        let count = checked_arithmetic::check_array_size(count, std::mem::size_of::<PreviewRgba>())?;
        let wire_size = encoded_size(count as u64)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or_else(|| ExrError::Overflow("Preview image wire size exceeds addressable memory".into()))?;
        xdr::checked_wire_len(wire_size)?;
        Ok(count)
    }

    /// Construct a `width × height` preview image.
    ///
    /// Mirrors `PreviewImage(width, height, pixels[])`: if `pixels` is `None`
    /// every pixel is initialised to the default `PreviewRgba` (`0,0,0,255`).
    ///
    /// # Errors
    /// [`ExrError::Overflow`] ("Integer multiplication overflow."), mirroring
    /// C++'s `_pixels = new PreviewRgba[checkArraySize(uiMult(_width, _height),
    /// sizeof(PreviewRgba))]` (`ImfPreviewImage.cpp`): `uiMult` first checks
    /// `width * height` in `unsigned int` (32-bit, even on a 64-bit `usize`
    /// build — a deliberate 1:1 divergence point from a plain `usize` multiply),
    /// then `checkArraySize` checks that count times `sizeof(PreviewRgba)`
    /// against `usize`. The encoded attribute must also fit its signed i32
    /// wire-length field; this is checked before accepting or allocating pixels.
    pub fn new(width: u32, height: u32, pixels: Option<Vec<PreviewRgba>>) -> Result<Self> {
        let count = Self::checked_pixel_count(width, height)?;

        let pixels = match pixels {
            Some(p) => {
                if p.len() != count {
                    return Err(ExrError::Arg(format!(
                        "Preview image pixel array length {} does not match width*height {count}",
                        p.len()
                    )));
                }
                p
            }
            None => vec![PreviewRgba::default(); count],
        };

        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Width in pixels — mirrors `PreviewImage::width()`.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels — mirrors `PreviewImage::height()`.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Read-only access to the row-major pixel array — mirrors `pixels()`.
    pub fn pixels(&self) -> &[PreviewRgba] {
        &self.pixels
    }

    /// Mutable access to the pixel array — mirrors the non-const `pixels()`.
    pub fn pixels_mut(&mut self) -> &mut [PreviewRgba] {
        &mut self.pixels
    }

    /// Access pixel `(x, y)` — mirrors `PreviewImage::pixel(x, y)`.
    ///
    /// Returns `None` if out of range (the C++ indexes unchecked; we bound it).
    pub fn pixel(&self, x: u32, y: u32) -> Option<&PreviewRgba> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.pixels.get((y * self.width + x) as usize)
    }
}

// ---------------------------------------------------------------------------
// AttrValue — byte-exact `"preview"` serialisation
// ---------------------------------------------------------------------------

impl AttrValue for PreviewImage {
    fn static_type_name() -> &'static str {
        // ImfPreviewImageAttribute.cpp: return "preview";
        "preview"
    }

    /// Mirrors `PreviewImageAttribute::writeValueTo`.
    ///
    /// Writes `width` (`u32`), `height` (`u32`), then `r, g, b, a` for each of
    /// the `width * height` pixels in row-major order.
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        xdr::write_u32(o, self.width)?;
        xdr::write_u32(o, self.height)?;

        for p in &self.pixels {
            xdr::write_u8_val(o, p.r)?;
            xdr::write_u8_val(o, p.g)?;
            xdr::write_u8_val(o, p.b)?;
            xdr::write_u8_val(o, p.a)?;
        }
        Ok(())
    }

    /// Mirrors `PreviewImageAttribute::readValueFrom`.
    ///
    /// Reads `width`/`height` as `int` (rejecting negatives), validates that
    /// `width * height * 4 + 8 == size` (the C++ dimension/size consistency
    /// check), then reads the `4 * width * height` pixel bytes.
    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        // C++ reads width/height as signed `int` and rejects negatives.
        let width = xdr::read_i32(i)?;
        let height = xdr::read_i32(i)?;

        if width < 0 || height < 0 {
            return Err(ExrError::Input(
                "Invalid dimensions in Preview Image Attribute".into(),
            ));
        }

        // total size == 4 bytes/pixel + 8 bytes for the two dimensions.
        let count = (width as u64) * (height as u64);
        let expected = encoded_size(count).ok_or_else(|| ExrError::Input(
            "Mismatch between Preview Image Attribute size and dimensions".into(),
        ))?;
        let bytes = expected - 8;
        if expected != (size as u64) {
            return Err(ExrError::Input(
                "Mismatch between Preview Image Attribute size and dimensions".into(),
            ));
        }
        // C++ 3.4 core `extract_attr_preview` additionally rejects a
        // 0x0 (zero-byte) preview outright (`bytes == 0` ->
        // `EXR_ERR_ATTR_SIZE_MISMATCH`), which the non-strict header reader
        // then drops. 3.3.11 `PreviewImageAttribute::readValueFrom` accepted
        // it (an empty `PreviewImage`).
        // <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXRCore/parse_header.c#L896-L905>
        if bytes == 0 {
            return Err(ExrError::Input(
                "Invalid size for preview: width or height is zero".into(),
            ));
        }

        let count = count as usize;

        // Refuse to pre-reserve more pixels than the stream can supply, and
        // reserve fallibly so a size-consistent but still-huge declaration errors
        // instead of aborting the process on OOM. Each pixel is 4 wire bytes.
        let byte_len = bytes;
        if let Some(total) = i.size() {
            let pos = i.tellg()?;
            let remaining = total.saturating_sub(pos);
            if byte_len > remaining {
                return Err(ExrError::Input(format!(
                    "preview image declares {byte_len} pixel bytes but only {remaining} remain"
                )));
            }
        }
        let mut pixels: Vec<PreviewRgba> = Vec::new();
        pixels.try_reserve_exact(count).map_err(|_| {
            ExrError::Input(format!("failed to allocate {count} preview pixels"))
        })?;
        for _ in 0..count {
            let r = xdr::read_u8_val(i)?;
            let g = xdr::read_u8_val(i)?;
            let b = xdr::read_u8_val(i)?;
            let a = xdr::read_u8_val(i)?;
            pixels.push(PreviewRgba::new(r, g, b, a));
        }

        PreviewImage::new(width as u32, height as u32, Some(pixels))
    }
}

/// `typedef TypedAttribute<PreviewImage> PreviewImageAttribute;`
pub type PreviewImageAttribute = TypedAttribute<PreviewImage>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(p: &PreviewImage) -> Vec<u8> {
        let mut o = MemOStream::new();
        p.write_value(&mut o, 0).expect("write");
        o.into_data()
    }

    #[test]
    fn wire_size_boundary_is_checked_without_allocating() {
        // The payload is 8 + 4*N, so the last fitting count leaves three bytes spare.
        assert_eq!(PreviewImage::checked_pixel_count(536_870_909, 1).unwrap(), 536_870_909);
        assert_eq!(encoded_size(536_870_909), Some(2_147_483_644));
        match PreviewImage::checked_pixel_count(536_870_910, 1).unwrap_err() {
            ExrError::Overflow(message) => assert_eq!(
                message, "value 2147483648 exceeds the i32 wire-length limit",
            ),
            other => panic!("expected Overflow, got {other:?}"),
        }
        assert_eq!(PreviewImage::checked_pixel_count(2, 3).unwrap(), 6);
        assert_eq!(PreviewImage::checked_pixel_count(0, 1).unwrap(), 0);
        assert!(matches!(PreviewImage::checked_pixel_count(u32::MAX, u32::MAX), Err(ExrError::Overflow(_))));
    }

    #[test]
    fn constructor_checks_wire_size_before_accepting_pixels() {
        // An empty supplied vector makes the regression safe: without the wire check
        // it returns Arg for the length mismatch, rather than allocating gigabytes.
        let err = PreviewImage::new(999, 749_250, Some(Vec::new())).unwrap_err();
        match err {
            ExrError::Overflow(message) => assert_eq!(
                message, "value 2994003008 exceeds the i32 wire-length limit",
            ),
            other => panic!("expected wire Overflow before pixel validation, got {other:?}"),
        }
    }

    #[test]
    fn type_name_is_preview() {
        assert_eq!(PreviewImage::static_type_name(), "preview");
    }

    /// `PreviewImage::new`'s dimension check must
    /// route through `checked_arithmetic` (`uiMult`/`checkArraySize`,
    /// `ImfPreviewImage.cpp`), giving `ExrError::Overflow`, not `Arg`.
    #[test]
    fn new_rejects_dimension_overflow_as_overflow_class() {
        match PreviewImage::new(u32::MAX, u32::MAX, None) {
            Err(ExrError::Overflow(_)) => {}
            other => panic!("expected Overflow error, got {other:?}"),
        }
    }

    /// A 0x0 preview is rejected outright, matching C++ 3.4 core
    /// `extract_attr_preview` (`bytes == 0` -> `EXR_ERR_ATTR_SIZE_MISMATCH`).
    #[test]
    fn read_value_rejects_zero_by_zero_preview() {
        // width=0, height=0, declared size = 8 (just the two dimensions).
        let mut raw = Vec::new();
        raw.extend_from_slice(&0i32.to_le_bytes());
        raw.extend_from_slice(&0i32.to_le_bytes());
        let mut i = MemIStream::new(raw);
        assert!(PreviewImage::read_value(&mut i, 8, 0).is_err());
    }

    #[test]
    fn default_pixels_are_opaque_black() {
        let p = PreviewImage::new(2, 1, None).unwrap();
        assert_eq!(p.pixels().len(), 2);
        assert_eq!(p.pixels()[0], PreviewRgba::new(0, 0, 0, 255));
    }

    #[test]
    fn exact_bytes_and_size() {
        let pixels = vec![
            PreviewRgba::new(1, 2, 3, 4),
            PreviewRgba::new(5, 6, 7, 8),
        ];
        let p = PreviewImage::new(2, 1, Some(pixels)).unwrap();
        let bytes = write_bytes(&p);

        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            // width = 2 (u32 LE)
            0x02, 0x00, 0x00, 0x00,
            // height = 1 (u32 LE)
            0x01, 0x00, 0x00, 0x00,
            // pixel 0: r,g,b,a
            1, 2, 3, 4,
            // pixel 1: r,g,b,a
            5, 6, 7, 8,
        ];
        assert_eq!(bytes, expected);

        // size == 4*w*h + 8, with w=2, h=1 -> 2*1*4 + 8 == 16
        let (w, h) = (2u64, 1u64);
        assert_eq!(bytes.len() as u64, w * h * 4 + 8);
    }

    #[test]
    fn round_trip() {
        let pixels = (0..(3 * 2))
            .map(|n| PreviewRgba::new(n as u8, (n * 2) as u8, (n * 3) as u8, 255))
            .collect();
        let p = PreviewImage::new(3, 2, Some(pixels)).unwrap();

        let bytes = write_bytes(&p);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = PreviewImage::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn size_mismatch_rejected() {
        // Claim 2x2 but provide a size that does not match.
        let mut o = MemOStream::new();
        xdr::write_u32(&mut o, 2).unwrap();
        xdr::write_u32(&mut o, 2).unwrap();
        // only write 1 pixel worth of data
        xdr::write_chars(&mut o, &[0, 0, 0, 0]).unwrap();
        let bytes = o.into_data();
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        assert!(PreviewImage::read_value(&mut i, size, 0).is_err());
    }

    // A header declaring a ~2 GiB preview (dimensions consistent with
    // `size`, but no pixel bytes present) must error via the remaining-bytes /
    // fallible-reservation guard rather than reserve 500M pixels and abort.
    #[test]
    fn huge_dimensions_rejected_without_alloc() {
        // w = h = 23170 -> w*h*4 + 8 == 2_147_395_608 (fits in a positive i32).
        let w: u32 = 23170;
        let h: u32 = 23170;
        let size = (w as u64 * h as u64 * 4 + 8) as i32;
        let mut o = MemOStream::new();
        xdr::write_u32(&mut o, w).unwrap();
        xdr::write_u32(&mut o, h).unwrap();
        // Deliberately provide no pixel bytes: only the 8-byte dimensions.
        let mut i = MemIStream::new(o.into_data());
        let err = PreviewImage::read_value(&mut i, size, 0)
            .expect_err("huge preview declaration must be rejected");
        assert!(matches!(err, ExrError::Input(_)));
    }
}
