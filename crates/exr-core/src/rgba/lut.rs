//! Port of OpenEXR's `ImfLut.h`/`ImfLut.cpp`.
//!
//! Lookup tables for fast application of a `half -> half` function to pixel
//! data. Both [`HalfLut`] and [`RgbaLut`] build on [`imath_rs::HalfFunction`]
//! (the already-ported `halfFunction<T>` — a precomputed 65,536-entry table,
//! one slot per `half` bit pattern), matching the C++ where `_lut` is a
//! `halfFunction<half>` built with the SAME five extra arguments C++ passes:
//! domain `[-HALF_MAX, HALF_MAX]`, `default = half(0)`, `posInf =
//! half::posInf()`, `negInf = half::negInf()`, `nan = half::qNan()`.
//!
//! Port target (pinned tag v3.4.15 — the repo's behavioural target; verified
//! via `git diff v3.3.11 v3.4.15 -- src/lib/OpenEXR/ImfLut.{cpp,h}` that both
//! files are byte-for-byte identical between the two tags, so nothing here
//! needed a behavioural update, only this citation):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfLut.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfLut.cpp>
//!
//! # Slice-based `apply`
//! C++'s `Slice`-taking overloads walk a raw `char* base` with `assert`s on
//! the sub-sampling alignment. This port's [`Slice`] has no raw pointer (see
//! `frame_buffer.rs`'s module doc): the caller's memory is a `&mut [u8]`
//! passed alongside the `Slice`, addressed via
//! `slice.base_offset + (x/x_sampling)*x_stride + (y/y_sampling)*y_stride`,
//! and every C++ `assert` becomes an [`ExrError::Arg`] (no-panic rule) rather
//! than being dropped. Half values inside a `FrameBuffer`-shaped buffer are
//! **native-endian** in memory (`misc.rs`'s convention — the wire's
//! little-endian XDR form only applies at the compressed-chunk boundary),
//! so this module reads/writes with `from_ne_bytes`/`to_ne_bytes`, matching
//! every other native-buffer accessor in the crate.

use imath_rs::half::HALF_MAX;
use imath_rs::{Box2i, Half, HalfFunction};

use crate::attr::PixelType;
use crate::error::{ExrError, Result};
use crate::frame_buffer::Slice;

use super::pixel::{Rgba, RgbaChannels};

/// Build a `halfFunction<half>` with the exact five extra arguments C++'s
/// `HalfLut`/`RgbaLut` constructors pass (`ImfLut.h:126-133`).
fn build_lut<F>(f: F) -> HalfFunction<Half>
where
    F: Fn(Half) -> Half,
{
    HalfFunction::new(f, Half::new(-HALF_MAX), Half::new(HALF_MAX), Half::new(0.0), Half::pos_inf(), Half::neg_inf(), Half::q_nan())
}

/// Byte offset of one `(x, y)` sample in a `Slice`'s native buffer, checked
/// against a negative result — the same formula `deep::sample_table`'s
/// `count_byte_offset` uses, specialised to a `Slice`'s sampling.
fn slice_byte_offset(slice: &Slice, x: i32, y: i32) -> Option<usize> {
    let off = slice.base_offset
        + (x / slice.x_sampling) as i64 * slice.x_stride as i64
        + (y / slice.y_sampling) as i64 * slice.y_stride as i64;
    usize::try_from(off).ok()
}

// ---------------------------------------------------------------------------
// HalfLut
// ---------------------------------------------------------------------------

/// Port of `Imf::HalfLut` — a lookup table for individual `half` channels.
pub struct HalfLut {
    lut: HalfFunction<Half>,
}

impl HalfLut {
    /// Port of the `template <class Function> HalfLut(Function f)` constructor.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(Half) -> Half,
    {
        Self { lut: build_lut(f) }
    }

    /// Port of `apply(half* data, int nData, int stride)`: applies the table
    /// in place to `data[0], data[stride], ..., data[(n_data-1)*stride]`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if an addressed index falls outside `data`
    /// (C++ trusts the caller's pointer arithmetic; this port bounds-checks
    /// instead of indexing unchecked).
    pub fn apply(&self, data: &mut [Half], n_data: usize, stride: usize) -> Result<()> {
        let mut idx = 0usize;
        for _ in 0..n_data {
            let slot = data
                .get_mut(idx)
                .ok_or_else(|| ExrError::Arg("HalfLut::apply: index out of range.".into()))?;
            *slot = self.lut.eval(*slot);
            idx += stride;
        }
        Ok(())
    }

    /// Port of `apply(const Slice&, const Box2i& dataWindow)`: applies the
    /// table in place to every sample of a `HALF` [`Slice`] inside
    /// `data_window`, honouring the slice's sub-sampling.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if `slice.type_` is not `HALF`, if `data_window` is
    /// not aligned to the slice's sub-sampling (the C++ `assert`s), or if an
    /// addressed byte range falls outside `buf`.
    pub fn apply_slice(&self, buf: &mut [u8], slice: &Slice, data_window: Box2i) -> Result<()> {
        if slice.type_ != PixelType::Half {
            return Err(ExrError::Arg("HalfLut::apply: slice is not HALF.".into()));
        }
        if data_window.min.x % slice.x_sampling != 0
            || data_window.min.y % slice.y_sampling != 0
            || (data_window.max.x - data_window.min.x + 1) % slice.x_sampling != 0
            || (data_window.max.y - data_window.min.y + 1) % slice.y_sampling != 0
        {
            return Err(ExrError::Arg("HalfLut::apply: data window is not aligned to the slice's sub-sampling.".into()));
        }

        for y in (data_window.min.y..=data_window.max.y).step_by(slice.y_sampling as usize) {
            for x in (data_window.min.x..=data_window.max.x).step_by(slice.x_sampling as usize) {
                let off = slice_byte_offset(slice, x, y)
                    .ok_or_else(|| ExrError::Arg("HalfLut::apply: negative byte offset.".into()))?;
                let bytes = buf
                    .get_mut(off..off + 2)
                    .ok_or_else(|| ExrError::Arg("HalfLut::apply: byte range out of range.".into()))?;
                let v = Half::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]]));
                bytes.copy_from_slice(&self.lut.eval(v).bits().to_ne_bytes());
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RgbaLut
// ---------------------------------------------------------------------------

/// Port of `Imf::RgbaLut` — a lookup table applied to selected channels of
/// combined RGBA data.
pub struct RgbaLut {
    lut: HalfFunction<Half>,
    chn: RgbaChannels,
}

impl RgbaLut {
    /// Port of `template <class Function> RgbaLut(Function f, RgbaChannels
    /// chn = WRITE_RGB)`.
    pub fn new<F>(f: F, chn: RgbaChannels) -> Self
    where
        F: Fn(Half) -> Half,
    {
        Self { lut: build_lut(f), chn }
    }

    /// Port of `apply(Rgba* data, int nData, int stride)`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if an addressed index falls outside `data`.
    pub fn apply(&self, data: &mut [Rgba], n_data: usize, stride: usize) -> Result<()> {
        let mut idx = 0usize;
        for _ in 0..n_data {
            let px = data
                .get_mut(idx)
                .ok_or_else(|| ExrError::Arg("RgbaLut::apply: index out of range.".into()))?;
            self.apply_pixel(px);
            idx += stride;
        }
        Ok(())
    }

    /// Port of `apply(Rgba* base, int xStride, int yStride, const Box2i&
    /// dataWindow)` — the frame-buffer-window overload. `base` is addressed
    /// the same way `RgbaOutputFile`/`RgbaInputFile` address their caller
    /// buffers in this crate: `base[(x - 0) * x_stride + (y - 0) * y_stride]`
    /// using signed strides in units of [`Rgba`], with `x`/`y` taken directly
    /// from `data_window` (matching C++'s `base + dataWindow.min.y * yStride`
    /// / `+ dataWindow.min.x * xStride` addressing from an ORIGIN-relative
    /// pointer).
    ///
    /// # Errors
    /// [`ExrError::Arg`] if an addressed index is negative or falls outside
    /// `base`.
    pub fn apply_window(&self, base: &mut [Rgba], x_stride: isize, y_stride: isize, data_window: Box2i) -> Result<()> {
        for y in data_window.min.y..=data_window.max.y {
            for x in data_window.min.x..=data_window.max.x {
                let idx = x as isize * x_stride + y as isize * y_stride;
                let idx = usize::try_from(idx)
                    .map_err(|_| ExrError::Arg("RgbaLut::apply: negative index.".into()))?;
                let px = base
                    .get_mut(idx)
                    .ok_or_else(|| ExrError::Arg("RgbaLut::apply: index out of range.".into()))?;
                self.apply_pixel(px);
            }
        }
        Ok(())
    }

    fn apply_pixel(&self, px: &mut Rgba) {
        if self.chn.intersects(RgbaChannels::WRITE_R) {
            px.r = self.lut.eval(px.r);
        }
        if self.chn.intersects(RgbaChannels::WRITE_G) {
            px.g = self.lut.eval(px.g);
        }
        if self.chn.intersects(RgbaChannels::WRITE_B) {
            px.b = self.lut.eval(px.b);
        }
        if self.chn.intersects(RgbaChannels::WRITE_A) {
            px.a = self.lut.eval(px.a);
        }
    }
}

// ---------------------------------------------------------------------------
// round12log / roundNBit
// ---------------------------------------------------------------------------

/// C++'s `int(double)` cast, on the MSVC x64/SSE2 target this crate pins for
/// float-path parity (the DWA codec follows the same "match the
/// platform's actual instruction, not the C standard's UB" rule):
/// `cvttsd2si` returns the "integer indefinite" `0x8000_0000` (`i32::MIN`)
/// for a NaN or an out-of-`i32`-range double, rather than saturating toward
/// the input's sign the way Rust's `as i32` does. `round12log`'s `int(...)`
/// cast only ever hits this path for a `+inf` input (any `NaN`/`-inf` input
/// is already routed to `x <= 0 -> 0` or produces a small in-range value by
/// the time it reaches the cast) — without this, `+inf` clamped to `4095`
/// (Rust's saturating cast) instead of C++'s `1` (`i32::MIN < 1`), a real
/// divergence caught by the oracle (`tests/lut_oracle.rs`, exhaustive over
/// all 65,536 half bit patterns).
fn cvttsd2si(x: f64) -> i32 {
    if x.is_finite() && x >= i32::MIN as f64 && x < 2147483648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

/// Port of `Imf::round12log`: 12-bit log rounding to 4096 steps (20 stops at
/// 200 steps/stop), with `0` mapping exactly to `0`.
pub fn round12_log(x: Half) -> Half {
    let middleval = 2.0f64.powf(-2.5);
    if x.to_f32() as f64 <= 0.0 {
        return Half::new(0.0);
    }
    let int12log = cvttsd2si(2000.5 + 200.0 * (x.to_f32() as f64 / middleval).ln() / 2.0f64.ln())
        // C++'s two sequential `if` clamps (`ImfLut.cpp:118-120`); `1..=4095`
        // is a fixed, always-valid range, so `clamp` (which panics on
        // `max < min`) is safe and exactly equivalent here.
        .clamp(1, 4095);
    Half::new((middleval * 2.0f64.powf((int12log - 2000) as f64 / 200.0)) as f32)
}

/// Port of `struct roundNBit`: rounds a `half` to `n`-bit precision
/// (`n` should be `0..=10`) via [`Half::round`].
#[derive(Clone, Copy, Debug)]
pub struct RoundNBit {
    pub n: u32,
}

impl RoundNBit {
    pub fn new(n: u32) -> Self {
        Self { n }
    }

    /// Port of `half operator()(half x)`.
    pub fn apply(&self, x: Half) -> Half {
        x.round(self.n)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use imath_rs::V2i;

    fn half_slice(base_offset: i64, x_stride: usize, y_stride: usize) -> Slice {
        Slice::new(PixelType::Half, base_offset, x_stride, y_stride, 1, 1, 0.0, false, false).unwrap()
    }

    #[test]
    fn half_lut_apply_doubles_via_array() {
        let lut = HalfLut::new(|x: Half| Half::new(x.to_f32() * 2.0));
        let mut data = vec![Half::new(1.0), Half::new(2.0), Half::new(3.0)];
        lut.apply(&mut data, 3, 1).unwrap();
        assert_eq!(data.iter().map(|h| h.to_f32()).collect::<Vec<_>>(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn half_lut_apply_respects_stride() {
        let lut = HalfLut::new(|x: Half| Half::new(x.to_f32() * 2.0));
        let mut data = vec![Half::new(1.0), Half::new(99.0), Half::new(2.0), Half::new(99.0)];
        lut.apply(&mut data, 2, 2).unwrap();
        assert_eq!(data[0].to_f32(), 2.0);
        assert_eq!(data[1].to_f32(), 99.0); // untouched (stride skipped it)
        assert_eq!(data[2].to_f32(), 4.0);
    }

    #[test]
    fn half_lut_apply_slice_walks_the_data_window() {
        let lut = HalfLut::new(|x: Half| Half::new(x.to_f32() + 1.0));
        let (w, h) = (4usize, 3usize);
        let mut buf = vec![0u8; w * h * 2];
        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 2;
                buf[off..off + 2].copy_from_slice(&Half::new((x + y) as f32).bits().to_ne_bytes());
            }
        }
        let slice = half_slice(0, 2, w * 2);
        let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: (w - 1) as i32, y: (h - 1) as i32 } };
        lut.apply_slice(&mut buf, &slice, dw).unwrap();
        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 2;
                let v = Half::from_bits(u16::from_ne_bytes([buf[off], buf[off + 1]]));
                assert_eq!(v.to_f32(), (x + y) as f32 + 1.0, "({x},{y})");
            }
        }
    }

    #[test]
    fn half_lut_apply_slice_rejects_non_half() {
        let lut = HalfLut::new(|x: Half| x);
        let mut buf = vec![0u8; 8];
        let slice = Slice::new(PixelType::Float, 0, 4, 4, 1, 1, 0.0, false, false).unwrap();
        let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 0, y: 0 } };
        assert!(matches!(lut.apply_slice(&mut buf, &slice, dw), Err(ExrError::Arg(_))));
    }

    #[test]
    fn half_lut_apply_slice_rejects_misaligned_window() {
        let lut = HalfLut::new(|x: Half| x);
        let mut buf = vec![0u8; 16];
        let slice = Slice::new(PixelType::Half, 0, 2, 8, 2, 1, 0.0, false, false).unwrap(); // xSampling = 2
        let dw = Box2i { min: V2i { x: 1, y: 0 }, max: V2i { x: 3, y: 0 } }; // min.x not aligned to xSampling
        assert!(matches!(lut.apply_slice(&mut buf, &slice, dw), Err(ExrError::Arg(_))));
    }

    #[test]
    fn rgba_lut_apply_only_touches_selected_channels() {
        let lut = RgbaLut::new(|x: Half| Half::new(x.to_f32() * 2.0), RgbaChannels::WRITE_R | RgbaChannels::WRITE_A);
        let mut data = vec![Rgba::new(Half::new(1.0), Half::new(1.0), Half::new(1.0), Half::new(1.0))];
        lut.apply(&mut data, 1, 1).unwrap();
        assert_eq!(data[0].r.to_f32(), 2.0);
        assert_eq!(data[0].g.to_f32(), 1.0); // untouched
        assert_eq!(data[0].b.to_f32(), 1.0); // untouched
        assert_eq!(data[0].a.to_f32(), 2.0);
    }

    #[test]
    fn rgba_lut_apply_window_walks_the_data_window() {
        let lut = RgbaLut::new(|x: Half| Half::new(x.to_f32() + 10.0), RgbaChannels::WRITE_RGB);
        let (w, h) = (3usize, 2usize);
        let mut base = vec![Rgba::default(); w * h];
        for (i, px) in base.iter_mut().enumerate() {
            *px = Rgba::new(Half::new(i as f32), Half::new(i as f32), Half::new(i as f32), Half::new(0.0));
        }
        let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: (w - 1) as i32, y: (h - 1) as i32 } };
        lut.apply_window(&mut base, 1, w as isize, dw).unwrap();
        for (i, px) in base.iter().enumerate() {
            assert_eq!(px.r.to_f32(), i as f32 + 10.0);
            assert_eq!(px.a.to_f32(), 0.0); // WRITE_RGB does not touch alpha
        }
    }

    #[test]
    fn round12_log_maps_zero_to_zero_and_is_monotonic() {
        assert_eq!(round12_log(Half::new(0.0)).to_f32(), 0.0);
        let a = round12_log(Half::new(0.1));
        let b = round12_log(Half::new(1.0));
        let c = round12_log(Half::new(10.0));
        assert!(a.to_f32() < b.to_f32());
        assert!(b.to_f32() < c.to_f32());
    }

    #[test]
    fn round_n_bit_matches_half_round() {
        let r = RoundNBit::new(4);
        let x = Half::new(1.23456);
        assert_eq!(r.apply(x).to_f32(), x.round(4).to_f32());
    }
}
