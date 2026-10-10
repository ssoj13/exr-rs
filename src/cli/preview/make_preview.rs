//! `makePreview.cpp`: add a preview image to an OpenEXR file (`openexr/src/bin/exrmakepreview`,
//! v3.4.15).
//!
//! The preview is a nearest-neighbour thumbnail of the image read as RGBA, with the exposure / knee /
//! gamma curve of `exrdisplay`'s image view (defog, knee low and knee high fixed at 0, 0 and 5). The
//! arithmetic keeps the C++ types and operation order (`float` for the curve, `double` for the
//! sampling grid), so the preview bytes are identical to `exrmakepreview`'s. The file is then copied
//! with its compressed chunks untouched, the preview inserted into the header.
//!
//! Deliberate differences from the C++:
//! * a preview height that does not fit an `int` is `1` like the x86 conversion the C++ relies on
//!   (undefined there), and a preview too large to allocate is an error, not a crash;
//! * the source file is read once per pass as in the C++, but only after the output name is
//!   checked against the input's (`main.cpp:164` compares strings).

use exr_core::header::is_deep_data;
use exr_core::io::StdIfStream;
use exr_core::rgba::{Rgba, RgbaInputFile};
use exr_core::scanline::{InputFile, OutputFile};
use exr_core::tiled::TiledOutputFile;
use exr_core::attr::{PreviewImage, PreviewRgba};
use exr_core::{ExrError, Result, Unsupported};
use imath_rs::Half;

/// Imath `clamp`: `a < l ? l : (a > h ? h : a)` (a NaN passes through).
fn clamp(a: f32, l: f32, h: f32) -> f32 {
    if a < l {
        l
    } else if a > h {
        h
    } else {
        a
    }
}

/// `knee`: the soft shoulder of the curve.
fn knee(x: f32, f: f32) -> f32 {
    (x * f + 1.0).ln() / f
}

/// `gamma`: one half sample to an 8-bit display value at exposure multiplier `m`.
fn gamma(h: Half, m: f32) -> u8 {
    // `std::max (0.f, h * m)`: `(a < b) ? b : a`, so a NaN becomes 0.
    let hm = h.to_f32() * m;
    let mut x = if 0.0 < hm { hm } else { 0.0 };
    if x > 1.0 {
        x = 1.0 + knee(x - 1.0, 0.184874);
    }
    clamp(x.powf(0.4545) * 84.66, 0.0, 255.0) as u8
}

/// The 8-bit alpha of a half sample.
fn alpha(a: Half) -> u8 {
    (clamp(a.to_f32() * 255.0, 0.0, 255.0) + 0.5) as i32 as u8
}

/// A float to `int` as the x86 conversion does it: anything out of range (or NaN) is `INT_MIN`.
fn to_int(v: f32) -> i32 {
    if (-2147483648.0..2147483648.0).contains(&v) { v as i32 } else { i32::MIN }
}

/// `generatePreview`: read `path` as RGBA and sample a `width`-wide preview; returns its height and
/// the pixels, row by row.
fn generate_preview(path: &str, exposure: f32, width: i32) -> Result<(i32, Vec<PreviewRgba>)> {
    let mut input = RgbaInputFile::open(path)?;
    let dw = input.data_window()?;
    let a = input.pixel_aspect_ratio()?;
    let extent = |lo: i32, hi: i32| {
        (hi as i64 - lo as i64 + 1)
            .try_into()
            .ok()
            .filter(|&n: &i32| n > 0)
            .ok_or_else(|| ExrError::Input(format!("invalid data window [{lo}, {hi}]")))
    };
    let (w, h) = (extent(dw.min.x, dw.max.x)?, extent(dw.min.y, dw.max.y)?);
    let height = to_int(h as f32 / (w as f32 * a) * width as f32 + 0.5).max(1);
    // Reject a preview that cannot be written before allocating or decoding the source.
    let n = PreviewImage::checked_pixel_count(width as u32, height as u32)?;
    let count = exr_core::checked_arithmetic::ui_mult(w as usize, h as usize)?;
    let count = exr_core::checked_arithmetic::check_array_size(count, std::mem::size_of::<Rgba>())?;
    let mut pixels: Vec<Rgba> = Vec::new();
    pixels.try_reserve_exact(count).map_err(|_| ExrError::Arg(format!("Cannot allocate {count} pixels for {path}")))?;
    pixels.resize(count, Rgba::default());
    input.read_pixels(dw.min.y, dw.max.y, &mut pixels, 1, w as isize)?;

    // Make a preview image.
    let mut preview: Vec<PreviewRgba> = Vec::new();
    preview.try_reserve_exact(n).map_err(|_| ExrError::Arg(format!("Cannot allocate a {width} x {height} preview")))?;

    let fx = if width > 1 { (w - 1) as f64 / (width - 1) as f64 } else { 1.0 };
    let fy = if height > 1 { (h - 1) as f64 / (height - 1) as f64 } else { 1.0 };
    // `pow (2.f, ..)`: the opaque base keeps the optimizer from rewriting it as `exp2`, which
    // rounds differently.
    let m = std::hint::black_box(2.0f32).powf(clamp(exposure + 2.47393, -20.0, 20.0));
    for y in 0..height {
        let row = (y as f64 * fy + 0.5) as i32 as usize;
        for x in 0..width {
            let col = (x as f64 * fx + 0.5) as i32 as usize;
            let p = pixels
                .get(row * w as usize + col)
                .ok_or_else(|| ExrError::Logic("a preview pixel maps outside the image".into()))?;
            preview.push(PreviewRgba::new(gamma(p.r, m), gamma(p.g, m), gamma(p.b, m), alpha(p.a)));
        }
    }
    Ok((height, preview))
}

/// `makePreview`: copy `input` to `output` with a `width`-wide preview in the header.
pub fn make_preview(input: &str, output: &str, width: i32, exposure: f32, verbose: bool) -> Result<()> {
    if verbose {
        outln!("generating preview image");
    }
    // A deep image has no RGBA picture to shrink (the C++ tool crashes on one). Deep-tiled files do not
    // even open as an `InputFile`; deep-scanline ones are flattened, so their type says it.
    let deep = || ExrError::Arg("Cannot make a preview of deep data".into());
    match InputFile::open(input) {
        Ok(probe) if probe.header().has_type() && is_deep_data(probe.header().type_()?) => return Err(deep()),
        Err(ExrError::Unsupported(Unsupported::Deep)) => return Err(deep()),
        Err(e) => return Err(e),
        Ok(_) => {}
    }
    let (height, pixels) = generate_preview(input, exposure, width)?;

    let mut in_stream = StdIfStream::open(input)?;
    let mut file = InputFile::new(&mut in_stream)?;
    let mut header = file.header().clone();
    header.set_preview_image(PreviewImage::new(width as u32, height as u32, Some(pixels))?)?;

    if verbose {
        outln!("copying {input} to {output}");
    }
    exr_core::io::write_file(output, |stream| -> exr_core::Result<_> {
        if header.has_tile_description() {
            TiledOutputFile::new(stream, &header)?.copy_pixels(&mut file)
        } else {
            OutputFile::new(stream, &header)?.copy_pixels(&mut file)
        }
    })?;
    if verbose {
        outln!("done.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(v: f32) -> Half {
        Half::from_f32(v)
    }

    #[test]
    fn oversized_preview_is_rejected_before_reading_pixel_chunks() {
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(self.0.join("bad-chunk.exr"));
                let _ = std::fs::remove_file(self.0.join("existing-output.exr"));
                let _ = std::fs::remove_dir(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = Cleanup(std::env::temp_dir().join(format!("exr-rs-preview-wire-{}-{nonce}", std::process::id())));
        std::fs::create_dir(&dir.0).unwrap();
        let input = dir.0.join("bad-chunk.exr");
        exr_core::write_rgba_file(&input, 64, 48, |_, _| (0.5, 0.5, 0.5, 1.0)).unwrap();
        let mut bytes = std::fs::read(&input).unwrap();
        let key = b"pixelAspectRatio\0float\0";
        let value = bytes.windows(key.len()).position(|b| b == key).unwrap() + key.len() + 4;
        bytes[value..value + 4].copy_from_slice(&0.001f32.to_le_bytes());

        // Walk the generated single-part header to find the first chunk offset.
        let mut at = 8;
        while bytes[at] != 0 {
            for _ in 0..2 {
                while bytes[at] != 0 { at += 1; }
                at += 1;
            }
            let size = i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            at += 4 + size;
        }
        at += 1;
        let chunk = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
        // Header and offset table remain valid; decoding rejects the negative data size.
        bytes[chunk + 4..chunk + 8].copy_from_slice(&(-1i32).to_le_bytes());
        std::fs::write(&input, bytes).unwrap();
        let input = input.to_str().unwrap();
        assert!(matches!(generate_preview(input, 0.0, 1), Err(ExrError::Input(_))),
            "the control must actually reach the invalid chunk");

        match generate_preview(input, 0.0, 999).unwrap_err() {
            ExrError::Overflow(message) => assert_eq!(
                message, "value 2994003008 exceeds the i32 wire-length limit",
            ),
            other => panic!("expected wire Overflow before chunk decoding, got {other:?}"),
        }
        let output = dir.0.join("existing-output.exr");
        std::fs::write(&output, b"unchanged").unwrap();
        assert!(matches!(make_preview(input, output.to_str().unwrap(), 999, 0.0, false), Err(ExrError::Overflow(_))));
        assert_eq!(std::fs::read(output).unwrap(), b"unchanged");
    }

    /// The display curve at exposure multiplier 1, against values worked out by hand (Python, double
    /// precision): below the knee (1.0 is not `> 1`), above it, clamped, and the odd inputs.
    #[test]
    fn gamma_follows_the_curve() {
        assert_eq!(gamma(h(0.0), 1.0), 0);
        assert_eq!(gamma(h(0.0001), 1.0), 1, "1.287");
        assert_eq!(gamma(h(0.001), 1.0), 3, "3.666");
        assert_eq!(gamma(h(0.5), 1.0), 61, "61.78");
        assert_eq!(gamma(h(1.0), 1.0), 84, "84.66: exactly 1 takes no knee");
        assert_eq!(gamma(h(2.0), 1.0), 113, "knee: 113.81");
        assert_eq!(gamma(h(100.0), 1.0), 255);
        // The multiplier scales first: 0.25 * 4 is 1.
        assert_eq!(gamma(h(0.25), 4.0), 84);
        // Negative, NaN and infinity: black, black (`max (0, NaN)` is 0), white.
        assert_eq!(gamma(h(-3.0), 1.0), 0);
        assert_eq!(gamma(h(f32::NAN), 1.0), 0);
        assert_eq!(gamma(h(f32::INFINITY), 1.0), 255);
    }

    /// `knee (x, f) = log (x f + 1) / f`: knee (0, f) = 0, and for small x it is x.
    #[test]
    fn the_knee_is_a_soft_shoulder() {
        assert_eq!(knee(0.0, 0.184874), 0.0);
        assert!((knee(1e-4, 0.184874) - 1e-4).abs() < 1e-5);
        assert!((knee(1.0, 0.184874) - 0.91755).abs() < 1e-4, "ln (1.184874) / 0.184874");
        assert!(knee(1e6, 0.184874) < 80.0, "log growth");
    }

    /// Alpha is clamped to [0, 1] and rounded to nearest (`+ .5`, truncated), NaN is 0.
    #[test]
    fn alpha_rounds_and_clamps() {
        for (a, want) in [(0.0, 0), (0.5, 128), (1.0, 255), (2.0, 255), (-1.0, 0), (0.001, 0), (0.004, 1), (f32::NAN, 0)] {
            assert_eq!(alpha(h(a)), want, "alpha {a}");
        }
    }

    /// A float that does not fit an `int` is `INT_MIN` (x86), so the preview height falls back to 1.
    #[test]
    fn out_of_range_heights_are_int_min() {
        assert_eq!(to_int(12.9), 12);
        assert_eq!(to_int(-5.9), -5);
        for v in [f32::NAN, 1e10, -1e10, f32::INFINITY, 2147483648.0] {
            assert_eq!(to_int(v), i32::MIN, "{v}");
        }
        assert_eq!(to_int(-2147483648.0), i32::MIN);
    }
}
