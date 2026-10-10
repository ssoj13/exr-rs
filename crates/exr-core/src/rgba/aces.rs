//! Port of OpenEXR's `ImfAcesFile.h`/`ImfAcesFile.cpp`.
//!
//! `AcesOutputFile`/`AcesInputFile` are thin wrappers around
//! [`RgbaOutputFile`]/[`RgbaInputFile`] that read/write RGB data in the ACES
//! (Academy Color Encoding Specification) color space:
//! - **Write:** forces the file's chromaticities to [`aces_chromaticities`]
//!   (the caller is expected to already supply ACES-space RGB) and pins the
//!   luminance/chroma rounding to `(7, 6)` — the values ACES files use,
//!   distinct from `RgbaOutputFile`'s own default `(7, 5)`.
//! - **Read:** if the file's own chromaticities differ from ACES, transforms
//!   every decoded pixel from the file's RGB space into ACES RGB via a
//!   Bradford-adapted 3x3 (stored as an affine `M44f`) matrix, computed once
//!   at open time from the header's `chromaticities`/`adoptedNeutral`.
//!
//! Port target (pinned tag v3.4.15 — the repo's behavioural target;
//! verified via `git diff v3.3.11 v3.4.15 --
//! src/lib/OpenEXR/ImfAcesFile.{cpp,h}` that `ImfAcesFile.cpp` is BYTE-FOR-BYTE
//! identical between the two tags and `ImfAcesFile.h`'s only change is a
//! UTF-8-path doc comment — so nothing here needed a behavioural update, only
//! this citation):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfAcesFile.cpp>
//!
//! This module follows the same "pixel data passed per call" convention as
//! [`super::output`]/[`super::input`]: neither wrapper
//! stores the caller's frame-buffer pointer between calls — `write_pixels`/
//! `read_pixels` take the buffer directly, unlike C++'s `setFrameBuffer` +
//! stateful `writePixels`/`readPixels`. This sidesteps needing unsafe raw
//! pointers to hold a borrow across calls and is exactly how
//! `RgbaOutputFile`/`RgbaInputFile` in this crate already work.
//!
//! Only the single stream-based constructor is ported (matching
//! `RgbaOutputFile::new`/`RgbaInputFile::new` in this crate) — C++'s other
//! three `AcesOutputFile` overloads (path name, explicit display/data
//! window, explicit width/height) are convenience constructors that all
//! bottom out in the same `Header` + chromaticities + rounding setup; a
//! caller here builds the `Header` itself and passes it in, same as for
//! `RgbaOutputFile`/`RgbaInputFile`.

use imath_rs::{M44f, V2f, V3f};

use crate::attr::{Chromaticities, Compression, LineOrder};
use crate::error::{ExrError, Result};
use crate::header::Header;
use crate::io::stream::{IStream, OStream};
use crate::sse::ss;

use super::input::RgbaInputFile;
use super::output::RgbaOutputFile;
use super::pixel::{Rgba, RgbaChannels};

/// Port of `Imf::acesChromaticities()` — the fixed ACES primaries/white
/// point every ACES file is defined against.
pub fn aces_chromaticities() -> Chromaticities {
    Chromaticities {
        red: [0.73470, 0.26530],
        green: [0.00000, 1.00000],
        blue: [0.00010, -0.07700],
        white: [0.32168, 0.33767],
    }
}

/// Port of the anonymous-namespace `checkCompression` — only these three
/// codecs are legal in an ACES file.
fn check_compression(compression: Compression) -> Result<()> {
    match compression {
        Compression::NoCompression | Compression::Piz | Compression::B44a => Ok(()),
        _ => Err(ExrError::Arg("Invalid compression type for ACES file.".into())),
    }
}

// ---------------------------------------------------------------------------
// AcesOutputFile
// ---------------------------------------------------------------------------

/// Port of `Imf::AcesOutputFile`.
pub struct AcesOutputFile<'a> {
    rgba: RgbaOutputFile<'a>,
}

impl<'a> AcesOutputFile<'a> {
    /// Port of `AcesOutputFile(OStream&, const Header&, RgbaChannels, int)`.
    /// Rejects a non-ACES-legal compression up front (`ArgExc` in C++),
    /// stamps the header with ACES chromaticities/adopted-neutral, and pins
    /// the YC rounding to `(7, 6)`.
    pub fn new(stream: &'a mut dyn OStream, header: &Header, rgba_channels: RgbaChannels) -> Result<Self> {
        check_compression(header.compression()?)?;

        let mut new_header = header.clone();
        new_header.add_chromaticities(aces_chromaticities())?;
        new_header.add_adopted_neutral(aces_chromaticities().white_v2f())?;

        let mut rgba = RgbaOutputFile::new(stream, &new_header, rgba_channels)?;
        rgba.set_yc_rounding(7, 6);
        Ok(Self { rgba })
    }

    /// Port of `setFrameBuffer` (minus the stored pointer — see module doc).
    pub fn set_frame_buffer(&mut self, x_stride: isize, y_stride: isize) -> Result<()> {
        self.rgba.set_frame_buffer(x_stride, y_stride)
    }

    /// Port of `writePixels` — straight passthrough, no color conversion
    /// (the caller is expected to already supply ACES-space RGB; C++ does
    /// the same, it never transforms on write).
    pub fn write_pixels(&mut self, pixels: &[Rgba], num_scan_lines: i32) -> Result<()> {
        self.rgba.write_pixels(pixels, num_scan_lines)
    }

    pub fn header(&self) -> &Header {
        self.rgba.header()
    }

    pub fn frame_buffer(&self) -> Option<(isize, isize)> {
        self.rgba.frame_buffer()
    }

    pub fn data_window(&self) -> imath_rs::Box2i {
        self.rgba.data_window()
    }

    pub fn display_window(&self) -> Result<imath_rs::Box2i> {
        self.rgba.display_window()
    }

    pub fn pixel_aspect_ratio(&self) -> Result<f32> {
        self.rgba.pixel_aspect_ratio()
    }

    pub fn screen_window_center(&self) -> Result<V2f> {
        self.rgba.screen_window_center()
    }

    pub fn screen_window_width(&self) -> Result<f32> {
        self.rgba.screen_window_width()
    }

    pub fn line_order(&self) -> LineOrder {
        self.rgba.line_order()
    }

    pub fn compression(&self) -> Result<Compression> {
        self.rgba.compression()
    }

    pub fn channels(&self) -> RgbaChannels {
        self.rgba.channels()
    }

    pub fn current_scan_line(&self) -> i32 {
        self.rgba.current_scan_line()
    }

    /// Replace the header's preview-image pixels and rewrite them in place
    /// on disk — port of `AcesOutputFile::updatePreviewImage`
    /// (`ImfAcesFile.cpp:265-268`, a one-line forward to
    /// `RgbaOutputFile::updatePreviewImage`). C++ has no `breakScanLine` on
    /// `AcesOutputFile` (`ImfAcesFile.h` declares only `updatePreviewImage`
    /// among the write-side test/preview helpers) — verified against the
    /// header, not assumed.
    pub fn update_preview_image(&mut self, new_pixels: &[crate::attr::PreviewRgba]) -> Result<()> {
        self.rgba.update_preview_image(new_pixels)
    }
}

// ---------------------------------------------------------------------------
// AcesInputFile
// ---------------------------------------------------------------------------

/// Port of `Imf::AcesInputFile`.
pub struct AcesInputFile<'a> {
    rgba: RgbaInputFile<'a>,
    /// `true` if the file's own RGB space differs from ACES and every
    /// decoded pixel must be transformed — port of `Data::mustConvertColor`.
    must_convert_color: bool,
    /// Port of `Data::fileToAces` — the file-RGB -> ACES-RGB matrix, valid
    /// only when `must_convert_color` is set.
    file_to_aces: M44f,
}

impl<'a> AcesInputFile<'a> {
    /// Port of `AcesInputFile(IStream&, int)`.
    pub fn new(stream: &'a mut dyn IStream) -> Result<Self> {
        let rgba = RgbaInputFile::new(stream)?;
        let (must_convert_color, file_to_aces) = init_color_conversion(rgba.header()?)?;
        Ok(Self { rgba, must_convert_color, file_to_aces })
    }

    /// Port of `setFrameBuffer` (minus the stored pointer — see module doc);
    /// this port has no persistent frame buffer to record, so it is a no-op
    /// kept only for API symmetry with `AcesOutputFile`/C++. The caller
    /// passes its buffer directly to [`read_pixels`](Self::read_pixels).
    pub fn set_frame_buffer(&mut self, _x_stride: isize, _y_stride: isize) {}

    /// Port of `readPixels(int, int)`: reads via the inner `RgbaInputFile`,
    /// then — if the file's RGB space is not already ACES — transforms every
    /// pixel in `[min_x, max_x] x [min(y1,y2), max(y1,y2)]` (the header's
    /// data window column range, C++'s `Data::minX`/`maxX`) from file-RGB
    /// into ACES-RGB in place.
    pub fn read_pixels(&mut self, y1: i32, y2: i32, base: &mut [Rgba], x_stride: isize, y_stride: isize) -> Result<()> {
        self.rgba.read_pixels(y1, y2, base, x_stride, y_stride)?;
        if !self.must_convert_color {
            return Ok(());
        }

        let dw = self.rgba.data_window()?;
        let (lo, hi) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        for y in lo..=hi {
            for x in dw.min.x..=dw.max.x {
                let idx = (x - dw.min.x) as isize * x_stride + (y - dw.min.y) as isize * y_stride;
                let idx = usize::try_from(idx)
                    .map_err(|_| ExrError::Arg("AcesInputFile::readPixels: destination index is negative.".into()))?;
                let px = base
                    .get_mut(idx)
                    .ok_or_else(|| ExrError::Arg("AcesInputFile::readPixels: destination index is out of range.".into()))?;
                let src = V3f { x: px.r.to_f32(), y: px.g.to_f32(), z: px.b.to_f32() };
                let dst = mul_v3_m44(&src, &self.file_to_aces);
                px.r = imath_rs::Half::from_f32(dst.x);
                px.g = imath_rs::Half::from_f32(dst.y);
                px.b = imath_rs::Half::from_f32(dst.z);
            }
        }
        Ok(())
    }

    /// Port of `readPixels(int)`.
    pub fn read_pixel_line(&mut self, scan_line: i32, base: &mut [Rgba], x_stride: isize, y_stride: isize) -> Result<()> {
        self.read_pixels(scan_line, scan_line, base, x_stride, y_stride)
    }

    pub fn header(&self) -> Result<&Header> {
        self.rgba.header()
    }

    pub fn display_window(&self) -> Result<imath_rs::Box2i> {
        self.rgba.display_window()
    }

    pub fn data_window(&self) -> Result<imath_rs::Box2i> {
        self.rgba.data_window()
    }

    pub fn pixel_aspect_ratio(&self) -> Result<f32> {
        self.rgba.pixel_aspect_ratio()
    }

    pub fn line_order(&self) -> Result<LineOrder> {
        self.rgba.line_order()
    }

    pub fn compression(&self) -> Result<Compression> {
        self.rgba.compression()
    }

    pub fn channels(&self) -> Result<RgbaChannels> {
        self.rgba.channels()
    }

    pub fn is_complete(&self) -> Result<bool> {
        self.rgba.is_complete()
    }

    pub fn version(&self) -> i32 {
        self.rgba.version()
    }
}

impl Chromaticities {
    /// Convenience: `white` as a `V2f` (the field is stored as `[f32; 2]`
    /// for wire layout, but `addAdoptedNeutral` in C++ takes a `V2f`).
    fn white_v2f(&self) -> V2f {
        V2f { x: self.white[0], y: self.white[1] }
    }
}

/// `V3f * M44f` (`ImathMatrix.h`'s `operator*(Vec3, Matrix44)`, what `AcesInputFile::readPixels`
/// calls) in the operation order of the MSVC x64 build, which only shows when an input is a NaN:
/// every term below is computed as the plain `x*m[0][c] + y*m[1][c] + z*m[2][c] + m[3][c]`, but
/// which NaN's payload survives depends on the operand that sits in the SSE destination register.
/// Oracle run (`exr2aces.exe` on a file whose pixels are the seven non-empty subsets of
/// `{R, G, B}` set to NaNs with payloads `0x7e01`, `0x7e02`, `0x7e03`): the output R takes G's
/// payload from `R+G` (and keeps G's for `R+G+B`), while the output G and B take R's; `R+B`
/// gives R and `G+B` gives G everywhere. That is `add(p1, p0)` for column 0 and `add(p0, p1)` for
/// columns 1 and 2, then the running sum as destination for `+ z*m[2][c]` and `+ m[3][c]`, and
/// the numerator as destination of the final divide. Without a NaN every order gives the same bits.
fn mul_v3_m44(v: &V3f, m: &M44f) -> V3f {
    let col = |c: usize| {
        let p = [ss::mul(v.x, m.x[0][c]), ss::mul(v.y, m.x[1][c]), ss::mul(v.z, m.x[2][c])];
        let s01 = if c == 0 { ss::add(p[1], p[0]) } else { ss::add(p[0], p[1]) };
        ss::add(ss::add(s01, p[2]), m.x[3][c])
    };
    let w = ss::add(ss::add(ss::add(ss::mul(v.x, m.x[0][3]), ss::mul(v.y, m.x[1][3])), ss::mul(v.z, m.x[2][3])), m.x[3][3]);
    V3f { x: ss::div(col(0), w), y: ss::div(col(1), w), z: ss::div(col(2), w) }
}

/// Port of `AcesInputFile::Data::initColorConversion`. Returns
/// `(must_convert_color, file_to_aces)`.
fn init_color_conversion(header: &Header) -> Result<(bool, M44f)> {
    let mut file_chr = if header.has_chromaticities() { *header.chromaticities()? } else { Chromaticities::default() };

    let mut file_neutral = file_chr.white_v2f();
    if header.has_adopted_neutral() {
        let n = *header.adopted_neutral()?;
        file_neutral = n;
        file_chr.white = [n.x, n.y]; // for rgb_to_xyz() purposes.
    }

    let aces_chr = aces_chromaticities();
    let aces_neutral = aces_chr.white_v2f();

    if file_chr.red == aces_chr.red
        && file_chr.green == aces_chr.green
        && file_chr.blue == aces_chr.blue
        && file_chr.white == aces_chr.white
    {
        // The file already contains ACES data, color conversion is not
        // necessary.
        return Ok((false, M44f::default()));
    }

    // Bradford cone-primary matrix and its inverse. Literals kept exactly as
    // C++ writes them (`internal_dwa_compressor.h`-style verbatim-constant
    // policy) rather than clippy's preferred shortest form.
    #[rustfmt::skip]
    #[allow(clippy::excessive_precision)]
    let bradford_cpm = M44f::new(
        0.895100, -0.750200, 0.038900, 0.000000,
        0.266400,  1.713500, -0.068500, 0.000000,
        -0.161400, 0.036700, 1.029600, 0.000000,
        0.000000,  0.000000, 0.000000, 1.000000,
    );
    #[rustfmt::skip]
    let inverse_bradford_cpm = M44f::new(
        0.986993,  0.432305, -0.008529, 0.000000,
        -0.147054, 0.518360,  0.040043, 0.000000,
        0.159963,  0.049291,  0.968487, 0.000000,
        0.000000,  0.000000,  0.000000, 1.000000,
    );

    // Convert the white points of the two RGB spaces to XYZ.
    let (fx, fy) = (file_neutral.x, file_neutral.y);
    let file_neutral_xyz = V3f { x: fx / fy, y: 1.0, z: (1.0 - fx - fy) / fy };

    let (ax, ay) = (aces_neutral.x, aces_neutral.y);
    let aces_neutral_xyz = V3f { x: ax / ay, y: 1.0, z: (1.0 - ax - ay) / ay };

    // Compute the Bradford transformation matrix.
    let mut a = V3f::default();
    bradford_cpm.mult_vec_matrix(&aces_neutral_xyz, &mut a);
    let mut b = V3f::default();
    bradford_cpm.mult_vec_matrix(&file_neutral_xyz, &mut b);
    let ratio = a / b;

    #[rustfmt::skip]
    let ratio_mat = M44f::new(
        ratio.x, 0.0, 0.0, 0.0,
        0.0, ratio.y, 0.0, 0.0,
        0.0, 0.0, ratio.z, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    let bradford_trans = bradford_cpm.mul_matrix(&ratio_mat).mul_matrix(&inverse_bradford_cpm);

    // Build a combined file-RGB-to-ACES-RGB conversion matrix.
    let file_to_aces = file_chr.rgb_to_xyz(1.0)?.mul_matrix(&bradford_trans).mul_matrix(&aces_chr.xyz_to_rgb(1.0)?);

    Ok((true, file_to_aces))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{MemIStream, MemOStream};
    use imath_rs::Half;

    fn test_header(w: i32, h: i32, compression: Compression) -> Header {
        Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, compression).unwrap()
    }

    /// C++ `checkCompression`: only NO/PIZ/B44A are legal in an ACES file.
    #[test]
    fn check_compression_allows_only_no_piz_b44a() {
        assert!(check_compression(Compression::NoCompression).is_ok());
        assert!(check_compression(Compression::Piz).is_ok());
        assert!(check_compression(Compression::B44a).is_ok());
        for bad in [Compression::Zip, Compression::Zips, Compression::Rle, Compression::Pxr24, Compression::B44, Compression::Dwaa] {
            assert!(matches!(check_compression(bad), Err(ExrError::Arg(_))), "{bad:?} should be rejected");
        }
    }

    /// `AcesOutputFile::new` must reject a header with a non-ACES-legal
    /// compression before touching anything else.
    #[test]
    fn new_rejects_bad_compression() {
        let h = test_header(4, 4, Compression::Zip);
        let mut out = MemOStream::new();
        let result = AcesOutputFile::new(&mut out, &h, RgbaChannels::WRITE_RGBA);
        assert!(matches!(result, Err(ExrError::Arg(_))));
    }

    /// `AcesOutputFile::new` must stamp `chromaticities`/`adoptedNeutral` with
    /// the ACES primaries regardless of what the caller's header carried.
    #[test]
    fn new_stamps_aces_chromaticities() {
        let h = test_header(4, 4, Compression::NoCompression); // default (Rec.709) chromaticities
        let mut out = MemOStream::new();
        let of = AcesOutputFile::new(&mut out, &h, RgbaChannels::WRITE_RGBA).unwrap();
        assert_eq!(*of.header().chromaticities().unwrap(), aces_chromaticities());
        assert_eq!(*of.header().adopted_neutral().unwrap(), aces_chromaticities().white_v2f());
    }

    /// Port of `AcesOutputFile::updatePreviewImage`
    /// (`ImfAcesFile.cpp:265-268`) -- a one-line forward to
    /// `RgbaOutputFile::updatePreviewImage`, wired now that
    /// `scanline::OutputFile` supports it. Writes a file whose header
    /// carries a preview image, rewrites it after the pixel data, and
    /// asserts the re-parsed header shows the new pixels.
    #[test]
    fn update_preview_image_rewrites_the_headers_preview() {
        use crate::attr::{PreviewImage, PreviewRgba};

        let mut h = test_header(2, 2, Compression::NoCompression);
        h.set_preview_image(
            PreviewImage::new(1, 1, Some(vec![PreviewRgba::new(1, 2, 3, 4)])).unwrap(),
        )
        .unwrap();

        let pixels = vec![Rgba::new(Half::from_f32(0.1), Half::from_f32(0.2), Half::from_f32(0.3), Half::from_f32(1.0)); 4];
        let mut out = MemOStream::new();
        {
            let mut of = AcesOutputFile::new(&mut out, &h, RgbaChannels::WRITE_RGBA).unwrap();
            of.set_frame_buffer(1, 2).unwrap();
            of.write_pixels(&pixels, 2).unwrap();
            of.update_preview_image(&[PreviewRgba::new(9, 9, 9, 9)]).unwrap();
        }
        let bytes = out.into_data();

        let mut is = MemIStream::new(bytes);
        let (reread, _version) = Header::read_checked(&mut is).unwrap();
        assert_eq!(
            reread.preview_image().unwrap().pixels(),
            &[PreviewRgba::new(9, 9, 9, 9)]
        );
    }

    /// If the file's chromaticities already equal ACES (as every
    /// `AcesOutputFile`-written file's do), `AcesInputFile` must not alter
    /// the decoded pixels at all.
    #[test]
    fn already_aces_file_applies_no_conversion() {
        let mut h = test_header(2, 2, Compression::NoCompression);
        h.add_chromaticities(aces_chromaticities()).unwrap();
        h.add_adopted_neutral(aces_chromaticities().white_v2f()).unwrap();

        let pixels = vec![Rgba::new(Half::from_f32(0.3), Half::from_f32(0.6), Half::from_f32(0.1), Half::from_f32(1.0)); 4];
        let mut out = MemOStream::new();
        RgbaOutputFile::write_stream(&mut out, &h, RgbaChannels::WRITE_RGBA, &pixels).unwrap();
        let bytes = out.into_data();

        let mut stream = MemIStream::new(bytes);
        let mut f = AcesInputFile::new(&mut stream).unwrap();
        assert!(!f.must_convert_color);
        let mut base = vec![Rgba::default(); 4];
        f.read_pixels(0, 1, &mut base, 1, 2).unwrap();
        assert_eq!(base, pixels);
    }

    /// The Bradford-adapted file-to-ACES matrix is constructed so that the
    /// FILE's own white point maps back to (approximately) white in ACES
    /// space (`RGBtoXYZ`/`XYZtoRGB` are each built to send their own white
    /// point to/from `(1,1,1)`, and the Bradford step only moves between the
    /// two white points) — an independent correctness check on the matrix
    /// assembly, not a restatement of the code under test.
    #[test]
    fn non_aces_file_maps_its_own_white_point_close_to_one() {
        let h = test_header(1, 1, Compression::NoCompression); // default (Rec.709) chromaticities, != ACES
        let white = vec![Rgba::new(Half::from_f32(1.0), Half::from_f32(1.0), Half::from_f32(1.0), Half::from_f32(1.0))];
        let mut out = MemOStream::new();
        RgbaOutputFile::write_stream(&mut out, &h, RgbaChannels::WRITE_RGBA, &white).unwrap();
        let bytes = out.into_data();

        let mut stream = MemIStream::new(bytes);
        let mut f = AcesInputFile::new(&mut stream).unwrap();
        assert!(f.must_convert_color);
        let mut base = vec![Rgba::default(); 1];
        f.read_pixels(0, 0, &mut base, 1, 0).unwrap();
        assert!((base[0].r.to_f32() - 1.0).abs() < 5e-3, "r={}", base[0].r.to_f32());
        assert!((base[0].g.to_f32() - 1.0).abs() < 5e-3, "g={}", base[0].g.to_f32());
        assert!((base[0].b.to_f32() - 1.0).abs() < 5e-3, "b={}", base[0].b.to_f32());
    }

    /// A non-white, non-trivial pixel must actually be transformed (not a
    /// silent no-op) when the file's chromaticities differ from ACES.
    #[test]
    fn non_aces_file_transforms_non_white_pixels() {
        let h = test_header(1, 1, Compression::NoCompression);
        let px = vec![Rgba::new(Half::from_f32(0.8), Half::from_f32(0.1), Half::from_f32(0.2), Half::from_f32(1.0))];
        let mut out = MemOStream::new();
        RgbaOutputFile::write_stream(&mut out, &h, RgbaChannels::WRITE_RGBA, &px).unwrap();
        let bytes = out.into_data();

        let mut stream = MemIStream::new(bytes);
        let mut f = AcesInputFile::new(&mut stream).unwrap();
        let mut base = vec![Rgba::default(); 1];
        f.read_pixels(0, 0, &mut base, 1, 0).unwrap();
        assert_ne!(base[0], px[0]);
    }

    /// Which NaN payload survives the colour matrix, pinned to what C++ `exr2aces` prints. Oracle:
    /// `target/parity-tools/exr2aces.exe in.exr out.aces` on a 7x1 Rec.709 HALF RGB file, NONE,
    /// pixel i = subset i of {R, G, B} holding NaN `0x7e01` (R) / `0x7e02` (G) / `0x7e03` (B), the
    /// rest 0.5. Its output payloads are the table below (R, G, B of each pixel); the ones an
    /// R-before-G order would give differ on `R+G` and `R+G+B` (output R would keep `0x7e01`).
    #[test]
    fn nan_payloads_follow_the_msvc_operand_order() {
        const FIN: u16 = 0x3800; // 0.5
        let nan = |on: bool, p: u16| Half::from_bits(if on { p } else { FIN });
        // (R, G, B NaN?) -> output (R, G, B) bits, one row per subset.
        #[rustfmt::skip]
        let cases: [([bool; 3], [u16; 3]); 7] = [
            ([true,  false, false], [0x7e01, 0x7e01, 0x7e01]),
            ([false, true,  false], [0x7e02, 0x7e02, 0x7e02]),
            ([false, false, true ], [0x7e03, 0x7e03, 0x7e03]),
            ([true,  true,  false], [0x7e02, 0x7e01, 0x7e01]),
            ([true,  false, true ], [0x7e01, 0x7e01, 0x7e01]),
            ([false, true,  true ], [0x7e02, 0x7e02, 0x7e02]),
            ([true,  true,  true ], [0x7e02, 0x7e01, 0x7e01]),
        ];
        let px: Vec<Rgba> = cases
            .iter()
            .map(|(n, _)| Rgba::new(nan(n[0], 0x7e01), nan(n[1], 0x7e02), nan(n[2], 0x7e03), Half::from_f32(1.0)))
            .collect();
        let h = Header::with_dimensions(7, 1, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::NoCompression).unwrap();
        let mut out = MemOStream::new();
        RgbaOutputFile::write_stream(&mut out, &h, RgbaChannels::WRITE_RGBA, &px).unwrap();

        let mut stream = MemIStream::new(out.into_data());
        let mut f = AcesInputFile::new(&mut stream).unwrap();
        let mut base = vec![Rgba::default(); 7];
        f.read_pixels(0, 0, &mut base, 1, 0).unwrap();
        for (i, (n, want)) in cases.iter().enumerate() {
            let got = [base[i].r.bits(), base[i].g.bits(), base[i].b.bits()];
            assert_eq!(got, *want, "NaN subset {n:?}: got {got:04x?}");
        }
    }
}
