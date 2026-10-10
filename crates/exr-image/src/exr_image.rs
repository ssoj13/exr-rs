//! OpenEXR for the `image` crate, decoded by `exr-core` (our 1:1 OpenEXR port) instead of the
//! crates.io `exr` (exrs) that `image`'s own `exr` feature pulls in.
//!
//! Why: the whole ecosystem keeps `image` for its other formats but must have exactly one OpenEXR
//! implementation. `image` lets a crate replace a format's decoder at runtime (`image::hooks`);
//! [`register`] installs this one for `.exr`, so `image::open`, `image::image_dimensions`,
//! `image::load_from_memory`, `ImageReader::open(..).decode()` and
//! `ImageReader::..with_guessed_format()` keep working with `image`'s `exr` feature turned off.
//!
//! Where it applies (verified against image 0.25.10 `io/image_reader_type.rs`, `dynimage.rs`) —
//! `image` consults the hooks inside `ImageReader`:
//! - by file extension, any casing (the lookup lowercases it since 0.25.10): `image::open`,
//!   `image_dimensions`, `ImageReader::open`;
//! - by content, the EXR magic `76 2F 31 01`: `ImageReader::with_guessed_format` and
//!   `image::load_from_memory` (which is built on it since 0.25.10).
//!
//! NOT covered, because `image` resolves a built-in `ImageFormat::OpenExr` there and never looks at
//! hooks: `load_from_memory_with_format(.., OpenExr)`, `image::load(reader, OpenExr)`,
//! `ImageReader::with_format`/`set_format(OpenExr)`. Such call sites must switch to
//! `image::load_from_memory` / `with_guessed_format()` or call [`decode`].
//! Writing EXR is not covered by `image`'s hooks at all: write through `exr-core`.
//!
//! Pixels: single-part flat images; the `R`, `G`, `B` channels (plus `A` when present) of any
//! sample type widened to `f32` — `Rgb32F` / `Rgba32F`, matching what the exrs-backed decoder
//! returned. The data window defines the size.

use std::io::{Read, Seek, SeekFrom};
use std::sync::OnceLock;

use exr_core::{ChannelData, ExrError, IStream, Image};
use image::error::{DecodingError, ImageFormatHint, LimitError, LimitErrorKind};
use image::hooks::{register_decoding_hook, register_format_detection_hook};
use image::{ColorType, DynamicImage, ImageDecoder, ImageError, ImageResult};

/// The OpenEXR magic number as it appears on disk (`MAGIC = 20000630`, little-endian).
const MAGIC: &[u8] = &[0x76, 0x2f, 0x31, 0x01];

/// Install the exr-core decoder into `image` for this process. Idempotent and thread-safe; call it
/// once from the crate's image-loading entry point before the first `image` decode.
pub fn register() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        // image 0.25.10+ lowercases the extension before the hook lookup, so one key serves
        // `.exr`, `.EXR`, `.Exr`.
        register_decoding_hook(
            "exr".into(),
            Box::new(|r| Ok(Box::new(ExrDecoder::new(r)?) as Box<dyn ImageDecoder + '_>)),
        );
        register_format_detection_hook("exr".into(), MAGIC, None);
    });
}

/// Decode an EXR straight into a [`DynamicImage`] without going through `image`'s format dispatch
/// (for call sites that used `ImageFormat::OpenExr` explicitly).
pub fn decode<R: Read + Seek>(reader: R) -> ImageResult<DynamicImage> {
    DynamicImage::from_decoder(ExrDecoder::new(reader)?)
}

/// `image::ImageDecoder` over exr-core. Construction parses the header and chunk table without
/// decoding any chunk; the pixels are decoded in `read_image`.
pub struct ExrDecoder<R> {
    stream: ReadSeekStream<R>,
    width: u32,
    height: u32,
    alpha: bool,
}

impl<R: Read + Seek> ExrDecoder<R> {
    /// Parse the header and validate that the image has the R, G, B channels this decoder emits.
    pub fn new(reader: R) -> ImageResult<Self> {
        let mut stream = ReadSeekStream::new(reader).map_err(io_err)?;
        let header = Image::read_header_only_from(&mut stream).map_err(exr_err)?;
        for name in ["R", "G", "B"] {
            if header.channel(name).is_none() {
                return Err(decoding(format!("no {name} channel: an RGB(A) image is required")));
            }
        }
        let alpha = header.channel("A").is_some();
        let (w, h) = (header.width(), header.height());
        let width = u32::try_from(w).map_err(|_| decoding(format!("bad width {w}")))?;
        let height = u32::try_from(h).map_err(|_| decoding(format!("bad height {h}")))?;
        Ok(Self { stream, width, height, alpha })
    }

    fn channels(&self) -> usize {
        if self.alpha { 4 } else { 3 }
    }
}

impl<R: Read + Seek> ImageDecoder for ExrDecoder<R> {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn color_type(&self) -> ColorType {
        if self.alpha { ColorType::Rgba32F } else { ColorType::Rgb32F }
    }

    fn read_image(mut self, buf: &mut [u8]) -> ImageResult<()> {
        let n = self.channels();
        let pixels = self.width as usize * self.height as usize;
        if buf.len() != pixels * n * 4 {
            return Err(ImageError::Limits(LimitError::from_kind(LimitErrorKind::DimensionError)));
        }
        self.stream.seekg(0).map_err(exr_err)?;
        let image = Image::read_from(&mut self.stream).map_err(exr_err)?;
        let names: &[&str] = if self.alpha { &["R", "G", "B", "A"] } else { &["R", "G", "B"] };
        // exr-core's flat reader already rejects subsampled channels, so every plane is w*h.
        for (c, name) in names.iter().enumerate() {
            let data = image
                .channel(name)
                .map(ChannelData::to_f32)
                .ok_or_else(|| decoding(format!("no {name} channel")))?;
            if data.len() != pixels {
                return Err(decoding(format!("channel {name}: {} samples for {pixels} pixels", data.len())));
            }
            // `image` expects native-endian samples, interleaved R,G,B[,A].
            for (i, v) in data.iter().enumerate() {
                let at = (i * n + c) * 4;
                buf[at..at + 4].copy_from_slice(&v.to_ne_bytes());
            }
        }
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }
}

/// exr-core input stream over any `Read + Seek` (what `image`'s hooks hand out), with the C++
/// `IStream` contract: an exact read returns `Ok(true)`, a short read is an error.
struct ReadSeekStream<R> {
    inner: R,
    pos: u64,
    size: u64,
}

impl<R: Read + Seek> ReadSeekStream<R> {
    fn new(mut inner: R) -> std::io::Result<Self> {
        let pos = inner.stream_position()?;
        let size = inner.seek(SeekFrom::End(0))?;
        inner.seek(SeekFrom::Start(pos))?;
        Ok(Self { inner, pos, size })
    }
}

impl<R: Read + Seek> IStream for ReadSeekStream<R> {
    fn file_name(&self) -> &str {
        "(image reader)"
    }

    fn read(&mut self, buf: &mut [u8]) -> exr_core::Result<bool> {
        self.inner.read_exact(buf).map_err(ExrError::from_io)?;
        self.pos += buf.len() as u64;
        Ok(true)
    }

    fn tellg(&mut self) -> exr_core::Result<u64> {
        Ok(self.pos)
    }

    fn seekg(&mut self, pos: u64) -> exr_core::Result<()> {
        if pos != self.pos {
            self.inner.seek(SeekFrom::Start(pos)).map_err(ExrError::from_io)?;
            self.pos = pos;
        }
        Ok(())
    }

    fn size(&self) -> Option<u64> {
        Some(self.size)
    }
}

fn decoding(msg: String) -> ImageError {
    ImageError::Decoding(DecodingError::new(ImageFormatHint::Name("OpenEXR".into()), msg))
}

fn exr_err(e: ExrError) -> ImageError {
    decoding(e.to_string())
}

fn io_err(e: std::io::Error) -> ImageError {
    ImageError::IoError(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use exr_core::attr::Compression;
    use imath_rs::{Box2i, V2i};

    fn write(path: &std::path::Path, w: i32, h: i32, alpha: bool) -> Vec<[f32; 4]> {
        let px: Vec<[f32; 4]> = (0..w * h)
            .map(|i| [i as f32 * 1.5, -0.25 * i as f32, 19.43, 0.5 + i as f32 / 100.0])
            .collect();
        let window = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w - 1, y: h - 1 } };
        let ch = |c: usize| ChannelData::Float(px.iter().map(|p| p[c]).collect());
        let mut img = Image::new(window)
            .with_channel("R", ch(0))
            .with_channel("G", ch(1))
            .with_channel("B", ch(2));
        if alpha {
            img = img.with_channel("A", ch(3));
        }
        img.write(path, Compression::Zip).unwrap();
        px
    }

    /// A per-test scratch dir: tests run in parallel and each removes its own.
    fn dir(test: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("exr-image-{}-{test}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn image_open_decodes_rgb_and_rgba_through_exr_core() {
        register();
        let d = dir("open");
        let rgb = d.join("rgb.exr");
        let px = write(&rgb, 3, 2, false);
        let img = image::open(&rgb).unwrap();
        assert_eq!(img.color(), ColorType::Rgb32F);
        let got = img.into_rgb32f();
        assert_eq!(got.dimensions(), (3, 2));
        for (g, p) in got.pixels().zip(&px) {
            assert_eq!(g.0, [p[0], p[1], p[2]], "values > 1 and negatives must survive bit-exactly");
        }
        let rgba = d.join("RGBA.EXR");
        let px = write(&rgba, 2, 2, true);
        let got = image::open(&rgba).unwrap().into_rgba32f();
        for (g, p) in got.pixels().zip(&px) {
            assert_eq!(g.0, *p);
        }
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn content_detection_and_explicit_decode() {
        register();
        let d = dir("detect");
        let path = d.join("no_extension");
        let px = write(&path, 2, 1, false);
        let bytes = std::fs::read(&path).unwrap();
        let guessed = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .into_rgb32f();
        assert_eq!(guessed.get_pixel(1, 0).0, [px[1][0], px[1][1], px[1][2]]);
        // image 0.25.10: `load_from_memory` guesses the format through `ImageReader`, so it reaches
        // the content hook too.
        let from_memory = image::load_from_memory(&bytes).unwrap().into_rgb32f();
        assert_eq!(from_memory, guessed);
        // Pinned limitation: an explicit built-in format bypasses the hooks, so with `image`'s exr
        // feature off it cannot decode EXR. If this starts passing, image changed again and the
        // module doc must be updated.
        assert!(image::load_from_memory_with_format(&bytes, image::ImageFormat::OpenExr).is_err());
        let direct = decode(std::io::Cursor::new(&bytes)).unwrap().into_rgb32f();
        assert_eq!(direct, guessed);
        let dims = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap();
        assert_eq!(dims, (2, 1));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn missing_rgb_channel_is_a_clear_error() {
        let d = dir("missing");
        let path = d.join("y.exr");
        let window = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 0, y: 0 } };
        Image::new(window)
            .with_channel("Y", ChannelData::Float(vec![1.0]))
            .write(&path, Compression::NoCompression)
            .unwrap();
        let err = decode(std::fs::File::open(&path).unwrap()).err().unwrap().to_string();
        assert!(err.contains("no R channel"), "{err}");
        std::fs::remove_dir_all(d).unwrap();
    }
}
