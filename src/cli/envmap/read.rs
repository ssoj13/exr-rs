//! `readInputImage.cpp/h`: load the input environment map, one file (optionally padded at the top
//! and bottom) or six cube-face files (`openexr/src/bin/exrenvmap/readInputImage.cpp`, v3.4.15).
//!
//! Pixels are read in bands of [`BAND`] rows, not in one call: the RGBA reader stages a whole
//! `readPixels` range before it scatters it, which for a 16k x 8k map would be a second copy of the
//! image. `BAND` is a multiple of every codec's scanlines per chunk (1, 16, 32, 256), so no chunk is
//! decoded twice.

use exr_core::attr::Envmap;
use exr_core::header::Header;
use exr_core::rgba::{Rgba, RgbaChannels, RgbaInputFile};
use exr_core::{ExrError, IStream, Result};
use imath_rs::{Box2i, V2i};

use super::image::{EnvmapImage, FACE_NAMES};

const BAND: i64 = 256;

/// What the input tells the output: its header (whose attributes the output inherits) and the
/// channels it was stored with.
pub struct Input {
    pub header: Header,
    pub channels: RgbaChannels,
}

/// Read scanlines `[y0, y1]` of `file` into `px`, a row-major array that starts at the window's
/// first row and is `w` pixels wide.
fn read_rows(file: &mut RgbaInputFile<'static, Box<dyn IStream>>, dw: &Box2i, px: &mut [Rgba], w: usize) -> Result<()> {
    let stride = isize::try_from(w).map_err(|_| ExrError::Arg("The image is too wide".into()))?;
    let mut y = i64::from(dw.min.y);
    while y <= i64::from(dw.max.y) {
        let end = (y + BAND - 1).min(i64::from(dw.max.y));
        file.read_pixels(y as i32, end as i32, px, 1, stride)?;
        y = end + 1;
    }
    Ok(())
}

/// `readSingleImage`: the map of one file, `pad_top`/`pad_bottom` times its height of extra lines
/// repeating the first and last row (lat-long maps only).
fn read_single(path: &str, pad: (f32, f32), over: Option<Envmap>, verbose: bool, image: &mut EnvmapImage) -> Result<Input> {
    let mut file = RgbaInputFile::open(path)?;
    if verbose {
        outln!("reading file {path}");
    }

    let mut header = file.header()?.clone();
    let channels = file.channels()?;

    let mut ty = Envmap::LatLong;
    if let Some(o) = over {
        ty = o;
        header.add_envmap(o)?;
    } else if file.header()?.has_envmap() {
        // The attribute reader rejects any value but the two envmap types, so this is one of them.
        ty = *file.header()?.envmap()?;
    }

    let dw = file.data_window()?;
    let w = usize::try_from(i64::from(dw.max.x) - i64::from(dw.min.x) + 1).map_err(|_| ExrError::Arg("Invalid data window".into()))?;
    let h = usize::try_from(i64::from(dw.max.y) - i64::from(dw.min.y) + 1).map_err(|_| ExrError::Arg("Invalid data window".into()))?;

    let (mut pt, mut pb) = (0usize, 0usize);
    if ty == Envmap::LatLong {
        // `int (pad * h + 0.5f)`
        let lines = |p: f32| {
            let n = p * h as f32 + 0.5;
            if n < i32::MAX as f32 { Ok(n as usize) } else { Err(ExrError::Arg("The padding is too large".into())) }
        };
        (pt, pb) = (lines(pad.0)?, lines(pad.1)?);
    }
    let grown = |y: i64| i32::try_from(y).map_err(|_| ExrError::Arg("The padding is too large".into()));
    let padded_dw = Box2i {
        min: V2i { x: dw.min.x, y: grown(i64::from(dw.min.y) - pt as i64)? },
        max: V2i { x: dw.max.x, y: grown(i64::from(dw.max.y) + pb as i64)? },
    };

    image.resize(ty, &padded_dw)?;
    // The file's first row lands `pt` rows down.
    read_rows(&mut file, &dw, &mut image.px_mut()[pt * w..], w)?;

    let px = image.px_mut();
    let (first, last) = (pt * w, (h + pt - 1) * w);
    for y in 0..pt {
        px.copy_within(first..first + w, y * w);
    }
    for y in h + pt..h + pt + pb {
        px.copy_within(last..last + w, y * w);
    }
    Ok(Input { header, channels })
}

/// `readSixImages`: the six square faces `path` names (its first `%` replaced by `+X` ... `-Z`),
/// stacked into one cube-face map.
fn read_six(path: &str, verbose: bool, image: &mut EnvmapImage) -> Result<Input> {
    let face_path = |face: &str| path.replacen('%', face, 1);

    let name = face_path(FACE_NAMES[0]);
    let (dw, w, h, input) = {
        let file = RgbaInputFile::open(&name)?;
        if verbose {
            outln!("reading cube face size from file {name}");
        }
        let dw = file.data_window()?;
        let (w, h) = (i64::from(dw.max.x) - i64::from(dw.min.x) + 1, i64::from(dw.max.y) - i64::from(dw.min.y) + 1);
        if w != h {
            return Err(ExrError::Input(format!("Cube face image {name} is not square.")));
        }
        let mut header = file.header()?.clone();
        header.add_envmap(Envmap::Cube)?;
        (dw, w, h, Input { header, channels: file.channels()? })
    };

    let side = usize::try_from(w).map_err(|_| ExrError::Arg("Invalid data window".into()))?;
    let image_dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: (w - 1) as i32, y: i32::try_from(6 * h - 1).map_err(|_| ExrError::Arg("The cube map is too large".into()))? } };
    image.resize(Envmap::Cube, &image_dw)?;

    for (face_name, face) in FACE_NAMES.iter().zip(image.px_mut().chunks_exact_mut(side * side)) {
        let name = face_path(face_name);
        let mut file = RgbaInputFile::open(&name)?;
        if verbose {
            outln!("reading file {name}");
        }
        if file.data_window()? != dw {
            return Err(ExrError::Input(format!("The data window of cube face {name} differs from the data window of other cube faces.")));
        }
        read_rows(&mut file, &dw, face, side)?;
    }
    Ok(input)
}

/// `readInputImage`: `path` with a `%` names six cube faces, anything else one file.
pub fn read_input(path: &str, pad: (f32, f32), over: Option<Envmap>, verbose: bool, image: &mut EnvmapImage) -> Result<Input> {
    if path.contains('%') { read_six(path, verbose, image) } else { read_single(path, pad, over, verbose, image) }
}
