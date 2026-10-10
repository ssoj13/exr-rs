//! `makeCubeMap.cpp/h` and `makeLatLongMap.cpp/h`: resample the input and write it as a cube-face
//! or lat-long map, in one file with all its levels or (cube faces only) in six files
//! (`openexr/src/bin/exrenvmap/makeCubeMap.cpp`, `makeLatLongMap.cpp`, v3.4.15).
//!
//! `makeCubeMapSingleFile` and `makeLatLongMap` differ only in the resampling function and the map's
//! proportions, so they are one function here, [`single`].

use exr_core::attr::{Compression, Envmap, LevelMode, LevelRoundingMode, LineOrder};
use exr_core::header::Header;
use exr_core::io::MemOStream;
use exr_core::rgba::{Rgba, RgbaChannels, TiledRgbaOutputFile};
use exr_core::{ExrError, Result};
use imath_rs::V2f;

use super::image::{EnvmapImage, FACE_NAMES, cube_window, window};
use super::resize::{resize_cube, resize_ll};

/// The output's shape and how to resample into it.
pub struct Opts {
    pub tile: (u32, u32),
    pub mode: LevelMode,
    pub rounding: LevelRoundingMode,
    pub comp: Compression,
    /// Width of the output map (of one cube face for a cube map).
    pub width: i32,
    pub radius: f32,
    pub samples: i32,
    pub verbose: bool,
}

/// Reject a header or channel set the output cannot take before the file is created: the C++
/// validates both first, so a rejected request leaves nothing behind. The library checks in its
/// constructor, which a throwaway in-memory stream lets run early.
fn check(header: &Header, channels: RgbaChannels, o: &Opts, mode: LevelMode, rounding: LevelRoundingMode) -> Result<()> {
    TiledRgbaOutputFile::new(&mut MemOStream::new(), header, channels, o.tile.0, o.tile.1, mode, rounding).map(drop)
}

/// Write every tile of `level` of `px`, an image `w` pixels wide whose row 0 is the level's first.
fn write_level(out: &mut TiledRgbaOutputFile<'_>, px: &[Rgba], w: usize, level: i32) -> Result<()> {
    out.set_frame_buffer(1, isize::try_from(w).map_err(|_| ExrError::Arg("The output image is too wide".into()))?)?;
    for tile_y in 0..out.num_y_tiles(level)? {
        for tile_x in 0..out.num_x_tiles(level)? {
            let t = out.data_window_for_tile(tile_x, tile_y, level, level)?;
            let at = usize::try_from(t.min.y).ok().zip(usize::try_from(t.min.x).ok()).and_then(|(y, x)| y.checked_mul(w)?.checked_add(x));
            let rest = at.and_then(|at| px.get(at..)).ok_or_else(|| ExrError::Logic(format!("tile ({tile_x}, {tile_y}) is outside the image")))?;
            out.write_tile_l(rest, tile_x, tile_y, level)?;
        }
    }
    Ok(())
}

/// The map of type `ty` in one file, all its levels: the top level resampled from `image`, each
/// other level from the one above.
fn single(ty: Envmap, image: &mut EnvmapImage, mut header: Header, channels: RgbaChannels, path: &str, o: &Opts) -> Result<()> {
    let dw = match ty {
        Envmap::Cube => cube_window(o.width)?,
        Envmap::LatLong => window(o.width, o.width / 2),
    };
    *header.data_window_mut()? = dw;
    *header.display_window_mut()? = dw;
    *header.compression_mut()? = o.comp;
    header.add_envmap(ty)?;

    check(&header, channels, o, o.mode, o.rounding)?;
    exr_core::io::write_file(path, |stream| -> exr_core::Result<_> {
        let mut out = TiledRgbaOutputFile::new(stream, &header, channels, o.tile.0, o.tile.1, o.mode, o.rounding)?;
        if o.verbose {
            outln!("writing file {path}");
        }

        // The two images trade places every level: the one just written is the source of the next, the
        // other (the input, after the first level) is overwritten.
        let mut spare = EnvmapImage::new();
        for level in 0..out.num_levels()? {
            if o.verbose {
                outln!("level {level}");
            }
            let dw = out.data_window_for_level_l(level)?;
            match ty {
                Envmap::Cube => resize_cube(image, &mut spare, &dw, o.radius, o.samples)?,
                Envmap::LatLong => resize_ll(image, &mut spare, &dw, o.radius, o.samples)?,
            }
            write_level(&mut out, spare.px(), spare.w(), level)?;
            std::mem::swap(image, &mut spare);
        }
        Ok(())
    })?;
    if o.verbose {
        outln!("done.");
    }
    Ok(())
}

/// `makeCubeMapSixFiles`: one resampled cube map, written as six square one-level files, the `%` of
/// `path` replaced by the face's name. Their headers are fresh (the input's attributes and the
/// level options do not apply).
fn six(image: &mut EnvmapImage, channels: RgbaChannels, path: &str, o: &Opts) -> Result<()> {
    let side = usize::try_from(o.width).map_err(|_| ExrError::Arg("Invalid output width".into()))?;
    let mut resized = EnvmapImage::new();
    resize_cube(image, &mut resized, &cube_window(o.width)?, o.radius, o.samples)?;

    let face_dw = window(o.width, o.width);
    for (face_name, face) in FACE_NAMES.iter().zip(resized.px().chunks_exact(side * side)) {
        let name = path.replacen('%', face_name, 1);
        if o.verbose {
            outln!("writing file {name}");
        }
        let header = Header::with_windows(face_dw, face_dw, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, o.comp)?;
        check(&header, channels, o, LevelMode::OneLevel, LevelRoundingMode::RoundDown)?;
        exr_core::io::write_file(&name, |stream| -> exr_core::Result<_> {
            let mut out = TiledRgbaOutputFile::new(stream, &header, channels, o.tile.0, o.tile.1, LevelMode::OneLevel, LevelRoundingMode::RoundDown)?;
            write_level(&mut out, face, side, 0)
        })?;
    }
    if o.verbose {
        outln!("done.");
    }
    Ok(())
}

/// `makeCubeMap`: a `%` in the file name asks for six files.
pub fn make_cube(image: &mut EnvmapImage, header: Header, channels: RgbaChannels, path: &str, o: &Opts) -> Result<()> {
    if path.contains('%') { six(image, channels, path, o) } else { single(Envmap::Cube, image, header, channels, path, o) }
}

/// `makeLatLongMap`.
pub fn make_ll(image: &mut EnvmapImage, header: Header, channels: RgbaChannels, path: &str, o: &Opts) -> Result<()> {
    single(Envmap::LatLong, image, header, channels, path, o)
}
