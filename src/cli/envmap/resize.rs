//! `resizeImage.cpp/h`: resample an environment map to a lat-long or cube-face window
//! (`openexr/src/bin/exrenvmap/resizeImage.cpp`, v3.4.15).

use exr_core::Result;
use exr_core::attr::Envmap;
use exr_core::attr::envmap::{cube_map, lat_long_map};
use imath_rs::{Box2i, V2f};

use super::image::{EnvmapImage, FACES};

/// `resizeLatLong`: `image2` becomes `image1` sampled over the lat-long window `dw2`.
pub fn resize_ll(image1: &EnvmapImage, image2: &mut EnvmapImage, dw2: &Box2i, filter_radius: f32, n: i32) -> Result<()> {
    image2.resize(Envmap::LatLong, dw2)?;
    let (w, h) = (image2.w(), image2.h());
    // `0.5f * 2 * M_PI * filterRadius / w` is a double expression, stored in a float.
    let radius = ((std::f64::consts::PI * f64::from(filter_radius)) / w as f64) as f32;
    for y in 0..h {
        for x in 0..w {
            let dir = lat_long_map::direction(*dw2, V2f { x: x as f32, y: y as f32 });
            *image2.cell(x as i32, y as i32)? = image1.look(dir, radius, n)?;
        }
    }
    Ok(())
}

/// `resizeCube`: `image2` becomes `image1` sampled over the cube-face window `dw2`; a cube map of
/// that very size is copied instead.
pub fn resize_cube(image1: &EnvmapImage, image2: &mut EnvmapImage, dw2: &Box2i, filter_radius: f32, n: i32) -> Result<()> {
    image2.resize(Envmap::Cube, dw2)?;
    if image1.ty() == Envmap::Cube && image1.dw() == *dw2 {
        image2.px_mut().copy_from_slice(image1.px());
        return Ok(());
    }

    let sof = cube_map::size_of_face(*dw2);
    let radius = 1.5f32 * filter_radius / sof as f32;
    for face in FACES {
        for y in 0..sof {
            for x in 0..sof {
                let in_face = V2f { x: x as f32, y: y as f32 };
                let dir = cube_map::direction(face, *dw2, in_face);
                let pos = cube_map::pixel_position(face, *dw2, in_face);
                *image2.cell((pos.x + 0.5) as i32, (pos.y + 0.5) as i32)? = image1.look(dir, radius, n)?;
            }
        }
    }
    Ok(())
}
