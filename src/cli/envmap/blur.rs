//! `blurImage.cpp/h`: the `-b` diffuse blur, a hemispherical convolution done on a small proxy
//! cube map (`openexr/src/bin/exrenvmap/blurImage.cpp`, v3.4.15).
//!
//! The input is converted to cube faces and shrunk to at most [`MAX_IN_WIDTH`] pixels per face, each
//! pixel is weighted by the solid angle it covers, and every pixel of an [`OUT_WIDTH`]-wide output
//! cube is the cosine-weighted sum of all input pixels. The sums keep the C++ order (faces, rows,
//! columns) and its `double` arithmetic. The C++ recomputes each input pixel's direction for every
//! output pixel (576 million `CubeMap::direction` calls); here they are computed once, which gives
//! the same floats.

use exr_core::Result;
use exr_core::ExrError;
use exr_core::attr::Envmap;
use exr_core::attr::envmap::{CubeMapFace, cube_map};
use exr_core::rgba::Rgba;
use imath_rs::{Half, V2f, V3f};

use super::image::{EnvmapImage, FACES, cube_window};
use super::resize::resize_cube;

const MAX_IN_WIDTH: i32 = 40;
const OUT_WIDTH: i32 = 100;

/// `toInt`: round a pixel position to its pixel.
fn to_int(x: f32) -> i32 {
    (x + 0.5f32) as i32
}

/// `p *= weight`: the C++ multiplies the half by the `double` through `half::operator*= (float)`.
fn scale(p: &mut Rgba, weight: f64) {
    let w = weight as f32;
    for c in [&mut p.r, &mut p.g, &mut p.b, &mut p.a] {
        *c = Half::from_f32(c.to_f32() * w);
    }
}

/// What face `f` contributes to a pixel's solid-angle weight: the face normal and which direction
/// components are "across" (`ix` is the face's own axis).
fn face_axes(face: CubeMapFace) -> (V3f, usize, usize, usize) {
    match face {
        CubeMapFace::PosX => (V3f::new(1.0, 0.0, 0.0), 0, 1, 2),
        CubeMapFace::NegX => (V3f::new(-1.0, 0.0, 0.0), 0, 1, 2),
        CubeMapFace::PosY => (V3f::new(0.0, 1.0, 0.0), 1, 0, 2),
        CubeMapFace::NegY => (V3f::new(0.0, -1.0, 0.0), 1, 0, 2),
        CubeMapFace::PosZ => (V3f::new(0.0, 0.0, 1.0), 2, 0, 1),
        CubeMapFace::NegZ => (V3f::new(0.0, 0.0, -1.0), 2, 0, 1),
    }
}

/// An input pixel as the blur sums it: its direction and its channels as the `double`s the C++
/// multiplies (`half` -> `float` -> `double` is exact).
struct Tap {
    dir: V3f,
    rgba: [f64; 4],
}

/// `blurImage`: replace `image` by its diffuse blur, a cube map `OUT_WIDTH` faces wide.
pub fn blur(image: &mut EnvmapImage, verbose: bool) -> Result<()> {
    if verbose {
        outln!("blurring map image");
    }
    let mut spare = EnvmapImage::new();
    // The C++ ping-pongs between two images and copies the result back when it ends up in the
    // second one; here `image` always holds the current one, and the count keeps "copying" honest.
    let mut swaps = 0;
    let mut w = i32::try_from(image.w()).map_err(|_| ExrError::Arg("The image is too wide".into()))?;

    if image.ty() == Envmap::LatLong {
        if verbose {
            outln!("    converting to cube-face format");
        }
        w /= 4;
        resize_cube(image, &mut spare, &cube_window(w)?, 1.0, 7)?;
        std::mem::swap(image, &mut spare);
        swaps += 1;
    }

    while w > MAX_IN_WIDTH {
        w = if w >= MAX_IN_WIDTH * 2 { w / 2 } else { MAX_IN_WIDTH };
        if verbose {
            outln!("    resizing cube faces to {w} by {w} pixels");
        }
        resize_cube(image, &mut spare, &cube_window(w)?, 1.0, 7)?;
        std::mem::swap(image, &mut spare);
        swaps += 1;
    }

    if verbose {
        outln!("    computing pixel weights");
    }
    weigh(image, verbose)?;

    if verbose {
        outln!("    generating blurred image");
    }
    let taps = taps(image)?;
    let dw2 = cube_window(OUT_WIDTH)?;
    spare.resize(Envmap::Cube, &dw2)?;
    let sof2 = cube_map::size_of_face(dw2);
    for face2 in FACES {
        if verbose {
            outln!("        face {}", face2 as i32);
        }
        for y2 in 0..sof2 {
            for x2 in 0..sof2 {
                let in_face = V2f { x: x2 as f32, y: y2 as f32 };
                let dir2 = cube_map::direction(face2, dw2, in_face);
                let pos2 = cube_map::pixel_position(face2, dw2, in_face);

                let (mut total, mut sums) = (0f64, [0f64; 4]);
                for t in &taps {
                    let weight = f64::from(t.dir.dot(&dir2));
                    if weight <= 0.0 {
                        continue;
                    }
                    total += weight;
                    for (s, c) in sums.iter_mut().zip(t.rgba) {
                        *s += c * weight;
                    }
                }
                let px = |i: usize| Half::from_f32((sums[i] / total) as f32);
                *spare.cell(to_int(pos2.x), to_int(pos2.y))? = Rgba::new(px(0), px(1), px(2), px(3));
            }
        }
    }
    std::mem::swap(image, &mut spare);
    swaps += 1;

    if swaps % 2 == 1 && verbose {
        outln!("    copying");
    }
    Ok(())
}

/// Multiply each pixel by the solid angle it subtends, then restore the brightness the weights
/// took away (the average weight is not 1).
fn weigh(image: &mut EnvmapImage, verbose: bool) -> Result<()> {
    let dw = image.dw();
    let sof = cube_map::size_of_face(dw);
    let mut total = 0f64;

    for face in FACES {
        if verbose {
            outln!("        face {}", face as i32);
        }
        let (face_dir, ix, iy, iz) = face_axes(face);
        for y in 0..sof {
            let y_edge = y == 0 || y == sof - 1;
            for x in 0..sof {
                let x_edge = x == 0 || x == sof - 1;
                let in_face = V2f { x: x as f32, y: y as f32 };
                let dir = cube_map::direction(face, dw, in_face).normalized();
                let pos = cube_map::pixel_position(face, dw, in_face);

                // The solid angle is proportional to the squared distance from the cube's centre and
                // to the cosine between the view direction and the face normal.
                let sqr = |v: f32| f64::from(v) * f64::from(v);
                let mut weight = f64::from(dir.dot(&face_dir)) * (sqr(dir[iy] / dir[ix]) + sqr(dir[iz] / dir[ix]) + 1.0);
                // Pixels on the edges and corners of the cube are duplicated.
                if x_edge && y_edge {
                    weight /= 3.0;
                } else if x_edge || y_edge {
                    weight /= 2.0;
                }
                scale(image.cell(to_int(pos.x), to_int(pos.y))?, weight);
                total += weight;
            }
        }
    }

    let correction = image.px().len() as f64 / total;
    for p in image.px_mut() {
        scale(p, correction);
    }
    Ok(())
}

/// Every pixel of the (weighted) cube with its direction, in the order the C++ visits them.
fn taps(image: &mut EnvmapImage) -> Result<Vec<Tap>> {
    let dw = image.dw();
    let sof = cube_map::size_of_face(dw);
    let mut taps = Vec::new();
    taps.try_reserve_exact(6 * usize::try_from(sof).unwrap_or(0).pow(2)).map_err(|_| ExrError::Arg("Cannot allocate the blur table".into()))?;
    for face in FACES {
        for y in 0..sof {
            for x in 0..sof {
                let in_face = V2f { x: x as f32, y: y as f32 };
                let dir = cube_map::direction(face, dw, in_face);
                let pos = cube_map::pixel_position(face, dw, in_face);
                let p = *image.cell(to_int(pos.x), to_int(pos.y))?;
                taps.push(Tap { dir, rgba: [p.r, p.g, p.b, p.a].map(|c| f64::from(c.to_f32())) });
            }
        }
    }
    Ok(taps)
}
