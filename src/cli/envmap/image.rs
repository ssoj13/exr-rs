//! `EnvmapImage.cpp/h`: an environment map held in memory as half RGBA, with the filtered lookup
//! the resampling is built on (`openexr/src/bin/exrenvmap/EnvmapImage.cpp`, v3.4.15).
//!
//! The geometry (direction <-> pixel position) is `exr_core::attr::envmap`, the port of
//! `ImfEnvmap.cpp`. Every float expression keeps the C++ operand order: the output files are
//! byte-identical to `exrenvmap`'s.

use exr_core::attr::Envmap;
use exr_core::attr::envmap::{CubeMapFace, cube_map, lat_long_map};
use exr_core::rgba::Rgba;
// `dst + src` in MSVC `exrenvmap`'s SSE operand order (oracle-verified on maps with NaN and
// infinite pixels); NaN picked by bits, so the optimizer cannot change the payload.
use exr_core::sse::ss::add;
use exr_core::{ExrError, Result};
use imath_rs::{Box2i, Half, V2f, V2i, V3f, fun};

/// The six faces in the order they are stacked in a cube-face image.
pub const FACES: [CubeMapFace; 6] =
    [CubeMapFace::PosX, CubeMapFace::NegX, CubeMapFace::PosY, CubeMapFace::NegY, CubeMapFace::PosZ, CubeMapFace::NegZ];

/// What replaces the `%` of a file name for each face, in face order.
pub const FACE_NAMES: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

/// `Box2i ((0, 0), (w - 1, h - 1))`.
pub fn window(w: i32, h: i32) -> Box2i {
    Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w - 1, y: h - 1 } }
}

/// The window of a cube-face image with faces `w` pixels wide (`w x 6w`).
pub fn cube_window(w: i32) -> Result<Box2i> {
    let h = w.checked_mul(6).ok_or_else(|| ExrError::Arg("The cube map is too large".into()))?;
    Ok(window(w, h))
}

/// A `w x h` raster of half RGBA pixels, row-major over `dw`, and what the pixels depict.
pub struct EnvmapImage {
    ty: Envmap,
    dw: Box2i,
    /// Width and height of `dw`: the row stride of `px` and its row count.
    w: usize,
    h: usize,
    px: Vec<Rgba>,
}

/// Width and height of `dw` (`max - min + 1`), which must be non-negative.
fn extent(dw: &Box2i) -> Result<(usize, usize)> {
    let side = |lo: i32, hi: i32| usize::try_from(i64::from(hi) - i64::from(lo) + 1).map_err(|_| ExrError::Arg(format!("Invalid image window [{lo}, {hi}]")));
    Ok((side(dw.min.x, dw.max.x)?, side(dw.min.y, dw.max.y)?))
}

impl EnvmapImage {
    /// `EnvmapImage()`: a single black lat-long pixel.
    pub fn new() -> Self {
        let dw = window(1, 1);
        Self { ty: Envmap::LatLong, dw, w: 1, h: 1, px: vec![Rgba::default()] }
    }

    pub fn ty(&self) -> Envmap {
        self.ty
    }

    pub fn dw(&self) -> Box2i {
        self.dw
    }

    /// Width in pixels.
    pub fn w(&self) -> usize {
        self.w
    }

    /// Height in pixels.
    pub fn h(&self) -> usize {
        self.h
    }

    pub fn px(&self) -> &[Rgba] {
        &self.px
    }

    pub fn px_mut(&mut self) -> &mut [Rgba] {
        &mut self.px
    }

    /// `resize (type, dataWindow)` (+ its `clear`): the window becomes `dw`, every pixel black. The
    /// storage is reused, so an image used as scratch for level after level allocates once.
    pub fn resize(&mut self, ty: Envmap, dw: &Box2i) -> Result<()> {
        let (w, h) = extent(dw)?;
        let n = w.checked_mul(h).ok_or_else(|| ExrError::Arg("The image is too large".into()))?;
        self.px.clear();
        self.px.try_reserve_exact(n).map_err(|_| ExrError::Arg(format!("Cannot allocate a {w} x {h} image")))?;
        self.px.resize(n, Rgba::default());
        self.ty = ty;
        self.dw = *dw;
        self.w = w;
        self.h = h;
        Ok(())
    }

    /// The pixel at `(x, y)` counted from the array's own corner, not the window's: the C++ indexes
    /// `pixels[y][x]` with positions that `ImfEnvmap` computes from 0.
    pub fn cell(&mut self, x: i32, y: i32) -> Result<&mut Rgba> {
        let w = self.w();
        let at = usize::try_from(x).ok().zip(usize::try_from(y).ok()).filter(|&(x, _)| x < w).and_then(|(x, y)| y.checked_mul(w)?.checked_add(x));
        at.and_then(|i| self.px.get_mut(i)).ok_or_else(|| ExrError::Logic(format!("pixel ({x}, {y}) is outside the image")))
    }

    /// `dirToPosLatLong` / `dirToPosCube`: where direction `d` lands in this map.
    fn to_pos(&self, d: V3f) -> V2f {
        match self.ty {
            Envmap::LatLong => lat_long_map::pixel_position_from_dir(self.dw, d),
            Envmap::Cube => {
                let (face, in_face) = cube_map::face_and_pixel_position(d, self.dw);
                cube_map::pixel_position(face, self.dw, in_face)
            }
        }
    }

    /// `filteredLookup`: `n x n` point samples clustered around direction `d` (within a pyramid of
    /// radius `r`), combined with a tent filter.
    pub fn look(&self, mut d: V3f, r: f32, n: i32) -> Result<Rgba> {
        // Two vectors of length r, orthogonal to d and to each other.
        d.normalize();
        let dx = if d.x.abs() > 0.707 { (d % V3f::new(0.0, 1.0, 0.0)).normalized() * r } else { (d % V3f::new(1.0, 0.0, 0.0)).normalized() * r };
        let dy = (d % dx).normalized() * r;

        let (mut wt, mut cr, mut cg, mut cb, mut ca) = (0f32, 0f32, 0f32, 0f32, 0f32);
        let taps = |i: i32| (((2 * i64::from(i) + 2) as f32) / ((i64::from(n) + 1) as f32)) - 1.0;
        for y in 0..n {
            let ry = taps(y);
            let wy = 1.0 - ry.abs();
            let ddy = dy * ry;
            for x in 0..n {
                let rx = taps(x);
                let wx = 1.0 - rx.abs();
                let ddx = dx * rx;

                let s = self.sample(self.to_pos(d + ddx + ddy))?;

                let w = wx * wy;
                wt += w;
                cr = add(cr, s.r.to_f32() * w);
                cg = add(cg, s.g.to_f32() * w);
                cb = add(cb, s.b.to_f32() * w);
                ca = add(ca, s.a.to_f32() * w);
            }
        }

        let wt = 1.0 / wt;
        Ok(Rgba::new(Half::from_f32(cr * wt), Half::from_f32(cg * wt), Half::from_f32(cb * wt), Half::from_f32(ca * wt)))
    }

    /// `sample`: bilinear interpolation between the four pixels nearest `pos`, clamped to the window.
    fn sample(&self, pos: V2f) -> Result<Rgba> {
        let (w, dw) = (self.w(), &self.dw);
        // The window's span fits `i32`, and `x1 + 1` wraps only for a position no window reaches.
        let x1 = fun::floor(pos.x);
        let x2 = x1.wrapping_add(1);
        let sx = x2 as f32 - pos.x;
        let tx = 1.0 - sx;
        let (x1, x2) = (fun::clamp(x1, dw.min.x, dw.max.x) - dw.min.x, fun::clamp(x2, dw.min.x, dw.max.x) - dw.min.x);

        let y1 = fun::floor(pos.y);
        let y2 = y1.wrapping_add(1);
        let sy = y2 as f32 - pos.y;
        let ty = 1.0 - sy;
        let (y1, y2) = (fun::clamp(y1, dw.min.y, dw.max.y) - dw.min.y, fun::clamp(y2, dw.min.y, dw.max.y) - dw.min.y);

        // Clamped to the window, so always inside `px`; checked all the same.
        let at = |x: i32, y: i32| {
            let i = usize::try_from(y).ok().zip(usize::try_from(x).ok()).and_then(|(y, x)| y.checked_mul(w)?.checked_add(x));
            i.and_then(|i| self.px.get(i)).copied().ok_or_else(|| ExrError::Logic(format!("sample ({x}, {y}) is outside the image")))
        };
        let (p11, p12, p21, p22) = (at(x1, y1)?, at(x2, y1)?, at(x1, y2)?, at(x2, y2)?);
        let mix = |a: Half, b: Half, c: Half, d: Half| Half::from_f32((a.to_f32() * sx + b.to_f32() * tx) * sy + (c.to_f32() * sx + d.to_f32() * tx) * ty);
        Ok(Rgba::new(mix(p11.r, p12.r, p21.r, p22.r), mix(p11.g, p12.g, p21.g, p22.g), mix(p11.b, p12.b, p21.b, p22.b), mix(p11.a, p12.a, p21.a, p22.a)))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use imath_rs::V2i;

    fn dw(w: i32, h: i32) -> Box2i {
        Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: w - 1, y: h - 1 } }
    }

    fn filled(ty: Envmap, w: i32, h: i32, v: f32) -> EnvmapImage {
        let mut im = EnvmapImage::new();
        im.resize(ty, &dw(w, h)).expect("resize");
        im.px_mut().fill(Rgba::from_f32(v, v * 2.0, v * 3.0, 1.0));
        im
    }

    /// Hand-computed ImfEnvmap values: the centre of a lat-long map looks along +Z, and the face
    /// centres of a cube map along their axes.
    #[test]
    fn geometry_hand_values() {
        let d = lat_long_map::direction(dw(101, 51), V2f { x: 50.0, y: 25.0 });
        assert!((d.z - 1.0).abs() < 1e-6 && d.x.abs() < 1e-6 && d.y.abs() < 1e-6, "{d:?}");
        let cube = dw(9, 54);
        let axes = [(0, V3f::new(1.0, 0.0, 0.0)), (1, V3f::new(-1.0, 0.0, 0.0)), (2, V3f::new(0.0, 1.0, 0.0)), (3, V3f::new(0.0, -1.0, 0.0)), (4, V3f::new(0.0, 0.0, 1.0)), (5, V3f::new(0.0, 0.0, -1.0))];
        for (f, want) in axes {
            let got = cube_map::direction(FACES[f], cube, V2f { x: 4.0, y: 4.0 });
            assert_eq!((got.x, got.y, got.z), (want.x, want.y, want.z), "face {f}");
        }
    }

    /// direction -> pixel position -> direction is the identity at every pixel, both map kinds.
    #[test]
    fn geometry_round_trips() {
        let ll = dw(64, 32);
        for y in 0..32 {
            for x in 0..64 {
                let p = V2f { x: x as f32, y: y as f32 };
                let q = lat_long_map::pixel_position_from_dir(ll, lat_long_map::direction(ll, p));
                // The poles collapse every longitude onto one point, and the first and last columns
                // are the same meridian (+-pi): only latitude must agree there.
                let same_x = (q.x - p.x).abs() < 2e-3 || (x % 63 == 0 && (q.x < 2e-3 || q.x > 63.0 - 2e-3));
                assert!((q.y - p.y).abs() < 2e-3 && (same_x || y == 0 || y == 31), "({x}, {y}) -> {q:?}");
            }
        }
        let cube = dw(8, 48);
        for face in FACES {
            for y in 0..8 {
                for x in 0..8 {
                    let p = V2f { x: x as f32, y: y as f32 };
                    let (f2, q) = cube_map::face_and_pixel_position(cube_map::direction(face, cube, p), cube);
                    // An edge or corner pixel belongs to several faces: the same direction is what counts.
                    let (a, b) = (cube_map::direction(face, cube, p).normalized(), cube_map::direction(f2, cube, q).normalized());
                    assert!((a - b).length() < 1e-5, "{face:?} ({x}, {y}) -> {f2:?} {q:?}");
                }
            }
        }
    }

    /// A filtered lookup of a constant map is that constant, for any direction and filter.
    #[test]
    fn look_of_a_constant_map_is_the_constant() {
        for ty in [Envmap::LatLong, Envmap::Cube] {
            let im = filled(ty, if ty == Envmap::Cube { 8 } else { 32 }, if ty == Envmap::Cube { 48 } else { 16 }, 0.5);
            for d in [V3f::new(0.3, -0.8, 0.5), V3f::new(1.0, 0.0, 0.0), V3f::new(0.0, 1.0, 0.0), V3f::new(-0.2, 0.1, -1.0)] {
                for (r, n) in [(0.0, 1), (0.1, 5), (0.7, 3)] {
                    let p = im.look(d, r, n).expect("look");
                    assert_eq!((p.r.to_f32(), p.g.to_f32(), p.b.to_f32(), p.a.to_f32()), (0.5, 1.0, 1.5, 1.0), "{ty:?} {d:?} {r} {n}");
                }
            }
        }
    }

    /// Bilinear interpolation between four pixels, clamped at the window's edge.
    #[test]
    fn sample_interpolates_and_clamps() {
        let mut im = EnvmapImage::new();
        im.resize(Envmap::LatLong, &dw(2, 2)).expect("resize");
        for (i, v) in [0.0, 1.0, 2.0, 3.0].into_iter().enumerate() {
            im.px_mut()[i] = Rgba::from_f32(v, v, v, v);
        }
        let s = |x: f32, y: f32| im.sample(V2f { x, y }).expect("sample").r.to_f32();
        assert_eq!((s(0.0, 0.0), s(1.0, 0.0), s(0.0, 1.0), s(1.0, 1.0)), (0.0, 1.0, 2.0, 3.0));
        assert_eq!(s(0.5, 0.5), 1.5);
        assert_eq!(s(-7.0, 0.0), 0.0, "clamped left of the window");
        assert_eq!(s(9.0, 9.0), 3.0, "clamped beyond the corner");
    }

    #[test]
    fn resize_reuses_storage_and_blanks() {
        let mut im = filled(Envmap::LatLong, 16, 16, 1.0);
        let cap = im.px.capacity();
        im.resize(Envmap::Cube, &dw(4, 24)).expect("resize");
        assert!(im.px.capacity() >= cap && im.px().iter().all(|p| p.r.to_f32() == 0.0) && (im.w(), im.h(), im.ty()) == (4, 24, Envmap::Cube));
        assert!(im.resize(Envmap::Cube, &Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: -2, y: 0 } }).is_err());
    }
}
