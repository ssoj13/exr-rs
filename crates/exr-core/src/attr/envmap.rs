//! Port of OpenEXR `ImfEnvmap.h/.cpp` and `ImfEnvmapAttribute.*`.
//!
//! Defines the [`Envmap`] enum and [`CubeMapFace`] enum, plus the geometry
//! helpers from the C++ `LatLongMap` and `CubeMap` namespaces, and the
//! 1-byte wire serialisation.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfEnvmap.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfEnvmap.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfEnvmapAttribute.cpp>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::Result;
use crate::io::stream::{IStream, OStream};


// ---------------------------------------------------------------------------
// Envmap enum — mirrors `enum Envmap` in ImfEnvmap.h
// ---------------------------------------------------------------------------

/// Supported environment-map projections.
///
/// Discriminants match the C++ enum exactly.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Envmap {
    /// Latitude-longitude environment map (codec id 0).
    LatLong = 0,
    /// Cube-map environment map (codec id 1).
    Cube = 1,
}

impl Envmap {
    /// Construct from the raw `u8` wire value.
    ///
    /// Returns `None` for any value `>= EXR_ENVMAP_LAST_TYPE` (2). The
    /// attribute reader ([`read_value`][AttrValue::read_value]) now rejects
    /// such values before they reach the header map, matching C++ 3.4
    /// core `extract_attr_uint8(.., EXR_ENVMAP_LAST_TYPE)`
    /// (`OpenEXRCore/parse_header.c`), rather than silently defaulting an
    /// out-of-range byte to `LatLong`.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::LatLong),
            1 => Some(Self::Cube),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CubeMapFace — mirrors `enum CubeMapFace` in ImfEnvmap.h
// ---------------------------------------------------------------------------

/// The six faces of a cube map, in the order they appear in the image strip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeMapFace {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

// ---------------------------------------------------------------------------
// LatLongMap helpers — port of `namespace LatLongMap` in ImfEnvmap.cpp
// ---------------------------------------------------------------------------

/// Geometry helpers for latitude-longitude environment maps.
///
/// Mirrors the C++ `namespace LatLongMap` in `ImfEnvmap.cpp`.
pub mod lat_long_map {
    use imath_rs::{Box2i, Vec2, Vec3};

    /// Convert a 3-D direction vector to (latitude, longitude) in radians.
    ///
    /// C++: `V2f LatLongMap::latLong(const V3f& dir)`
    ///
    /// Uses Imath's `V3f::length()` (denormal-safe `lengthTiny` fallback) and
    /// `sign()` (0 for zero *and* NaN) rather than a plain `sqrt`/`signum`,
    /// matching C++ exactly for denormal-range or NaN directions:
    /// `f32::signum` instead propagates NaN and never returns 0.
    pub fn lat_long_from_dir(dir: Vec3<f32>) -> Vec2<f32> {
        let r = (dir.z * dir.z + dir.x * dir.x).sqrt();
        let len = dir.length();
        let latitude = if r < dir.y.abs() {
            // Use acos formula
            (r / len).acos() * imath_rs::fun::sign(dir.y) as f32
        } else {
            (dir.y / len).asin()
        };
        let longitude = if dir.z == 0.0 && dir.x == 0.0 {
            0.0
        } else {
            dir.x.atan2(dir.z)
        };
        Vec2 { x: latitude, y: longitude }
    }

    /// Convert a pixel position to (latitude, longitude).
    ///
    /// C++: `V2f LatLongMap::latLong(const Box2i& dw, const V2f& pixelPos)`
    pub fn lat_long_from_pixel(data_window: Box2i, pixel_pos: Vec2<f32>) -> Vec2<f32> {
        let latitude = if data_window.max.y > data_window.min.y {
            -std::f32::consts::PI
                * ((pixel_pos.y - data_window.min.y as f32)
                    / (data_window.max.y - data_window.min.y) as f32
                    - 0.5)
        } else {
            0.0
        };
        let longitude = if data_window.max.x > data_window.min.x {
            -2.0 * std::f32::consts::PI
                * ((pixel_pos.x - data_window.min.x as f32)
                    / (data_window.max.x - data_window.min.x) as f32
                    - 0.5)
        } else {
            0.0
        };
        Vec2 { x: latitude, y: longitude }
    }

    /// Convert (latitude, longitude) to a pixel position.
    ///
    /// C++: `V2f LatLongMap::pixelPosition(const Box2i& dw, const V2f& ll)`
    pub fn pixel_position_from_ll(data_window: Box2i, ll: Vec2<f32>) -> Vec2<f32> {
        let x = ll.y / (-2.0 * std::f32::consts::PI) + 0.5;
        let y = ll.x / (-std::f32::consts::PI) + 0.5;
        Vec2 {
            x: x * (data_window.max.x - data_window.min.x) as f32 + data_window.min.x as f32,
            y: y * (data_window.max.y - data_window.min.y) as f32 + data_window.min.y as f32,
        }
    }

    /// Convert a 3-D direction to a pixel position.
    ///
    /// C++: `V2f LatLongMap::pixelPosition(const Box2i& dw, const V3f& dir)`
    pub fn pixel_position_from_dir(data_window: Box2i, dir: Vec3<f32>) -> Vec2<f32> {
        pixel_position_from_ll(data_window, lat_long_from_dir(dir))
    }

    /// Convert a pixel position to a 3-D direction.
    ///
    /// C++: `V3f LatLongMap::direction(const Box2i& dw, const V2f& pixelPos)`
    pub fn direction(data_window: Box2i, pixel_pos: Vec2<f32>) -> Vec3<f32> {
        let ll = lat_long_from_pixel(data_window, pixel_pos);
        Vec3 {
            x: ll.y.sin() * ll.x.cos(),
            y: ll.x.sin(),
            z: ll.y.cos() * ll.x.cos(),
        }
    }
}

// ---------------------------------------------------------------------------
// CubeMap helpers — port of `namespace CubeMap` in ImfEnvmap.cpp
// ---------------------------------------------------------------------------

/// Geometry helpers for cube-map environment maps.
///
/// Mirrors the C++ `namespace CubeMap` in `ImfEnvmap.cpp`.
pub mod cube_map {
    use super::CubeMapFace;
    use imath_rs::{Box2i, Vec2, Vec3};

    /// Width/height of one cube face in pixels.
    ///
    /// C++: `int CubeMap::sizeOfFace(const Box2i& dw)`
    pub fn size_of_face(data_window: Box2i) -> i32 {
        // Widen the extents to i64 so an extreme window can't panic (debug) /
        // wrap (release); for a valid window the min fits i32, so the narrowed
        // result is identical to the C++ `int` computation.
        let w = data_window.max.x as i64 - data_window.min.x as i64 + 1;
        let h = (data_window.max.y as i64 - data_window.min.y as i64 + 1) / 6;
        w.min(h) as i32
    }

    /// The sub-window (in environment-map pixel coords) for `face`.
    ///
    /// C++: `Box2i CubeMap::dataWindowForFace(CubeMapFace, const Box2i& dw)`
    pub fn data_window_for_face(face: CubeMapFace, data_window: Box2i) -> Box2i {
        let sof = size_of_face(data_window);
        let min_y = face as i32 * sof;
        Box2i {
            min: Vec2 { x: 0, y: min_y },
            max: Vec2 { x: sof - 1, y: min_y + sof - 1 },
        }
    }

    /// Convert a position within a face to environment-map pixel coords.
    ///
    /// C++: `V2f CubeMap::pixelPosition(CubeMapFace, const Box2i& dw, V2f pif)`
    pub fn pixel_position(
        face: CubeMapFace,
        data_window: Box2i,
        pos_in_face: Vec2<f32>,
    ) -> Vec2<f32> {
        let dwf = data_window_for_face(face, data_window);
        match face {
            CubeMapFace::PosX => Vec2 {
                x: dwf.min.x as f32 + pos_in_face.y,
                y: dwf.max.y as f32 - pos_in_face.x,
            },
            CubeMapFace::NegX => Vec2 {
                x: dwf.max.x as f32 - pos_in_face.y,
                y: dwf.max.y as f32 - pos_in_face.x,
            },
            CubeMapFace::PosY => Vec2 {
                x: dwf.min.x as f32 + pos_in_face.x,
                y: dwf.max.y as f32 - pos_in_face.y,
            },
            CubeMapFace::NegY => Vec2 {
                x: dwf.min.x as f32 + pos_in_face.x,
                y: dwf.min.y as f32 + pos_in_face.y,
            },
            CubeMapFace::PosZ => Vec2 {
                x: dwf.max.x as f32 - pos_in_face.x,
                y: dwf.max.y as f32 - pos_in_face.y,
            },
            CubeMapFace::NegZ => Vec2 {
                x: dwf.min.x as f32 + pos_in_face.x,
                y: dwf.max.y as f32 - pos_in_face.y,
            },
        }
    }

    /// Convert a 3-D direction to a face + position within that face.
    ///
    /// C++: `void CubeMap::faceAndPixelPosition(const V3f& dir, ...)`
    pub fn face_and_pixel_position(
        dir: Vec3<f32>,
        data_window: Box2i,
    ) -> (CubeMapFace, Vec2<f32>) {
        let sof = size_of_face(data_window);
        let abs_x = dir.x.abs();
        let abs_y = dir.y.abs();
        let abs_z = dir.z.abs();

        if abs_x >= abs_y && abs_x >= abs_z {
            if abs_x == 0.0 {
                return (CubeMapFace::PosX, Vec2 { x: 0.0, y: 0.0 });
            }
            let pif = Vec2 {
                x: (dir.y / abs_x + 1.0) / 2.0 * (sof - 1) as f32,
                y: (dir.z / abs_x + 1.0) / 2.0 * (sof - 1) as f32,
            };
            let face = if dir.x > 0.0 { CubeMapFace::PosX } else { CubeMapFace::NegX };
            (face, pif)
        } else if abs_y >= abs_z {
            let pif = Vec2 {
                x: (dir.x / abs_y + 1.0) / 2.0 * (sof - 1) as f32,
                y: (dir.z / abs_y + 1.0) / 2.0 * (sof - 1) as f32,
            };
            let face = if dir.y > 0.0 { CubeMapFace::PosY } else { CubeMapFace::NegY };
            (face, pif)
        } else {
            let pif = Vec2 {
                x: (dir.x / abs_z + 1.0) / 2.0 * (sof - 1) as f32,
                y: (dir.y / abs_z + 1.0) / 2.0 * (sof - 1) as f32,
            };
            let face = if dir.z > 0.0 { CubeMapFace::PosZ } else { CubeMapFace::NegZ };
            (face, pif)
        }
    }

    /// Convert a face + position within that face to a 3-D direction.
    ///
    /// C++: `V3f CubeMap::direction(CubeMapFace, const Box2i& dw, const V2f& pif)`
    pub fn direction(
        face: CubeMapFace,
        data_window: Box2i,
        pos_in_face: Vec2<f32>,
    ) -> Vec3<f32> {
        let sof = size_of_face(data_window);
        let pos = if sof > 1 {
            Vec2 {
                x: pos_in_face.x / (sof - 1) as f32 * 2.0 - 1.0,
                y: pos_in_face.y / (sof - 1) as f32 * 2.0 - 1.0,
            }
        } else {
            Vec2 { x: 0.0, y: 0.0 }
        };

        match face {
            CubeMapFace::PosX => Vec3 { x: 1.0, y: pos.x, z: pos.y },
            CubeMapFace::NegX => Vec3 { x: -1.0, y: pos.x, z: pos.y },
            CubeMapFace::PosY => Vec3 { x: pos.x, y: 1.0, z: pos.y },
            CubeMapFace::NegY => Vec3 { x: pos.x, y: -1.0, z: pos.y },
            CubeMapFace::PosZ => Vec3 { x: pos.x, y: pos.y, z: 1.0 },
            CubeMapFace::NegZ => Vec3 { x: pos.x, y: pos.y, z: -1.0 },
        }
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (1 byte on disk)
// ---------------------------------------------------------------------------

impl AttrValue for Envmap {
    fn static_type_name() -> &'static str {
        // ImfEnvmapAttribute.cpp: return "envmap";
        "envmap"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: unsigned char tmp = _value; Xdr::write<StreamIO>(os, tmp);
        xdr::write_u8_val(o, *self as u8)
    }

    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        // C++ 3.4 core `extract_attr_uint8(.., EXR_ENVMAP_LAST_TYPE=2)`
        // (`OpenEXRCore/parse_header.c`): `attrsz != 1` -> size-mismatch error,
        // `value >= 2` -> invalid-attr error. Both fail the attribute so the
        // header reader's non-strict loop drops it instead of
        // relabelling an ordinary image as a lat-long environment map.
        if size != 1 {
            return Err(crate::error::ExrError::Input(format!(
                "invalid size {size} for envmap attribute (expected 1)"
            )));
        }
        let raw = xdr::read_u8_val(i)?;
        Self::from_u8(raw).ok_or_else(|| {
            crate::error::ExrError::Input(format!(
                "envmap value {raw} is out of range (expected 0 or 1)"
            ))
        })
    }
}

/// `typedef TypedAttribute<Envmap> EnvmapAttribute;`
pub type EnvmapAttribute = crate::attr::attribute::TypedAttribute<Envmap>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(e: Envmap) -> Vec<u8> {
        let mut o = MemOStream::new();
        e.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(Envmap::static_type_name(), "envmap");
    }

    #[test]
    fn exact_bytes() {
        assert_eq!(write_bytes(Envmap::LatLong), vec![0]);
        assert_eq!(write_bytes(Envmap::Cube), vec![1]);
    }

    #[test]
    fn round_trip() {
        for raw in 0u8..2 {
            let e = Envmap::from_u8(raw).unwrap();
            let mut i = MemIStream::new(vec![raw]);
            let back = Envmap::read_value(&mut i, 1, 0).unwrap();
            assert_eq!(back, e);
        }
    }

    /// An out-of-range raw value must be rejected, not silently
    /// remapped to `LatLong`.
    #[test]
    fn read_value_rejects_out_of_range() {
        let mut i = MemIStream::new(vec![2u8]);
        assert!(Envmap::read_value(&mut i, 1, 0).is_err());
    }

    /// A declared size other than 1 must be rejected.
    #[test]
    fn read_value_rejects_wrong_size() {
        let mut i = MemIStream::new(vec![0u8, 0u8]);
        assert!(Envmap::read_value(&mut i, 2, 0).is_err());
    }

    /// An extreme (public) window must not panic when computing the face size;
    /// for a valid window the narrowed result matches the C++ `int` computation.
    #[test]
    fn size_of_face_no_panic_on_extreme_window() {
        use imath_rs::{Box2i, Vec2};
        // Full i32 span: the old i32 subtraction would overflow/panic in debug.
        let dw = Box2i {
            min: Vec2 { x: i32::MIN, y: i32::MIN },
            max: Vec2 { x: i32::MAX, y: i32::MAX },
        };
        // w = 2^32, h = 2^32 / 6 = 715_827_882, min fits i32.
        assert_eq!(super::cube_map::size_of_face(dw), 715_827_882);

        // A normal cube window (6 stacked 64x64 faces) is unchanged.
        let dw = Box2i {
            min: Vec2 { x: 0, y: 0 },
            max: Vec2 { x: 63, y: 6 * 64 - 1 },
        };
        assert_eq!(super::cube_map::size_of_face(dw), 64);
    }
}
