//! Port of OpenEXR `ImfStandardAttributes.h` / `ImfStandardAttributes.cpp`.
//!
//! These are the *optional standard* attributes — not every header has them,
//! but they define the standard wire representation for commonly used metadata
//! (chromaticities, white luminance, camera/lens info, time/key codes, etc.).
//!
//! In C++ each attribute is declared with the `IMF_STD_ATTRIBUTE_DEF(name,
//! suffix, object)` macro, which expands to six free functions:
//!
//! ```cpp
//! void                       addFoo  (Header&, const T&);
//! bool                       hasFoo  (const Header&);
//! const TypedAttribute<T>&   fooAttribute (const Header&);
//! TypedAttribute<T>&         fooAttribute (Header&);
//! const T&                   foo     (const Header&);
//! T&                         foo     (Header&);
//! ```
//!
//! This port reproduces that exactly via the `std_attribute` macro, but emits
//! the accessors as inherent methods on [`Header`] (Rust's natural equivalent
//! of the C++ free functions that take a `Header&` first argument):
//!
//! | C++                  | Rust (`Header` method)            |
//! |----------------------|-----------------------------------|
//! | `addFoo(h, v)`       | `h.add_foo(v)`                    |
//! | `hasFoo(h)`          | `h.has_foo()`                     |
//! | `foo(h)` (const)     | `h.foo()` → value (copy/borrow)   |
//! | `foo(h)` (mutable)   | `h.foo_mut()` → `&mut T`          |
//!
//! The on-disk attribute **name** and value **type** are exactly the C++ ones,
//! so files written/read through these helpers are byte-identical to IlmImf.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfStandardAttributes.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfStandardAttributes.cpp>

use imath_rs::{Box2i, M44f, V2f};

use crate::attr::{
    Chromaticities, CompressedIDManifest, DeepImageState, Envmap, ExrString, KeyCode, Rational,
    TimeCode, TypedAttribute,
};
use crate::error::Result;
use crate::header::Header;

/// Generate the six C++ `IMF_STD_ATTRIBUTE_DEF` accessors for one standard
/// attribute, as inherent methods on [`Header`].
///
/// * `$add` — `add<Suffix>(value)`  → `Header::insert`.
/// * `$has` — `has<Suffix>()`       → `findTypedAttribute<T>() != 0`.
/// * `$get` — value accessor        → `typedAttribute<T>().value()`.
/// * `$get_mut` — mutable accessor  → `&mut typedAttribute<T>().value()`.
///
/// `$attr_name` is the **exact** on-disk attribute name; `$ty` is its value
/// type (which must implement [`AttrValue`][crate::attr::AttrValue]).
macro_rules! std_attribute {
    (
        $(#[$meta:meta])*
        $attr_name:literal, $ty:ty, $add:ident, $has:ident, $get:ident, $get_mut:ident
    ) => {
        $(#[$meta])*
        impl Header {
            #[doc = concat!("`add` for the standard `", $attr_name, "` attribute (C++ `add", stringify!($add), "`).")]
            pub fn $add(&mut self, v: $ty) -> Result<()> {
                self.insert($attr_name, TypedAttribute::<$ty>::new(v))
            }

            #[doc = concat!("`has` for the standard `", $attr_name, "` attribute (C++ `has", stringify!($has), "`).")]
            pub fn $has(&self) -> bool {
                self.find_typed_attribute::<$ty>($attr_name).is_some()
            }

            #[doc = concat!("Value accessor for the standard `", $attr_name, "` attribute (C++ `", stringify!($get), "`).")]
            pub fn $get(&self) -> Result<&$ty> {
                Ok(&self.typed_attribute::<$ty>($attr_name)?.value)
            }

            #[doc = concat!("Mutable value accessor for the standard `", $attr_name, "` attribute.")]
            pub fn $get_mut(&mut self) -> Result<&mut $ty> {
                Ok(&mut self.typed_attribute_mut::<$ty>($attr_name)?.value)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// One std_attribute! per IMF_STD_ATTRIBUTE_DEF in ImfStandardAttributes.h.
// The attribute name + value type are taken verbatim from the C++ macro args
// (the C++ `Imath::Box2i`/`M44f`/`V2f` map to imath-rs `Box2i`/`M44f`/`V2f`;
// `std::string`→`ExrString`; `StringVector`→`Vec<ExrString>`).
// ---------------------------------------------------------------------------

std_attribute!("originalDataWindow", Box2i, add_original_data_window, has_original_data_window, original_data_window, original_data_window_mut);
std_attribute!("worldToCamera", M44f, add_world_to_camera, has_world_to_camera, world_to_camera, world_to_camera_mut);
std_attribute!("worldToNDC", M44f, add_world_to_ndc, has_world_to_ndc, world_to_ndc, world_to_ndc_mut);

std_attribute!("sensorCenterOffset", V2f, add_sensor_center_offset, has_sensor_center_offset, sensor_center_offset, sensor_center_offset_mut);
std_attribute!("sensorOverallDimensions", V2f, add_sensor_overall_dimensions, has_sensor_overall_dimensions, sensor_overall_dimensions, sensor_overall_dimensions_mut);
std_attribute!("sensorPhotositePitch", f32, add_sensor_photosite_pitch, has_sensor_photosite_pitch, sensor_photosite_pitch, sensor_photosite_pitch_mut);
std_attribute!("sensorAcquisitionRectangle", Box2i, add_sensor_acquisition_rectangle, has_sensor_acquisition_rectangle, sensor_acquisition_rectangle, sensor_acquisition_rectangle_mut);

std_attribute!("ascFramingDecisionList", ExrString, add_asc_framing_decision_list, has_asc_framing_decision_list, asc_framing_decision_list, asc_framing_decision_list_mut);

std_attribute!("xDensity", f32, add_x_density, has_x_density, x_density, x_density_mut);

std_attribute!("longitude", f32, add_longitude, has_longitude, longitude, longitude_mut);
std_attribute!("latitude", f32, add_latitude, has_latitude, latitude, latitude_mut);
std_attribute!("altitude", f32, add_altitude, has_altitude, altitude, altitude_mut);

std_attribute!("cameraMake", ExrString, add_camera_make, has_camera_make, camera_make, camera_make_mut);
std_attribute!("cameraModel", ExrString, add_camera_model, has_camera_model, camera_model, camera_model_mut);
std_attribute!("cameraSerialNumber", ExrString, add_camera_serial_number, has_camera_serial_number, camera_serial_number, camera_serial_number_mut);
std_attribute!("cameraFirmwareVersion", ExrString, add_camera_firmware_version, has_camera_firmware_version, camera_firmware_version, camera_firmware_version_mut);
std_attribute!("cameraUuid", ExrString, add_camera_uuid, has_camera_uuid, camera_uuid, camera_uuid_mut);
std_attribute!("cameraLabel", ExrString, add_camera_label, has_camera_label, camera_label, camera_label_mut);
std_attribute!("cameraCCTSetting", f32, add_camera_cct_setting, has_camera_cct_setting, camera_cct_setting, camera_cct_setting_mut);
std_attribute!("cameraTintSetting", f32, add_camera_tint_setting, has_camera_tint_setting, camera_tint_setting, camera_tint_setting_mut);
std_attribute!("cameraColorBalance", V2f, add_camera_color_balance, has_camera_color_balance, camera_color_balance, camera_color_balance_mut);

std_attribute!("isoSpeed", f32, add_iso_speed, has_iso_speed, iso_speed, iso_speed_mut);
std_attribute!("expTime", f32, add_exp_time, has_exp_time, exp_time, exp_time_mut);
std_attribute!("shutterAngle", f32, add_shutter_angle, has_shutter_angle, shutter_angle, shutter_angle_mut);
std_attribute!("captureRate", Rational, add_capture_rate, has_capture_rate, capture_rate, capture_rate_mut);

std_attribute!("lensMake", ExrString, add_lens_make, has_lens_make, lens_make, lens_make_mut);
std_attribute!("lensModel", ExrString, add_lens_model, has_lens_model, lens_model, lens_model_mut);
std_attribute!("lensSerialNumber", ExrString, add_lens_serial_number, has_lens_serial_number, lens_serial_number, lens_serial_number_mut);
std_attribute!("lensFirmwareVersion", ExrString, add_lens_firmware_version, has_lens_firmware_version, lens_firmware_version, lens_firmware_version_mut);
std_attribute!("nominalFocalLength", f32, add_nominal_focal_length, has_nominal_focal_length, nominal_focal_length, nominal_focal_length_mut);
std_attribute!("pinholeFocalLength", f32, add_pinhole_focal_length, has_pinhole_focal_length, pinhole_focal_length, pinhole_focal_length_mut);
std_attribute!("effectiveFocalLength", f32, add_effective_focal_length, has_effective_focal_length, effective_focal_length, effective_focal_length_mut);
std_attribute!("entrancePupilOffset", f32, add_entrance_pupil_offset, has_entrance_pupil_offset, entrance_pupil_offset, entrance_pupil_offset_mut);
std_attribute!("aperture", f32, add_aperture, has_aperture, aperture, aperture_mut);
std_attribute!("tStop", f32, add_t_stop, has_t_stop, t_stop, t_stop_mut);
std_attribute!("focus", f32, add_focus, has_focus, focus, focus_mut);

std_attribute!("owner", ExrString, add_owner, has_owner, owner, owner_mut);
std_attribute!("comments", ExrString, add_comments, has_comments, comments, comments_mut);
std_attribute!("capDate", ExrString, add_cap_date, has_cap_date, cap_date, cap_date_mut);
std_attribute!("utcOffset", f32, add_utc_offset, has_utc_offset, utc_offset, utc_offset_mut);

std_attribute!("keyCode", KeyCode, add_key_code, has_key_code, key_code, key_code_mut);
std_attribute!("timeCode", TimeCode, add_time_code, has_time_code, time_code, time_code_mut);
std_attribute!("framesPerSecond", Rational, add_frames_per_second, has_frames_per_second, frames_per_second, frames_per_second_mut);
std_attribute!("imageCounter", i32, add_image_counter, has_image_counter, image_counter, image_counter_mut);
std_attribute!("reelName", ExrString, add_reel_name, has_reel_name, reel_name, reel_name_mut);

std_attribute!("chromaticities", Chromaticities, add_chromaticities, has_chromaticities, chromaticities, chromaticities_mut);
std_attribute!("whiteLuminance", f32, add_white_luminance, has_white_luminance, white_luminance, white_luminance_mut);
std_attribute!("adoptedNeutral", V2f, add_adopted_neutral, has_adopted_neutral, adopted_neutral, adopted_neutral_mut);

// Deprecated in OpenEXR v3.2 but still defined/read; ported for round-trip fidelity.
std_attribute!("renderingTransform", ExrString, add_rendering_transform, has_rendering_transform, rendering_transform, rendering_transform_mut);
std_attribute!("lookModTransform", ExrString, add_look_mod_transform, has_look_mod_transform, look_mod_transform, look_mod_transform_mut);
// `colorInteropID` — OpenEXR 3.4 addition
// (<https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfStandardAttributes.cpp#L136>),
// an OpenColorIO Interop Config identifier.
std_attribute!("colorInteropID", ExrString, add_color_interop_id, has_color_interop_id, color_interop_id, color_interop_id_mut);

std_attribute!("envmap", Envmap, add_envmap, has_envmap, envmap, envmap_mut);
std_attribute!("wrapmodes", ExrString, add_wrapmodes, has_wrapmodes, wrapmodes, wrapmodes_mut);
std_attribute!("multiView", Vec<ExrString>, add_multi_view, has_multi_view, multi_view, multi_view_mut);
std_attribute!("deepImageState", DeepImageState, add_deep_image_state, has_deep_image_state, deep_image_state, deep_image_state_mut);

// Deprecated alias for the header's DWA compression-level setting; the
// attribute itself still exists for files that carry it explicitly.
std_attribute!("dwaCompressionLevel", f32, add_dwa_compression_level, has_dwa_compression_level, dwa_compression_level, dwa_compression_level_mut);

// --- exr-rs EXTENSION (NOT an OpenEXR standard attribute) -------------------
// HTJ2K quantization step. OpenEXR's HTJ2K is reversible-only (lossless) and
// defines no quality attribute; this exr-rs extension lets a writer request a
// LOSSY HTJ2K encode (irreversible 9/7 wavelet + this quantization step). When
// the attribute is absent the encode stays lossless, matching stock OpenEXR.
// Files carrying it are an exr-rs / private-container extension; standard OpenEXR
// readers ignore the unknown attribute, and any HTJ2K decoder recovers the step
// from the codestream's own QCD marker regardless (so decode needs no attribute).
std_attribute!("htj2kQuantStep", f32, add_htj2k_quant_step, has_htj2k_quant_step, htj2k_quant_step, htj2k_quant_step_mut);

std_attribute!("idManifest", CompressedIDManifest, add_id_manifest, has_id_manifest, id_manifest, id_manifest_mut);

// ---------------------------------------------------------------------------
// Tests — exercise add/has/get for a representative sample of standard
// attributes (string, float, V2f, M44f, Chromaticities) and confirm the
// on-disk attribute name they use round-trips through the Header writer/reader.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    #[test]
    fn add_has_get_owner_and_comments() {
        let mut h = Header::new();
        assert!(!h.has_owner());
        h.add_owner(ExrString::from("alice")).unwrap();
        assert!(h.has_owner());
        assert_eq!(h.owner().unwrap().as_str(), Some("alice"));
        // comments is a separate attribute.
        assert!(!h.has_comments());
        h.add_comments(ExrString::from("rendered overnight")).unwrap();
        assert_eq!(h.comments().unwrap().as_str(), Some("rendered overnight"));
    }

    /// `colorInteropID` round-trips through the standard accessors.
    #[test]
    fn add_has_get_color_interop_id() {
        let mut h = Header::new();
        assert!(!h.has_color_interop_id());
        h.add_color_interop_id(ExrString::from("ocio-interop-v1")).unwrap();
        assert!(h.has_color_interop_id());
        assert_eq!(h.color_interop_id().unwrap().as_str(), Some("ocio-interop-v1"));
    }

    #[test]
    fn add_has_get_world_to_camera() {
        let mut h = Header::new();
        assert!(!h.has_world_to_camera());
        let mut m = M44f::default();
        m.x[0][3] = 5.0;
        h.add_world_to_camera(m).unwrap();
        assert!(h.has_world_to_camera());
        assert_eq!(h.world_to_camera().unwrap().x[0][3], 5.0);
    }

    #[test]
    fn add_has_get_chromaticities() {
        let mut h = Header::new();
        assert!(!h.has_chromaticities());
        let c = Chromaticities::default();
        h.add_chromaticities(c).unwrap();
        assert!(h.has_chromaticities());
        assert_eq!(h.chromaticities().unwrap().red, c.red);
        assert_eq!(h.chromaticities().unwrap().white, c.white);
    }

    #[test]
    fn add_has_get_scalar_and_vector_attrs() {
        let mut h = Header::new();
        h.add_white_luminance(120.0).unwrap();
        assert_eq!(*h.white_luminance().unwrap(), 120.0);
        h.add_adopted_neutral(V2f { x: 0.3127, y: 0.3290 }).unwrap();
        assert_eq!(h.adopted_neutral().unwrap().x, 0.3127);
    }

    #[test]
    fn mutable_accessor_updates_value() {
        let mut h = Header::new();
        h.add_x_density(72.0).unwrap();
        *h.x_density_mut().unwrap() = 300.0;
        assert_eq!(*h.x_density().unwrap(), 300.0);
    }

    #[test]
    fn standard_attribute_round_trips_through_header() {
        // Confirm the standard attribute uses its exact C++ on-disk name and
        // survives a write/read cycle.
        let mut h = Header::new();
        h.add_owner(ExrString::from("studio")).unwrap();
        h.add_chromaticities(Chromaticities::default()).unwrap();
        h.add_world_to_camera(M44f::default()).unwrap();

        let mut o = MemOStream::new();
        h.write_to(&mut o, 0).unwrap();
        let mut i = MemIStream::new(o.into_data());
        let (h2, _) = Header::read_from(&mut i).unwrap();

        assert!(h2.has_owner());
        assert_eq!(h2.owner().unwrap().as_str(), Some("studio"));
        assert!(h2.has_chromaticities());
        assert!(h2.has_world_to_camera());
        // The raw attribute names match the C++ spelling.
        assert!(h2.contains("owner"));
        assert!(h2.contains("chromaticities"));
        assert!(h2.contains("worldToCamera"));
    }
}
