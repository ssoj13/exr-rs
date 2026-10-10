//! Port of OpenEXR `ImfKeyCode.h/.cpp` and `ImfKeyCodeAttribute.*`.
//!
//! Defines [`KeyCode`] — a motion-picture film frame identifier — with field
//! validation matching the C++ setters, and a 28-byte on-disk layout
//! (7 × `i32` LE).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfKeyCode.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfKeyCode.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfKeyCodeAttribute.cpp>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// KeyCode — mirrors `class KeyCode` in ImfKeyCode.h
// ---------------------------------------------------------------------------

/// Motion-picture film frame identifier (SMPTE 254/268M/270/271).
///
/// All seven fields are stored as `i32` on disk (LE), for a total of 28 bytes.
///
/// Field validation mirrors the C++ setters in `ImfKeyCode.cpp`:
/// - `film_mfc_code`: 0–99
/// - `film_type`: 0–99
/// - `prefix`: 0–999 999
/// - `count`: 0–9 999
/// - `perf_offset`: 0–119
/// - `perfs_per_frame`: 1–15
/// - `perfs_per_count`: 20–120
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyCode {
    film_mfc_code:   i32,
    film_type:       i32,
    prefix:          i32,
    count:           i32,
    perf_offset:     i32,
    perfs_per_frame: i32,
    perfs_per_count: i32,
}

impl KeyCode {
    /// Construct with validation — mirrors `KeyCode(...)` in `ImfKeyCode.cpp`.
    ///
    /// C++ defaults: `filmMfcCode=0, filmType=0, prefix=0, count=0,
    /// perfOffset=0, perfsPerFrame=4, perfsPerCount=64`.
    pub fn new(
        film_mfc_code:   i32,
        film_type:       i32,
        prefix:          i32,
        count:           i32,
        perf_offset:     i32,
        perfs_per_frame: i32,
        perfs_per_count: i32,
    ) -> Result<Self> {
        let mut kc = Self {
            film_mfc_code:   0,
            film_type:       0,
            prefix:          0,
            count:           0,
            perf_offset:     0,
            perfs_per_frame: 4,
            perfs_per_count: 64,
        };
        kc.set_film_mfc_code(film_mfc_code)?;
        kc.set_film_type(film_type)?;
        kc.set_prefix(prefix)?;
        kc.set_count(count)?;
        kc.set_perf_offset(perf_offset)?;
        kc.set_perfs_per_frame(perfs_per_frame)?;
        kc.set_perfs_per_count(perfs_per_count)?;
        Ok(kc)
    }

    // -- Accessors ---------------------------------------------------------

    /// Film manufacturer code (0–99).
    pub fn film_mfc_code(&self) -> i32 { self.film_mfc_code }
    /// Film type code (0–99).
    pub fn film_type(&self) -> i32 { self.film_type }
    /// Prefix identifying the film roll (0–999 999).
    pub fn prefix(&self) -> i32 { self.prefix }
    /// Count (0–9 999).
    pub fn count(&self) -> i32 { self.count }
    /// Perforation offset within the count (0–119).
    pub fn perf_offset(&self) -> i32 { self.perf_offset }
    /// Perforations per frame (1–15).
    pub fn perfs_per_frame(&self) -> i32 { self.perfs_per_frame }
    /// Perforations per count (20–120).
    pub fn perfs_per_count(&self) -> i32 { self.perfs_per_count }

    // -- Setters with validation (mirrors C++ set*() methods) -------------

    /// Set film manufacturer code; range 0–99.
    pub fn set_film_mfc_code(&mut self, v: i32) -> Result<()> {
        if !(0..=99).contains(&v) {
            return Err(ExrError::Arg(
                "Invalid key code film manufacturer code (must be between 0 and 99).".into(),
            ));
        }
        self.film_mfc_code = v;
        Ok(())
    }

    /// Set film type code; range 0–99.
    pub fn set_film_type(&mut self, v: i32) -> Result<()> {
        if !(0..=99).contains(&v) {
            return Err(ExrError::Arg(
                "Invalid key code film type (must be between 0 and 99).".into(),
            ));
        }
        self.film_type = v;
        Ok(())
    }

    /// Set prefix; range 0–999 999.
    pub fn set_prefix(&mut self, v: i32) -> Result<()> {
        if !(0..=999_999).contains(&v) {
            return Err(ExrError::Arg(
                "Invalid key code prefix (must be between 0 and 999999).".into(),
            ));
        }
        self.prefix = v;
        Ok(())
    }

    /// Set count; range 0–9 999.
    pub fn set_count(&mut self, v: i32) -> Result<()> {
        if !(0..=9_999).contains(&v) {
            return Err(ExrError::Arg(
                "Invalid key code count (must be between 0 and 9999).".into(),
            ));
        }
        self.count = v;
        Ok(())
    }

    /// Set perforation offset; range 0–119.
    pub fn set_perf_offset(&mut self, v: i32) -> Result<()> {
        if !(0..=119).contains(&v) {
            return Err(ExrError::Arg(
                "Invalid key code perforation offset (must be between 0 and 119).".into(),
            ));
        }
        self.perf_offset = v;
        Ok(())
    }

    /// Set perforations per frame; range 1–15.
    pub fn set_perfs_per_frame(&mut self, v: i32) -> Result<()> {
        if !(1..=15).contains(&v) {
            return Err(ExrError::Arg(
                "Invalid key code number of perforations per frame (must be between 1 and 15)."
                    .into(),
            ));
        }
        self.perfs_per_frame = v;
        Ok(())
    }

    /// Set perforations per count; range 20–120.
    pub fn set_perfs_per_count(&mut self, v: i32) -> Result<()> {
        if !(20..=120).contains(&v) {
            return Err(ExrError::Arg(
                "Invalid key code number of perforations per count (must be between 20 and 120)."
                    .into(),
            ));
        }
        self.perfs_per_count = v;
        Ok(())
    }
}

impl Default for KeyCode {
    /// C++ defaults: all zeros except `perfsPerFrame=4`, `perfsPerCount=64`.
    fn default() -> Self {
        Self {
            film_mfc_code:   0,
            film_type:       0,
            prefix:          0,
            count:           0,
            perf_offset:     0,
            perfs_per_frame: 4,
            perfs_per_count: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (7 × i32 = 28 bytes on disk)
// ---------------------------------------------------------------------------

impl AttrValue for KeyCode {
    fn static_type_name() -> &'static str {
        // ImfKeyCodeAttribute.cpp: return "keycode";
        "keycode"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: Xdr::write(os, filmMfcCode()); ... Xdr::write(os, perfsPerCount());
        xdr::write_i32(o, self.film_mfc_code)?;
        xdr::write_i32(o, self.film_type)?;
        xdr::write_i32(o, self.prefix)?;
        xdr::write_i32(o, self.count)?;
        xdr::write_i32(o, self.perf_offset)?;
        xdr::write_i32(o, self.perfs_per_frame)?;
        xdr::write_i32(o, self.perfs_per_count)
    }

    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        // C++: int tmp; Xdr::read(is, tmp); _value.setFilmMfcCode(tmp); ...
        let mut kc = KeyCode::default();
        kc.set_film_mfc_code(xdr::read_i32(i)?)?;
        kc.set_film_type(xdr::read_i32(i)?)?;
        kc.set_prefix(xdr::read_i32(i)?)?;
        kc.set_count(xdr::read_i32(i)?)?;
        kc.set_perf_offset(xdr::read_i32(i)?)?;
        kc.set_perfs_per_frame(xdr::read_i32(i)?)?;
        kc.set_perfs_per_count(xdr::read_i32(i)?)?;
        Ok(kc)
    }
}

/// `typedef TypedAttribute<KeyCode> KeyCodeAttribute;`
pub type KeyCodeAttribute = crate::attr::attribute::TypedAttribute<KeyCode>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(kc: &KeyCode) -> Vec<u8> {
        let mut o = MemOStream::new();
        kc.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(KeyCode::static_type_name(), "keycode");
    }

    #[test]
    fn size_is_28_bytes() {
        assert_eq!(write_bytes(&KeyCode::default()).len(), 28);
    }

    #[test]
    fn round_trip() {
        let kc = KeyCode::new(1, 2, 12345, 9999, 100, 4, 64).unwrap();
        let bytes = write_bytes(&kc);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = KeyCode::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, kc);
    }

    #[test]
    fn validation_errors() {
        assert!(KeyCode::new(100, 0, 0, 0, 0, 4, 64).is_err()); // film_mfc_code > 99
        assert!(KeyCode::new(0, 0, 0, 0, 0, 0, 64).is_err());   // perfs_per_frame = 0
        assert!(KeyCode::new(0, 0, 0, 0, 0, 4, 19).is_err());   // perfs_per_count < 20
    }
}
