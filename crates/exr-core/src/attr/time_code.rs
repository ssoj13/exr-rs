//! Port of OpenEXR `ImfTimeCode.h/.cpp` and `ImfTimeCodeAttribute.*`.
//!
//! Defines [`TimeCode`] — SMPTE 12M-1999 time and control codes — with the
//! full bit-packing logic from the C++ implementation and an 8-byte on-disk
//! layout (2 × `u32` LE).
//!
//! The internal representation mirrors C++: two packed `u32` fields `_time`
//! and `_user`.  All accessor and packing methods are 1:1 ports.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTimeCode.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTimeCode.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfTimeCodeAttribute.cpp>

use crate::attr::attribute::AttrValue;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

// ---------------------------------------------------------------------------
// Packing enum
// ---------------------------------------------------------------------------

/// Bit-packing variants for the time-and-flags word.
///
/// Mirrors `enum TimeCode::Packing` in `ImfTimeCode.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// Packing for 60-field television (default).
    Tv60,
    /// Packing for 50-field television.
    Tv50,
    /// Packing for 24-frame film.
    Film24,
}

// ---------------------------------------------------------------------------
// Bit helpers (private, mirrors anonymous namespace in ImfTimeCode.cpp)
// ---------------------------------------------------------------------------

/// Extract bits `[min_bit..=max_bit]` from `value`.
///
/// C++: `unsigned int bitField(value, minBit, maxBit)`
#[inline]
fn bit_field(value: u32, min_bit: u32, max_bit: u32) -> u32 {
    let mask = (!(!0u32 << (max_bit - min_bit + 1))) << min_bit;
    (value & mask) >> min_bit
}

/// Write `field` into `value` at bits `[min_bit..=max_bit]`.
///
/// C++: `void setBitField(value, minBit, maxBit, field)`
#[inline]
fn set_bit_field(value: &mut u32, min_bit: u32, max_bit: u32, field: u32) {
    let mask = (!(!0u32 << (max_bit - min_bit + 1))) << min_bit;
    *value = (*value & !mask) | ((field << min_bit) & mask);
}

/// BCD-encoded nibbles → binary integer.
///
/// C++: `int bcdToBinary(unsigned int bcd)`
#[inline]
fn bcd_to_binary(bcd: u32) -> i32 {
    ((bcd & 0x0f) + 10 * ((bcd >> 4) & 0x0f)) as i32
}

/// Binary integer → BCD-encoded nibbles.
///
/// C++: `unsigned int binaryToBcd(int binary)`
#[inline]
fn binary_to_bcd(binary: i32) -> u32 {
    let units = (binary % 10) as u32;
    let tens  = ((binary / 10) % 10) as u32;
    units | (tens << 4)
}

// ---------------------------------------------------------------------------
// TimeCode — mirrors `class TimeCode` in ImfTimeCode.h
// ---------------------------------------------------------------------------

/// SMPTE 12M-1999 time and control codes.
///
/// **On-disk layout (8 bytes):**
/// `timeAndFlags` (TV60 packing) as `u32` LE, then `userData` as `u32` LE.
///
/// All 31 bits of `_time` and 32 bits of `_user` are stored internally in
/// "TV60 normal" form (the same packing that `writeValueTo` uses).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeCode {
    // Internal representation — the C++ `_time` and `_user` fields.
    // `_time` is always in TV60 canonical form internally;
    // TV50 / Film24 re-packing happens on read/write via `time_and_flags()`.
    time: u32,
    user: u32,
}

impl Default for TimeCode {
    /// All fields zero / false.
    fn default() -> Self {
        Self { time: 0, user: 0 }
    }
}

impl TimeCode {
    // -- Constructors -------------------------------------------------------

    /// Construct from individual fields with validation.
    ///
    /// Mirrors `TimeCode(int hours, int minutes, ...)` in `ImfTimeCode.cpp`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_fields(
        hours:         i32,
        minutes:       i32,
        seconds:       i32,
        frame:         i32,
        drop_frame:    bool,
        color_frame:   bool,
        field_phase:   bool,
        bgf0:          bool,
        bgf1:          bool,
        bgf2:          bool,
        binary_group1: i32,
        binary_group2: i32,
        binary_group3: i32,
        binary_group4: i32,
        binary_group5: i32,
        binary_group6: i32,
        binary_group7: i32,
        binary_group8: i32,
    ) -> Result<Self> {
        let mut tc = Self::default();
        tc.set_hours(hours)?;
        tc.set_minutes(minutes)?;
        tc.set_seconds(seconds)?;
        tc.set_frame(frame)?;
        tc.set_drop_frame(drop_frame);
        tc.set_color_frame(color_frame);
        tc.set_field_phase(field_phase);
        tc.set_bgf0(bgf0);
        tc.set_bgf1(bgf1);
        tc.set_bgf2(bgf2);
        tc.set_binary_group(1, binary_group1)?;
        tc.set_binary_group(2, binary_group2)?;
        tc.set_binary_group(3, binary_group3)?;
        tc.set_binary_group(4, binary_group4)?;
        tc.set_binary_group(5, binary_group5)?;
        tc.set_binary_group(6, binary_group6)?;
        tc.set_binary_group(7, binary_group7)?;
        tc.set_binary_group(8, binary_group8)?;
        Ok(tc)
    }

    /// Construct from packed `timeAndFlags` + `userData`.
    ///
    /// Mirrors `TimeCode(unsigned int timeAndFlags, unsigned int userData,
    /// Packing packing)`.
    pub fn from_packed(time_and_flags: u32, user_data: u32, packing: Packing) -> Self {
        let mut tc = Self::default();
        tc.set_time_and_flags(time_and_flags, packing);
        tc.set_user_data(user_data);
        tc
    }

    // -- Time-address accessors ---------------------------------------------

    /// Hours field (0–23).
    pub fn hours(&self) -> i32 { bcd_to_binary(bit_field(self.time, 24, 29)) }

    /// Set hours; range 0–23.
    pub fn set_hours(&mut self, v: i32) -> Result<()> {
        if !(0..=23).contains(&v) {
            return Err(ExrError::Arg(
                "Cannot set hours field in time code. New value is out of range.".into(),
            ));
        }
        set_bit_field(&mut self.time, 24, 29, binary_to_bcd(v));
        Ok(())
    }

    /// Minutes field (0–59).
    pub fn minutes(&self) -> i32 { bcd_to_binary(bit_field(self.time, 16, 22)) }

    /// Set minutes; range 0–59.
    pub fn set_minutes(&mut self, v: i32) -> Result<()> {
        if !(0..=59).contains(&v) {
            return Err(ExrError::Arg(
                "Cannot set minutes field in time code. New value is out of range.".into(),
            ));
        }
        set_bit_field(&mut self.time, 16, 22, binary_to_bcd(v));
        Ok(())
    }

    /// Seconds field (0–59).
    pub fn seconds(&self) -> i32 { bcd_to_binary(bit_field(self.time, 8, 14)) }

    /// Set seconds; range 0–59.
    pub fn set_seconds(&mut self, v: i32) -> Result<()> {
        if !(0..=59).contains(&v) {
            return Err(ExrError::Arg(
                "Cannot set seconds field in time code. New value is out of range.".into(),
            ));
        }
        set_bit_field(&mut self.time, 8, 14, binary_to_bcd(v));
        Ok(())
    }

    /// Frame field (0–29).
    pub fn frame(&self) -> i32 { bcd_to_binary(bit_field(self.time, 0, 5)) }

    /// Set frame; range 0–29.
    pub fn set_frame(&mut self, v: i32) -> Result<()> {
        if !(0..=29).contains(&v) {
            return Err(ExrError::Arg(
                "Cannot set frame field in time code. New value is out of range.".into(),
            ));
        }
        set_bit_field(&mut self.time, 0, 5, binary_to_bcd(v));
        Ok(())
    }

    // -- Flag accessors -----------------------------------------------------

    /// Drop-frame flag (bit 6).
    pub fn drop_frame(&self) -> bool { bit_field(self.time, 6, 6) != 0 }
    /// Set drop-frame flag.
    pub fn set_drop_frame(&mut self, v: bool) {
        set_bit_field(&mut self.time, 6, 6, v as u32);
    }

    /// Color-frame flag (bit 7).
    pub fn color_frame(&self) -> bool { bit_field(self.time, 7, 7) != 0 }
    /// Set color-frame flag.
    pub fn set_color_frame(&mut self, v: bool) {
        set_bit_field(&mut self.time, 7, 7, v as u32);
    }

    /// Field/phase flag (bit 15).
    pub fn field_phase(&self) -> bool { bit_field(self.time, 15, 15) != 0 }
    /// Set field/phase flag.
    pub fn set_field_phase(&mut self, v: bool) {
        set_bit_field(&mut self.time, 15, 15, v as u32);
    }

    /// BGF0 flag (bit 23).
    pub fn bgf0(&self) -> bool { bit_field(self.time, 23, 23) != 0 }
    /// Set BGF0 flag.
    pub fn set_bgf0(&mut self, v: bool) {
        set_bit_field(&mut self.time, 23, 23, v as u32);
    }

    /// BGF1 flag (bit 30).
    pub fn bgf1(&self) -> bool { bit_field(self.time, 30, 30) != 0 }
    /// Set BGF1 flag.
    pub fn set_bgf1(&mut self, v: bool) {
        set_bit_field(&mut self.time, 30, 30, v as u32);
    }

    /// BGF2 flag (bit 31).
    pub fn bgf2(&self) -> bool { bit_field(self.time, 31, 31) != 0 }
    /// Set BGF2 flag.
    pub fn set_bgf2(&mut self, v: bool) {
        set_bit_field(&mut self.time, 31, 31, v as u32);
    }

    // -- Binary group accessors (1–8) --------------------------------------

    /// Binary group `group` (1–8), each stores a 4-bit nibble (0–15).
    pub fn binary_group(&self, group: i32) -> Result<i32> {
        if !(1..=8).contains(&group) {
            return Err(ExrError::Arg(
                "Cannot extract binary group from time code user data. Group number is out of range."
                    .into(),
            ));
        }
        let min_bit = 4 * (group - 1) as u32;
        let max_bit = min_bit + 3;
        Ok(bit_field(self.user, min_bit, max_bit) as i32)
    }

    /// Set binary group `group` (1–8) to `value` (stored as 4 bits).
    pub fn set_binary_group(&mut self, group: i32, value: i32) -> Result<()> {
        if !(1..=8).contains(&group) {
            return Err(ExrError::Arg(
                "Cannot extract binary group from time code user data. Group number is out of range."
                    .into(),
            ));
        }
        let min_bit = 4 * (group - 1) as u32;
        let max_bit = min_bit + 3;
        set_bit_field(&mut self.user, min_bit, max_bit, value as u32);
        Ok(())
    }

    // -- Packed representations --------------------------------------------

    /// Return the packed time-and-flags word under the given packing scheme.
    ///
    /// Mirrors `unsigned int TimeCode::timeAndFlags(Packing) const`.
    pub fn time_and_flags(&self, packing: Packing) -> u32 {
        match packing {
            Packing::Tv50 => {
                // Clear bits 6, 15, 23, 30, 31; re-map bgf0/bgf2/bgf1/fieldPhase
                let mut t = self.time;
                t &= !((1 << 6) | (1 << 15) | (1 << 23) | (1 << 30) | (1 << 31));
                t |= (self.bgf0() as u32) << 15;
                t |= (self.bgf2() as u32) << 23;
                t |= (self.bgf1() as u32) << 30;
                t |= (self.field_phase() as u32) << 31;
                t
            }
            Packing::Film24 => {
                // Clear bits 6 and 7 (drop-frame and color-frame unused in film)
                self.time & !((1 << 6) | (1 << 7))
            }
            Packing::Tv60 => self.time,
        }
    }

    /// Set the packed time-and-flags word, re-mapping fields if needed.
    ///
    /// Mirrors `void TimeCode::setTimeAndFlags(unsigned int, Packing)`.
    pub fn set_time_and_flags(&mut self, value: u32, packing: Packing) {
        match packing {
            Packing::Tv50 => {
                self.time =
                    value & !((1 << 6) | (1 << 15) | (1 << 23) | (1 << 30) | (1 << 31));
                if value & (1 << 15) != 0 { self.set_bgf0(true); }
                if value & (1 << 23) != 0 { self.set_bgf2(true); }
                if value & (1 << 30) != 0 { self.set_bgf1(true); }
                if value & (1 << 31) != 0 { self.set_field_phase(true); }
            }
            Packing::Film24 => {
                self.time = value & !((1 << 6) | (1 << 7));
            }
            Packing::Tv60 => {
                self.time = value;
            }
        }
    }

    /// Raw user-data word (32 bits of binary groups).
    pub fn user_data(&self) -> u32 { self.user }

    /// Set the raw user-data word.
    pub fn set_user_data(&mut self, value: u32) { self.user = value; }
}

// ---------------------------------------------------------------------------
// AttrValue — wire serialisation (2 × u32 = 8 bytes on disk)
// ---------------------------------------------------------------------------

impl AttrValue for TimeCode {
    fn static_type_name() -> &'static str {
        // ImfTimeCodeAttribute.cpp: return "timecode";
        "timecode"
    }

    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        // C++: Xdr::write(os, _value.timeAndFlags());  (TV60, the default)
        //      Xdr::write(os, _value.userData());
        xdr::write_u32(o, self.time_and_flags(Packing::Tv60))?;
        xdr::write_u32(o, self.user_data())
    }

    fn read_value(i: &mut dyn IStream, _size: i32, _version: i32) -> Result<Self> {
        // C++: unsigned int tmp; Xdr::read(is, tmp); _value.setTimeAndFlags(tmp);
        //      Xdr::read(is, tmp); _value.setUserData(tmp);
        let time_and_flags = xdr::read_u32(i)?;
        let user_data      = xdr::read_u32(i)?;
        let mut tc = Self::default();
        tc.set_time_and_flags(time_and_flags, Packing::Tv60);
        tc.set_user_data(user_data);
        Ok(tc)
    }
}

/// `typedef TypedAttribute<TimeCode> TimeCodeAttribute;`
pub type TimeCodeAttribute = crate::attr::attribute::TypedAttribute<TimeCode>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(tc: &TimeCode) -> Vec<u8> {
        let mut o = MemOStream::new();
        tc.write_value(&mut o, 0).unwrap();
        o.into_data()
    }

    #[test]
    fn type_name() {
        assert_eq!(TimeCode::static_type_name(), "timecode");
    }

    #[test]
    fn size_is_8_bytes() {
        assert_eq!(write_bytes(&TimeCode::default()).len(), 8);
    }

    #[test]
    fn round_trip_all_zero() {
        let tc = TimeCode::default();
        let bytes = write_bytes(&tc);
        let mut i = MemIStream::new(bytes);
        let back = TimeCode::read_value(&mut i, 8, 0).unwrap();
        assert_eq!(back, tc);
    }

    #[test]
    fn exact_bytes_known_value() {
        // Build TimeCode(1h 2m 3s frame 4) with TV60 packing and verify the
        // EXACT on-disk bytes, not just a round-tripped field readback (which
        // an all-zero or shuffled-field encoder could also satisfy).
        //
        // `set_bit_field`'s layout (all verified against this file's own
        // `hours`/`minutes`/`seconds`/`frame` setters, which mirror
        // `ImfTimeCode.h`'s documented bit positions): frame BCD at bits
        // 0-5, seconds BCD at bits 8-14, minutes BCD at bits 16-22, hours
        // BCD at bits 24-29. Every field here is a single BCD digit (< 10),
        // so `binary_to_bcd` is the identity and the packed word is simply
        // hours<<24 | minutes<<16 | seconds<<8 | frame:
        //   1<<24 | 2<<16 | 3<<8 | 4 == 0x0102_0304
        let tc = TimeCode::from_fields(
            1, 2, 3, 4, false, false, false, false, false, false,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        .unwrap();
        assert_eq!(
            tc.time_and_flags(Packing::Tv60),
            0x0102_0304,
            "packed time-and-flags word must match the known BCD layout exactly"
        );

        let bytes = write_bytes(&tc);
        assert_eq!(
            bytes,
            vec![0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00],
            "on-disk bytes: LE u32 time-and-flags (0x01020304) then LE u32 userData (0)"
        );

        // Read back and check fields.
        let mut i = MemIStream::new(bytes);
        let back = TimeCode::read_value(&mut i, 8, 0).unwrap();
        assert_eq!(back.hours(), 1);
        assert_eq!(back.minutes(), 2);
        assert_eq!(back.seconds(), 3);
        assert_eq!(back.frame(), 4);
    }

    #[test]
    fn bit_packing_flags() {
        let mut tc = TimeCode::default();
        tc.set_drop_frame(true);
        tc.set_bgf1(true);
        assert!(tc.drop_frame());
        assert!(tc.bgf1());
        assert!(!tc.color_frame());

        // Round-trip preserves flags
        let bytes = write_bytes(&tc);
        let mut i = MemIStream::new(bytes);
        let back = TimeCode::read_value(&mut i, 8, 0).unwrap();
        assert!(back.drop_frame());
        assert!(back.bgf1());
    }

    #[test]
    fn bcd_round_trip() {
        // 23:59:59 frame 29 — max valid values
        let tc = TimeCode::from_fields(
            23, 59, 59, 29, false, false, false, false, false, false,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        .unwrap();
        let bytes = write_bytes(&tc);
        let mut i = MemIStream::new(bytes);
        let back = TimeCode::read_value(&mut i, 8, 0).unwrap();
        assert_eq!(back.hours(), 23);
        assert_eq!(back.minutes(), 59);
        assert_eq!(back.seconds(), 59);
        assert_eq!(back.frame(), 29);
    }

    #[test]
    fn binary_groups() {
        let mut tc = TimeCode::default();
        tc.set_binary_group(1, 0xA).unwrap();
        tc.set_binary_group(8, 0xF).unwrap();
        assert_eq!(tc.binary_group(1).unwrap(), 0xA);
        assert_eq!(tc.binary_group(8).unwrap(), 0xF);

        let bytes = write_bytes(&tc);
        let mut i = MemIStream::new(bytes);
        let back = TimeCode::read_value(&mut i, 8, 0).unwrap();
        assert_eq!(back.binary_group(1).unwrap(), 0xA);
        assert_eq!(back.binary_group(8).unwrap(), 0xF);
    }

    #[test]
    fn tv50_packing() {
        // BGF0 lives at bit 15 in TV50, not at bit 23
        let mut tc = TimeCode::default();
        tc.set_bgf0(true);
        let tf60 = tc.time_and_flags(Packing::Tv60);
        let tf50 = tc.time_and_flags(Packing::Tv50);
        // bit 23 set in TV60 form
        assert_ne!(tf60 & (1 << 23), 0);
        // bit 15 set in TV50 form
        assert_ne!(tf50 & (1 << 15), 0);
    }
}
