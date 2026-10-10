//! DWA v2 channel-rules section (read + write) — port of
//! `DwaCompressor_writeRelevantChannelRules` / `DwaCompressor_readChannelRules`
//! and the `Classifier_write` / `Classifier_read` helpers.
//!
//! # Wire format (little-endian)
//!
//! ```text
//! u16  total_section_bytes        (includes the 2-byte size field itself)
//! [ per emitted classifier rule: ]
//!   bytes  null-terminated suffix
//!   u8     flags: bits[7:4]=csc_idx+1, bits[3:2]=scheme, bit[0]=case_insensitive
//!   u8     pixel type (0=UINT, 1=HALF, 2=FLOAT)
//! ```
//!
//! On **write** (v2), C++ emits, for each rule in `sDefaultChannelRules` order,
//! the rule itself iff at least one image channel matches it (suffix + type) —
//! `DwaCompressor_writeRelevantChannelRules`.  We reproduce that exactly by
//! walking [`classifier::default_rules`] (the same table the encoder classifies
//! with — there is no second copy of it here).
//!
//! On **read**, [`read_channel_rules`] returns the parsed rules; the decoder
//! classifies the image's channels with exactly those, as C++
//! `DwaCompressor_uncompress` does (`internal_dwa_compressor.h:800-805`).
//! Discarding them (the old `read_and_skip`) made us mis-read every C++ v2 chunk
//! whose channels do not match our static default table.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_classifier.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_compressor.h>

use std::borrow::Cow;

use super::classifier::{self, find_suffix, CompressionScheme, CscIndex, Rule};
use crate::attr::{ChannelList, PixelType};
use crate::error::{ExrError, Result};

/// Pixel type as stored on the wire (matches `exr_pixel_type_t`).
fn pixel_type_to_u8(t: PixelType) -> u8 {
    match t {
        PixelType::Uint => 0,
        PixelType::Half => 1,
        PixelType::Float => 2,
        PixelType::NumPixelTypes => 0,
    }
}

/// Validate a wire pixel-type byte.
fn pixel_type_from_u8(v: u8) -> Result<PixelType> {
    match v {
        0 => Ok(PixelType::Uint),
        1 => Ok(PixelType::Half),
        2 => Ok(PixelType::Float),
        _ => Err(ExrError::Input("DWA classifier: invalid pixel type".into())),
    }
}

/// Scheme → 2-bit wire tag (`CompressorScheme`).
///
/// **Must match the C++ `CompressorScheme` enum exactly** (internal_dwa_helpers.h):
/// `UNKNOWN = 0, LOSSY_DCT = 1, RLE = 2`.  The decoder reads these codes from the
/// v2 channel-rules section to re-classify each channel; an off-by-enum mapping
/// makes real C++ OpenEXR mis-classify R/G/B (LOSSY_DCT) as UNKNOWN and reject
/// the chunk with `EXR_ERR_CORRUPT_CHUNK`.
fn scheme_to_u8(s: CompressionScheme) -> u8 {
    match s {
        CompressionScheme::Unknown => 0,
        CompressionScheme::LossyDct => 1,
        CompressionScheme::Rle => 2,
    }
}

/// CSC slot -> wire index (`Classifier::_cscIdx`: 0=R, 1=G, 2=B, -1=none).
fn csc_to_i8(c: CscIndex) -> i8 {
    match c {
        CscIndex::Red => 0,
        CscIndex::Green => 1,
        CscIndex::Blue => 2,
        CscIndex::None => -1,
    }
}

/// Wire index -> CSC slot; `Err` for anything outside `-1..=2`.
fn csc_from_i8(v: i8) -> Result<CscIndex> {
    match v {
        -1 => Ok(CscIndex::None),
        0 => Ok(CscIndex::Red),
        1 => Ok(CscIndex::Green),
        2 => Ok(CscIndex::Blue),
        _ => Err(ExrError::Input("DWA classifier: invalid csc_idx".into())),
    }
}

/// Wire 2-bit tag -> scheme (inverse of [`scheme_to_u8`]).
fn scheme_from_u8(v: u8) -> Result<CompressionScheme> {
    match v {
        0 => Ok(CompressionScheme::Unknown),
        1 => Ok(CompressionScheme::LossyDct),
        2 => Ok(CompressionScheme::Rle),
        _ => Err(ExrError::Input("DWA classifier: invalid scheme".into())),
    }
}

/// Append one rule to `out` — `Classifier_write`.
fn write_rule(rule: &Rule, out: &mut Vec<u8>) {
    out.extend_from_slice(&rule.suffix);
    out.push(0u8); // null terminator

    let csc_field = ((csc_to_i8(rule.csc) + 1) as u8) & 0x0f;
    let flags = (csc_field << 4)
        | ((scheme_to_u8(rule.scheme) & 3) << 2)
        | u8::from(rule.case_insensitive);
    out.push(flags);
    out.push(pixel_type_to_u8(rule.type_));
}

/// Write the v2 channel-rules section for `channels` — port of
/// `DwaCompressor_writeRelevantChannelRules`.
///
/// Emits each default rule that at least one channel matches, in
/// `sDefaultChannelRules` order, prefixed by a LE u16 total size (incl. the
/// size field).  An all-rules-miss image yields a 2-byte section (size == 2).
///
/// # Errors
/// If the section would need more than `u16::MAX` bytes, C++
/// `DwaCompressor_writeRelevantChannelRules` returns `EXR_ERR_OUT_OF_MEMORY`
/// (`internal_dwa_compressor.h:1554`, its `nOut` overflow check) rather than
/// truncating the size field — this cannot happen with the 15 default rules
/// (well under 2 KB total), but silently clamping instead of erroring would
/// have written a size field that undercounts the actual bytes.
pub fn write_for_channels(channels: &ChannelList, out: &mut Vec<u8>) -> Result<()> {
    // Collect (suffix, type) of every channel for matching. Matches on
    // raw bytes, exactly like C++ (`Classifier_find_suffix`/`Classifier_match`
    // operate on `const char*`, never checking encoding), so a non-UTF-8
    // channel name classifies the same way a UTF-8 one does.
    let chan_suffixes: Vec<(&[u8], PixelType)> = channels
        .iter()
        .map(|(name, ch)| (find_suffix(name.as_bytes()), ch.type_))
        .collect();

    // Reserve the u16 size slot; patch it after writing records.
    let size_offset = out.len();
    out.push(0);
    out.push(0);

    for rule in classifier::default_rules() {
        let matches = chan_suffixes
            .iter()
            .any(|&(s, t)| classifier::rule_matches(rule, s, t));
        if matches {
            write_rule(rule, out);
        }
    }

    let total = out.len() - size_offset;
    if total > 0xffff {
        return Err(ExrError::Logic("DWA classifier: channel rules section exceeds 65535 bytes".into()));
    }
    let size_le = (total as u16).to_le_bytes();
    out[size_offset] = size_le[0];
    out[size_offset + 1] = size_le[1];
    Ok(())
}

/// Parse the v2 channel-rules section at `data[*offset..]`, advancing `*offset`
/// past it and returning the rules — port of `DwaCompressor_readChannelRules` +
/// `Classifier_read`.
///
/// The returned rules are the ONLY ones the decoder may classify with for a v2
/// stream (`internal_dwa_compressor.h:800-805`).
pub fn read_channel_rules(data: &[u8], offset: &mut usize) -> Result<Vec<Rule>> {
    let size_end = offset
        .checked_add(2)
        .ok_or_else(|| ExrError::Input("DWA v2: channel rules offset overflow".into()))?;
    if size_end > data.len() {
        return Err(ExrError::Input("DWA v2: missing channel rules size".into()));
    }
    let section_size = u16::from_le_bytes([data[*offset], data[*offset + 1]]) as usize;
    if section_size < 2 {
        return Err(ExrError::Input("DWA v2: channel rules size < 2".into()));
    }
    let section_end = offset
        .checked_add(section_size)
        .ok_or_else(|| ExrError::Input("DWA v2: channel rules offset overflow".into()))?;
    if section_end > data.len() {
        return Err(ExrError::Input("DWA v2: channel rules section truncated".into()));
    }

    let mut rules = Vec::new();
    let mut p = *offset + 2; // past the size field
    let mut remaining = section_size - 2;

    // Parse each record while more than 2 bytes remain — C++
    // `DwaCompressor_readChannelRules`'s loop condition is `*nAvail > 2`
    // (`internal_dwa_compressor.h:1570`), the minimum a record can be (a
    // 0-length suffix's null terminator + 1 flags byte + 1 type byte). A
    // trailing 1- or 2-byte gap after the last real record therefore ends the
    // loop cleanly in C++; our previous `remaining <= 3` check (requiring
    // `remaining > 3`, one byte stricter than C++) turned that benign gap
    // into a hard "record too short" error C++ does not raise.
    while remaining > 2 {
        // suffix: null-terminated (<= 128 chars), then 2 flag bytes.
        let start = p;
        // C++ `Classifier_read` scans at most 128 suffix bytes and never past
        // the declared section end.
        let scan_end = (start + 129).min(section_end);
        let null_pos = data
            .get(start..scan_end)
            .and_then(|w| w.iter().position(|&b| b == 0))
            .map(|rel| start + rel)
            .ok_or_else(|| ExrError::Input("DWA classifier: missing terminator".into()))?;
        let len_with_null = null_pos - start + 1;
        let flags_pos = start + len_with_null;
        if flags_pos + 2 > section_end {
            return Err(ExrError::Input("DWA classifier: truncated flags".into()));
        }
        let flags = data[flags_pos];
        let type_byte = data[flags_pos + 1];

        // The suffix is a raw byte string on the wire, matching C++
        // `Classifier_read`, which reads it via `memcpy` into `char[]` with no
        // encoding check at all.
        let suffix = &data[start..null_pos];
        let csc = csc_from_i8((flags >> 4) as i8 - 1)?;
        let scheme = scheme_from_u8((flags >> 2) & 3)?;
        let type_ = pixel_type_from_u8(type_byte)?;

        rules.push(Rule {
            suffix: Cow::Owned(suffix.to_vec()),
            scheme,
            type_,
            csc,
            case_insensitive: (flags & 1) != 0,
        });

        let rec_size = len_with_null + 2;
        // Defensive: never advance past the declared section.
        if rec_size > remaining {
            return Err(ExrError::Input("DWA classifier: record overruns section".into()));
        }
        p += rec_size;
        remaining -= rec_size;
    }

    *offset = section_end;
    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList};

    fn cl(chans: &[(&str, PixelType)]) -> ChannelList {
        let mut c = ChannelList::new();
        for (n, t) in chans {
            c.insert(*n, Channel::new(*t, 1, 1, false)).unwrap();
        }
        c
    }

    /// Mutation-testing gap fix: wire tag values are C++'s own
    /// `CompressorScheme` enum (OpenEXRCore v3.4.15,
    /// `internal_dwa_helpers.h:21-28`: `UNKNOWN=0, LOSSY_DCT=1, RLE=2,
    /// NUM_COMPRESSOR_SCHEMES=3`), not a value derived from this function
    /// itself. Every valid tag maps to its specific scheme (not just
    /// "doesn't error"), and `3` (the sentinel `NUM_COMPRESSOR_SCHEMES`,
    /// never a real scheme) is rejected.
    #[test]
    fn scheme_from_u8_matches_documented_wire_values() {
        assert_eq!(scheme_from_u8(0).unwrap(), CompressionScheme::Unknown);
        assert_eq!(scheme_from_u8(1).unwrap(), CompressionScheme::LossyDct);
        assert_eq!(scheme_from_u8(2).unwrap(), CompressionScheme::Rle);
        assert!(scheme_from_u8(3).is_err(), "3 is not a valid wire scheme tag");
    }

    /// Mutation-testing gap fix: the loop boundary is C++'s own
    /// `DwaCompressor_readChannelRules` condition `*nAvail > 2`
    /// (OpenEXRCore v3.4.15, `internal_dwa_compressor.h:1570`), not this
    /// port's own historical `remaining <= 3` check (already fixed, see the
    /// comment above the loop) - a trailing 2-byte gap after the last
    /// real record must end the loop cleanly, not be misread as a new
    /// record. A section with one real record followed by exactly 2
    /// non-null trailing bytes exercises this exact boundary.
    #[test]
    fn read_channel_rules_stops_cleanly_at_exactly_two_trailing_bytes() {
        // size(2) + record("A\0" + flags=0x00 + type=1) + 2 trailing bytes.
        let data: Vec<u8> = vec![8, 0, b'A', 0x00, 0x00, 1, 0xFF, 0xFF];
        let mut offset = 0usize;
        let rules = read_channel_rules(&data, &mut offset).expect("a benign 2-byte trailing gap must not error");
        assert_eq!(rules.len(), 1, "the 2 trailing bytes must not be parsed as a second record");
        assert_eq!(rules[0].suffix.as_ref(), b"A");
        assert_eq!(rules[0].scheme, CompressionScheme::Unknown);
        assert_eq!(rules[0].type_, PixelType::Half);
        assert_eq!(rules[0].csc, CscIndex::None);
        assert!(!rules[0].case_insensitive);
    }

    /// Mutation-testing gap fix: `flags_pos = start + len_with_null` locates
    /// the flags/type bytes right after a record's null-terminated suffix.
    /// Uses a nonzero starting `offset` (so `start + len_with_null` and
    /// `start * len_with_null` diverge sharply) and checks the parsed rule's
    /// SPECIFIC fields against independently-known wire values (not a round
    /// trip), so reading the flags/type from the wrong offset - garbage or
    /// out-of-bounds - is caught.
    #[test]
    fn read_channel_rules_locates_flags_after_suffix_correctly() {
        let mut data = vec![0u8; 16];
        data[10] = 6; // section_size low byte = 6 (2 size + 4 record).
        data[11] = 0;
        data[12] = b'A';
        data[13] = 0x00; // null terminator: len_with_null = 2.
        data[14] = 0x00; // flags: csc=None, scheme=Unknown, case-sensitive.
        data[15] = 1; // type = HALF.
        let mut offset = 10usize;
        let rules = read_channel_rules(&data, &mut offset).expect("well-formed record at a nonzero offset must parse");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].suffix.as_ref(), b"A");
        assert_eq!(rules[0].scheme, CompressionScheme::Unknown);
        assert_eq!(rules[0].type_, PixelType::Half);
        assert_eq!(rules[0].csc, CscIndex::None);
    }

    #[test]
    fn write_then_skip_roundtrips_rgb_az() {
        let channels = cl(&[
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
            ("A", PixelType::Float),
            ("Z", PixelType::Float),
        ]);
        let mut out = Vec::new();
        write_for_channels(&channels, &mut out).unwrap();

        // Section size matches the written length.
        let size = u16::from_le_bytes([out[0], out[1]]) as usize;
        assert_eq!(size, out.len());

        // R/G/B HALF (3 rules) + A FLOAT (1 rule) match; Z matches nothing.
        // Each rule: suffix(1) + null(1) + 2 flag bytes = 4; 4 rules = 16.
        assert_eq!(out.len(), 2 + 16);

        // Byte-exact match to real C++ OpenEXR 3.4's channel-rules section for
        // R,G,B(half)+A(float). The flags byte packs `(cscIdx+1)<<4 | scheme<<2`
        // with the C++ `CompressorScheme` order UNKNOWN=0/LOSSY_DCT=1/RLE=2, so
        // R/G/B (LOSSY_DCT) -> scheme 1 (low nibble 0x4|cscIdx<<4) and A (RLE) ->
        // scheme 2 (0x08).  This guards a past regression: a wrong enum mapping
        // here made genuine C++ reject our tiled-DWAA chunks as CORRUPT.
        assert_eq!(
            out,
            vec![
                0x12, 0x00, // section size = 18
                b'R', 0x00, 0x14, 0x01, // R: csc 0, LOSSY_DCT, HALF
                b'G', 0x00, 0x24, 0x01, // G: csc 1, LOSSY_DCT, HALF
                b'B', 0x00, 0x34, 0x01, // B: csc 2, LOSSY_DCT, HALF
                b'A', 0x00, 0x08, 0x02, // A: csc -1, RLE, FLOAT
            ]
        );

        let mut off = 0usize;
        let rules = read_channel_rules(&out, &mut off).unwrap();
        assert_eq!(off, out.len());
        // The parsed rules must round-trip back to what we wrote.
        assert_eq!(rules.len(), 4);
        assert_eq!(rules[0].suffix.as_ref(), b"R");
        assert_eq!(rules[0].scheme, CompressionScheme::LossyDct);
        assert_eq!(rules[0].csc, CscIndex::Red);
        assert_eq!(rules[0].type_, PixelType::Half);
        assert_eq!(rules[3].suffix.as_ref(), b"A");
        assert_eq!(rules[3].scheme, CompressionScheme::Rle);
        assert_eq!(rules[3].csc, CscIndex::None);
        assert_eq!(rules[3].type_, PixelType::Float);
        assert!(rules.iter().all(|r| !r.case_insensitive));
    }

    #[test]
    fn empty_when_no_rules_match() {
        let channels = cl(&[("Z", PixelType::Float), ("depth", PixelType::Float)]);
        let mut out = Vec::new();
        write_for_channels(&channels, &mut out).unwrap();
        assert_eq!(out, vec![2u8, 0u8]); // size = 2, no records
        let mut off = 0usize;
        assert!(read_channel_rules(&out, &mut off).unwrap().is_empty());
        assert_eq!(off, 2);
    }

    #[test]
    fn skip_rejects_truncated() {
        let bad = [0u8; 1];
        let mut off = 0usize;
        assert!(read_channel_rules(&bad, &mut off).is_err());
    }
}
