//! DWA channel classification — port of `sDefaultChannelRules` /
//! `sLegacyChannelRules` + `Classifier_match` / `Classifier_find_suffix` from
//! `OpenEXRCore/internal_dwa_classifier.h`.
//!
//! Every channel is routed to one of three sub-schemes by matching its *suffix*
//! (the part after the last `.`) and pixel type against an ordered rule table:
//! * `R/G/B/Y/RY/BY` (HALF or FLOAT) → [`CompressionScheme::LossyDct`]
//!   (`R/G/B` additionally carry a CSC index so RGB triplets can be grouped);
//! * `A` (UINT/HALF/FLOAT) → [`CompressionScheme::Rle`];
//! * everything else → [`CompressionScheme::Unknown`] (zlib).
//!
//! **Exactly one rule set is active at a time**, as in C++:
//! * the **encoder** always uses [`default_rules`] (`sDefaultChannelRules`) —
//!   `DwaCompressor_compress`, `internal_dwa_compressor.h:256-258`;
//! * the **decoder** uses [`legacy_rules`] (`sLegacyChannelRules`) for a v0/v1
//!   stream and the rules **stored in the chunk** for a v2 stream —
//!   `DwaCompressor_uncompress`, `internal_dwa_compressor.h:792-805`.
//!
//! Chaining the two sets (the old behaviour) made us write chunks that real
//! OpenEXR rejects (`red/green/blue`, `r/g/b/a` -> `EXR_ERR_CORRUPT_CHUNK`) and
//! made us mis-read C++'s v2 chunks for the same channel names.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXRCore/internal_dwa_classifier.h>

use std::borrow::Cow;

use crate::attr::PixelType;

/// Per-channel DWA sub-scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionScheme {
    /// Lossy DCT (color channels).  Wire tag 0.
    LossyDct,
    /// RLE (alpha).  Wire tag 1.
    Rle,
    /// Unknown → zlib (depth, normals, …).  Wire tag 2.
    Unknown,
}

/// CSC group role of a channel (which slot of an RGB triplet it fills).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CscIndex {
    /// Red (csc index 0).
    Red,
    /// Green (csc index 1).
    Green,
    /// Blue (csc index 2).
    Blue,
    /// Not part of a CSC group (csc index -1).
    None,
}

/// One classifier rule — a row of `sDefaultChannelRules` /
/// `sLegacyChannelRules`, or a rule parsed out of a v2 chunk's rules section.
///
/// `suffix` is a `Cow` so the static tables stay allocation-free while rules
/// read from a chunk ([`super::channel_rules::read_channel_rules`]) own theirs.
///
/// `PartialEq` lets `decompress_dwa` compare the just-read
/// per-chunk rule set against the previous chunk cached one, so the
/// expensive `classify_channels` output can be reused when the rules truly
/// match -- see `DwaScratch::plan`'s doc. Structural equality only.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    /// Channel-name suffix this rule matches (`Classifier::_suffix`).
    ///
    /// Bytes, not `str` — C++ `Classifier::_suffix` is a raw byte
    /// buffer (`Classifier_construct` takes `const char* suffix`; channel
    /// names are never UTF-8-checked anywhere in OpenEXR), so suffix matching
    /// must work on a non-UTF-8 channel name the same way it works on any
    /// other.
    pub suffix: Cow<'static, [u8]>,
    /// Sub-scheme the channel is routed to (`Classifier::_scheme`).
    pub scheme: CompressionScheme,
    /// Pixel type the rule applies to (`Classifier::_type`).
    pub type_: PixelType,
    /// CSC slot this channel fills, if any (`Classifier::_cscIdx`).
    pub csc: CscIndex,
    /// Compare `suffix` case-insensitively (`Classifier::_caseInsensitive`).
    pub case_insensitive: bool,
}

/// `sDefaultChannelRules` (v2, case-sensitive) — the authoritative encode rules.
static DEFAULT_RULES: &[Rule] = &[
    Rule { suffix: Cow::Borrowed(b"R"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Red,   case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"R"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Red,   case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"G"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Green, case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"G"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Green, case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"B"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Blue,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"B"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Blue,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"Y"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"Y"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"BY"), scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"BY"), scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"RY"), scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"RY"), scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"A"),  scheme: CompressionScheme::Rle,      type_: PixelType::Uint,  csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"A"),  scheme: CompressionScheme::Rle,      type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: false },
    Rule { suffix: Cow::Borrowed(b"A"),  scheme: CompressionScheme::Rle,      type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: false },
];

/// `sLegacyChannelRules` (case-insensitive lowercase) — used for v0/v1 decode.
static LEGACY_RULES: &[Rule] = &[
    Rule { suffix: Cow::Borrowed(b"r"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Red,   case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"r"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Red,   case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"red"),   scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Red,   case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"red"),   scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Red,   case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"g"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Green, case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"g"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Green, case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"grn"),   scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Green, case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"grn"),   scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Green, case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"green"), scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Green, case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"green"), scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Green, case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"b"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Blue,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"b"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Blue,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"blu"),   scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Blue,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"blu"),   scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Blue,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"blue"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::Blue,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"blue"),  scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::Blue,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"y"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"y"),     scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"by"),    scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"by"),    scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"ry"),    scheme: CompressionScheme::LossyDct, type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"ry"),    scheme: CompressionScheme::LossyDct, type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"a"),     scheme: CompressionScheme::Rle,      type_: PixelType::Uint,  csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"a"),     scheme: CompressionScheme::Rle,      type_: PixelType::Half,  csc: CscIndex::None,  case_insensitive: true },
    Rule { suffix: Cow::Borrowed(b"a"),     scheme: CompressionScheme::Rle,      type_: PixelType::Float, csc: CscIndex::None,  case_insensitive: true },
];

/// `Classifier_find_suffix`: the part of `name` after the last `.`, else `name`.
///
/// Operates on raw bytes so a non-UTF-8 channel name still splits
/// correctly on the ASCII `.` byte, matching C++'s `strrchr(name, '.')`.
pub fn find_suffix(name: &[u8]) -> &[u8] {
    match name.iter().rposition(|&b| b == b'.') {
        Some(pos) => &name[pos + 1..],
        None => name,
    }
}

/// `sDefaultChannelRules` — the ONLY rule set the encoder may use, and the
/// fallback when a v2 chunk carries no rules of its own.
pub fn default_rules() -> &'static [Rule] {
    DEFAULT_RULES
}

/// `sLegacyChannelRules` — used by the decoder for v0/v1 streams only.
pub fn legacy_rules() -> &'static [Rule] {
    LEGACY_RULES
}

/// `Classifier_match`: type must match; suffix compared case-(in)sensitively.
pub fn rule_matches(rule: &Rule, suffix: &[u8], type_: PixelType) -> bool {
    if rule.type_ != type_ {
        return false;
    }
    if rule.case_insensitive {
        suffix.eq_ignore_ascii_case(&rule.suffix)
    } else {
        suffix == rule.suffix.as_ref()
    }
}

/// Classify one channel against `rules` — the inner loop of
/// `DwaCompressor_classifyChannels` (`internal_dwa_compressor.h:1626-1639`).
///
/// C++ does **not** stop at the first match: it walks the whole active rule set,
/// so the *last* matching rule wins for the scheme, while the CSC slot is set by
/// the last match that carries one.  The starting value is `UNKNOWN` / no slot
/// (`internal_dwa_compressor.h:135`).
pub fn classify_with(
    rules: &[Rule],
    name: &[u8],
    type_: PixelType,
) -> (CompressionScheme, CscIndex) {
    let suffix = find_suffix(name);
    let mut scheme = CompressionScheme::Unknown;
    let mut csc = CscIndex::None;
    for rule in rules {
        if rule_matches(rule, suffix, type_) {
            scheme = rule.scheme;
            if rule.csc != CscIndex::None {
                csc = rule.csc;
            }
        }
    }
    (scheme, csc)
}

/// Classify with the **default** rule set — the encoder's only entry point.
pub fn classify_channel(name: &[u8], type_: PixelType) -> (CompressionScheme, CscIndex) {
    classify_with(DEFAULT_RULES, name, type_)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_uppercase() {
        assert_eq!(classify_channel(b"R", PixelType::Half), (CompressionScheme::LossyDct, CscIndex::Red));
        assert_eq!(classify_channel(b"G", PixelType::Half), (CompressionScheme::LossyDct, CscIndex::Green));
        assert_eq!(classify_channel(b"B", PixelType::Float), (CompressionScheme::LossyDct, CscIndex::Blue));
    }

    #[test]
    fn layered_prefix() {
        assert_eq!(classify_channel(b"diffuse.R", PixelType::Half), (CompressionScheme::LossyDct, CscIndex::Red));
    }

    #[test]
    fn alpha_is_rle_all_types() {
        assert_eq!(classify_channel(b"A", PixelType::Half), (CompressionScheme::Rle, CscIndex::None));
        assert_eq!(classify_channel(b"A", PixelType::Uint), (CompressionScheme::Rle, CscIndex::None));
        assert_eq!(classify_channel(b"A", PixelType::Float), (CompressionScheme::Rle, CscIndex::None));
    }

    #[test]
    fn uint_rgb_falls_through_to_unknown() {
        // No default rule for UINT R/G/B → Unknown (zlib).
        assert_eq!(classify_channel(b"R", PixelType::Uint), (CompressionScheme::Unknown, CscIndex::None));
    }

    #[test]
    fn depth_is_unknown() {
        assert_eq!(classify_channel(b"Z", PixelType::Float), (CompressionScheme::Unknown, CscIndex::None));
    }

    #[test]
    fn legacy_lowercase_only_under_legacy_rules() {
        // The DEFAULT (encode) rules are case-sensitive, so lowercase and
        // long-form names classify as UNKNOWN there.  Chaining the legacy table
        // in made us write chunks real OpenEXR rejects as CORRUPT_CHUNK.
        for n in ["r", "green", "blu", "red", "b", "a"] {
            assert_eq!(
                classify_channel(n.as_bytes(), PixelType::Half).0,
                CompressionScheme::Unknown,
                "{n} must not classify under the default rules"
            );
        }
        let legacy = legacy_rules();
        assert_eq!(
            classify_with(legacy, b"r", PixelType::Half),
            (CompressionScheme::LossyDct, CscIndex::Red)
        );
        assert_eq!(
            classify_with(legacy, b"green", PixelType::Half),
            (CompressionScheme::LossyDct, CscIndex::Green)
        );
        assert_eq!(
            classify_with(legacy, b"blu", PixelType::Half),
            (CompressionScheme::LossyDct, CscIndex::Blue)
        );
        assert_eq!(classify_with(legacy, b"a", PixelType::Half).0, CompressionScheme::Rle);
    }
}
