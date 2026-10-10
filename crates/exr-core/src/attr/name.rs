//! Port of `Imf::Name` — <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfName.h>
//!
//! C++ `Name` is a fixed `char _text[256]` buffer: construction is
//! `strncpy(_text, text, MAX_LENGTH); _text[MAX_LENGTH] = 0`, i.e. the stored
//! text is cut at the first NUL of the input **and** at 255 bytes, whichever
//! comes first. Comparison is `strcmp`, which is unsigned-byte lexicographic.
//! There is no encoding requirement anywhere in the C++ type or its use in
//! `parse_header.c`'s `read_text` — attribute, type and channel names are byte
//! strings, not necessarily UTF-8. `Name` here is that same byte
//! string: no allocation-free unsized borrowed form is provided (that would
//! need an `unsafe` `repr(transparent)` cast, which this crate avoids outside
//! SIMD), so a borrowed name is just `&[u8]` and lookups
//! take `impl AsRef<[u8]>`.

use std::borrow::Borrow;
use std::fmt;

/// A 1:1 port of `Imf::Name`: raw bytes, no encoding, `strncpy`/`strcmp`
/// semantics. Invariant: never contains a NUL byte and is at most
/// [`Name::MAX_LENGTH`] bytes long (both enforced by [`Name::new`]).
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name(Box<[u8]>);

impl Name {
    /// C++ `Name::SIZE` — the fixed on-disk/in-memory buffer size (255 bytes
    /// of text plus the terminating NUL C++ always appends).
    pub const SIZE: usize = 256;
    /// C++ `Name::MAX_LENGTH` — the longest text a `Name` can hold.
    pub const MAX_LENGTH: usize = Self::SIZE - 1;

    /// Build a `Name` from arbitrary bytes, applying the exact C++
    /// `strncpy(_text, text, MAX_LENGTH)` truncation: cut at the first NUL
    /// byte (a C string never contains one past the terminator), then at
    /// [`Self::MAX_LENGTH`] bytes. Unlike the old `xdr::truncate_name`, this
    /// cuts at the exact byte, not at a UTF-8 character boundary.
    pub fn new(text: impl AsRef<[u8]>) -> Self {
        Name(Self::normalize(text.as_ref()).into())
    }

    /// Apply the same cut as [`Self::new`] without allocating — used by every
    /// lookup, mirroring C++ building a temporary `Name` for `map::find(const
    /// char*)` / `operator==(const Name&, const char*)`.
    pub(crate) fn normalize(b: &[u8]) -> &[u8] {
        let cut_at_nul = match b.iter().position(|&c| c == 0) {
            Some(i) => &b[..i],
            None => b,
        };
        if cut_at_nul.len() > Self::MAX_LENGTH {
            &cut_at_nul[..Self::MAX_LENGTH]
        } else {
            cut_at_nul
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `None` when the stored bytes are not valid UTF-8 — for consumers that
    /// need text (openexr-py's strict `py::str` mapping, CLI display of the
    /// common case, etc.). Never used for identity/lookup.
    pub fn to_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }
}

impl Borrow<[u8]> for Name {
    fn borrow(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for Name {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Name::new(s.as_bytes())
    }
}

impl From<String> for Name {
    fn from(s: String) -> Self {
        Name::new(s.as_bytes())
    }
}

impl From<&String> for Name {
    fn from(s: &String) -> Self {
        Name::new(s.as_bytes())
    }
}

impl From<&[u8]> for Name {
    fn from(b: &[u8]) -> Self {
        Name::new(b)
    }
}

impl From<Vec<u8>> for Name {
    fn from(b: Vec<u8>) -> Self {
        Name::new(b.as_slice())
    }
}

impl From<&Name> for Name {
    fn from(n: &Name) -> Self {
        n.clone()
    }
}

impl PartialEq<str> for Name {
    fn eq(&self, other: &str) -> bool {
        self.0.as_ref() == other.as_bytes()
    }
}

impl PartialEq<&str> for Name {
    fn eq(&self, other: &&str) -> bool {
        self.0.as_ref() == other.as_bytes()
    }
}

impl PartialEq<[u8]> for Name {
    fn eq(&self, other: &[u8]) -> bool {
        self.0.as_ref() == other
    }
}

impl PartialEq<Name> for str {
    fn eq(&self, other: &Name) -> bool {
        other == self
    }
}

/// Verbatim for valid UTF-8; every invalid byte is escaped as `\xNN` and a
/// literal backslash as `\\`, so the printed text is unambiguous and can be
/// parsed back byte-for-byte. Never feed this back into a lookup — the
/// escaping is a presentation choice, not identity; keep the `Name` itself
/// for that (see the module doc).
impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rest = self.0.as_ref();
        loop {
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    for ch in valid.chars() {
                        if ch == '\\' {
                            write!(f, "\\\\")?;
                        } else {
                            write!(f, "{ch}")?;
                        }
                    }
                    return Ok(());
                }
                Err(e) => {
                    let (valid, after_valid) = rest.split_at(e.valid_up_to());
                    for ch in std::str::from_utf8(valid).unwrap_or("").chars() {
                        if ch == '\\' {
                            write!(f, "\\\\")?;
                        } else {
                            write!(f, "{ch}")?;
                        }
                    }
                    let bad_len = e.error_len().unwrap_or(after_valid.len().max(1));
                    for &b in &after_valid[..bad_len.min(after_valid.len())] {
                        write!(f, "\\x{b:02x}")?;
                    }
                    rest = &after_valid[bad_len.min(after_valid.len())..];
                    if rest.is_empty() {
                        return Ok(());
                    }
                }
            }
        }
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b\"")?;
        for &b in self.0.iter() {
            match b {
                b'"' => write!(f, "\\\"")?,
                b'\\' => write!(f, "\\\\")?,
                0x20..=0x7e => write!(f, "{}", b as char)?,
                _ => write!(f, "\\x{b:02x}")?,
            }
        }
        write!(f, "\"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_at_first_nul() {
        assert_eq!(Name::new(b"a\0b" as &[u8]).as_bytes(), b"a");
        assert_eq!(Name::new(b"a\0" as &[u8]).as_bytes(), b"a");
        assert_eq!(Name::new(b"abc" as &[u8]).as_bytes(), b"abc");
    }

    #[test]
    fn truncates_at_max_length_exact_byte() {
        let long = vec![b'x'; 300];
        let n = Name::new(long.as_slice());
        assert_eq!(n.len(), Name::MAX_LENGTH);
        assert!(n.as_bytes().iter().all(|&b| b == b'x'));
    }

    #[test]
    fn exact_length_unchanged() {
        let s = vec![b'y'; Name::MAX_LENGTH];
        assert_eq!(Name::new(s.as_slice()).as_bytes(), s.as_slice());
    }

    #[test]
    fn non_utf8_round_trips_as_bytes() {
        let raw: &[u8] = &[0x5a, 0xe9]; // "Z\xE9" — not valid UTF-8
        let n = Name::new(raw);
        assert_eq!(n.as_bytes(), raw);
        assert_eq!(n.to_str(), None);
    }

    #[test]
    fn ordering_matches_strcmp_unsigned_byte() {
        // strcmp compares bytes as unsigned char; 0xFF > 0x7F > 0x00.
        let low = Name::new(&[0x01][..]);
        let high = Name::new(&[0xff][..]);
        assert!(low < high);
        let a = Name::new("a");
        let hi_byte = Name::new(&[0xff][..]);
        assert!(a < hi_byte); // 'a' = 0x61 < 0xff unsigned
    }

    #[test]
    fn display_escapes_invalid_bytes_and_backslash() {
        let n = Name::new([0x5a, 0xe9, b'\\', b'a'].as_slice());
        assert_eq!(n.to_string(), "Z\\xe9\\\\a");
    }

    #[test]
    fn display_valid_utf8_verbatim() {
        assert_eq!(Name::new("hello.world").to_string(), "hello.world");
    }

    #[test]
    fn equality_with_str() {
        let n = Name::new("channels");
        assert_eq!(n, *"channels");
        assert_eq!(n, "channels");
    }
}
