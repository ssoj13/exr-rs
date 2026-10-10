//! Binary primitives from `ImfIDManifest.cpp`'s anonymous namespace: the
//! variable-length integer, string-list, and Pascal-string encodings that
//! `IDManifest::init`/`serialize` use for the decompressed manifest payload.
//!
//! `Xdr::write<CharPtrIO>`/`read<CharPtrIO>` in C++ still go through the same
//! portable little-endian `Xdr` template as every other attribute — only the
//! byte SINK changes (a raw `char*` instead of a stream) — so these operate
//! directly on a `Vec<u8>` / `&[u8]` cursor with the SAME little-endian
//! encoding [`crate::container::xdr`] uses, just without the `IStream`/
//! `OStream` trait indirection (there is no stream here: the whole manifest
//! is decompressed into memory up front).
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.cpp#L1-L335>

use crate::error::{ExrError, Result};

// ---------------------------------------------------------------------------
// Variable-length integer — `writeVariableLengthInteger`/`readVariableLengthInteger`
// ---------------------------------------------------------------------------

/// `getVariableLengthIntegerSize` — not needed for writing (we grow a `Vec`
/// instead of pre-sizing a fixed buffer the way C++ must), but kept for
/// parity with the C++ API surface and because callers may want it to reason
/// about a manifest's on-disk size without serializing.
pub fn varint_size(value: u64) -> usize {
    match value {
        v if v < (1 << 7) => 1,
        v if v < (1 << 14) => 2,
        v if v < (1 << 21) => 3,
        v if v < (1 << 28) => 4,
        v if v < (1 << 35) => 5,
        v if v < (1 << 42) => 6,
        v if v < (1 << 49) => 7,
        v if v < (1 << 56) => 8,
        v if v < (1 << 63) => 9,
        _ => 10,
    }
}

/// `writeVariableLengthInteger` — LSB-first base-128 varint (the top bit of
/// each byte marks "more bytes follow").
pub fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 127) as u8;
        value >>= 7;
        if value > 0 {
            byte |= 128;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// `readVariableLengthInteger` — the read side, bounds-checked against `buf`
/// and rejecting a >64-bit-wide encoding (matching C++'s `shift >= 64` guard,
/// which exists because `(byte & 127) << shift` is undefined behaviour in C++
/// once `shift >= 64`; the equivalent Rust panic this guards against is a
/// shift-amount overflow, caught here before it happens).
pub fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut shift: u32 = 0;
    let mut value: u64 = 0;
    loop {
        let byte = *buf.get(*pos).ok_or_else(|| {
            ExrError::Input("IDManifest too small for variable length integer".into())
        })?;
        *pos += 1;
        if shift >= 64 {
            return Err(ExrError::Input(
                "Invalid variable-length integer in IDManifest".into(),
            ));
        }
        value |= ((byte & 127) as u64) << shift;
        shift += 7;
        if byte & 128 == 0 {
            break;
        }
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Fixed-width little-endian scalars (the `Xdr::write<CharPtrIO>(outPtr, T)` /
// `Xdr::read<CharPtrIO>` calls in ImfIDManifest.cpp for `int`/`char`).
// ---------------------------------------------------------------------------

/// Write an `i32` LE, matching `Xdr::write<CharPtrIO>(outPtr, int(...))`.
pub fn write_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Read an `i32` LE, bounds-checked.
pub fn read_i32(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let bytes = buf
        .get(*pos..*pos + 4)
        .ok_or_else(|| ExrError::Input("IDManifest too small".into()))?;
    *pos += 4;
    Ok(i32::from_le_bytes(bytes.try_into().expect("checked 4-byte slice")))
}

/// Write a single byte (`Xdr::write<CharPtrIO>(outPtr, char(...))`).
pub fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

/// Read a single byte, bounds-checked.
pub fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8> {
    let byte = *buf
        .get(*pos)
        .ok_or_else(|| ExrError::Input("IDManifest too small".into()))?;
    *pos += 1;
    Ok(byte)
}

// ---------------------------------------------------------------------------
// Pascal string — `readPascalString`/`writePascalString`: `u32` LE length +
// raw bytes (no NUL terminator, unlike the header attribute name encoding).
// ---------------------------------------------------------------------------

pub fn write_pascal_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

pub fn read_pascal_string(buf: &[u8], pos: &mut usize) -> Result<String> {
    let len_bytes = buf
        .get(*pos..*pos + 4)
        .ok_or_else(|| ExrError::Input("IDManifest too small for string size".into()))?;
    let len = u32::from_le_bytes(len_bytes.try_into().expect("checked 4-byte slice")) as usize;
    *pos += 4;
    let bytes = buf
        .get(*pos..*pos + len)
        .ok_or_else(|| ExrError::Input("IDManifest too small for string".into()))?;
    *pos += len;
    String::from_utf8(bytes.to_vec())
        .map_err(|e| ExrError::Input(format!("Non-UTF-8 string in IDManifest: {e}")))
}

// ---------------------------------------------------------------------------
// String list — `writeStringList`/`readStringList`: sizes come first (as
// varints), then the concatenated string bytes, so a compressor sees runs of
// similar-sized-integer bytes together instead of interleaved with text.
// ---------------------------------------------------------------------------

/// `writeStringList<T>(outPtr, stringList, entries=0)`.
///
/// `entries`: `None` writes an explicit `i32` count first (the top-level
/// string table and a `ChannelGroupManifest`'s `_channels` set); `Some(n)`
/// omits the count and instead confirms `list.len() == n` (a
/// `ChannelGroupManifest`'s `_components`, whose length is implied by the
/// table entries that follow and must not silently change).
pub fn write_string_list(out: &mut Vec<u8>, list: &[String], entries: Option<usize>) -> Result<()> {
    match entries {
        None => {
            let n = i32::try_from(list.len()).map_err(|_| {
                ExrError::Overflow("IDManifest string list too long to encode".into())
            })?;
            write_i32(out, n);
        }
        Some(n) => {
            if list.len() != n {
                return Err(ExrError::Input(
                    "Incorrect number of components stored in ID Manifest".into(),
                ));
            }
        }
    }
    for s in list {
        write_varint(out, s.len() as u64);
    }
    for s in list {
        out.extend_from_slice(s.as_bytes());
    }
    Ok(())
}

/// `readStringList<T>(readPtr, endPtr, outputVector, numberOfStrings=0)`.
///
/// `count`: `None` reads the `i32` count first; `Some(n)` uses `n` directly
/// (the caller already knows it, e.g. from `getComponents().len()`).
pub fn read_string_list(buf: &[u8], pos: &mut usize, count: Option<usize>) -> Result<Vec<String>> {
    let n = match count {
        Some(n) => n,
        None => {
            let v = read_i32(buf, pos)?;
            if v < 0 {
                return Err(ExrError::Input("Negative count for number of strings".into()));
            }
            v as usize
        }
    };

    // Coarse sanity bound: C++ requires at least `n` bytes to remain before
    // even trying to decode `n` varint lengths (each length is >= 1 byte).
    if pos.checked_add(n).is_none_or(|end| end > buf.len()) {
        return Err(ExrError::Input(
            "IDManifest too small for string length table".into(),
        ));
    }

    // First pass: decode the `n` varint lengths, advancing `pos` past the
    // whole length table (matches C++ advancing `readPtr` in this loop).
    let mut lengths = Vec::with_capacity(n);
    let mut total: usize = 0;
    for _ in 0..n {
        let len = read_varint(buf, pos)?;
        let len = usize::try_from(len)
            .map_err(|_| ExrError::Input("IDManifest string length too large".into()))?;
        total = total
            .checked_add(len)
            .ok_or_else(|| ExrError::Input("IDManifest string table size overflow".into()))?;
        lengths.push(len);
    }

    if pos.checked_add(total).is_none_or(|end| end > buf.len()) {
        return Err(ExrError::Input("IDManifest too small for string table".into()));
    }

    // Second pass: slice out each string's bytes from the position the first
    // pass left `pos` at (the start of the string data, right after the
    // length table).
    let mut result = Vec::with_capacity(n);
    for &len in &lengths {
        let bytes = &buf[*pos..*pos + len];
        let s = String::from_utf8(bytes.to_vec())
            .map_err(|e| ExrError::Input(format!("Non-UTF-8 string in IDManifest: {e}")))?;
        result.push(s);
        *pos += len;
    }
    Ok(result)
}

/// Byte-string variant of [`write_string_list`] — the top-level "all text
/// used in the manifest" table stores PREFIX-COMPRESSED entries (a 1- or
/// 2-byte "characters in common with the previous entry" header glued to a
/// tail substring), which are not valid UTF-8 in general even though every
/// entry decodes to a UTF-8 string once expanded. Same wire framing as
/// [`write_string_list`], just without the UTF-8 requirement on each entry.
pub fn write_byte_string_list(out: &mut Vec<u8>, list: &[Vec<u8>]) -> Result<()> {
    let n = i32::try_from(list.len())
        .map_err(|_| ExrError::Overflow("IDManifest string list too long to encode".into()))?;
    write_i32(out, n);
    for s in list {
        write_varint(out, s.len() as u64);
    }
    for s in list {
        out.extend_from_slice(s);
    }
    Ok(())
}

/// Read side of [`write_byte_string_list`].
pub fn read_byte_string_list(buf: &[u8], pos: &mut usize) -> Result<Vec<Vec<u8>>> {
    let v = read_i32(buf, pos)?;
    if v < 0 {
        return Err(ExrError::Input("Negative count for number of strings".into()));
    }
    let n = v as usize;

    if pos.checked_add(n).is_none_or(|end| end > buf.len()) {
        return Err(ExrError::Input(
            "IDManifest too small for string length table".into(),
        ));
    }

    let mut lengths = Vec::with_capacity(n);
    let mut total: usize = 0;
    for _ in 0..n {
        let len = read_varint(buf, pos)?;
        let len = usize::try_from(len)
            .map_err(|_| ExrError::Input("IDManifest string length too large".into()))?;
        total = total
            .checked_add(len)
            .ok_or_else(|| ExrError::Input("IDManifest string table size overflow".into()))?;
        lengths.push(len);
    }

    if pos.checked_add(total).is_none_or(|end| end > buf.len()) {
        return Err(ExrError::Input("IDManifest too small for string table".into()));
    }

    let mut result = Vec::with_capacity(n);
    for &len in &lengths {
        result.push(buf[*pos..*pos + len].to_vec());
        *pos += len;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trip_and_size() {
        for &v in &[0u64, 1, 127, 128, 16383, 16384, 1 << 30, u64::MAX] {
            let mut out = Vec::new();
            write_varint(&mut out, v);
            assert_eq!(out.len(), varint_size(v), "size mismatch for {v}");
            let mut pos = 0;
            assert_eq!(read_varint(&out, &mut pos).unwrap(), v);
            assert_eq!(pos, out.len());
        }
    }

    #[test]
    fn varint_size_thresholds() {
        assert_eq!(varint_size(0), 1);
        assert_eq!(varint_size(127), 1);
        assert_eq!(varint_size(128), 2);
        assert_eq!(varint_size((1 << 14) - 1), 2);
        assert_eq!(varint_size(1 << 14), 3);
    }

    #[test]
    fn read_varint_rejects_truncated_input() {
        // A byte with the continuation bit set but nothing after it.
        let mut pos = 0;
        assert!(read_varint(&[0x80], &mut pos).is_err());
    }

    #[test]
    fn pascal_string_round_trip() {
        let mut out = Vec::new();
        write_pascal_string(&mut out, "hello");
        write_pascal_string(&mut out, "");
        let mut pos = 0;
        assert_eq!(read_pascal_string(&out, &mut pos).unwrap(), "hello");
        assert_eq!(read_pascal_string(&out, &mut pos).unwrap(), "");
        assert_eq!(pos, out.len());
    }

    #[test]
    fn string_list_round_trip_with_explicit_count() {
        let list = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];
        let mut out = Vec::new();
        write_string_list(&mut out, &list, None).unwrap();
        let mut pos = 0;
        let back = read_string_list(&out, &mut pos, None).unwrap();
        assert_eq!(back, list);
        assert_eq!(pos, out.len());
    }

    #[test]
    fn string_list_round_trip_with_known_count() {
        let list = vec!["x".to_string(), "y".to_string()];
        let mut out = Vec::new();
        write_string_list(&mut out, &list, Some(2)).unwrap();
        let mut pos = 0;
        let back = read_string_list(&out, &mut pos, Some(2)).unwrap();
        assert_eq!(back, list);
    }

    #[test]
    fn write_string_list_rejects_count_mismatch() {
        let list = vec!["x".to_string()];
        let mut out = Vec::new();
        assert!(write_string_list(&mut out, &list, Some(2)).is_err());
    }

    #[test]
    fn byte_string_list_round_trip_with_non_utf8_bytes() {
        let list: Vec<Vec<u8>> = vec![vec![0xFF, 0xFE], vec![], vec![1, 2, 3]];
        let mut out = Vec::new();
        write_byte_string_list(&mut out, &list).unwrap();
        let mut pos = 0;
        let back = read_byte_string_list(&out, &mut pos).unwrap();
        assert_eq!(back, list);
        assert_eq!(pos, out.len());
    }

    #[test]
    fn empty_string_list_round_trips() {
        let list: Vec<String> = vec![];
        let mut out = Vec::new();
        write_string_list(&mut out, &list, None).unwrap();
        let mut pos = 0;
        assert_eq!(read_string_list(&out, &mut pos, None).unwrap(), list);
    }
}
