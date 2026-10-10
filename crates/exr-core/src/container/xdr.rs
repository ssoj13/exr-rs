//! Port of OpenEXR `ImfXdr.h` — machine-independent external data representation.
//!
//! EXR stores all multi-byte scalars in **little-endian** byte order.  The C++
//! `Xdr` namespace implements this via byte-by-byte decomposition (no
//! `to_le_bytes` intrinsic) exactly as shown in the template specialisations.
//! This port replicates every specialisation verbatim so that the on-disk
//! layout is byte-exact.
//!
//! # IO adapter
//! Instead of C++'s template parameter `S` (a helper struct with `writeChars` /
//! `readChars` static methods), we operate directly on the [`OStream`] /
//! [`IStream`] traits via free functions that take `&mut dyn OStream` /
//! `&mut dyn IStream`.  The `StreamIO` helper class in `ImfIO.h` is subsumed
//! here — callers pass the stream directly.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfXdr.h>

use imath_rs::Half;

use crate::{
    error::{ExrError, Result},
    io::stream::{IStream, OStream},
};

// ---------------------------------------------------------------------------
// Write helpers
// ---------------------------------------------------------------------------

/// Write a single byte — `Xdr::write<S>(out, char v)` / `write<S>(out, bool v)`.
#[inline]
fn write_u8(out: &mut dyn OStream, b: u8) -> Result<()> {
    out.write(&[b])
}

/// Write two bytes little-endian — internal helper.
#[inline]
fn write_u16_le(out: &mut dyn OStream, v: u16) -> Result<()> {
    let b = [v as u8, (v >> 8) as u8];
    out.write(&b)
}

/// Write four bytes little-endian — internal helper.
#[inline]
fn write_u32_le(out: &mut dyn OStream, v: u32) -> Result<()> {
    let b = [v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8];
    out.write(&b)
}

/// Write eight bytes little-endian — internal helper.
#[inline]
fn write_u64_le(out: &mut dyn OStream, v: u64) -> Result<()> {
    let b = [
        v as u8,
        (v >> 8) as u8,
        (v >> 16) as u8,
        (v >> 24) as u8,
        (v >> 32) as u8,
        (v >> 40) as u8,
        (v >> 48) as u8,
        (v >> 56) as u8,
    ];
    out.write(&b)
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

/// Read a single byte — `Xdr::read<S>(in, char/bool/u8/i8)`.
#[inline]
fn read_u8(inp: &mut dyn IStream) -> Result<u8> {
    let mut b = [0u8; 1];
    inp.read(&mut b)?;
    Ok(b[0])
}

/// Read two bytes little-endian — internal helper.
#[inline]
fn read_u16_le(inp: &mut dyn IStream) -> Result<u16> {
    let mut b = [0u8; 2];
    inp.read(&mut b)?;
    Ok((b[0] as u16) | ((b[1] as u16) << 8))
}

/// Read four bytes little-endian — internal helper.
#[inline]
fn read_u32_le(inp: &mut dyn IStream) -> Result<u32> {
    let mut b = [0u8; 4];
    inp.read(&mut b)?;
    Ok((b[0] as u32)
        | ((b[1] as u32) << 8)
        | ((b[2] as u32) << 16)
        | ((b[3] as u32) << 24))
}

/// Read eight bytes little-endian — internal helper.
#[inline]
fn read_u64_le(inp: &mut dyn IStream) -> Result<u64> {
    let mut b = [0u8; 8];
    inp.read(&mut b)?;
    Ok((b[0] as u64)
        | ((b[1] as u64) << 8)
        | ((b[2] as u64) << 16)
        | ((b[3] as u64) << 24)
        | ((b[4] as u64) << 32)
        | ((b[5] as u64) << 40)
        | ((b[6] as u64) << 48)
        | ((b[7] as u64) << 56))
}

// ---------------------------------------------------------------------------
// Public write API — mirrors every `Xdr::write` specialisation in ImfXdr.h
// ---------------------------------------------------------------------------

/// `Xdr::write<S>(T& out, bool v)` — writes `!!v` as a single byte.
pub fn write_bool(out: &mut dyn OStream, v: bool) -> Result<()> {
    write_u8(out, v as u8)
}

/// `Xdr::write<S>(T& out, char v)` — writes one byte verbatim.
pub fn write_char(out: &mut dyn OStream, v: u8) -> Result<()> {
    write_u8(out, v)
}

/// `Xdr::write<S>(T& out, signed char v)` — writes one byte verbatim.
pub fn write_i8(out: &mut dyn OStream, v: i8) -> Result<()> {
    write_u8(out, v as u8)
}

/// `Xdr::write<S>(T& out, unsigned char v)` — writes one byte verbatim.
pub fn write_u8_val(out: &mut dyn OStream, v: u8) -> Result<()> {
    write_u8(out, v)
}

/// `Xdr::write<S>(T& out, signed short v)` — writes 2 bytes LE.
///
/// ```text
/// // On disk: b[0] = v & 0xFF, b[1] = (v >> 8) & 0xFF
/// ```
pub fn write_i16(out: &mut dyn OStream, v: i16) -> Result<()> {
    write_u16_le(out, v as u16)
}

/// `Xdr::write<S>(T& out, unsigned short v)` — writes 2 bytes LE.
pub fn write_u16(out: &mut dyn OStream, v: u16) -> Result<()> {
    write_u16_le(out, v)
}

/// `Xdr::write<S>(T& out, signed int v)` — writes 4 bytes LE.
pub fn write_i32(out: &mut dyn OStream, v: i32) -> Result<()> {
    write_u32_le(out, v as u32)
}

/// `Xdr::write<S>(T& out, unsigned int v)` — writes 4 bytes LE.
pub fn write_u32(out: &mut dyn OStream, v: u32) -> Result<()> {
    write_u32_le(out, v)
}

/// `Xdr::write<S>(T& out, int64_t v)` — writes 8 bytes LE.
pub fn write_i64(out: &mut dyn OStream, v: i64) -> Result<()> {
    write_u64_le(out, v as u64)
}

/// `Xdr::write<S>(T& out, uint64_t v)` — writes 8 bytes LE.
pub fn write_u64(out: &mut dyn OStream, v: u64) -> Result<()> {
    write_u64_le(out, v)
}

/// `Xdr::write<S>(T& out, float v)` — writes the IEEE 754 bit pattern 4 bytes LE.
///
/// Replicates the C++ union `{ unsigned int i; float f }` reinterpret cast.
pub fn write_f32(out: &mut dyn OStream, v: f32) -> Result<()> {
    write_u32_le(out, v.to_bits())
}

/// `Xdr::write<S>(T& out, double v)` — writes the IEEE 754 bit pattern 8 bytes LE.
///
/// Replicates the C++ union `{ uint64_t i; double d }` reinterpret cast.
pub fn write_f64(out: &mut dyn OStream, v: f64) -> Result<()> {
    write_u64_le(out, v.to_bits())
}

/// `Xdr::write<S>(T& out, half v)` — writes the 16-bit half bit pattern LE.
///
/// Uses [`Half::bits()`] to extract the raw `u16`, then writes 2 bytes LE.
/// Matches: `b[0] = v.bits() & 0xFF; b[1] = v.bits() >> 8`.
pub fn write_half(out: &mut dyn OStream, v: Half) -> Result<()> {
    write_u16_le(out, v.bits())
}

/// `Xdr::write<S>(T& out, const char v[], int n)` — write a fixed-size char array verbatim.
pub fn write_chars(out: &mut dyn OStream, v: &[u8]) -> Result<()> {
    out.write(v)
}

/// `Xdr::write<S>(T& out, const char v[])` — write a zero-terminated string (NUL included).
///
/// Mirrors the C++ loop that writes each character then appends the `\0`.
pub fn write_string(out: &mut dyn OStream, s: &str) -> Result<()> {
    // Write the string bytes then the terminating NUL.
    out.write(s.as_bytes())?;
    out.write(&[0u8])
}

/// `Name::MAX_LENGTH` — the largest identifier `Name::operator=(const char*)`
/// will keep before truncating. Kept as a plain constant for callers that
/// still size fixed buffers against it; the truncation itself now lives on
/// [`crate::attr::Name::new`] (`truncate_name` used to back off to a
/// UTF-8 character boundary instead of cutting at the exact byte C++ does).
pub const NAME_TRUNCATE_LEN: usize = 255;

/// `Xdr::write<S>(T& out, const char v[])` for a [`crate::attr::Name`] — a
/// zero-terminated byte string, no encoding assumed. This is the wire
/// form every attribute/type/channel/part name uses; [`write_string`] stays
/// for fixed ASCII literals (type-name tags, the end-of-list marker, …).
pub fn write_name(out: &mut dyn OStream, name: &[u8]) -> Result<()> {
    out.write(name)?;
    out.write(&[0u8])
}

/// `Xdr::pad<S>(T& out, int n)` — write `n` zero bytes.
pub fn pad(out: &mut dyn OStream, n: usize) -> Result<()> {
    // Write in chunks to avoid large stack allocations.
    const CHUNK: usize = 256;
    let zeros = [0u8; CHUNK];
    let mut remaining = n;
    while remaining >= CHUNK {
        out.write(&zeros)?;
        remaining -= CHUNK;
    }
    if remaining > 0 {
        out.write(&zeros[..remaining])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public read API — mirrors every `Xdr::read` specialisation in ImfXdr.h
// ---------------------------------------------------------------------------

/// `Xdr::read<S>(T& in, bool& v)` — reads one byte, `v = !!c`.
pub fn read_bool(inp: &mut dyn IStream) -> Result<bool> {
    Ok(read_u8(inp)? != 0)
}

/// `Xdr::read<S>(T& in, char& v)` — reads one byte verbatim.
pub fn read_char(inp: &mut dyn IStream) -> Result<u8> {
    read_u8(inp)
}

/// `Xdr::read<S>(T& in, signed char& v)`.
pub fn read_i8(inp: &mut dyn IStream) -> Result<i8> {
    Ok(read_u8(inp)? as i8)
}

/// `Xdr::read<S>(T& in, unsigned char& v)`.
pub fn read_u8_val(inp: &mut dyn IStream) -> Result<u8> {
    read_u8(inp)
}

/// `Xdr::read<S>(T& in, signed short& v)` — 2 bytes LE.
///
/// C++: `v = (b[0] & 0x00ff) | (b[1] << 8)` with sign propagation.
pub fn read_i16(inp: &mut dyn IStream) -> Result<i16> {
    Ok(read_u16_le(inp)? as i16)
}

/// `Xdr::read<S>(T& in, unsigned short& v)` — 2 bytes LE.
pub fn read_u16(inp: &mut dyn IStream) -> Result<u16> {
    read_u16_le(inp)
}

/// `Xdr::read<S>(T& in, signed int& v)` — 4 bytes LE.
///
/// C++: assembles as unsigned then assigns to signed int (defined behaviour in
/// C++ for values representable in the signed type; we use `as i32` which is
/// equivalent via two's-complement wrapping, matching the C++ intent).
pub fn read_i32(inp: &mut dyn IStream) -> Result<i32> {
    Ok(read_u32_le(inp)? as i32)
}

/// `Xdr::read<S>(T& in, unsigned int& v)` — 4 bytes LE.
pub fn read_u32(inp: &mut dyn IStream) -> Result<u32> {
    read_u32_le(inp)
}

/// `Xdr::read<S>(T& in, int64_t& v)` — 8 bytes LE.
pub fn read_i64(inp: &mut dyn IStream) -> Result<i64> {
    Ok(read_u64_le(inp)? as i64)
}

/// `Xdr::read<S>(T& in, uint64_t& v)` — 8 bytes LE.
pub fn read_u64(inp: &mut dyn IStream) -> Result<u64> {
    read_u64_le(inp)
}

/// `Xdr::read<S>(T& in, float& v)` — 4 bytes LE, reinterpreted as `f32`.
///
/// Replicates the C++ union: assemble unsigned int, then `u.f`.
pub fn read_f32(inp: &mut dyn IStream) -> Result<f32> {
    Ok(f32::from_bits(read_u32_le(inp)?))
}

/// `Xdr::read<S>(T& in, double& v)` — 8 bytes LE, reinterpreted as `f64`.
///
/// Replicates the C++ union: assemble uint64_t, then `u.d`.
pub fn read_f64(inp: &mut dyn IStream) -> Result<f64> {
    Ok(f64::from_bits(read_u64_le(inp)?))
}

/// `Xdr::read<S>(T& in, half& v)` — 2 bytes LE → half bit pattern.
///
/// C++: `v.setBits((b[0] & 0x00ff) | (b[1] << 8))`.
pub fn read_half(inp: &mut dyn IStream) -> Result<Half> {
    let bits = read_u16_le(inp)?;
    Ok(Half::from_bits(bits))
}

/// `Xdr::read<S>(T& in, char v[], int n)` — read exactly `n` bytes into `buf`.
pub fn read_chars(inp: &mut dyn IStream, buf: &mut [u8]) -> Result<()> {
    inp.read(buf)?;
    Ok(())
}

/// Read exactly `size` raw bytes into a freshly-allocated buffer, refusing to
/// pre-allocate more than the stream can actually supply.
///
/// Variable-size attribute readers take an attacker-controlled `size` field
/// straight off the wire. Allocating it eagerly (`vec![0u8; size]`) lets a tiny
/// truncated file declare a multi-gigabyte attribute and abort the process
/// before the short read is ever detected. This helper closes that hole in two
/// independent layers:
///
/// * When the stream reports its total length (`size()` is `Some`), any `size`
///   larger than the bytes remaining from the current position is rejected
///   **before** allocating — the declaration cannot possibly be satisfied.
/// * The buffer is grown fallibly (`try_reserve_exact`), so even a
///   within-remaining but still-huge request surfaces as an `Input` error
///   instead of aborting on allocation failure.
///
/// Streams that cannot report their size (`size()` is `None`, e.g. a pure
/// streaming source) skip the remaining-bytes check but keep the
/// fallible-allocation guard. This subsumes [`read_chars`] for the raw,
/// size-delimited byte payloads that attribute readers use.
pub fn read_bounded_bytes(inp: &mut dyn IStream, size: usize) -> Result<Vec<u8>> {
    if let Some(total) = inp.size() {
        let pos = inp.tellg()?;
        let remaining = total.saturating_sub(pos);
        if size as u64 > remaining {
            return Err(ExrError::Input(format!(
                "attribute declares {size} bytes but only {remaining} remain"
            )));
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(size).map_err(|_| {
        ExrError::Input(format!(
            "failed to allocate {size} bytes for attribute value"
        ))
    })?;
    buf.resize(size, 0);

    if size > 0 {
        inp.read(&mut buf)?;
    }
    Ok(buf)
}

/// `Xdr::read<S>(T& in, int n, char v[])` — read a NUL-terminated string into `buf`.
///
/// Reads bytes one at a time until a NUL is found or `max_len` bytes have been
/// read (not counting the NUL).  Mirrors the C++ loop:
/// ```text
/// while (n >= 0) { read 1 byte; if (*v == 0) break; --n; ++v; }
/// ```
/// The buffer receives the string content **without** the NUL terminator; the
/// returned `usize` is the number of bytes written (not including NUL).
///
/// Returns `ExrError::Input` if the string exceeds `max_len` bytes without a
/// NUL (the C++ simply stops reading, leaving the string unterminated — we
/// surface this as an error to avoid silent truncation).
pub fn read_string(inp: &mut dyn IStream, buf: &mut Vec<u8>, max_len: usize) -> Result<()> {
    if read_name(inp, buf, max_len)? {
        Ok(())
    } else {
        Err(ExrError::Input(format!(
            "String exceeds maximum length of {max_len} bytes."
        )))
    }
}

/// [`read_string`] that reports an over-long name as `Ok(false)` instead of an error, having
/// consumed exactly `max_len + 1` bytes -- where C++ `parse_header.c::read_text` stops and
/// where its non-strict header loop resumes parsing (`buf` then holds the bytes read so far).
pub fn read_name(inp: &mut dyn IStream, buf: &mut Vec<u8>, max_len: usize) -> Result<bool> {
    buf.clear();
    loop {
        let b = read_u8(inp)?;
        if b == 0 {
            return Ok(true);
        }
        // `max_len` is the maximum PAYLOAD length (OpenEXR `Name::MAX_LENGTH` =
        // `SIZE - 1`; the fixed C++ `char[SIZE]` buffer reserves the last slot for
        // the NUL terminator). The `max_len + 1`-th byte is where C++ gives up, so
        // accepting at most `max_len` payload bytes is identical for every name C++
        // accepts.
        if buf.len() >= max_len {
            return Ok(false);
        }
        buf.push(b);
    }
}

/// Write a length or count as an Xdr `int` (i32), rejecting a value that does
/// not fit the signed wire field instead of narrowing it with `as i32`.
///
/// A raw `len as i32` on a `usize` past `i32::MAX` would write a negative or
/// truncated length and then emit the full payload — a corrupt, unreadable
/// stream produced by an otherwise-fallible API. This is the single checked
/// conversion for every size-prefixed wire field (attribute value sizes,
/// string-vector element lengths, …).
pub fn write_wire_len(out: &mut dyn OStream, len: usize) -> Result<()> {
    write_i32(out, checked_wire_len(len)?)
}

/// Validate a length before allocating or generating its size-prefixed payload.
pub fn checked_wire_len(len: usize) -> Result<i32> {
    i32::try_from(len).map_err(|_| {
        ExrError::Overflow(format!("value {len} exceeds the i32 wire-length limit"))
    })
}

/// `Xdr::skip<S>(T& in, int n)` — skip (consume) `n` padding bytes.
///
/// Mirrors the C++ implementation: reads in 1024-byte chunks to discard data.
pub fn skip(inp: &mut dyn IStream, n: usize) -> Result<()> {
    const CHUNK: usize = 1024;
    let mut buf = [0u8; CHUNK];
    let mut remaining = n;
    while remaining >= CHUNK {
        inp.read(&mut buf)?;
        remaining -= CHUNK;
    }
    if remaining > 0 {
        inp.read(&mut buf[..remaining])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Size constants — port of `Xdr::size<S>()` template specialisations
// ---------------------------------------------------------------------------

/// On-disk byte size of a `bool` — `Xdr::size<bool>()` → 1.
pub const SIZE_BOOL: usize = 1;
/// On-disk byte size of a `char` / `i8` / `u8` — `Xdr::size<char>()` → 1.
pub const SIZE_I8: usize = 1;
/// On-disk byte size of a `short` / `u16` — `Xdr::size<short>()` → 2.
pub const SIZE_I16: usize = 2;
/// On-disk byte size of an `int` / `u32` — `Xdr::size<int>()` → 4.
pub const SIZE_I32: usize = 4;
/// On-disk byte size of a `long` / `u64` — `Xdr::size<long>()` → 8.
pub const SIZE_I64: usize = 8;
/// On-disk byte size of a `float` — `Xdr::size<float>()` → 4.
pub const SIZE_F32: usize = 4;
/// On-disk byte size of a `double` — `Xdr::size<double>()` → 8.
pub const SIZE_F64: usize = 8;
/// On-disk byte size of a `half` — `Xdr::size<half>()` → 2.
pub const SIZE_HALF: usize = 2;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    // The truncate-at-first-NUL / cap-at-255-bytes coverage lives in
    // `crate::attr::name::tests` (`Name::new` is now the single
    // truncation point `truncate_name` used to be).

    /// Helper: write with a closure, get the bytes, read back.
    fn round_trip<W, R, T>(write_fn: W, read_fn: R, value: T) -> (Vec<u8>, T)
    where
        W: Fn(&mut dyn OStream, T) -> Result<()>,
        R: Fn(&mut dyn IStream) -> Result<T>,
        T: Copy,
    {
        let mut out = MemOStream::new();
        write_fn(&mut out, value).unwrap();
        let bytes = out.into_data();

        let mut inp = MemIStream::new(bytes.clone());
        let back = read_fn(&mut inp).unwrap();
        (bytes, back)
    }

    // -----------------------------------------------------------------------
    // Exact on-disk byte checks (the gold standard for byte-exactness)
    // -----------------------------------------------------------------------

    #[test]
    fn i32_one_le_bytes() {
        // i32 = 1 → LE bytes 01 00 00 00
        let mut out = MemOStream::new();
        write_i32(&mut out, 1).unwrap();
        assert_eq!(out.data(), &[0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn i32_neg_one_le_bytes() {
        // i32 = -1 → LE bytes FF FF FF FF
        let mut out = MemOStream::new();
        write_i32(&mut out, -1).unwrap();
        assert_eq!(out.data(), &[0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn f32_one_le_bytes() {
        // f32 = 1.0 → IEEE 754 bits 0x3F800000 → LE 00 00 80 3F
        let mut out = MemOStream::new();
        write_f32(&mut out, 1.0f32).unwrap();
        assert_eq!(out.data(), &[0x00, 0x00, 0x80, 0x3F]);
    }

    #[test]
    fn f32_zero_le_bytes() {
        // f32 = 0.0 → bits 0x00000000
        let mut out = MemOStream::new();
        write_f32(&mut out, 0.0f32).unwrap();
        assert_eq!(out.data(), &[0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn half_one_le_bytes() {
        // half = 1.0 → bits 0x3C00 → LE 00 3C
        let h = Half::new(1.0f32);
        assert_eq!(h.bits(), 0x3C00, "Half::new(1.0) should be 0x3C00");
        let mut out = MemOStream::new();
        write_half(&mut out, h).unwrap();
        assert_eq!(out.data(), &[0x00, 0x3C]);
    }

    #[test]
    fn half_zero_le_bytes() {
        // half = 0.0 → bits 0x0000 → LE 00 00
        let h = Half::new(0.0f32);
        let mut out = MemOStream::new();
        write_half(&mut out, h).unwrap();
        assert_eq!(out.data(), &[0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // Round-trip tests for every type
    // -----------------------------------------------------------------------

    #[test]
    fn bool_round_trip() {
        let (bytes, back) = round_trip(write_bool, read_bool, true);
        assert_eq!(bytes, &[1u8]);
        assert!(back);

        let (bytes2, back2) = round_trip(write_bool, read_bool, false);
        assert_eq!(bytes2, &[0u8]);
        assert!(!back2);
    }

    #[test]
    fn i8_round_trip() {
        let (_, back) = round_trip(write_i8, read_i8, -42i8);
        assert_eq!(back, -42i8);
    }

    #[test]
    fn u8_round_trip() {
        let (_, back) = round_trip(write_u8_val, read_u8_val, 200u8);
        assert_eq!(back, 200u8);
    }

    #[test]
    fn i16_round_trip() {
        let (_, back) = round_trip(write_i16, read_i16, -1234i16);
        assert_eq!(back, -1234i16);
    }

    #[test]
    fn u16_round_trip() {
        let (_, back) = round_trip(write_u16, read_u16, 0xABCDu16);
        assert_eq!(back, 0xABCDu16);
    }

    #[test]
    fn i32_round_trip() {
        let (_, back) = round_trip(write_i32, read_i32, -100_000i32);
        assert_eq!(back, -100_000i32);
    }

    #[test]
    fn u32_round_trip() {
        let (_, back) = round_trip(write_u32, read_u32, 0xDEAD_BEEFu32);
        assert_eq!(back, 0xDEAD_BEEFu32);
    }

    #[test]
    fn i64_round_trip() {
        let (_, back) = round_trip(write_i64, read_i64, i64::MIN);
        assert_eq!(back, i64::MIN);
    }

    #[test]
    fn u64_round_trip() {
        let (_, back) = round_trip(write_u64, read_u64, u64::MAX);
        assert_eq!(back, u64::MAX);
    }

    #[test]
    fn f32_round_trip() {
        let (_, back) = round_trip(write_f32, read_f32, core::f32::consts::PI);
        assert_eq!(back.to_bits(), core::f32::consts::PI.to_bits());
    }

    #[test]
    fn f64_round_trip() {
        let (_, back) = round_trip(write_f64, read_f64, core::f64::consts::E);
        assert_eq!(back.to_bits(), core::f64::consts::E.to_bits());
    }

    #[test]
    fn half_round_trip() {
        let h = Half::new(2.5f32);
        let mut out = MemOStream::new();
        write_half(&mut out, h).unwrap();
        let bytes = out.into_data();
        let mut inp = MemIStream::new(bytes);
        let back = read_half(&mut inp).unwrap();
        assert_eq!(back.bits(), h.bits());
    }

    #[test]
    fn chars_round_trip() {
        let data = b"hello";
        let mut out = MemOStream::new();
        write_chars(&mut out, data).unwrap();
        assert_eq!(out.data(), data);

        let mut buf = vec![0u8; 5];
        let mut inp = MemIStream::new(out.into_data());
        read_chars(&mut inp, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn string_write_has_nul() {
        let mut out = MemOStream::new();
        write_string(&mut out, "abc").unwrap();
        assert_eq!(out.data(), b"abc\0");
    }

    #[test]
    fn string_round_trip() {
        let mut out = MemOStream::new();
        write_string(&mut out, "rgba").unwrap();
        let mut inp = MemIStream::new(out.into_data());
        let mut buf = Vec::new();
        read_string(&mut inp, &mut buf, 255).unwrap();
        assert_eq!(&buf, b"rgba");
    }

    #[test]
    fn pad_skip_round_trip() {
        let mut out = MemOStream::new();
        write_i32(&mut out, 42).unwrap();
        pad(&mut out, 4).unwrap();
        write_i32(&mut out, 99).unwrap();

        let mut inp = MemIStream::new(out.into_data());
        let a = read_i32(&mut inp).unwrap();
        skip(&mut inp, 4).unwrap();
        let b = read_i32(&mut inp).unwrap();
        assert_eq!(a, 42);
        assert_eq!(b, 99);
    }

    #[test]
    fn read_string_caps_at_max_len() {
        // `max_len` = 255 payload bytes (Name::MAX_LENGTH). Lengths up to 255 are
        // accepted; the 256th payload byte is rejected (C++ accepted it and left
        // its fixed buffer unterminated).
        let mk = |n: usize| {
            let mut v = vec![b'a'; n];
            v.push(0); // NUL terminator
            v
        };
        for n in [0usize, 1, 254, 255] {
            let mut inp = MemIStream::new(mk(n));
            let mut buf = Vec::new();
            read_string(&mut inp, &mut buf, 255).unwrap();
            assert_eq!(buf.len(), n);
        }
        let mut inp = MemIStream::new(mk(256));
        let mut buf = Vec::new();
        assert!(read_string(&mut inp, &mut buf, 255).is_err());
    }

    #[test]
    fn checked_wire_len_preserves_boundary_and_error() {
        assert_eq!(checked_wire_len(0).unwrap(), 0);
        assert_eq!(checked_wire_len(i32::MAX as usize).unwrap(), i32::MAX);
        let invalid = i32::MAX as usize + 1;
        let mut out = MemOStream::new();
        match write_wire_len(&mut out, invalid).unwrap_err() {
            ExrError::Overflow(message) => assert_eq!(
                message, "value 2147483648 exceeds the i32 wire-length limit",
            ),
            other => panic!("expected Overflow, got {other:?}"),
        }
        assert!(out.into_data().is_empty(), "a rejected length writes no bytes");
    }

    #[test]
    fn write_wire_len_rejects_over_i32() {
        let mut out = MemOStream::new();
        assert!(write_wire_len(&mut out, i32::MAX as usize).is_ok());
        let mut out2 = MemOStream::new();
        assert!(write_wire_len(&mut out2, i32::MAX as usize + 1).is_err());
    }
}
