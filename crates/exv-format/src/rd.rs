//! Bounded little-endian reader — the single field-parsing primitive for every EXV parser.
//!
//! # Why
//! Header, container prelude/index and frame-body index all parse untrusted LE fields. Each used to
//! carry its own `rd_u32`/`rd_u64` copy (some with unchecked `p + n`, wrapping on 32-bit targets).
//! One reader means one place where "never index past the slice, never overflow the cursor" is true.
//! Every method returns `None` instead of panicking when the input is short.
//!
//! # Where used
//! `Header::read_with_meta`, `exv-video` (`VideoIndex::parse`, frame headers), and any other crate
//! that parses an EXV structure (e.g. `exv-codec::frame`).

/// Cursor over a byte slice with bounds-checked LE reads.
#[derive(Clone, Copy, Debug)]
pub struct Rd<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    /// Reader positioned at the start of `d`.
    pub fn new(d: &'a [u8]) -> Self {
        Self { d, p: 0 }
    }

    /// Reader positioned at byte `p` of `d` (`p` may exceed `d.len()`; the next read then fails).
    pub fn at(d: &'a [u8], p: usize) -> Self {
        Self { d, p }
    }

    /// Current byte offset from the start of the underlying slice.
    pub fn pos(&self) -> usize {
        self.p
    }

    /// Bytes remaining after the cursor (`0` when the cursor is past the end).
    pub fn left(&self) -> usize {
        self.d.len().saturating_sub(self.p)
    }

    /// Take the next `n` bytes and advance. `None` (cursor unchanged) if fewer remain or `p + n` overflows.
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.d.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }

    /// Take exactly `N` bytes as an array.
    fn arr<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    pub fn u8(&mut self) -> Option<u8> {
        Some(self.arr::<1>()?[0])
    }
    pub fn u16(&mut self) -> Option<u16> {
        self.arr().map(u16::from_le_bytes)
    }
    pub fn i16(&mut self) -> Option<i16> {
        self.arr().map(i16::from_le_bytes)
    }
    pub fn u32(&mut self) -> Option<u32> {
        self.arr().map(u32::from_le_bytes)
    }
    pub fn u64(&mut self) -> Option<u64> {
        self.arr().map(u64::from_le_bytes)
    }
}

/// Narrow a file-supplied `u64` offset/length to `usize` (`None` on a 32-bit overflow instead of a wrap).
pub fn off(v: u64) -> Option<usize> {
    usize::try_from(v).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_le_and_tracks_pos() {
        let d = [1u8, 2, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0xfe, 0xff];
        let mut r = Rd::new(&d);
        assert_eq!(r.u8(), Some(1));
        assert_eq!(r.u16(), Some(2));
        assert_eq!(r.u32(), Some(3));
        assert_eq!(r.u64(), Some(4));
        assert_eq!(r.i16(), Some(-2));
        assert_eq!((r.pos(), r.left()), (d.len(), 0));
    }

    #[test]
    fn short_or_overflowing_reads_fail_without_advancing() {
        let d = [1u8, 2, 3];
        let mut r = Rd::new(&d);
        assert_eq!(r.u32(), None);
        assert_eq!(r.pos(), 0, "failed read leaves the cursor in place");
        assert_eq!(r.take(usize::MAX), None, "p + n overflow is a clean None");
        assert_eq!(Rd::at(&d, usize::MAX).u8(), None);
        assert_eq!(Rd::at(&d, 99).left(), 0);
    }
}
