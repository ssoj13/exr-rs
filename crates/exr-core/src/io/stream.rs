//! Port of OpenEXR `ImfIO.h` — abstract I/O stream traits.
//!
//! `IStream` and `OStream` are the abstract base classes that all of IlmImf's
//! I/O goes through.  Every reader and writer accepts a reference to one of
//! these traits rather than a concrete file type, allowing in-memory streams,
//! network streams, etc.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfIO.h>

use crate::error::Result;

/// Abstract input stream — port of `Imf::IStream`.
///
/// Every method that can fail returns `Result<_>` rather than throwing C++
/// exceptions.  The semantics mirror the C++ class exactly:
///
/// * [`IStream::read`] must read **exactly** `buf.len()` bytes or return an error
///   (matching the C++ contract "if the stream contains less than n bytes, …
///   throws an exception").
/// * Returns `Ok(false)` when the **last** byte of the file was just consumed,
///   `Ok(true)` otherwise — matching the C++ `bool` return of `read(c, n)`.
/// * [`IStream::seekg`] / [`IStream::tellg`] mirror `seekg(uint64_t)` / `uint64_t tellg()`.
///
/// The optional memory-mapped interface (`is_memory_mapped`, `read_memory_mapped`)
/// mirrors the C++ virtual methods; both have default `false`/error
/// implementations, so implementors only override them when relevant.
pub trait IStream {
    /// Name of the underlying file or stream (mirrors `IStream::fileName()`).
    fn file_name(&self) -> &str;

    /// Read exactly `buf.len()` bytes from the stream.
    ///
    /// Returns `Ok(false)` if the last byte in the stream was just read (EOF
    /// reached), `Ok(true)` otherwise.  Returns `Err` on short read or I/O
    /// error, mirroring the C++ `InputExc` throw.
    fn read(&mut self, buf: &mut [u8]) -> Result<bool>;

    /// Current read position in bytes from the start of the stream.
    ///
    /// Mirrors `uint64_t IStream::tellg()`.
    fn tellg(&mut self) -> Result<u64>;

    /// Seek to an absolute byte offset from the start of the stream.
    ///
    /// Mirrors `void IStream::seekg(uint64_t pos)`.
    fn seekg(&mut self, pos: u64) -> Result<()>;

    /// Clear any error state on the stream.
    ///
    /// Default no-op; override for streams backed by `std::io::BufReader` etc.
    /// Mirrors `virtual void IStream::clear()`.
    fn clear(&mut self) {}

    /// Return the total size of the stream in bytes, or `None` if unknown.
    ///
    /// Mirrors `virtual int64_t IStream::size()` which returns `-1` when
    /// unknown.  `None` here skips the same consistency checks.
    fn size(&self) -> Option<u64> {
        None
    }

    /// Whether this stream supports memory-mapped reads.
    ///
    /// Default `false`.  Override to return `true` and implement
    /// [`read_memory_mapped`][IStream::read_memory_mapped] when the backing
    /// store is already mapped into virtual memory.
    fn is_memory_mapped(&self) -> bool {
        false
    }

    /// Read `n` bytes from a memory-mapped stream, returning a slice into the
    /// mapping.
    ///
    /// Only called when [`is_memory_mapped`][IStream::is_memory_mapped] returns
    /// `true`.  The slice must remain valid for the lifetime of the stream.
    /// Default implementation always returns an error.
    fn read_memory_mapped(&mut self, _n: usize) -> Result<&[u8]> {
        Err(crate::error::ExrError::Input(
            "stream is not memory-mapped".into(),
        ))
    }

    /// Whether this stream supports *stateless* reads via
    /// [`read_stateless`][IStream::read_stateless] — mirrors
    /// `virtual bool IStream::isStatelessRead() const`.
    ///
    /// A stateless-capable stream lets multiple threads read from different
    /// offsets concurrently, since the read does not depend on or move a
    /// shared cursor (unlike [`read`][IStream::read]/[`seekg`][IStream::seekg]).
    /// Default `false`, matching the C++ default.
    fn is_stateless_read(&self) -> bool {
        false
    }

    /// Read up to `buf.len()` bytes starting at the absolute byte `offset`,
    /// without touching the stream's shared read cursor — mirrors
    /// `virtual int64_t IStream::read(void* buf, uint64_t sz, uint64_t offset)`.
    ///
    /// Only called when [`is_stateless_read`][IStream::is_stateless_read]
    /// returns `true`. Unlike [`read`][IStream::read], reading past the end of
    /// the stream is **not** an error: C++ documents "if the offset is past
    /// the end, it should read 0", and a request that partially overruns the
    /// end returns however many bytes were actually available. Returns the
    /// number of bytes read (`0..=buf.len()`).
    ///
    /// Default implementation mirrors the C++ default body — an
    /// unconditional `InputExc` ("Attempt to perform a stateless read on a
    /// stream without support") — via [`ExrError::Input`][crate::error::ExrError::Input].
    fn read_stateless(&self, _buf: &mut [u8], _offset: u64) -> Result<i64> {
        Err(crate::error::ExrError::Input(
            "Attempt to perform a stateless read on a stream without support.".into(),
        ))
    }
}

/// Abstract output stream — port of `Imf::OStream`.
///
/// Mirrors the C++ class: [`write`] stores bytes and throws on I/O error
/// (here returns `Err`).  [`OStream::seekp`] / [`OStream::tellp`] mirror the C++ counterparts.
pub trait OStream {
    /// Name of the underlying file or stream (mirrors `OStream::fileName()`).
    fn file_name(&self) -> &str;

    /// Write all bytes in `buf` to the stream.
    ///
    /// Returns `Err` on any I/O failure, mirroring the C++ `write()` throw.
    fn write(&mut self, buf: &[u8]) -> Result<()>;

    /// Current write position in bytes from the start of the stream.
    ///
    /// Mirrors `uint64_t OStream::tellp()`.
    fn tellp(&mut self) -> Result<u64>;

    /// Seek to an absolute byte offset from the start of the stream.
    ///
    /// Mirrors `void OStream::seekp(uint64_t pos)`.
    fn seekp(&mut self, pos: u64) -> Result<()>;

    /// Push everything still buffered by this stream to its destination and report a failure.
    ///
    /// Writers that must hand back a COMPLETE file (e.g. [`crate::tiled::TiledOutputFile::finish`]) call
    /// this, so a failed final write is returned instead of only logged by a destructor. The default is a
    /// no-op for unbuffered streams such as in-memory ones; buffered streams override it. C++ `OStream`
    /// has no equivalent.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
