//! Port of OpenEXR `ImfStdIO.h` / `ImfStdIO.cpp` — concrete stream adaptors.
//!
//! Provides four concrete [`IStream`] / [`OStream`] implementations backed by
//! Rust's standard library types, mirroring the four C++ classes:
//!
//! | C++            | Rust           | Backed by                      |
//! |----------------|----------------|--------------------------------|
//! | `StdIFStream`  | [`StdIfStream`]| `std::fs::File` (read)         |
//! | `StdISStream`  | [`MemIStream`] | `std::io::Cursor<&[u8]>`       |
//! | `StdOFStream`  | [`StdOfStream`]| `std::fs::File` (write)        |
//! | `StdOSStream`  | [`MemOStream`] | `std::io::Cursor<Vec<u8>>`     |
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfStdIO.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfStdIO.cpp>

use std::{
    fs::File,
    io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    error::{ExrError, Result},
    io::stream::{IStream, OStream},
};

// ---------------------------------------------------------------------------
// StdIfStream — port of C++ StdIFStream
// ---------------------------------------------------------------------------

/// File-backed input stream — port of `Imf::StdIFStream`.
///
/// Opens a file for binary reading on construction.  The file is closed when
/// the struct is dropped.
pub struct StdIfStream {
    file_name: String,
    /// A raw `File` here made every `seekg` a real syscall even when the
    /// target offset was already inside the OS's page cache / this
    /// process's most recent read — and scanline/tiled readers issue one
    /// `seekg` PER CHUNK (`fetch_chunk`/`fetch_tile`, unconditionally, even
    /// for a freshly-written, purely-sequential file where the cursor is
    /// already exactly there). `BufReader` coalesces the actual `read()`
    /// syscalls that follow a seek (e.g. a chunk's small leader read then
    /// its data read), but does NOT by itself fix the redundant seek: `Seek
    /// for BufReader<R>`'s `SeekFrom::Start` arm unconditionally delegates
    /// to the inner reader and discards the buffer, regardless of whether
    /// the target is the position already sitting in it (confirmed in
    /// std's source). `pos` (below) is what
    /// actually eliminates the redundant syscall: `seekg` compares against
    /// it and skips the underlying `Seek::seek` call — and therefore the
    /// buffer discard — entirely when the caller is already there.
    file: BufReader<File>,
    /// This crate's own tracked logical read position (bytes from the
    /// start), updated after every successful `read`/`seekg` on this
    /// struct. `seekg` uses it to skip the underlying seek when the target
    /// is already the current position (see `file`'s doc) — a real,
    /// deterministic optimization, unlike relying on `BufReader` to notice
    /// this on its own (it can't, since `Seek::seek` with `SeekFrom::Start`
    /// gives it no way to know a seek is a no-op before discarding).
    pos: u64,
    /// Total file length in bytes, cached at open time.
    ///
    /// Backs [`IStream::size`] so the disk-file path can apply the
    /// remaining-bytes bound that guards variable-size attribute allocation
    /// (the C++ `StdIFStream` exposes the same via `size()`).
    len: u64,
}

impl StdIfStream {
    /// Open the named file for reading (mirrors `StdIFStream(const char[])`).
    ///
    /// Returns `ExrError::Errno` if the file cannot be opened, matching the C++
    /// `IEX_NAMESPACE::throwErrnoExc()` call.
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path).map_err(ExrError::from_io)?;
        // Cache the length up front (via metadata, no seek) so `size()` is O(1)
        // and never perturbs the read cursor.
        let len = file.metadata().map_err(ExrError::from_io)?.len();
        Ok(Self {
            file_name: path.to_owned(),
            file: BufReader::with_capacity(256 * 1024, file),
            pos: 0,
            len,
        })
    }
}

impl IStream for StdIfStream {
    fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Read exactly `buf.len()` bytes.
    ///
    /// Mirrors the C++ `StdIFStream::read` (`ImfStdIO.cpp:124-131`):
    /// `_is->read(c, n)` sets `failbit`/`eofbit` only when it extracts
    /// *fewer* than `n` characters — reading exactly to the last byte of the
    /// file is a normal, fully successful `istream::read`. `checkError` then
    /// returns `true` whenever the stream is still good, and only throws
    /// (never silently returns `false`) on a short read:
    /// * `InputExc("Unexpected end of file.")` when the stream is already at
    ///   EOF before the read even starts.
    /// * `InputExc("Early end of file: …")` on a short read mid-buffer.
    ///
    /// This used to disagree with C++ by probing one byte past the
    /// requested data (plus a `seek(-1)`) to decide whether the *last*
    /// successful read should report `false` — three syscalls per read for a
    /// distinction the real reader never observes, since no caller inspects
    /// the bool for anything other than "did this fail" (`Xdr::skip` is the
    /// one C++ consumer, and this crate's `xdr::skip` already ignores it).
    fn read(&mut self, buf: &mut [u8]) -> Result<bool> {
        let n = buf.len();
        let mut total = 0usize;

        // Read until the buffer is full or an error / EOF occurs.
        while total < n {
            match self.file.read(&mut buf[total..]) {
                Ok(0) => {
                    // EOF before we filled the buffer.
                    if total == 0 {
                        // Mirrors: if (!*_is) throw InputExc("Unexpected end of file.")
                        return Err(ExrError::Input("Unexpected end of file.".into()));
                    } else {
                        // Mirrors: "Early end of file: read X out of Y requested bytes."
                        return Err(ExrError::Input(format!(
                            "Early end of file: read {total} out of {n} requested bytes."
                        )));
                    }
                }
                Ok(k) => total += k,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(ExrError::from_io(e)),
            }
        }
        self.pos += n as u64;

        // A full, exact read of `n` bytes is always `true` in C++, even when
        // it consumed the file's last byte.
        Ok(true)
    }

    fn tellg(&mut self) -> Result<u64> {
        // `self.pos` is this struct's own tracked position (see its doc),
        // kept exactly in sync with every `read`/`seekg` — no need to ask
        // the underlying `BufReader` (which would cost nothing here anyway,
        // but this keeps the single source of truth in one place).
        Ok(self.pos)
    }

    fn seekg(&mut self, pos: u64) -> Result<()> {
        // Skip the underlying seek (and the buffer discard it always causes
        // — see `file`'s doc) when we are already there. The common case:
        // `fetch_chunk`/`fetch_tile` seek to a chunk before every read, but
        // a purely sequential/in-order chunk walk (the default write order)
        // means the cursor is usually already sitting exactly at that chunk.
        if pos == self.pos {
            return Ok(());
        }
        self.file
            .seek(SeekFrom::Start(pos))
            .map_err(ExrError::from_io)?;
        self.pos = pos;
        Ok(())
    }

    /// Total file length, cached at open time (mirrors `StdIFStream::size()`).
    fn size(&self) -> Option<u64> {
        Some(self.len)
    }
}

// ---------------------------------------------------------------------------
// StdOfStream — port of C++ StdOFStream
// ---------------------------------------------------------------------------

/// File-backed output stream — port of `Imf::StdOFStream`.
///
/// Creates (or truncates) a file for binary writing on construction.  The file
/// is flushed and closed when the struct is dropped.
pub struct StdOfStream {
    file_name: String,
    /// Same rationale as `StdIfStream::file` — a raw
    /// `File` made every `seekp` (the writer back-patches offset-table
    /// entries and, for tiled/deep, out-of-order chunk positions)
    /// a real syscall even when the target offset sits inside the write
    /// buffer BufWriter would already hold. As with `StdIfStream`, `Seek for
    /// BufWriter<W>`'s `SeekFrom::Start` arm unconditionally flushes and
    /// delegates to a real underlying seek regardless of whether the target
    /// is already the current position — `pos` (below) is what actually
    /// skips the redundant syscall.
    file: BufWriter<File>,
    /// This crate's own tracked logical write position — see
    /// `StdIfStream::pos`'s doc for why `BufWriter` cannot notice a no-op
    /// seek on its own.
    pos: u64,
}

impl StdOfStream {
    /// Create or truncate the named file for writing (mirrors `StdOFStream(const char[])`).
    ///
    /// Returns `ExrError::Errno` on failure, matching C++ `throwErrnoExc()`.
    pub fn create(path: &str) -> Result<Self> {
        let file = File::create(path).map_err(ExrError::from_io)?;
        Ok(Self {
            file_name: path.to_owned(),
            file: BufWriter::with_capacity(256 * 1024, file),
            pos: 0,
        })
    }

    /// Write everything still buffered to the file and report a failure. Call it after the
    /// last write: `Drop` flushes too, but can only log a failure, so a file that could not be
    /// completed (disk full, network drive gone) would otherwise look written. C++
    /// `StdOFStream` has no such call (its destructor also ignores the error); this is the
    /// seam every file writer here ends with.
    pub fn flush(&mut self) -> Result<()> {
        self.file.flush().map_err(ExrError::from_io)
    }
}

/// Lets a non-EXR encoder (PNG, ...) write through the same all-or-nothing `write_file` stream.
/// Writes go through `OStream::write`, so the tracked position stays right.
impl std::io::Write for StdOfStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        OStream::write(self, buf).map_err(std::io::Error::other)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        StdOfStream::flush(self).map_err(std::io::Error::other)
    }
}

impl OStream for StdOfStream {
    fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Write all bytes in `buf` to the file.
    ///
    /// Mirrors `StdOFStream::write`: calls `checkError(*_os)` after the write,
    /// which throws `ErrnoExc("File output failed.")` on any failure.
    fn write(&mut self, buf: &[u8]) -> Result<()> {
        self.file.write_all(buf).map_err(ExrError::from_io)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn tellp(&mut self) -> Result<u64> {
        Ok(self.pos)
    }

    fn seekp(&mut self, pos: u64) -> Result<()> {
        // Skip the underlying seek (and the flush-then-discard it always
        // causes) when we are already there — see `file`'s doc.
        if pos == self.pos {
            return Ok(());
        }
        self.file
            .seek(SeekFrom::Start(pos))
            .map_err(ExrError::from_io)?;
        self.pos = pos;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        StdOfStream::flush(self)
    }
}

impl Drop for StdOfStream {
    /// `BufWriter`'s own `Drop` flushes but silently swallows the result (the
    /// stdlib docs explicitly warn against relying on it) — this crate's
    /// "never silently discard a fallible result" rule still applies even
    /// though `Drop::drop` cannot return `Result`: an explicit flush here at
    /// least surfaces a failed final write as a loud `log::error!` instead of
    /// disappearing entirely. Every normal call path (writer destructors,
    /// tests) drops this stream only after the last `write`/`seekp`, so a
    /// flush failure here means the file is genuinely incomplete on disk.
    fn drop(&mut self) {
        if let Err(e) = self.file.flush() {
            log::error!(
                "StdOfStream({}): final flush failed, file is likely incomplete: {e}",
                self.file_name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// write_file — the one way to write a file
// ---------------------------------------------------------------------------

/// Distinguishes the temporary files of concurrent writes within one process.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Write the file at `path` through `write`, all or nothing: the bytes go to a temporary file
/// next to `path` (same directory, so the final rename stays on one volume), which is flushed
/// with its error checked and then renamed over `path`. On any failure — `write` returning an
/// error, a failed final flush (disk full), a failed rename — the temporary file is removed and
/// `path` is left exactly as it was: absent, or the previous file untouched.
///
/// Every file writer in the workspace goes through this function (a test in the root crate
/// fails on any other production `StdOfStream::create`), so none can report success for an
/// incomplete file or leave a truncated one behind. C++ OpenEXR writes in place and ignores the
/// destructor's flush error; a failed C++ write leaves a partial file (e.g. `exrmakepreview` on
/// a deep image) — this is the deliberate difference.
///
/// The error type is the caller's (`E: From<ExrError>`), so crates with their own error type
/// (the CLI, Cryptomatte, the Python module) use it directly.
pub fn write_file<T, E: From<ExrError>>(
    path: impl AsRef<Path>,
    write: impl FnOnce(&mut StdOfStream) -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    let path = path.as_ref();
    let tmp = tmp_path(path)?;
    let tmp_str = tmp
        .to_str()
        .ok_or_else(|| ExrError::Arg(format!("Path is not valid UTF-8: {}", path.display())))?;
    let result = StdOfStream::create(tmp_str).map_err(E::from).and_then(|mut stream| {
        let value = write(&mut stream)?;
        stream.flush()?;
        Ok(value)
    });
    // The stream is closed here (dropped inside `and_then`), so the rename and the removal
    // below act on a closed file (Windows refuses both on an open one).
    match result.and_then(|value| std::fs::rename(&tmp, path).map(|()| value).map_err(|e| E::from(ExrError::from_io(e)))) {
        Ok(value) => Ok(value),
        Err(e) => {
            if tmp.exists()
                && let Err(rm) = std::fs::remove_file(&tmp)
            {
                log::error!("write_file: could not remove the temporary file {}: {rm}", tmp.display());
            }
            Err(e)
        }
    }
}

/// `<dir>/<name>.<pid>.<seq>.tmp`: next to `path`, unique within and across processes.
fn tmp_path(path: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| ExrError::Arg(format!("Not a file path: {}", path.display())))?;
    let mut tmp_name = name.to_os_string();
    tmp_name.push(format!(".{}.{}.tmp", std::process::id(), TMP_SEQ.fetch_add(1, Ordering::Relaxed)));
    Ok(path.with_file_name(tmp_name))
}

// ---------------------------------------------------------------------------
// MemIStream — port of C++ StdISStream (in-memory input)
// ---------------------------------------------------------------------------

/// In-memory input stream — port of `Imf::StdISStream`.
///
/// Reads from a byte buffer via a `Cursor`: an owned `Vec<u8>` (the default, as `StdISStream` owns its
/// string) or ANY `AsRef<[u8]>` — a `&[u8]` borrows the caller's bytes, so decoding a file that is
/// already in memory (a pooled read buffer, a memory map) copies nothing.
pub struct MemIStream<D = Vec<u8>> {
    file_name: String,
    cursor: Cursor<D>,
}

impl<D: AsRef<[u8]>> MemIStream<D> {
    /// Construct over `data` (an owned `Vec<u8>` or a borrowed `&[u8]`).
    pub fn new(data: D) -> Self {
        Self {
            file_name: "(memory)".into(),
            cursor: Cursor::new(data),
        }
    }

    /// Construct with an explicit name (useful for error messages).
    pub fn with_name(data: D, name: &str) -> Self {
        Self {
            file_name: name.to_owned(),
            cursor: Cursor::new(data),
        }
    }

    /// Return a reference to the underlying data (mirrors `StdISStream::str()`).
    pub fn data(&self) -> &[u8] {
        self.cursor.get_ref().as_ref()
    }
}

impl MemIStream<Vec<u8>> {
    /// Replace the buffer and reset the position (mirrors `StdISStream::str(const string&)`).
    pub fn set_data(&mut self, data: Vec<u8>) {
        self.cursor = Cursor::new(data);
    }
}

impl<D: AsRef<[u8]> + Send> IStream for MemIStream<D> {
    fn file_name(&self) -> &str {
        &self.file_name
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<bool> {
        let n = buf.len();
        let pos = self.cursor.position() as usize;
        let data = self.cursor.get_ref().as_ref();
        let available = data.len().saturating_sub(pos);

        if available == 0 && n > 0 {
            return Err(ExrError::Input("Unexpected end of file.".into()));
        }
        if available < n {
            return Err(ExrError::Input(format!(
                "Early end of file: read {available} out of {n} requested bytes."
            )));
        }

        // Perform the read via Cursor so the position advances.
        self.cursor.read_exact(buf).map_err(ExrError::from_io)?;

        // A full, exact read of `n` bytes is always `true` in C++, even when
        // it consumed the buffer's last byte (matches `StdIfStream`).
        Ok(true)
    }

    fn tellg(&mut self) -> Result<u64> {
        Ok(self.cursor.position())
    }

    fn seekg(&mut self, pos: u64) -> Result<()> {
        self.cursor.set_position(pos);
        Ok(())
    }

    fn size(&self) -> Option<u64> {
        Some(self.cursor.get_ref().as_ref().len() as u64)
    }

    fn is_memory_mapped(&self) -> bool {
        false
    }

    /// A `Vec<u8>` backing buffer needs no shared cursor to read an offset
    /// range, so `MemIStream` genuinely supports stateless reads
    /// — unlike `StdIfStream`, whose backing `std::fs::File`
    /// would need a platform-specific positional read (`pread`/`ReadFileEx`)
    /// to implement this without disturbing the shared file cursor; that is
    /// not ported here (see the module doc), so `StdIfStream` keeps the
    /// trait's default "unsupported" behaviour.
    fn is_stateless_read(&self) -> bool {
        true
    }

    fn read_stateless(&self, buf: &mut [u8], offset: u64) -> Result<i64> {
        let data = self.cursor.get_ref().as_ref();
        let offset = offset as usize;
        if offset >= data.len() {
            // "if the offset is past the end, it should read 0" — not an error.
            return Ok(0);
        }
        let available = data.len() - offset;
        let n = buf.len().min(available);
        buf[..n].copy_from_slice(&data[offset..offset + n]);
        Ok(n as i64)
    }
}

// ---------------------------------------------------------------------------
// MemOStream — port of C++ StdOSStream (in-memory output)
// ---------------------------------------------------------------------------

/// In-memory output stream — port of `Imf::StdOSStream`.
///
/// Accumulates written bytes into a `Vec<u8>`.  Call [`MemOStream::into_data`] /
/// [`MemOStream::data`] to retrieve the result.
pub struct MemOStream {
    file_name: String,
    cursor: Cursor<Vec<u8>>,
}

impl MemOStream {
    /// Construct an empty in-memory output stream.
    pub fn new() -> Self {
        Self {
            file_name: "(memory)".into(),
            cursor: Cursor::new(Vec::new()),
        }
    }

    /// Return the accumulated bytes (mirrors `StdOSStream::str()`).
    pub fn data(&self) -> &[u8] {
        self.cursor.get_ref()
    }

    /// Consume the stream and return the underlying buffer.
    pub fn into_data(self) -> Vec<u8> {
        self.cursor.into_inner()
    }
}

impl Default for MemOStream {
    fn default() -> Self {
        Self::new()
    }
}

impl OStream for MemOStream {
    fn file_name(&self) -> &str {
        &self.file_name
    }

    fn write(&mut self, buf: &[u8]) -> Result<()> {
        self.cursor.write_all(buf).map_err(ExrError::from_io)
    }

    fn tellp(&mut self) -> Result<u64> {
        Ok(self.cursor.position())
    }

    fn seekp(&mut self, pos: u64) -> Result<()> {
        self.cursor.set_position(pos);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{write_file, IStream, MemIStream, OStream, StdIfStream, StdOfStream};
    use crate::error::ExrError;
    use std::fs::File;
    use std::io::Write;

    /// A fresh empty directory for one test.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("exr-core-write-file-{name}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entries(dir: &std::path::Path) -> Vec<String> {
        let mut v: Vec<String> =
            std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    }

    /// Success: the file holds exactly what was written, no temporary file is left.
    #[test]
    fn write_file_writes_and_leaves_no_temporary() {
        let dir = scratch("ok");
        let out = dir.join("a.exr");
        let n = write_file(&out, |s| {
            OStream::write(s, b"hello")?;
            OStream::write(s, b" world")?;
            Ok::<_, ExrError>(11)
        })
        .unwrap();
        assert_eq!(n, 11);
        assert_eq!(std::fs::read(&out).unwrap(), b"hello world");
        assert_eq!(entries(&dir), ["a.exr"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Failure inside the write: no new file appears, an existing one keeps its bytes, and no
    /// temporary file is left — the output is never truncated or half written.
    #[test]
    fn write_file_failure_leaves_the_target_as_it_was() {
        let dir = scratch("fail");
        let fresh = dir.join("fresh.exr");
        let err = write_file(&fresh, |s| {
            OStream::write(s, b"partial")?;
            Err::<(), _>(ExrError::Arg("boom".into()))
        });
        assert!(matches!(err, Err(ExrError::Arg(ref m)) if m == "boom"), "{err:?}");
        assert!(!fresh.exists());

        let existing = dir.join("existing.exr");
        std::fs::write(&existing, b"keep me").unwrap();
        let err = write_file(&existing, |s| {
            OStream::write(s, b"partial")?;
            Err::<(), _>(ExrError::Arg("boom".into()))
        });
        assert!(err.is_err());
        assert_eq!(std::fs::read(&existing).unwrap(), b"keep me");
        assert_eq!(entries(&dir), ["existing.exr"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `StdOfStream::flush` puts the buffered bytes on disk while the stream is still open
    /// (the buffer is 256 KiB, so without it nothing would be there yet).
    #[test]
    fn std_of_stream_flush_writes_the_buffer() {
        let dir = scratch("flush");
        let out = dir.join("f.bin");
        let mut s = StdOfStream::create(out.to_str().unwrap()).unwrap();
        OStream::write(&mut s, b"abc").unwrap();
        s.flush().unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"abc");
        drop(s);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The `std::io::Write` view (what PNG-style encoders get) writes through the same stream,
    /// keeps `tellp` in step and its `flush` is the real one.
    #[test]
    fn std_of_stream_is_an_io_write() {
        let dir = scratch("iowrite");
        let out = dir.join("w.bin");
        let mut s = StdOfStream::create(out.to_str().unwrap()).unwrap();
        s.write_all(b"ab").unwrap();
        OStream::write(&mut s, b"c").unwrap();
        assert_eq!(s.tellp().unwrap(), 3);
        Write::flush(&mut s).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"abc");
        drop(s);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `MemIStream::read` shares the same fix: an exact read to the
    /// end of the buffer returns `true`, not `false`.
    #[test]
    fn mem_istream_read_exactly_to_eof_returns_true() {
        let mut s = MemIStream::new(vec![1u8, 2, 3, 4]);
        let mut buf = [0u8; 4];
        assert!(s.read(&mut buf).unwrap());
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    /// `MemIStream` supports stateless reads: an in-range
    /// offset read, a partial read at the tail, and a fully-past-the-end
    /// offset (which reads 0 bytes, not an error).
    #[test]
    fn mem_istream_stateless_read() {
        let s = MemIStream::new(vec![10u8, 20, 30, 40, 50]);
        assert!(s.is_stateless_read());

        let mut buf = [0u8; 2];
        assert_eq!(s.read_stateless(&mut buf, 1).unwrap(), 2);
        assert_eq!(buf, [20, 30]);

        // Request overruns the end: reads only what's available.
        let mut buf2 = [0u8; 4];
        assert_eq!(s.read_stateless(&mut buf2, 3).unwrap(), 2);
        assert_eq!(&buf2[..2], &[40, 50]);

        // Offset past the end: 0 bytes, not an error.
        let mut buf3 = [0u8; 4];
        assert_eq!(s.read_stateless(&mut buf3, 100).unwrap(), 0);
    }

    /// `StdIfStream` keeps the trait's default
    /// "unsupported" stateless-read behaviour.
    #[test]
    fn std_if_stream_stateless_read_unsupported_by_default() {
        let mut path = std::env::temp_dir();
        path.push("exr_core_stdifstream_stateless_test.bin");
        let path_str = path.to_str().expect("utf-8 temp path");
        {
            let mut f = File::create(&path).expect("create temp file");
            f.write_all(&[1u8, 2, 3]).expect("write temp file");
        }
        let s = StdIfStream::open(path_str).expect("open temp file");
        assert!(!s.is_stateless_read());
        let mut buf = [0u8; 1];
        assert!(s.read_stateless(&mut buf, 0).is_err());
        std::fs::remove_file(&path).expect("remove temp file");
    }

    // The disk-file stream must report its length via `size()` so
    // the header parser can apply the remaining-bytes bound on the primary path.
    #[test]
    fn std_if_stream_reports_size() {
        let mut path = std::env::temp_dir();
        path.push("exr_core_stdifstream_size_test.bin");
        let path_str = path.to_str().expect("utf-8 temp path");

        {
            let mut f = File::create(&path).expect("create temp file");
            f.write_all(&[0u8; 123]).expect("write temp file");
        }

        let s = StdIfStream::open(path_str).expect("open temp file");
        assert_eq!(s.size(), Some(123));

        std::fs::remove_file(&path).expect("remove temp file");
    }

    /// A read of exactly the remaining bytes returns `true`, matching
    /// C++ `StdIFStream::read` (`istream::read` only sets `eofbit` on a
    /// *short* read, not on one that lands exactly on the last byte).
    #[test]
    fn read_exactly_to_eof_returns_true() {
        let mut path = std::env::temp_dir();
        path.push("exr_core_stdifstream_eof_test.bin");
        let path_str = path.to_str().expect("utf-8 temp path");

        {
            let mut f = File::create(&path).expect("create temp file");
            f.write_all(&[1u8, 2, 3, 4]).expect("write temp file");
        }

        let mut s = StdIfStream::open(path_str).expect("open temp file");
        let mut buf = [0u8; 4];
        assert!(s.read(&mut buf).unwrap());
        assert_eq!(buf, [1, 2, 3, 4]);

        std::fs::remove_file(&path).expect("remove temp file");
    }

    /// A short read (past what the file actually holds) is a hard
    /// error, never a silent `false`.
    #[test]
    fn read_past_eof_errors() {
        let mut path = std::env::temp_dir();
        path.push("exr_core_stdifstream_short_read_test.bin");
        let path_str = path.to_str().expect("utf-8 temp path");

        {
            let mut f = File::create(&path).expect("create temp file");
            f.write_all(&[1u8, 2]).expect("write temp file");
        }

        let mut s = StdIfStream::open(path_str).expect("open temp file");
        let mut buf = [0u8; 4];
        assert!(s.read(&mut buf).is_err());

        std::fs::remove_file(&path).expect("remove temp file");
    }
}

#[cfg(test)]
mod borrowed_tests {
    use super::{IStream, MemIStream};

    /// A `MemIStream` over a borrowed slice reads, seeks and reports its size like the owned one.
    #[test]
    fn borrowed_slice_stream_behaves_like_the_owned_one() {
        let bytes = [1u8, 2, 3, 4, 5, 6];
        let mut owned = MemIStream::new(bytes.to_vec());
        let mut borrowed = MemIStream::new(&bytes[..]);
        assert_eq!(borrowed.size(), owned.size());
        let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
        assert!(owned.read(&mut a).unwrap() && borrowed.read(&mut b).unwrap());
        assert_eq!(a, b);
        borrowed.seekg(5).unwrap();
        assert_eq!(borrowed.tellg().unwrap(), 5);
        assert!(borrowed.read(&mut [0u8; 2]).is_err(), "reading past the end is an error");
        let mut c = [0u8; 3];
        assert_eq!(borrowed.read_stateless(&mut c, 1).unwrap(), 3);
        assert_eq!(c, [2, 3, 4]);
    }
}
