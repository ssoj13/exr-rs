//! Small, safe file IO for large media files: ONE read-only loader and ONE atomic writer.
//!
//! # Why
//! Every tool that reads big containers and writes outputs ends up re-implementing the same two things:
//! "map the input (or just read it)" and "write to a temp, rename over the destination". Each copy gets
//! cleanup, permissions, Windows sharing and durability slightly differently wrong. This crate is the one
//! owner of both, extracted from `exv-rs` (`exv_video::fsio`), where it serves the CLI, the C ABI and the
//! viewer.
//!
//! # Read: [`open_ro`]
//! Files below [`MAP_MIN_BYTES`] are READ into memory; bigger ones are memory-mapped (demand paged, so opening
//! a 40 GB container costs only the pages actually touched). Why not always map: a long-lived map of a small
//! file buys nothing but (a) the Windows "user-mapped section" lock (`ERROR_USER_MAPPED_FILE`), which makes
//! replacing a file we still hold fail at rename time, and (b) a SIGBUS/access violation if another process
//! truncates the file. A mapped (>= 64 MiB) [`Src`] still blocks replacing that file on Windows: drop it first.
//!
//! # Write: [`OutFile`] / [`write_atomic`] / [`atomic_path`]
//! Output goes to a randomly named temp file in the DESTINATION directory (same volume, so the final rename is
//! atomic) and is renamed over the destination only on commit. Dropping an uncommitted [`OutFile`] (error,
//! panic, cancelled render) deletes the temp: a failed write never leaves a truncated file under the final
//! name and never destroys the previous one. A hard process kill leaves a `.<name>.<random>.part` temp next to
//! the destination.
//!
//! # Guard: [`same_file`] / [`ensure_distinct`]
//! Converting a file onto itself is legal with atomic output on Unix but fails at the very end on Windows when
//! the input is mapped; the guard lets a tool refuse it before doing the work.

use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};

use tempfile::{Builder, NamedTempFile, TempPath};

/// Files below this size are read into memory by [`open_ro`]; files of this size or bigger are mapped.
///
/// 64 MiB: big enough that every "small" file (stills, sidecars, configs) is lock-free and truncation-safe,
/// small enough that reading it costs a few milliseconds and never doubles the footprint of a big container.
pub const MAP_MIN_BYTES: u64 = 64 << 20;

/// Typed failure of an fsio call: what failed, on which path, and the OS error underneath.
///
/// Converts into [`io::Error`] (keeping [`Error::kind`]) so `?` works in `io::Result` code.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Opening, reading or mapping a source file ([`open_ro`], [`same_file`]).
    Read {
        /// The file being read.
        path: PathBuf,
        /// OS error.
        source: io::Error,
    },
    /// Creating the temp file next to the destination: the directory is missing or not writable.
    Create {
        /// The destination the temp was for.
        path: PathBuf,
        /// OS error.
        source: io::Error,
    },
    /// Flushing, syncing or preparing the temp file before the rename. The destination is untouched.
    Write {
        /// The destination.
        path: PathBuf,
        /// OS error.
        source: io::Error,
    },
    /// Renaming the temp over the destination failed. The destination is untouched and the temp is deleted.
    /// Usual Windows cause: the destination is memory-mapped (or opened without delete sharing) by someone.
    Replace {
        /// The destination.
        path: PathBuf,
        /// OS error.
        source: io::Error,
    },
    /// The rename HAPPENED (the destination holds the new bytes), but syncing its directory failed, so the
    /// rename may not survive a power loss. Only [`Durability::Full`] reports it, only on Unix.
    NotDurable {
        /// The destination.
        path: PathBuf,
        /// OS error.
        source: io::Error,
    },
    /// [`ensure_distinct`]: the output names the same file as the input.
    SameFile {
        /// The input path as given.
        input: PathBuf,
        /// The output path as given.
        output: PathBuf,
    },
}

impl Error {
    /// The [`io::ErrorKind`] of the underlying OS error ([`io::ErrorKind::InvalidInput`] for [`Error::SameFile`]).
    pub fn kind(&self) -> io::ErrorKind {
        match self {
            Self::Read { source, .. }
            | Self::Create { source, .. }
            | Self::Write { source, .. }
            | Self::Replace { source, .. }
            | Self::NotDurable { source, .. } => source.kind(),
            Self::SameFile { .. } => io::ErrorKind::InvalidInput,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "cannot read '{}': {source}", path.display()),
            Self::Create { path, source } => {
                write!(f, "cannot create a temp file next to '{}': {source}", path.display())
            }
            Self::Write { path, source } => write!(f, "cannot write '{}': {source}", path.display()),
            Self::Replace { path, source } => {
                write!(f, "cannot replace '{}': {source}", path.display())?;
                if cfg!(windows) {
                    f.write_str(" (close any viewer/host that still has it open)")?;
                }
                Ok(())
            }
            Self::NotDurable { path, source } => {
                write!(f, "'{}' was replaced, but syncing its directory failed: {source}", path.display())
            }
            Self::SameFile { input, output } => {
                write!(f, "output '{}' is the same file as input '{}'", output.display(), input.display())
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. }
            | Self::Create { source, .. }
            | Self::Write { source, .. }
            | Self::Replace { source, .. }
            | Self::NotDurable { source, .. } => Some(source),
            Self::SameFile { .. } => None,
        }
    }
}

impl From<Error> for io::Error {
    fn from(e: Error) -> Self {
        io::Error::new(e.kind(), e)
    }
}

/// Read-only file bytes: read through (small) or memory-mapped (big). Derefs to `[u8]`.
///
/// Produced by [`open_ro`]. Hold it as long as the bytes are needed; for a [`Src::Map`] that is also how long
/// the file cannot be replaced on Windows.
pub enum Src {
    /// Small file (< [`MAP_MIN_BYTES`]), read into RAM: no lock, immune to external truncation.
    Mem(Vec<u8>),
    /// Big file, read-only map.
    Map(memmap2::Mmap),
}

impl Deref for Src {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Src::Mem(v) => v,
            Src::Map(m) => m,
        }
    }
}

impl AsRef<[u8]> for Src {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl fmt::Debug for Src {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Src::Mem(_) => "Mem",
            Src::Map(_) => "Map",
        };
        f.debug_struct("Src").field("kind", &kind).field("len", &self.len()).finish()
    }
}

/// Open `path` read-only: read-through below [`MAP_MIN_BYTES`], read-only map at or above it.
pub fn open_ro(path: &Path) -> Result<Src, Error> {
    open_with(path, MAP_MIN_BYTES)
}

/// [`open_ro`] with an explicit threshold (tests exercise both branches without 64 MiB files).
fn open_with(path: &Path, map_min: u64) -> Result<Src, Error> {
    let err = |source| Error::Read { path: path.to_path_buf(), source };
    let mut file = File::open(path).map_err(err)?;
    let len = file.metadata().map_err(err)?.len();
    if len < map_min {
        // `len` is only a capacity hint: `read_to_end` stays correct if the file changes size meanwhile.
        let mut v = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
        file.read_to_end(&mut v).map_err(err)?;
        return Ok(Src::Mem(v));
    }
    map::ro(&file).map(Src::Map).map_err(err)
}

/// The crate's only `unsafe`: kept in one tiny module so the rest stays `deny(unsafe_code)`.
#[allow(unsafe_code)]
mod map {
    use std::fs::File;
    use std::io;

    /// Read-only map of `file`.
    pub(super) fn ro(file: &File) -> io::Result<memmap2::Mmap> {
        // SAFETY: a read-only map; we never write through it and never hand out `&mut`. The remaining hazard is
        // external: if another process truncates the file while it is mapped, touching the lost pages raises
        // SIGBUS (Unix) / an in-page access violation (Windows). That is the contract of every mmap reader and is
        // accepted only for files >= MAP_MIN_BYTES; smaller ones are read through. While mapped, Windows also
        // refuses to rename over or delete the file (ERROR_USER_MAPPED_FILE): callers drop the `Src` first.
        unsafe { memmap2::Mmap::map(file) }
    }
}

/// How hard a commit pushes the bytes to stable storage before returning.
///
/// Both levels are equally ATOMIC for other processes (they see the old file or the whole new one); they differ
/// only in what survives a power loss / kernel crash.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Durability {
    /// fsync the file before the rename and (Unix) its directory after it: the new file survives a power loss.
    #[default]
    Full,
    /// No fsync: the OS writes back when it likes. After a power loss the destination may hold the old file, the
    /// new one, or (on some filesystems) an empty file. For scratch/preview outputs where speed matters more.
    Fast,
}

/// Directory that holds `dest`'s temp: its parent (so the rename stays on one volume), `.` for a bare name.
fn dir_of(dest: &Path) -> &Path {
    dest.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."))
}

/// Create the temp next to `dest`: `.<name>.<random>.part` (random, so concurrent writers never clash; the name
/// is cut to 32 chars so a long destination name cannot push the temp over the filesystem's name limit).
fn temp_for(dest: &Path) -> Result<NamedTempFile, Error> {
    let mut prefix = OsString::from(".");
    if let Some(name) = dest.file_name() {
        prefix.push(name.to_string_lossy().chars().take(32).collect::<String>());
        prefix.push(".");
    }
    Builder::new()
        .prefix(&prefix)
        .suffix(".part")
        .tempfile_in(dir_of(dest))
        .map_err(|source| Error::Create { path: dest.to_path_buf(), source })
}

/// Rename the finished (already synced, if `Full`) temp over `dest`. On failure the temp is deleted.
fn publish(tmp: TempPath, dest: &Path, dur: Durability) -> Result<(), Error> {
    #[cfg(unix)]
    unix_mode(&tmp, dest)?;
    tmp.persist(dest).map_err(|e| Error::Replace { path: dest.to_path_buf(), source: e.error })?;
    if dur == Durability::Full {
        sync_dir(dest)?;
    }
    Ok(())
}

/// `rename` does not sync the directory entry; without this a power loss can lose the rename itself.
#[cfg(unix)]
fn sync_dir(dest: &Path) -> Result<(), Error> {
    File::open(dir_of(dest))
        .and_then(|d| d.sync_all())
        .map_err(|source| Error::NotDurable { path: dest.to_path_buf(), source })
}

/// Windows has no directory fsync (a directory handle cannot be flushed); NTFS journals the rename metadata.
#[cfg(not(unix))]
fn sync_dir(_: &Path) -> Result<(), Error> {
    Ok(())
}

/// Give the temp the destination's mode: tempfile creates 0600, but an output is an ordinary shared file.
/// Replacing an existing file keeps its mode; a new file gets 0644.
#[cfg(unix)]
fn unix_mode(tmp: &Path, dest: &Path) -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;
    let err = |source| Error::Write { path: dest.to_path_buf(), source };
    let mode = match fs::metadata(dest) {
        Ok(m) => m.permissions().mode() & 0o7777,
        Err(e) if e.kind() == io::ErrorKind::NotFound => 0o644,
        Err(e) => return Err(err(e)),
    };
    fs::set_permissions(tmp, fs::Permissions::from_mode(mode)).map_err(err)
}

/// Atomic, buffered output file: `Write + Seek`, committed explicitly.
///
/// Created as a temp next to the destination; [`OutFile::commit`] renames it over the destination, dropping it
/// uncommitted deletes it. Seekable, so writers can back-patch headers/counts/offset tables.
pub struct OutFile {
    w: BufWriter<NamedTempFile>,
    dest: PathBuf,
}

impl OutFile {
    /// Create the temp next to `dest`. Fails EARLY (missing/unwritable directory) instead of at commit time.
    pub fn create(dest: &Path) -> Result<Self, Error> {
        Ok(Self { w: BufWriter::new(temp_for(dest)?), dest: dest.to_path_buf() })
    }

    /// The final destination path.
    pub fn dest(&self) -> &Path {
        &self.dest
    }

    /// [`OutFile::commit_with`] at [`Durability::Full`].
    pub fn commit(self) -> Result<(), Error> {
        self.commit_with(Durability::Full)
    }

    /// Flush, sync (per `dur`) and atomically replace the destination. The temp is deleted if any step fails.
    pub fn commit_with(self, dur: Durability) -> Result<(), Error> {
        let Self { w, dest } = self;
        let err = |source| Error::Write { path: dest.clone(), source };
        let tmp = w.into_inner().map_err(|e| err(e.into_error()))?;
        if dur == Durability::Full {
            tmp.as_file().sync_all().map_err(err)?;
        }
        // Close our handle before the rename (Windows sharing); the `TempPath` still deletes on drop.
        publish(tmp.into_temp_path(), &dest, dur)
    }
}

impl fmt::Debug for OutFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutFile").field("dest", &self.dest).field("temp", &self.w.get_ref().path()).finish()
    }
}

impl Write for OutFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.w.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}

impl Seek for OutFile {
    /// Seeking flushes the buffer first ([`BufWriter`] semantics), so a back-patch lands on disk in order.
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.w.seek(pos)
    }
}

/// Atomically write `bytes` to `dest` at [`Durability::Full`] (small, whole-in-RAM outputs).
pub fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut f = OutFile::create(dest)?;
    f.write_all(bytes).map_err(|source| Error::Write { path: dest.to_path_buf(), source })?;
    f.commit()
}

/// Atomic output for writers that can only open the file BY PATH (third-party encoders).
///
/// Runs `body` against a temp path next to `dest`, then syncs it and renames it over `dest` only on `Ok(Some(_))`.
/// On `Err` or `Ok(None)` ("not handled": the caller falls back to another writer) the temp is deleted and `dest`
/// is untouched. `body`'s own error type passes through unchanged; fsio failures convert via `From<Error>`.
/// Our temp handle is closed before `body` runs, so `body` may create/truncate the path (Windows sharing).
pub fn atomic_path<T, E: From<Error>>(
    dest: &Path,
    body: impl FnOnce(&Path) -> Result<Option<T>, E>,
) -> Result<Option<T>, E> {
    let path = temp_for(dest)?.into_temp_path();
    let Some(v) = body(&path)? else { return Ok(None) };
    // `body` closed its own handle; reopen for writing because Windows `FlushFileBuffers` needs write access.
    OpenOptions::new()
        .write(true)
        .open(&path)
        .and_then(|f| f.sync_all())
        .map_err(|source| Error::Write { path: dest.to_path_buf(), source })?;
    publish(path, dest, Durability::Full)?;
    Ok(Some(v))
}

/// Do `a` and `b` name the same existing file? A path that does not exist is never the same as anything.
///
/// Unix compares (device, inode), so hard links and symlinks are caught. Elsewhere it compares canonical paths
/// (symlinks, `..`, 8.3 short names and letter case resolved), which does NOT see through hard links.
pub fn same_file(a: &Path, b: &Path) -> Result<bool, Error> {
    let (Some(ma), Some(mb)) = (meta(a)?, meta(b)?) else { return Ok(false) };
    Ok(identity(a, &ma)? == identity(b, &mb)?)
}

/// Refuse `output` when it is the same file as `input` ([`Error::SameFile`]); call before converting in place.
pub fn ensure_distinct(input: &Path, output: &Path) -> Result<(), Error> {
    if same_file(input, output)? {
        return Err(Error::SameFile { input: input.to_path_buf(), output: output.to_path_buf() });
    }
    Ok(())
}

/// Metadata of `p` (following symlinks), `None` if it does not exist.
fn meta(p: &Path) -> Result<Option<Metadata>, Error> {
    match fs::metadata(p) {
        Ok(m) => Ok(Some(m)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(Error::Read { path: p.to_path_buf(), source }),
    }
}

#[cfg(unix)]
fn identity(_: &Path, m: &Metadata) -> Result<(u64, u64), Error> {
    use std::os::unix::fs::MetadataExt;
    Ok((m.dev(), m.ino()))
}

/// Stable std has no file id on Windows (`windows_by_handle` is unstable), so fall back to the canonical path.
#[cfg(not(unix))]
fn identity(p: &Path, _: &Metadata) -> Result<PathBuf, Error> {
    fs::canonicalize(p).map_err(|source| Error::Read { path: p.to_path_buf(), source })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    fn leftovers(d: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(d)
            .expect("read_dir")
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect();
        v.sort();
        v
    }

    /// Small files are read through; a missing file is a typed `Read` error naming the path.
    #[test]
    fn open_ro_small_is_mem() {
        let d = dir();
        let p = d.path().join("a.bin");
        fs::write(&p, b"hello").expect("write");
        let s = open_ro(&p).expect("open");
        assert!(matches!(s, Src::Mem(_)));
        assert_eq!(&*s, b"hello");
        let e = open_ro(&d.path().join("missing")).expect_err("missing");
        assert!(matches!(e, Error::Read { .. }) && e.kind() == io::ErrorKind::NotFound, "{e:?}");
        assert!(e.to_string().contains("missing"), "{e}");
    }

    /// Threshold: `len < map_min` reads through, `len >= map_min` maps; both deref to the same bytes.
    #[test]
    fn map_threshold() {
        let d = dir();
        let p = d.path().join("a.bin");
        fs::write(&p, b"hello").expect("write");
        let mapped = open_with(&p, 5).expect("map");
        let read = open_with(&p, 6).expect("read");
        assert!(matches!(mapped, Src::Map(_)) && matches!(read, Src::Mem(_)));
        assert_eq!((&*mapped, &*read), (b"hello".as_slice(), b"hello".as_slice()));
    }

    /// A read-through source does not lock the file: it can be replaced while the `Src` is alive.
    #[test]
    fn replacing_an_open_small_source_works() {
        let d = dir();
        let p = d.path().join("a.bin");
        fs::write(&p, b"old").expect("write");
        let held = open_ro(&p).expect("open");
        write_atomic(&p, b"new").expect("replace while a read-through Src is alive");
        assert_eq!((&*held, fs::read(&p).expect("read").as_slice()), (b"old".as_slice(), b"new".as_slice()));
    }

    /// The reason for the read-through policy, pinned per OS: Windows refuses to replace a MAPPED file (typed
    /// `Replace` error, destination and directory untouched); Unix replaces it and the map keeps the old inode.
    #[test]
    fn replacing_a_mapped_source() {
        let d = dir();
        let p = d.path().join("a.bin");
        fs::write(&p, b"old").expect("write");
        let held = open_with(&p, 0).expect("map");
        assert!(matches!(held, Src::Map(_)));
        let r = write_atomic(&p, b"new");
        if cfg!(windows) {
            let e = r.expect_err("Windows cannot rename over a mapped file");
            assert!(matches!(e, Error::Replace { .. }), "{e:?}");
            assert_eq!(fs::read(&p).expect("read"), b"old");
        } else {
            r.expect("Unix replaces a mapped file");
            assert_eq!(fs::read(&p).expect("read"), b"new");
        }
        assert_eq!(&*held, b"old");
        assert_eq!(leftovers(d.path()), ["a.bin"]);
        drop(held);
        write_atomic(&p, b"new").expect("replace after the map is dropped");
        assert_eq!(fs::read(&p).expect("read"), b"new");
    }

    /// Commit replaces atomically (seek back-patch included) and leaves no temp behind; `Fast` too.
    #[test]
    fn commit_replaces_and_cleans() {
        let d = dir();
        let p = d.path().join("out.exv");
        fs::write(&p, b"previous").expect("write");
        let mut f = OutFile::create(&p).expect("create");
        assert_eq!(f.dest(), p);
        f.write_all(b"abc").expect("w");
        f.seek(SeekFrom::Start(1)).expect("seek");
        f.write_all(b"X").expect("patch");
        assert_eq!(fs::read(&p).expect("read"), b"previous", "destination untouched before commit");
        assert_eq!(leftovers(d.path()).len(), 2, "temp lives next to the destination");
        f.commit().expect("commit");
        assert_eq!(fs::read(&p).expect("read"), b"aXc");
        assert_eq!(leftovers(d.path()), ["out.exv"]);

        let mut f = OutFile::create(&p).expect("create");
        f.write_all(b"fast").expect("w");
        f.commit_with(Durability::Fast).expect("commit fast");
        assert_eq!(fs::read(&p).expect("read"), b"fast");
        assert_eq!(leftovers(d.path()), ["out.exv"]);
    }

    /// The temp is named after the destination (hidden, `.part`), cut so long names stay valid.
    #[test]
    fn temp_name() {
        let d = dir();
        let long = "x".repeat(200);
        let f = OutFile::create(&d.path().join(format!("{long}.exv"))).expect("create a temp for a long name");
        let name = f.w.get_ref().path().file_name().expect("name").to_string_lossy().into_owned();
        assert!(name.starts_with(&format!(".{}.", &long[..32])) && name.ends_with(".part"), "{name}");
        let f = OutFile::create(&d.path().join("out.exv")).expect("create");
        let name = f.w.get_ref().path().file_name().expect("name").to_string_lossy().into_owned();
        assert!(name.starts_with(".out.exv.") && name.ends_with(".part"), "{name}");
    }

    /// Dropping without commit = abort: nothing under the destination name, previous file preserved, no temp.
    #[test]
    fn drop_aborts() {
        let d = dir();
        let p = d.path().join("out.exv");
        {
            let mut f = OutFile::create(&p).expect("create");
            f.write_all(b"partial").expect("w");
        }
        assert!(leftovers(d.path()).is_empty());
        fs::write(&p, b"keep").expect("write");
        {
            let mut f = OutFile::create(&p).expect("create");
            f.write_all(&[7; 1 << 16]).expect("w past the buffer");
        }
        assert_eq!(fs::read(&p).expect("read"), b"keep");
        assert_eq!(leftovers(d.path()), ["out.exv"]);
    }

    /// A missing destination directory fails at `create` (typed `Create`), not at commit.
    #[test]
    fn missing_dir_fails_early() {
        let d = dir();
        let p = d.path().join("nope").join("x.exv");
        assert!(matches!(OutFile::create(&p), Err(Error::Create { .. })));
        assert!(matches!(write_atomic(&p, b"x"), Err(Error::Create { .. })));
    }

    /// A destination that cannot be replaced (a directory) fails the commit with a typed `Replace` error naming
    /// the path, and leaves no temp behind.
    #[test]
    fn blocked_commit_reports_and_cleans() {
        let d = dir();
        let p = d.path().join("blocked");
        fs::create_dir(&p).expect("mkdir");
        let e = write_atomic(&p, b"x").expect_err("a directory cannot be replaced");
        assert!(matches!(e, Error::Replace { .. }), "{e:?}");
        assert!(e.to_string().contains("blocked"), "{e}");
        let io: io::Error = e.into();
        assert!(io.to_string().contains("blocked"), "the io::Error keeps the message: {io}");
        assert_eq!(leftovers(d.path()), ["blocked"]);
    }

    /// Concurrent writers to one path: every commit is atomic, so the survivor is exactly one writer's whole
    /// payload and no temp is left. On Windows a rename racing another rename may be refused (typed `Replace`).
    #[test]
    fn concurrent_writers_last_wins() {
        let d = dir();
        let p = d.path().join("race.bin");
        const N: u8 = 8;
        const LEN: usize = 256 << 10;
        let results: Vec<Result<(), Error>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..N)
                .map(|i| {
                    s.spawn({
                        let p = &p;
                        move || write_atomic(p, &vec![i; LEN])
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("writer thread")).collect()
        });
        assert!(results.iter().any(Result::is_ok), "{results:?}");
        for r in &results {
            if let Err(e) = r {
                assert!(matches!(e, Error::Replace { .. }), "{e:?}");
            }
        }
        let got = fs::read(&p).expect("read");
        assert_eq!(got.len(), LEN);
        assert!(got.iter().all(|&b| b == got[0]) && got[0] < N, "torn file");
        assert_eq!(leftovers(d.path()), ["race.bin"]);
    }

    /// `atomic_path`: renamed only on `Ok(Some)`; `Ok(None)` and `Err` delete the temp and keep the old file;
    /// the body's own error type passes through.
    #[test]
    fn atomic_path_commit_rules() {
        let d = dir();
        let p = d.path().join("o.exr");
        let r: io::Result<_> = atomic_path(&p, |t| fs::write(t, b"1").map(|()| Some(7)));
        assert_eq!(r.expect("ok"), Some(7));
        assert_eq!(fs::read(&p).expect("read"), b"1");
        let r: io::Result<Option<()>> = atomic_path(&p, |t| fs::write(t, b"2").map(|()| None));
        assert_eq!(r.expect("ok"), None);
        let r: Result<Option<()>, String> = atomic_path(&p, |t| {
            fs::write(t, b"3").map_err(|e| e.to_string())?;
            Err("boom".to_owned())
        });
        assert_eq!(r, Err("boom".to_owned()));
        assert_eq!(fs::read(&p).expect("read"), b"1", "old file survives None/Err");
        assert_eq!(leftovers(d.path()), ["o.exr"]);
    }

    impl From<Error> for String {
        fn from(e: Error) -> Self {
            e.to_string()
        }
    }

    /// `same_file` sees through `..` spellings, refuses nothing that does not exist; `ensure_distinct` is typed.
    #[test]
    fn same_file_and_guard() {
        let d = dir();
        let a = d.path().join("a.exv");
        let b = d.path().join("b.exv");
        fs::write(&a, b"x").expect("write");
        fs::write(&b, b"x").expect("write");
        fs::create_dir(d.path().join("sub")).expect("mkdir");
        let spelled = d.path().join("sub").join("..").join("a.exv");
        assert!(same_file(&a, &spelled).expect("same"));
        assert!(!same_file(&a, &b).expect("different"));
        assert!(!same_file(&a, &d.path().join("new.exv")).expect("missing output"));
        let e = ensure_distinct(&a, &spelled).expect_err("in place");
        assert!(matches!(e, Error::SameFile { .. }) && e.kind() == io::ErrorKind::InvalidInput, "{e:?}");
        ensure_distinct(&a, &b).expect("distinct");
    }

    /// Unix only: hard links are the same file; committed mode is 0644 for a new file, preserved on replace.
    #[cfg(unix)]
    #[test]
    fn unix_identity_and_mode() {
        use std::os::unix::fs::PermissionsExt;
        let d = dir();
        let a = d.path().join("a");
        write_atomic(&a, b"x").expect("write");
        assert_eq!(fs::metadata(&a).expect("meta").permissions().mode() & 0o777, 0o644);
        fs::set_permissions(&a, fs::Permissions::from_mode(0o600)).expect("chmod");
        write_atomic(&a, b"y").expect("replace");
        assert_eq!(fs::metadata(&a).expect("meta").permissions().mode() & 0o777, 0o600);
        let l = d.path().join("l");
        fs::hard_link(&a, &l).expect("link");
        assert!(same_file(&a, &l).expect("hard link"));
    }
}
