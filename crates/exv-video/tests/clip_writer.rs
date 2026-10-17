//! `ClipWriter` / `write_clip*` / `transcode` behavior: byte-identity across front doors and thread counts,
//! counter patching, hostile sinks, poisoning.

mod common;

use std::io::{self, Cursor, Seek, SeekFrom, Write};

use common::{clip, frame};
use exv_codec::Profile;
use exv_codec::image::Image;
use exv_video::{
    ClipWriter, Decoder, VideoError, VideoIndex, WriteOpts, read_video, transcode, write_clip, write_clip_pull, write_video,
};

fn profiles() -> Vec<(&'static str, Profile)> {
    vec![("fast", Profile::lossless_fast()), ("wave", Profile::lossless_wave()), ("lossy2", Profile::lossy_wave(2))]
}

fn new_writer(frames: &[Image], prof: &Profile, opts: &WriteOpts) -> ClipWriter<Cursor<Vec<u8>>> {
    ClipWriter::new(Cursor::new(Vec::new()), &frames[0].header, prof, opts).expect("new")
}

fn bytes_of(w: ClipWriter<Cursor<Vec<u8>>>) -> Vec<u8> {
    w.finish().expect("finish").0.into_inner()
}

/// push one by one == push_win with several (misaligned) window sizes == write_clip_pull (window 1 and
/// large) == the in-RAM `write_video`, for every I/P configuration.
#[test]
fn stream_equals_inram() {
    let frames = clip(7, 16);
    for (name, prof) in profiles() {
        for gop in [1u32, 3, 8] {
            for search in [0i32, 4] {
                let want = write_video(&frames, &prof, gop, search, 0).expect("in-RAM");
                let opts = WriteOpts { gop, search, ..WriteOpts::default() };
                let tag = format!("{name} gop {gop} search {search}");

                let mut w = new_writer(&frames, &prof, &opts);
                frames.iter().cloned().try_for_each(|f| w.push(f)).expect("push");
                assert_eq!(bytes_of(w), want, "push one-by-one: {tag}");

                for win in [2usize, 3, 7] {
                    let mut w = new_writer(&frames, &prof, &opts);
                    for c in frames.chunks(win) {
                        w.push_win(c.to_vec(), &|_| {}).expect("push_win");
                    }
                    assert_eq!(bytes_of(w), want, "push_win({win}): {tag}");
                }

                let get = |i: usize| frames.get(i).cloned().ok_or_else(|| format!("no frame {i}"));
                for budget in [1u64, u64::MAX] {
                    let (c, n) = write_clip_pull(Cursor::new(Vec::new()), frames.len(), &prof, &opts, budget, &get, &|_| {})
                        .expect("pull");
                    let got = c.into_inner();
                    assert_eq!(n as usize, got.len());
                    assert_eq!(got, want, "pull budget {budget}: {tag}");
                }
            }
        }
    }
}

/// The bytes do not depend on the rayon thread count (1 vs 8), for the in-RAM writer incl. B-frames and for
/// windowed pushes.
#[test]
fn thread_count_independent() {
    let frames = clip(9, 16);
    let prof = Profile::lossless_fast();
    let run = |threads: usize| -> (Vec<u8>, Vec<u8>) {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("pool").install(|| {
            let b = write_video(&frames, &prof, 4, 4, 1).expect("with B");
            let opts = WriteOpts { gop: 4, search: 4, ..WriteOpts::default() };
            let mut w = new_writer(&frames, &prof, &opts);
            w.push_win(frames.clone(), &|_| {}).expect("push_win");
            (b, bytes_of(w))
        })
    };
    assert_eq!(run(1), run(8));
}

/// `finish` patches BOTH stored frame counters (prelude @9, header @23); the result parses, decodes, and
/// equals the source. A clip with no frame is `Empty`.
#[test]
fn finish_patches_counts() {
    let frames = clip(5, 16);
    let opts = WriteOpts { gop: 2, search: 2, ..WriteOpts::default() };
    let mut w = new_writer(&frames, &Profile::lossless_fast(), &opts);
    assert_eq!((w.frames(), w.bytes() > 0), (0, true));
    for f in frames.iter().cloned() {
        w.push(f).expect("push");
    }
    assert_eq!(w.frames(), 5);
    w.flush().expect("flush");
    let enc = bytes_of(w);
    assert_eq!(enc[9..13], 5u32.to_le_bytes(), "prelude count");
    assert_eq!(enc[23..27], 5u32.to_le_bytes(), "header count");
    assert_eq!(VideoIndex::parse(&enc).expect("parse").frame_count(), 5);
    let mut dec = Decoder::open(&enc).expect("decoder");
    for (k, f) in frames.iter().enumerate() {
        assert_eq!(dec.frame(k).expect("frame").planes, f.planes, "frame {k}");
    }
    let w = new_writer(&frames, &Profile::lossless_fast(), &opts);
    assert!(matches!(w.finish(), Err(VideoError::Empty)));
    assert!(matches!(write_clip(Cursor::new(Vec::new()), &[], &Profile::lossless_fast(), &opts), Err(VideoError::Empty)));
}

/// A shape mismatch names the frame, B-frames are refused on the push path, and a failed call poisons.
#[test]
fn errors_are_typed_and_poison() {
    let frames = clip(3, 16);
    let prof = Profile::lossless_fast();
    let opts = WriteOpts { gop: 4, ..WriteOpts::default() };
    let mut w = new_writer(&frames, &prof, &opts);
    w.push(frames[0].clone()).expect("push");
    let mut odd = frame(1, 16);
    odd.header.width = 24;
    odd.planes = odd.planes.iter().map(|p| p[..p.len() / 2].to_vec()).collect();
    assert!(matches!(w.push(odd), Err(VideoError::Mismatch { frame: 1 })));
    assert!(matches!(w.push(frames[1].clone()), Err(VideoError::Poisoned)));
    assert!(matches!(w.flush(), Err(VideoError::Poisoned)));
    assert!(matches!(w.finish(), Err(VideoError::Poisoned)));

    let bopts = WriteOpts { b_frames: 1, ..opts };
    let mut w = new_writer(&frames, &prof, &bopts);
    assert!(matches!(w.push(frames[0].clone()), Err(VideoError::Unsupported(_))));
    // The loader's own failure keeps its frame and message.
    let bad = |i: usize| if i == 2 { Err("disk on fire".to_string()) } else { Ok(frames[i].clone()) };
    let r = write_clip_pull(Cursor::new(Vec::new()), 3, &prof, &WriteOpts::default(), 1, &bad, &|_| {});
    assert!(matches!(&r, Err(VideoError::Source { frame: 2, msg }) if msg == "disk on fire"), "{r:?}");
    // A sink that does not start at byte 0 is refused (the counter patch needs absolute offsets).
    let mut c = Cursor::new(Vec::new());
    c.write_all(b"x").expect("write");
    assert!(matches!(ClipWriter::new(c, &frames[0].header, &prof, &WriteOpts::default()), Err(VideoError::Unsupported(_))));
}

/// A sink that fails after `left` bytes (disk full); seeks are free.
struct Flaky {
    inner: Cursor<Vec<u8>>,
    left: usize,
}

impl Write for Flaky {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self.left == 0 {
            return Err(io::Error::other("disk full"));
        }
        let n = b.len().min(self.left);
        self.left -= n;
        self.inner.write(&b[..n])
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for Flaky {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        self.inner.seek(p)
    }
}

/// Disk-full at EVERY byte offset (prelude, frames, index, counter patch; the patch rewrites 8 bytes, so a
/// full write costs `total + 8` sink bytes): the write fails with a typed
/// `Io` error (never a panic, never `Ok` on a short file), and the writer is poisoned afterwards.
#[test]
fn failing_sink_sweep() {
    let frames = clip(3, 16);
    let prof = Profile::lossless_fast();
    let opts = WriteOpts { gop: 2, search: 2, ..WriteOpts::default() };
    let total = write_video(&frames, &prof, 2, 2, 0).expect("reference").len();
    let run = |left: usize| -> Result<(), VideoError> {
        let mut w = ClipWriter::new(Flaky { inner: Cursor::new(Vec::new()), left }, &frames[0].header, &prof, &opts)?;
        for f in frames.iter().cloned() {
            if let Err(e) = w.push(f.clone()) {
                assert!(matches!(w.push(f), Err(VideoError::Poisoned)), "poisoned after {e}");
                return Err(e);
            }
        }
        w.finish().map(drop)
    };
    let cost = total + 8;
    // Every byte of the prelude and of the tail (index, trailer, counter patch), a stride through the frames.
    for left in (0..cost).filter(|&l| l < 64 || l + 64 >= cost || l % 13 == 0) {
        let e = run(left).expect_err("a short sink must fail");
        assert!(matches!(e, VideoError::Io(_)), "left {left}: {e:?}");
    }
    run(cost).expect("exactly enough room");
}

/// `.exv -> .exv`: re-encoding a B-frame clip with a new profile/GOP gives the same bytes as decoding it
/// frame by frame and writing the frames (the old CLI path), for any window size.
#[test]
fn transcode_equals_frame_by_frame() {
    let frames = clip(9, 16);
    let src = write_video(&frames, &Profile::lossless_fast(), 4, 2, 1).expect("source with B");
    let idx = VideoIndex::parse(&src).expect("idx");
    let decoded = read_video(&src).expect("decode");
    for (name, prof) in profiles() {
        for gop in [1u32, 3, 20] {
            let want = write_video(&decoded, &prof, gop, 2, 0).expect("reference");
            let opts = WriteOpts { gop, search: 2, ..WriteOpts::default() };
            for budget in [1u64, u64::MAX] {
                let (c, _) = transcode(Cursor::new(Vec::new()), &src, &idx, &prof, &opts, budget, &|i| i).expect("transcode");
                assert_eq!(c.into_inner(), want, "{name} gop {gop} budget {budget}");
            }
        }
    }
}
