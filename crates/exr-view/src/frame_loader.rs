//! The sequence's decode loader and its memory of frames that failed to decode.
//!
//! `scancache` retries a failed frame (250 ms, 500 ms, then dead) because a frame may fail only
//! while a render is still writing it. A frame that is broken for good (a colour tag nobody can
//! decode, a corrupt file) would be fully decoded again on every retry, 9.7 MB of PNG each time.
//! [`Fails`] makes the failure terminal for as long as the file stays as it was: the loader
//! answers a known-bad `(path, modified, size)` with the stored error without touching the
//! decoder, and the file changing on disk (or a rescan rewriting it) is the one way back.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

use scanseq::Seq;

use crate::image::DecodedFrame;

/// What identifies a file's content between two looks: modification time and size.
type Stamp = (SystemTime, u64);

fn stamp(path: &Path) -> Option<Stamp> {
    let m = std::fs::metadata(path).ok()?;
    Some((m.modified().ok()?, m.len()))
}

struct Fail {
    path: PathBuf,
    stamp: Stamp,
    msg: Arc<str>,
}

/// Frames whose decode failed, by frame number.
#[derive(Default)]
pub(crate) struct Fails(Mutex<HashMap<i32, Fail>>);

impl Fails {
    fn map(&self) -> std::sync::MutexGuard<'_, HashMap<i32, Fail>> {
        // A poisoned map still holds valid entries (every writer is a single insert/remove).
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The stored error of `frame` when its file is still the one that failed.
    fn known(&self, frame: i32, path: &Path) -> Option<Arc<str>> {
        let map = self.map();
        let f = map.get(&frame)?;
        (f.path == path && stamp(path) == Some(f.stamp)).then(|| Arc::clone(&f.msg))
    }

    /// Remember that `path` (looked at as `stamp`, taken BEFORE the decode so a file rewritten
    /// meanwhile is not mistaken for the failed one) did not decode.
    fn record(&self, frame: i32, path: &Path, stamp: Option<Stamp>, msg: &str) {
        // An unreadable file has no identity to compare: leave it to scancache's retries.
        let Some(stamp) = stamp else { return };
        self.map().insert(frame, Fail { path: path.to_path_buf(), stamp, msg: msg.into() });
    }

    /// Forget `frame` (it decoded, or its file was rewritten).
    pub(crate) fn clear(&self, frame: i32) {
        self.map().remove(&frame);
    }

    /// Whether `frame` has a recorded failure.
    pub(crate) fn has(&self, frame: i32) -> bool {
        self.map().contains_key(&frame)
    }

    /// The error of `frame`, for the UI.
    pub(crate) fn msg(&self, frame: i32) -> Option<Arc<str>> {
        self.map().get(&frame).map(|f| Arc::clone(&f.msg))
    }
}

/// How one file becomes a frame (production: [`DecodedFrame::open`]; tests count calls).
pub(crate) type Decode = Arc<dyn Fn(&Path) -> Result<DecodedFrame, String> + Send + Sync>;

/// The cache's loader: frame number -> file via the shared sequence (a rescan may add frames),
/// then `decode`. `cancel` is honoured where it is cheap: before starting, and before publishing
/// a result that went stale (`Scancache` drops a cancelled `Err` with no failure record).
pub(crate) fn loader(seq: Arc<RwLock<Seq>>, fails: Arc<Fails>, decode: Decode) -> scancache::Loader<DecodedFrame> {
    Arc::new(move |frame: i32, cancel: &scancache::Cancel| {
        if cancel.is_cancelled() {
            return Err("cancelled".to_owned());
        }
        let file = seq.read().ok().and_then(|s| s.get_file(i64::from(frame)));
        let Some(file) = file else {
            return Err(format!("frame {frame}: not present in the sequence"));
        };
        let path = Path::new(&file);
        if let Some(msg) = fails.known(frame, path) {
            return Err(msg.to_string());
        }
        let before = stamp(path);
        let result = decode(path);
        let result = match result {
            Ok(f) => {
                fails.clear(frame);
                Ok(f)
            }
            Err(e) => {
                // The same text the single-file open shows (`path: reason`).
                let msg = format!("{}: {e}", path.display());
                log::warn!("frame {frame} failed to decode, not retried until the file changes: {msg}");
                fails.record(frame, path, before, &msg);
                Err(msg)
            }
        };
        if cancel.is_cancelled() {
            return Err("cancelled".to_owned());
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// `n` tiny PNG frames `shot.0001.png`..; frame `bad` is garbage.
    fn frames(dir: &Path, n: i32, bad: i32) {
        std::fs::create_dir_all(dir).unwrap();
        for f in 1..=n {
            let p = dir.join(format!("shot.{f:04}.png"));
            if f == bad {
                std::fs::write(&p, b"this is not a png").unwrap();
            } else {
                crate::image::tests::write_png(&p, [f as u8 * 20, 0, 0, 255]);
            }
        }
    }

    /// A cache over `dir` whose loader counts decodes per frame.
    fn cache(dir: &Path, n: i32) -> (scancache::Scancache<DecodedFrame>, Arc<Fails>, Arc<Vec<AtomicUsize>>) {
        let seq = scanseq::Scanner::from_file(dir.join("shot.0001.png")).expect("sequence");
        let seq = Arc::new(RwLock::new(seq));
        let fails = Arc::new(Fails::default());
        let loads: Arc<Vec<AtomicUsize>> = Arc::new((0..=n).map(|_| AtomicUsize::new(0)).collect());
        let counted = Arc::clone(&loads);
        // The frame number is the digits in the name; count per frame.
        let decode: Decode = Arc::new(move |p: &Path| {
            let f: usize = p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.rsplit('.').next()).unwrap().parse().unwrap();
            counted[f].fetch_add(1, Ordering::SeqCst);
            DecodedFrame::open(p, u64::MAX).map_err(|e| e.status())
        });
        let cache = scancache::Scancache::new(
            (1..=n).collect(),
            1 << 30,
            2,
            Arc::new(|f: &DecodedFrame| f.byte_size()),
            loader(seq, Arc::clone(&fails), decode),
            Arc::new(|| {}),
        );
        cache.set_retry_backoff(Duration::from_millis(1), 3);
        (cache, fails, loads)
    }

    fn settle(cache: &scancache::Scancache<DecodedFrame>) {
        let end = Instant::now() + Duration::from_secs(30);
        while cache.has_pending() {
            assert!(Instant::now() < end, "the cache settles");
            cache.poll();
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// One garbage frame among good ones: the good ones load, the bad one is decoded exactly once
    /// (scancache asks three times), and its error is recorded with the path.
    #[test]
    fn a_failed_frame_is_decoded_once() {
        let tmp = tempfile::Builder::new().prefix("exr-view-failonce-").tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        frames(&dir, 5, 3);
        let (cache, fails, loads) = cache(&dir, 5);
        cache.set_playhead(scancache::Playhead::at(1));
        settle(&cache);
        let counts: Vec<usize> = loads.iter().map(|c| c.load(Ordering::SeqCst)).collect();
        assert_eq!(counts, [0, 1, 1, 1, 1, 1], "every frame decoded once, the bad one included");
        assert!((1..=5).filter(|f| *f != 3).all(|f| cache.contains(f)));
        assert!(!cache.contains(3));
        let msg = fails.msg(3).expect("the failure is recorded");
        assert!(msg.contains("shot.0003.png"), "{msg}");
        assert!(fails.msg(1).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Rewriting the file is the way back: the stamp differs, the frame decodes again.
    #[test]
    fn a_rewritten_frame_is_tried_again() {
        let tmp = tempfile::Builder::new().prefix("exr-view-failretry-").tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        frames(&dir, 2, 2);
        let (cache, fails, loads) = cache(&dir, 2);
        cache.set_playhead(scancache::Playhead::at(1));
        settle(&cache);
        assert_eq!(loads[2].load(Ordering::SeqCst), 1);
        crate::image::tests::write_png(&dir.join("shot.0002.png"), [9, 9, 9, 255]);
        cache.invalidate(2);
        cache.poll();
        settle(&cache);
        assert_eq!(loads[2].load(Ordering::SeqCst), 2, "a changed file is decoded again");
        assert!(cache.contains(2) && fails.msg(2).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
