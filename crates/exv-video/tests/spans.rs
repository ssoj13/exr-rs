//! tracing contract of exv-video: span names exist, `transcode` decodes each frame exactly once, and
//! `decode_frame` decodes only a frame's dependency ancestors (bit-exact vs a full decode).
//!
//! The span counter is a process-global subscriber (spans are created on rayon workers, so a thread-local
//! default would miss them); tests that count are serialized by `LOCK`.

mod common;

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Mutex, Once};

use common::clip;
use exv_codec::Profile;
use exv_video::{VideoIndex, WriteOpts, decode_frame, read_video, transcode, write_video};
use tracing::span;

static COUNTS: Mutex<Option<HashMap<&'static str, usize>>> = Mutex::new(None);
static LOCK: Mutex<()> = Mutex::new(());

struct Count;

impl tracing::Subscriber for Count {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attrs: &span::Attributes<'_>) -> span::Id {
        if let Ok(mut g) = COUNTS.lock() {
            *g.get_or_insert_with(HashMap::new).entry(attrs.metadata().name()).or_default() += 1;
        }
        span::Id::from_u64(1)
    }
    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}
}

/// Serialize a test: every test of this binary holds it for its whole body (an un-counted encode in one
/// test would otherwise leak spans into another test's counters).
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::subscriber::set_global_default(Count).expect("global subscriber");
    });
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run `f` with fresh counters; returns `f`'s value + the counts. Call only while holding [`serial`].
fn counted<R>(f: impl FnOnce() -> R) -> (R, HashMap<&'static str, usize>) {
    *COUNTS.lock().expect("counts") = Some(HashMap::new());
    let r = f();
    let counts = COUNTS.lock().expect("counts").take().unwrap_or_default();
    (r, counts)
}

fn n(c: &HashMap<&'static str, usize>, name: &str) -> usize {
    c.get(name).copied().unwrap_or(0)
}

/// An encode + decode emits the documented span names (clip / frame / GOP at info, `decode_one` at debug).
#[test]
fn span_names_smoke() {
    let _s = serial();
    let frames = clip(6, 16);
    let (_, c) = counted(|| {
        let enc = write_video(&frames, &Profile::lossless_fast(), 3, 2, 0).expect("enc");
        let idx = VideoIndex::parse(&enc).expect("idx");
        decode_frame(&enc, &idx, 4, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("decode_frame");
        let mut dec = exv_video::Decoder::open(&enc).expect("dec");
        dec.frame(1).expect("frame");
        exv_video::read_video_streaming(&enc, &idx, u64::MAX, &|_, _| Ok(())).expect("stream");
    });
    assert_eq!(n(&c, "write_clip"), 1);
    assert_eq!(n(&c, "encode_frame"), 6);
    assert_eq!(n(&c, "decode_frame"), 1);
    assert_eq!(n(&c, "decoder"), 1);
    assert_eq!(n(&c, "read_clip"), 1);
    assert!(n(&c, "decode_gop") >= 3, "decoder GOP + 2 streaming GOPs: {c:?}");
    assert!(n(&c, "decode_one") > 0, "{c:?}");
}

/// `.exv -> .exv` through `read_windows` decodes every source frame exactly once (the old per-frame
/// `decode_frame` loop decoded O(gop^2)), and the output equals the frame-by-frame result.
#[test]
fn transcode_decodes_each_frame_once() {
    let _s = serial();
    let frames = clip(9, 16);
    let src = write_video(&frames, &Profile::lossless_fast(), 4, 2, 1).expect("source");
    let idx = VideoIndex::parse(&src).expect("idx");
    let opts = WriteOpts { gop: 3, search: 2, ..WriteOpts::default() };
    let (out, c) = counted(|| {
        transcode(Cursor::new(Vec::new()), &src, &idx, &Profile::lossless_wave(), &opts, 1, &|i| i).expect("transcode").0.into_inner()
    });
    assert_eq!(n(&c, "decode_one"), 9, "each source frame decoded once: {c:?}");
    assert_eq!(out, write_video(&read_video(&src).expect("decode"), &Profile::lossless_wave(), 3, 2, 0).expect("ref"));
}

/// `decode_frame` is bit-exact vs a full decode over gop x B x MV configurations, and decodes only the
/// frame's ancestor closure: in a `gop 8, b 1` clip display frame 1 (a B between I0 and P2) needs {I0,P2,B1}.
#[test]
fn decode_frame_ancestors_only() {
    let _s = serial();
    let frames = clip(10, 16);
    for gop in [1u32, 3, 5, 8] {
        for b in [0u32, 1, 2] {
            for search in [0i32, 3] {
                let enc = write_video(&frames, &Profile::lossless_fast(), gop, search, b).expect("enc");
                let all = read_video(&enc).expect("full decode");
                let idx = VideoIndex::parse(&enc).expect("idx");
                for (k, want) in all.iter().enumerate() {
                    let got = decode_frame(&enc, &idx, k, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("decode_frame");
                    assert_eq!(got.planes, want.planes, "gop {gop} b {b} search {search} frame {k}");
                }
            }
        }
    }
    let enc = write_video(&frames, &Profile::lossless_fast(), 8, 0, 1).expect("enc");
    let idx = VideoIndex::parse(&enc).expect("idx");
    // display 0..8 = GOP 0: anchors I0 P2 P4 P6 P7; Bs: 1 (0,2), 3 (2,4), 5 (4,6).
    for (k, ancestors) in [(0usize, 1usize), (2, 2), (1, 3), (3, 4), (5, 5), (7, 5)] {
        let (_, c) = counted(|| decode_frame(&enc, &idx, k, exv_video::Chans::All).and_then(exv_video::Frame::into_image).expect("decode_frame"));
        assert_eq!(n(&c, "decode_one"), ancestors, "display frame {k}: {c:?}");
    }
}
