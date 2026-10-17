//! Clip writing: ONE container assembler ([`Asm`]), ONE frame encoder ([`Enc`]), and the front doors on top.
//!
//! ```text
//!  write_video / write_video_meta / write_still (Vec<u8>)        write_clip (in-RAM, I/P/B, sink)
//!  write_clip_pull (loader callback, I/P, sink)                    transcode (.exv -> .exv, I/P, sink)
//!                       \                                          /
//!                        ClipWriter<W: Write + Seek>  --- push / push_win (parallel indexed encode,
//!                                  |                       sequential ordered append)
//!                                  v
//!                 Enc::frame  (MV search + MC reference + exv_codec tile encode)
//!                 Asm         (prelude | frames | outer index | trailer)  <- also write_still_windowed
//! ```
//!
//! # Streaming and the frame counter
//! The count is stored twice (prelude `@9`, header `@23`) and a caller such as After Effects does not know it
//! when the first frame arrives, so [`ClipWriter`] writes `0` placeholders and PATCHES both at
//! [`ClipWriter::finish`] (the index already sits at the END of the file, so frames stream out with no
//! rewind). The sink therefore must be `Write + Seek`; the known-count still writer ([`write_still_windowed`])
//! only needs `Write` and shares the same [`Asm`].
//!
//! # Determinism
//! Frames of a window encode in parallel through an INDEXED rayon collect (order preserved) and are appended
//! serially, so the file is byte-identical for any thread count. Temporal prediction is open-loop (frame
//! `f` is predicted from the SOURCE planes of its reference, never from encoded bytes), which is what makes
//! the parallel encode legal. Do not use `par_bridge` here (unordered).

use std::io::{self, Cursor, Seek, SeekFrom, Write};

use exv_codec::Profile;
use exv_codec::frame::{self, RefPlanes};
use exv_codec::image::Image;
use exv_format::header::FRAME_COUNT_OFF;
use exv_format::{Header, Meta, WriteError};
use exv_tile::{Budget, PlanePool};
use rayon::prelude::*;
use tracing::{Span, field, info_span, trace_span};

use crate::container::{
    FrameKind, IDX_ENTRY, MAGIC, Mv, PRELUDE_COUNT_OFF, PRELUDE_LEN, RefChan, VERSION, VideoIndex, mc_avg, mc_shift,
    planes_match, same_shape, write_frame_hdr,
};
use crate::error::VideoError;
use crate::motion;

/// Encoder options shared by every write path.
#[derive(Clone, Debug)]
pub struct WriteOpts {
    /// I-frame cadence in DISPLAY order (`0` is treated as `1` = all-intra).
    pub gop: u32,
    /// Global-MV search radius; `<= 0` disables the search (MV = 0).
    pub search: i32,
    /// B-frames between anchors (`0` => plain I/P). Whole-clip writers only ([`write_clip`]; the streaming
    /// ones refuse `> 0`); a lossy profile always stays I/P.
    pub b_frames: u32,
    /// Self-describing clip metadata (fps / color / HDR-transfer / timecode); empty => byte-identical header.
    pub meta: Meta,
}

impl Default for WriteOpts {
    /// All-intra, no search, no B-frames, no metadata (what a still uses).
    fn default() -> Self {
        Self { gop: 1, search: 0, b_frames: 0, meta: Meta::default() }
    }
}

// ───────────────────────────── layout helpers ─────────────────────────────

/// The authoritative clip header: `base` geometry + the write profile's codec id/params baked in +
/// `frame_count = n`. The single codec/geometry source for every frame (shared by all write paths).
fn resolve_header(base: &Header, n: u32, profile: &Profile) -> Header {
    let mut hdr = base.clone();
    hdr.codec_id = profile.codec_id().0;
    hdr.codec_params = profile.params();
    hdr.frame_count = n;
    hdr
}

/// Container prelude `MAGIC | version | gop | frame_count | Header(+Meta)`. `frame_count` comes from
/// `hdr` so the prelude and the embedded header can never disagree. The single prelude writer.
fn prelude(gop: u32, hdr: &Header, meta: &Meta) -> Result<Vec<u8>, WriteError> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.push(VERSION);
    out.extend_from_slice(&gop.to_le_bytes());
    out.extend_from_slice(&hdr.frame_count.to_le_bytes());
    hdr.write_with_meta(meta, &mut out)?;
    Ok(out)
}

/// Outer seek index `{ n_frames, (kind, offset, display)* }` + the `index_off` trailer, as written right
/// after the frames. `entries` are in DECODE order; `index_off` = byte offset of this index.
/// The single index writer (mirror of the parse in [`VideoIndex::parse`]).
fn index_bytes(index_off: u64, entries: &[(FrameKind, u64, u32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + entries.len() * IDX_ENTRY + 8);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes()); // `Asm::put_with` bounds `n` to u32
    for &(kind, off, disp) in entries {
        out.push(kind as u8);
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&disp.to_le_bytes());
    }
    out.extend_from_slice(&index_off.to_le_bytes());
    out
}

/// Container assembler: lays out `prelude | frames | outer index | trailer` on a plain `Write`. The ONE
/// place that records a frame's index entry and advances the byte position (every write path, incl. the
/// out-of-core still, goes through [`Asm::put_with`]).
struct Asm<W: Write> {
    out: W,
    /// Bytes written so far (== the next frame's offset).
    pos: u64,
    /// `(kind, offset, display)` per frame in DECODE order.
    entries: Vec<(FrameKind, u64, u32)>,
}

impl<W: Write> Asm<W> {
    /// Write the prelude; the frame count inside `hdr` is whatever the caller wants stored for now.
    fn begin(mut out: W, gop: u32, hdr: &Header, meta: &Meta) -> Result<Self, VideoError> {
        let head = prelude(gop, hdr, meta)?;
        out.write_all(&head)?;
        Ok(Self { out, pos: head.len() as u64, entries: Vec::new() })
    }

    /// Append one frame whose bytes `write` emits (returns the byte count). The entry is recorded only
    /// after the write succeeded.
    fn put_with(
        &mut self,
        kind: FrameKind,
        display: u32,
        write: impl FnOnce(&mut W) -> io::Result<u64>,
    ) -> Result<(), VideoError> {
        if self.entries.len() >= u32::MAX as usize {
            return Err(VideoError::TooManyFrames);
        }
        let at = self.pos;
        let n = write(&mut self.out)?;
        self.entries.push((kind, at, display));
        self.pos += n;
        Ok(())
    }

    /// Append one already-encoded frame (`frame_hdr + body`).
    fn put(&mut self, kind: FrameKind, display: u32, bytes: &[u8]) -> Result<(), VideoError> {
        self.put_with(kind, display, |o| o.write_all(bytes).map(|()| bytes.len() as u64))
    }

    /// Write the outer index + trailer; returns the sink and the total bytes written.
    fn end(mut self) -> Result<(W, u64), VideoError> {
        let idx = index_bytes(self.pos, &self.entries);
        self.out.write_all(&idx)?;
        Ok((self.out, self.pos + idx.len() as u64))
    }
}

// ───────────────────────────── frame encoder ─────────────────────────────

/// The global MV of `cur` vs `refp` (channel 0), or `(0,0)` for a lossy profile / no-search.
fn mv_search(cur: &[Vec<u8>], refp: &[Vec<u8>], header: &Header, lossless: bool, search: i32) -> Mv {
    if lossless && search > 0 && !header.channels.is_empty() {
        let sp = trace_span!("mv_search", search, dx = field::Empty, dy = field::Empty);
        let _g = sp.enter();
        let (dx, dy) =
            motion::search_global_mv(&cur[0], &refp[0], header.width, header.height, header.channels[0].dtype, search);
        sp.record("dx", dx).record("dy", dy);
        (dx as i16, dy as i16)
    } else {
        (0, 0)
    }
}

/// One encoded frame: its kind and `frame_hdr + tiled body` bytes.
type Coded = (FrameKind, Vec<u8>);
/// What a frame header + codec call needs about the references: decode indices and MVs (the MC planes live in `frame`).
type RefSet = (Vec<u32>, Vec<Mv>);

/// Everything needed to encode a frame; shared read-only by the parallel encode tasks.
struct Enc {
    /// Resolved clip header (geometry + codec; its `frame_count` is irrelevant to encoding).
    hdr: Header,
    profile: Profile,
    search: i32,
    lossless: bool,
    /// Recycler of the motion-compensated reference planes (a shifted / averaged COPY per P / B frame: a full frame
    /// of fresh zeroed pages each time otherwise). Thread-safe; the parallel encode tasks share it.
    pool: PlanePool,
    /// Idle cap of `pool`, re-sized per window by [`Enc::keep`].
    idle: Budget,
}

impl Enc {
    fn new(base: &Header, profile: &Profile, search: i32) -> Self {
        let hdr = resolve_header(base, 0, profile);
        let idle = Budget::new(0);
        Self { hdr, profile: *profile, search, lossless: profile.is_lossless(), pool: PlanePool::new(idle.clone()), idle }
    }

    /// Let the MC pool keep what `inflight` concurrent frames of the window can hold idle: one frame of planes per
    /// P frame, three per B frame (two shifted references + their mean). That is at most the window's own frames, so
    /// the clip's peak (window + their MC copies, which exist while encoding anyway) does not grow; between windows the
    /// shelf is what makes the next window allocate nothing.
    fn keep(&self, inflight: usize, sets: u64) {
        let frame: u64 = self.hdr.channels.iter().filter_map(|c| self.hdr.plane_len(c.dtype)).map(|n| n as u64).sum();
        let par = inflight.min(rayon::current_num_threads()).max(1) as u64;
        self.idle.set(frame.saturating_mul(par).saturating_mul(sets));
    }

    /// Encode one frame of `kind` against `refs` = `(source planes, DECODE index)` of its references
    /// (none for Intra, one for Pred, two for Bidir). The motion-compensated / averaged reference is built
    /// here from the SOURCE planes (open loop), so any frame can be encoded independently of the others.
    fn frame(&self, kind: FrameKind, cur: &[Vec<u8>], refs: &[(&[Vec<u8>], u32)]) -> Result<Coded, VideoError> {
        let hdr = &self.hdr;
        let pool = &self.pool;
        let mc: Option<Vec<RefChan>>; // the reconstructed (MC / mean) reference, pooled planes returning on drop
        let (refs_dec, mvs): RefSet = match (kind, refs) {
            (FrameKind::Intra, []) => {
                mc = None;
                (vec![], vec![])
            }
            (FrameKind::Pred, &[(p, di)]) => {
                let mv = mv_search(cur, p, hdr, self.lossless, self.search);
                mc = Some(mc_shift(pool, p, hdr, mv)?);
                (vec![di], vec![mv])
            }
            (FrameKind::Bidir, &[(a, da), (b, db)]) => {
                let (mv0, mv1) = (
                    mv_search(cur, a, hdr, self.lossless, self.search),
                    mv_search(cur, b, hdr, self.lossless, self.search),
                );
                let (sa, sb) = (mc_shift(pool, a, hdr, mv0)?, mc_shift(pool, b, hdr, mv1)?);
                mc = Some(mc_avg(pool, &sa, &sb, hdr)?);
                (vec![da, db], vec![mv0, mv1])
            }
            _ => return Err(VideoError::Shape("reference count does not match the frame kind")),
        };
        let reference: Option<RefPlanes> = mc.as_ref().map(|r| r.iter().map(|c| Some(c.bytes())).collect());
        let mut out = Vec::new();
        write_frame_hdr(&mut out, kind, &refs_dec, &mvs);
        out.extend_from_slice(&frame::write_frame_tiles_ref(cur, reference.as_ref(), hdr, &self.profile)?);
        Ok((kind, out))
    }

    /// I/P frame `fi` (decode order == display order): Intra at a GOP anchor (`fi % gop == 0`) or with no
    /// predecessor, else Pred against `prev` (frame `fi - 1`).
    fn frame_ip(&self, cur: &[Vec<u8>], prev: Option<&[Vec<u8>]>, fi: usize, gop: u32) -> Result<Coded, VideoError> {
        match prev {
            Some(p) if !(fi as u32).is_multiple_of(gop) => self.frame(FrameKind::Pred, cur, &[(p, (fi - 1) as u32)]),
            _ => self.frame(FrameKind::Intra, cur, &[]),
        }
    }
}

/// A per-frame coding plan in DECODE order. `refs_display` are DISPLAY indices (converted to decode
/// indices at write time); `Intra` has none, `Pred` one (past), `Bidir` two (past, future).
struct Plan {
    display: usize,
    kind: FrameKind,
    refs_display: Vec<usize>,
}

/// Build the decode-order coding plan for `n` display frames. `gop` = I-frame cadence (display order);
/// `b` = B-frames between anchors (`0` => plain I/P, decode == display). Each display-GOP `[g, g+gop)` is
/// planned INDEPENDENTLY (a CLOSED GOP — no ref crosses an I -> bounded random access): its first frame
/// is I, anchors sit every `b+1` positions with the GOP's LAST frame forced to be an anchor (so trailing
/// frames are P, not dangling B), subsequent anchors are P (ref = previous anchor), and each B strictly
/// between two anchors references both (past + future), emitted AFTER the future anchor.
fn plan_frames(n: usize, gop: usize, b: usize) -> Vec<Plan> {
    let gop = gop.max(1);
    let step = b + 1;
    let mut plans = Vec::with_capacity(n);
    let mut g = 0;
    while g < n {
        let gend = (g + gop).min(n); // exclusive; a fresh I starts at `gend` if < n
        let mut anchors: Vec<usize> = (g..gend).step_by(step).collect();
        if anchors.last() != Some(&(gend - 1)) {
            anchors.push(gend - 1); // the GOP's last frame must anchor B's on both sides
        }
        // anchors[0] == g by construction: the GOP's I-frame.
        plans.push(Plan { display: g, kind: FrameKind::Intra, refs_display: vec![] });
        for w in anchors.windows(2) {
            let (pa, a) = (w[0], w[1]);
            plans.push(Plan { display: a, kind: FrameKind::Pred, refs_display: vec![pa] });
            for bp in (pa + 1)..a {
                plans.push(Plan { display: bp, kind: FrameKind::Bidir, refs_display: vec![pa, a] });
            }
        }
        g = gend;
    }
    plans
}

// ───────────────────────────── ClipWriter ─────────────────────────────

/// Streaming clip writer: frames are encoded and appended as they arrive, memory is independent of the
/// clip length (the caller's frame + the previous frame when `gop > 1` + one encoded window).
///
/// Lifecycle: [`new`](Self::new) (writes the prelude) -> [`push`](Self::push) / [`push_win`](Self::push_win)
/// repeatedly -> [`finish`](Self::finish) (index + trailer + frame-count patch). Dropping it unfinished
/// leaves an incomplete file: put an [`exv_format::fsio::OutFile`] underneath and the drop deletes it.
/// Any failed call POISONS the writer ([`VideoError::Poisoned`] afterwards): the output is incomplete.
///
/// Only I/P clips stream (B needs the whole clip to reorder: use [`write_clip`]). The sink must start at
/// byte 0.
pub struct ClipWriter<W: Write + Seek> {
    asm: Asm<W>,
    enc: Enc,
    gop: u32,
    /// `opts.b_frames`: refused by the push paths, honored by [`write_clip`].
    b_frames: u32,
    /// Source planes of the previous frame (the P reference); kept only when `gop > 1`.
    prev: Option<Vec<Vec<u8>>>,
    failed: bool,
    /// Clip-level span, stored (not an entered guard: guards are thread-affine, the writer may move
    /// between threads, e.g. the C ABI) and entered per call.
    span: Span,
}

impl<W: Write + Seek> ClipWriter<W> {
    /// Start a clip on `out` (which must be positioned at 0): resolves the header from `base` + `profile`
    /// and writes the prelude with frame-count placeholders.
    ///
    /// # Errors
    /// [`VideoError::Header`] if the header is not representable, [`VideoError::Io`] on a sink error.
    pub fn new(mut out: W, base: &Header, profile: &Profile, opts: &WriteOpts) -> Result<Self, VideoError> {
        if out.stream_position()? != 0 {
            return Err(VideoError::Unsupported("the sink must start at byte 0"));
        }
        let gop = opts.gop.max(1);
        let enc = Enc::new(base, profile, opts.search);
        let span = info_span!(
            "write_clip",
            w = enc.hdr.width,
            h = enc.hdr.height,
            nch = enc.hdr.channels.len(),
            gop,
            codec = enc.hdr.codec_id,
            n = field::Empty,
            bytes = field::Empty
        );
        let asm = span.in_scope(|| Asm::begin(out, gop, &enc.hdr, &opts.meta))?;
        Ok(Self { asm, enc, gop, b_frames: opts.b_frames, prev: None, failed: false, span })
    }

    /// Frames appended so far.
    pub fn frames(&self) -> usize {
        self.asm.entries.len()
    }

    /// Bytes written so far (prelude + frames; the index is added at [`finish`](Self::finish)).
    pub fn bytes(&self) -> u64 {
        self.asm.pos
    }

    /// Run `f` inside the clip span; a failure poisons the writer.
    fn guard<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, VideoError>) -> Result<T, VideoError> {
        if self.failed {
            return Err(VideoError::Poisoned);
        }
        let span = self.span.clone();
        let r = span.in_scope(|| f(self));
        self.failed |= r.is_err();
        r
    }

    /// Append one I/P frame, taking its planes (they become the next P reference without a copy).
    pub fn push(&mut self, img: Image) -> Result<(), VideoError> {
        self.push_win(vec![img], &|_| {})
    }

    /// Append a window of consecutive I/P frames: they encode IN PARALLEL (the P reference is the previous
    /// SOURCE frame, so frames are independent), then append in order — the bytes do not depend on the
    /// thread count. `on_frame(n)` is called from the worker when frame number `n` (global, 0-based) is
    /// encoded (progress).
    ///
    /// # Errors
    /// [`VideoError::Mismatch`] if a frame differs from the clip's shape, [`VideoError::Unsupported`] if the
    /// writer was opened with `b_frames > 0`, codec / sink errors otherwise.
    pub fn push_win(&mut self, win: Vec<Image>, on_frame: &(dyn Fn(usize) + Sync)) -> Result<(), VideoError> {
        self.guard(|s| s.push_win_inner(win, on_frame))
    }

    fn push_win_inner(&mut self, mut win: Vec<Image>, on_frame: &(dyn Fn(usize) + Sync)) -> Result<(), VideoError> {
        if self.b_frames > 0 {
            return Err(VideoError::Unsupported("B-frames need the whole clip: use write_clip"));
        }
        let base = self.frames();
        for (i, img) in win.iter().enumerate() {
            if !same_shape(&img.header, &self.enc.hdr) || !planes_match(&img.planes, &img.header) {
                return Err(VideoError::Mismatch { frame: base + i });
            }
        }
        let carry = self.prev.take();
        self.enc.keep(win.len(), 1);
        let (enc, gop, span) = (&self.enc, self.gop, &self.span);
        let coded = win
            .par_iter()
            .enumerate()
            .map(|(i, img)| {
                let fi = base + i;
                let sp = info_span!(parent: span, "encode_frame", frame = fi, bytes = field::Empty);
                let _g = sp.enter();
                // First frame of the window predicts from the previous window's last frame (`carry`).
                let prev = if i == 0 { carry.as_deref() } else { win.get(i - 1).map(|p| p.planes.as_slice()) };
                let c = enc.frame_ip(&img.planes, prev, fi, gop)?;
                sp.record("bytes", c.1.len());
                on_frame(fi);
                Ok(c)
            })
            .collect::<Result<Vec<Coded>, VideoError>>()?;
        for (i, (kind, bytes)) in coded.into_iter().enumerate() {
            // I/P: decode order == display order, so display_index = the running frame number.
            self.asm.put(kind, (base + i) as u32, &bytes)?;
        }
        if self.gop > 1 {
            self.prev = win.pop().map(|i| i.planes);
        }
        Ok(())
    }

    /// Durability point: flush the sink. Writes NO index (the file is not valid until [`finish`](Self::finish)).
    pub fn flush(&mut self) -> Result<(), VideoError> {
        self.guard(|s| Ok(s.asm.out.flush()?))
    }

    /// Append one already-encoded frame in DECODE order (the whole-clip writer's append).
    fn append(&mut self, kind: FrameKind, display: u32, bytes: &[u8]) -> Result<(), VideoError> {
        self.guard(|s| s.asm.put(kind, display, bytes))
    }

    /// Write the outer index + trailer, patch both frame counters, flush; returns the sink and the total
    /// file size in bytes.
    ///
    /// # Errors
    /// [`VideoError::Empty`] if no frame was appended (a clip needs one), [`VideoError::Poisoned`] after an
    /// earlier failure, [`VideoError::Io`] on a sink error.
    pub fn finish(self) -> Result<(W, u64), VideoError> {
        if self.failed {
            return Err(VideoError::Poisoned);
        }
        let span = self.span.clone();
        span.in_scope(|| self.finish_inner())
    }

    fn finish_inner(self) -> Result<(W, u64), VideoError> {
        let n = u32::try_from(self.asm.entries.len()).map_err(|_| VideoError::TooManyFrames)?;
        if n == 0 {
            return Err(VideoError::Empty);
        }
        let index_off = self.asm.pos;
        let (mut out, total) = self.asm.end()?;
        tracing::debug!(n, index_off, "write_index");
        for at in [PRELUDE_COUNT_OFF, PRELUDE_LEN + FRAME_COUNT_OFF] {
            out.seek(SeekFrom::Start(at as u64))?;
            out.write_all(&n.to_le_bytes())?;
        }
        out.seek(SeekFrom::End(0))?;
        out.flush()?;
        self.span.record("n", n).record("bytes", total);
        Ok((out, total))
    }
}

// ───────────────────────────── front doors ─────────────────────────────

/// Check every frame is the same shape as the first and internally consistent (a hand-built [`Image`] can
/// be inconsistent), so the codec below never indexes blindly.
fn check_frames(frames: &[Image]) -> Result<&Image, VideoError> {
    let first = frames.first().ok_or(VideoError::Empty)?;
    u32::try_from(frames.len()).map_err(|_| VideoError::TooManyFrames)?;
    for (i, f) in frames.iter().enumerate() {
        if !same_shape(&f.header, &first.header) || !planes_match(&f.planes, &f.header) {
            return Err(VideoError::Mismatch { frame: i });
        }
    }
    Ok(first)
}

/// Encode a whole in-RAM clip (frames in DISPLAY order) onto `out`: planned GOPs and B-frames, frames
/// encoded in parallel chunks and appended in decode order. MV/PRED/B apply to a LOSSLESS profile only; a
/// lossy profile stays I/P, MV=0, INTRA/SKIP. Returns the sink and the file size.
///
/// # Errors
/// [`VideoError::Empty`] / [`VideoError::Mismatch`] on bad input, otherwise as [`ClipWriter`].
pub fn write_clip<W: Write + Seek>(
    out: W,
    frames: &[Image],
    profile: &Profile,
    opts: &WriteOpts,
) -> Result<(W, u64), VideoError> {
    let first = check_frames(frames)?;
    // B needs a future reference -> open-loop bit-pattern mean -> LOSSLESS only (else drift).
    let b = if profile.is_lossless() { opts.b_frames as usize } else { 0 };
    let mut w = ClipWriter::new(out, &first.header, profile, opts)?;
    let plans = plan_frames(frames.len(), w.gop as usize, b);
    // display -> decode index (decode index = position in `plans`).
    let mut decode_of = vec![0u32; frames.len()];
    for (i, p) in plans.iter().enumerate() {
        *decode_of.get_mut(p.display).ok_or(VideoError::Shape("planned display index out of range"))? = i as u32;
    }
    let span = w.span.clone();
    // Chunked so only O(threads) encoded frames are resident; the chunk size cannot change the bytes.
    for chunk in plans.chunks((4 * rayon::current_num_threads()).max(1)) {
        let enc = &w.enc;
        enc.keep(chunk.len(), if b > 0 { 3 } else { 1 });
        let coded = chunk
            .par_iter()
            .map(|plan| {
                let sp = info_span!(parent: &span, "encode_frame", frame = plan.display, kind = ?plan.kind, bytes = field::Empty);
                let _g = sp.enter();
                let planes = |d: usize| frames.get(d).map(|f| f.planes.as_slice());
                let refs = plan
                    .refs_display
                    .iter()
                    .map(|&d| Some((planes(d)?, *decode_of.get(d)?)))
                    .collect::<Option<Vec<_>>>()
                    .ok_or(VideoError::Shape("reference display index out of range"))?;
                let cur = planes(plan.display).ok_or(VideoError::Shape("display index out of range"))?;
                let c = enc.frame(plan.kind, cur, &refs)?;
                sp.record("bytes", c.1.len());
                Ok(c)
            })
            .collect::<Result<Vec<Coded>, VideoError>>()?;
        for (plan, (kind, bytes)) in chunk.iter().zip(coded) {
            w.append(kind, plan.display as u32, &bytes)?;
        }
    }
    w.finish()
}

/// In-RAM clip to a `Vec<u8>` (thin wrapper over [`write_clip`] on a `Cursor`).
pub fn write_video_meta(frames: &[Image], profile: &Profile, opts: &WriteOpts) -> Result<Vec<u8>, VideoError> {
    write_clip(Cursor::new(Vec::new()), frames, profile, opts).map(|(c, _)| c.into_inner())
}

/// As [`write_video_meta`] with positional `gop` / `search` / `b_frames` and no metadata.
pub fn write_video(
    frames: &[Image],
    profile: &Profile,
    gop: u32,
    search: i32,
    b_frames: u32,
) -> Result<Vec<u8>, VideoError> {
    write_video_meta(frames, profile, &WriteOpts { gop, search, b_frames, meta: Meta::default() })
}

/// A STILL is a 1-frame `.exv` (the unified model — no separate still format): one all-INTRA frame.
pub fn write_still(img: &Image, profile: &Profile) -> Result<Vec<u8>, VideoError> {
    write_video(std::slice::from_ref(img), profile, 1, 0, 0)
}

/// Pull-model I/P clip writer for sources larger than RAM: frames come from `get(i)`, `W = budget_bytes /
/// frame_bytes` of them are loaded + encoded in parallel at a time ([`ClipWriter::push_win`]), so any frame
/// count fits a fixed budget. `on_frame(i)` reports encoded frames (progress). The loader's failure is
/// returned as [`VideoError::Source`] with its message.
///
/// # Errors
/// [`VideoError::Empty`] for `n == 0`, [`VideoError::Unsupported`] for `opts.b_frames > 0`.
pub fn write_clip_pull<W: Write + Seek>(
    out: W,
    n: usize,
    profile: &Profile,
    opts: &WriteOpts,
    budget_bytes: u64,
    get: &(dyn Fn(usize) -> Result<Image, String> + Sync),
    on_frame: &(dyn Fn(usize) + Sync),
) -> Result<(W, u64), VideoError> {
    if n == 0 {
        return Err(VideoError::Empty);
    }
    let load = |frame: usize| get(frame).map_err(|msg| VideoError::Source { frame, msg });
    let f0 = load(0)?;
    let frame_bytes = f0.header.frame_bytes().unwrap_or(u64::MAX).max(1);
    let win = exv_tile::par_for(budget_bytes, frame_bytes, n);
    let mut w = ClipWriter::new(out, &f0.header, profile, opts)?;
    drop(f0); // the first window reloads frame 0 (one extra load — negligible for large n)
    let mut s = 0usize;
    while s < n {
        let e = (s + win).min(n);
        let frames = (s..e).into_par_iter().map(load).collect::<Result<Vec<_>, _>>()?;
        w.push_win(frames, on_frame)?;
        s = e;
    }
    w.finish()
}

/// Re-encode an existing `.exv` (new profile / GOP / metadata) without decoding any frame twice: the
/// source is read as whole-GOP windows ([`crate::read_windows`]) and each window feeds
/// [`ClipWriter::push_win`] (the destination GOP cadence is independent of the source's). `map` is applied
/// to every decoded frame first (identity `&|i| i`, or e.g. a dtype downcast; it is also the progress hook:
/// it runs once per frame, on a worker); the clip header comes from the first MAPPED frame. I/P output only.
///
/// # Errors
/// [`VideoError::Decode`] for a source frame that fails to decode, [`VideoError::Empty`] for a clip with no
/// frame, otherwise as [`ClipWriter`].
pub fn transcode<W: Write + Seek>(
    out: W,
    data: &[u8],
    idx: &VideoIndex,
    profile: &Profile,
    opts: &WriteOpts,
    budget_bytes: u64,
    map: &(dyn Fn(Image) -> Image + Sync),
) -> Result<(W, u64), VideoError> {
    let mut out = Some(out);
    let mut w: Option<ClipWriter<W>> = None;
    for win in crate::container::read_windows(data, idx, budget_bytes) {
        let win: Vec<Image> = win?.into_par_iter().map(map).collect();
        if w.is_none() {
            let first = win.first().ok_or(VideoError::Empty)?;
            let sink = out.take().ok_or(VideoError::Poisoned)?;
            w = Some(ClipWriter::new(sink, &first.header, profile, opts)?);
        }
        w.as_mut().ok_or(VideoError::Poisoned)?.push_win(win, &|_| {})?;
    }
    w.ok_or(VideoError::Empty)?.finish()
}

impl From<VideoError> for io::Error {
    /// For the `io::Result` writers: a sink error passes through, a refused allocation is `OutOfMemory`,
    /// everything else is `InvalidInput` (an unrepresentable header / bad frame) with the typed message.
    fn from(e: VideoError) -> Self {
        match e {
            VideoError::Io(e) => e,
            e @ VideoError::Alloc(_) => io::Error::new(io::ErrorKind::OutOfMemory, e),
            other => io::Error::new(io::ErrorKind::InvalidInput, other),
        }
    }
}

/// Streaming, windowed **out-of-core** still writer (a still = a 1-frame all-INTRA `.exv`) — the
/// mirror of [`write_still`] that NEVER materializes the whole image in RAM. Wraps the container
/// (via the shared [`Asm`]) around [`frame::write_intra_frame_body_windowed`]. `produce(t0,t1)` yields the raw
/// per-channel bytes of tiles `[t0,t1)` IN ORDER (peak heap = one encode window, not `O(image)`). `budget` is the
/// byte budget of that window (an `exv_tile::Budget` cap): the tile window is `budget / bytes-per-tile`.
/// The count is known (1), so the sink only needs `Write`. `meta` is recorded in the header like every other
/// writer's (an empty [`Meta`] is byte-identical to none). Returns total bytes.
pub fn write_still_windowed(
    header: &Header,
    profile: &Profile,
    meta: &Meta,
    budget: u64,
    out: &mut impl Write,
    produce: impl FnMut(u64, u64) -> io::Result<Vec<Vec<Vec<u8>>>>,
) -> io::Result<u64> {
    let hdr = resolve_header(header, 1, profile);
    let mut asm = Asm::begin(out, 1, &hdr, meta)?;
    asm.put_with(FrameKind::Intra, 0, |o| {
        let mut fh = Vec::new();
        write_frame_hdr(&mut fh, FrameKind::Intra, &[], &[]);
        o.write_all(&fh)?;
        Ok(fh.len() as u64 + frame::write_intra_frame_body_windowed(&hdr, profile, budget, o, produce)?)
    })?;
    Ok(asm.end()?.1)
}
