//! Playback clock (VW-3), driven over the frames ACTUALLY PRESENT on disk.
//!
//! The clock idea is **lifted from the operator's debugged `playa`**
//! (`playa-engine/src/core/player.rs::update`): an FPS-based, real-time clock
//! that advances by `elapsed * fps` each tick, looping or clamping at the ends.
//! We take ONLY the clock — none of playa's `Project`/event-bus/comp machinery.
//!
//! What changed for the scanseq port (VW-3 bug fix): playback no longer walks a
//! contiguous `first..=last` integer range. It steps over the **present frame
//! set** — `scanseq::Seq::indices` — so missing frames (gaps) are simply skipped
//! and an unpadded sequence (`heart.0.exr … heart.249.exr`) plays its real
//! frames in order. The cursor is a position into `indices`; `current()` maps it
//! back to the on-disk frame number the cache/viewport key on.
//!
//! Frame advance is fractional-accumulator based (not "one frame per tick") so
//! the playback rate is independent of the UI redraw rate.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// FPS-based playback over a sorted set of **present** frame numbers.
///
/// `indices` holds the frames that actually exist on disk (from
/// `scanseq::Seq::indices`); `pos` is the cursor INTO that set, so stepping and
/// the slider naturally skip gaps. `current()` returns `indices[pos]` — the
/// on-disk frame number the cache and viewport use. Scrubbing snaps to the
/// nearest present frame via [`Playback::scrub`].
#[derive(Debug, Clone)]
pub struct Playback {
    /// Whether the clock is advancing.
    pub playing: bool,
    /// Playback rate in frames per second.
    pub fps: f32,
    /// Loop back to the first present frame at the end (vs. stop on the last).
    pub loop_: bool,
    /// Playing towards lower frame numbers (Shift+Space); the stepping keys ignore it.
    pub reverse: bool,
    /// The frames actually present on disk (sorted, deduped, non-empty).
    indices: Vec<i32>,
    /// Cursor into `indices` — the source of truth for "current frame".
    pos: usize,
    /// Wall-clock timestamp of the previous `tick`. `None` until the first tick.
    last_tick: Option<Instant>,
    /// Fractional-frame accumulator (carries the sub-frame remainder).
    accum: f32,
    /// Clock steps taken since [`Playback::take_stepped`] last read them: the
    /// frames the clock moved while the viewport showed one ([`ShowMeter`]).
    stepped: u32,
}

impl Playback {
    /// New paused clock over the present frame set `indices`, at `fps`, parked
    /// on the first present frame. `indices` MUST be non-empty and sorted (it
    /// comes straight from `scanseq::Seq::indices`); an empty set degenerates to
    /// a single frame `0` so the viewer never indexes out of bounds.
    pub fn new(indices: Vec<i32>, fps: f32) -> Self {
        let indices = if indices.is_empty() { vec![0] } else { indices };
        Self {
            playing: false,
            fps: fps.max(0.001),
            loop_: true,
            reverse: false,
            indices,
            pos: 0,
            last_tick: None,
            accum: 0.0,
            stepped: 0,
        }
    }

    /// The clock steps taken since the last call (and reset them). The viewport
    /// reads it when it swaps in a new frame: more than one step means frames
    /// went by unshown.
    pub fn take_stepped(&mut self) -> u32 {
        std::mem::take(&mut self.stepped)
    }

    /// The current on-disk frame number (`indices[pos]`). Always valid: `pos` is
    /// kept in-bounds and `indices` is non-empty by construction.
    pub fn current(&self) -> i32 {
        // `pos` is clamped on every mutation; `indices` is non-empty.
        self.indices.get(self.pos).copied().unwrap_or(0)
    }

    /// First present frame number.
    pub fn first(&self) -> i32 {
        self.indices.first().copied().unwrap_or(0)
    }

    /// Last present frame number.
    pub fn last(&self) -> i32 {
        self.indices.last().copied().unwrap_or(0)
    }

    /// Number of present frames.
    pub fn frame_count(&self) -> usize {
        self.indices.len()
    }

    /// The present frame set (read-only) — used to prefetch neighbours.
    pub fn indices(&self) -> &[i32] {
        &self.indices
    }

    /// Current cursor position (for a position-based scrub slider).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Current cursor as a `0..=1` fraction across the present frames — drives
    /// the on-screen scrubber playhead. `0.0` for a single-frame sequence.
    pub fn position(&self) -> f32 {
        let last = self.indices.len().saturating_sub(1);
        if last == 0 {
            0.0
        } else {
            self.pos as f32 / last as f32
        }
    }

    /// Scrub to a `0..=1` fraction across the present frames (the on-screen LMB
    /// scrubber). `frac` is clamped; it maps to the nearest present frame index.
    pub fn scrub_fraction(&mut self, frac: f32) {
        let last = self.indices.len().saturating_sub(1);
        let pos = (frac.clamp(0.0, 1.0) * last as f32).round() as usize;
        self.set_pos(pos);
    }

    /// Toggle play/pause. Starting playback re-arms the clock so the first tick
    /// measures from *now* (no spurious large jump after a long pause).
    pub fn toggle(&mut self) {
        self.playing = !self.playing;
        self.last_tick = None;
        self.accum = 0.0;
        self.stepped = 0;
    }

    /// Play in direction `reverse`: pause when already playing that way, else
    /// (re)start that way - so Space and Shift+Space each toggle their own direction
    /// and switch between them without a stop.
    pub fn play(&mut self, reverse: bool) {
        if self.playing && self.reverse == reverse {
            self.toggle();
        } else {
            self.reverse = reverse;
            self.playing = true;
            self.last_tick = None;
            self.accum = 0.0;
            self.stepped = 0;
        }
    }

    /// Jump to the first present frame.
    pub fn go_first(&mut self) {
        self.set_pos(0);
    }

    /// Jump to the last present frame.
    pub fn go_last(&mut self) {
        self.set_pos(self.indices.len().saturating_sub(1));
    }

    /// Jump to cursor position `pos` (e.g. from a position slider). Clamps into
    /// the present set, stops the accumulator drift, and re-arms the clock.
    pub fn set_pos(&mut self, pos: usize) {
        let max = self.indices.len().saturating_sub(1);
        self.pos = pos.min(max);
        self.accum = 0.0;
        self.last_tick = None;
    }

    /// Snap to the present frame nearest `frame` (scrub by frame number). Clamps
    /// into the present set; with gaps this lands on the closest existing frame.
    pub fn scrub(&mut self, frame: i32) {
        // Find the nearest present frame by absolute distance (indices sorted).
        let nearest = self
            .indices
            .iter()
            .enumerate()
            .min_by_key(|&(_, &f)| (f - frame).unsigned_abs())
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.set_pos(nearest);
    }

    /// Step one present frame forward (wrapping when `loop_`, else clamping at
    /// the last present frame). Skips gaps — moves to the next existing frame.
    pub fn step_next(&mut self) {
        let last = self.indices.len().saturating_sub(1);
        if self.pos >= last {
            if self.loop_ {
                self.pos = 0;
            }
        } else {
            self.pos += 1;
        }
        self.accum = 0.0;
        self.last_tick = None;
    }

    /// Step one present frame back (wrapping when `loop_`, else clamping at the
    /// first present frame). Skips gaps — moves to the previous existing frame.
    pub fn step_prev(&mut self) {
        if self.pos == 0 {
            if self.loop_ {
                self.pos = self.indices.len().saturating_sub(1);
            }
        } else {
            self.pos -= 1;
        }
        self.accum = 0.0;
        self.last_tick = None;
    }

    /// Re-fit to a (re)detected present set, keeping the current frame number on
    /// the nearest still-present frame.
    pub fn set_indices(&mut self, indices: Vec<i32>) {
        let want = self.current();
        self.indices = if indices.is_empty() { vec![0] } else { indices };
        self.scrub(want);
    }

    /// Advance the clock by real elapsed time.
    ///
    /// Returns `Some(new_frame)` when the present frame changed (so the caller
    /// can request that frame from the cache), `None` otherwise. A no-op while
    /// paused or when only one frame is present.
    ///
    /// Lifted from playa's `Player::update`: measure `elapsed` since the last
    /// tick, convert to frames via `fps`, and step. We accumulate the fractional
    /// part so playback stays rate-accurate across variable redraw intervals.
    pub fn tick(&mut self) -> Option<i32> {
        if !self.playing || self.indices.len() < 2 {
            return None;
        }

        let now = Instant::now();
        let Some(last) = self.last_tick else {
            // First tick after (re)start: just arm the clock.
            self.last_tick = Some(now);
            return None;
        };

        let elapsed = now.duration_since(last).as_secs_f32();
        self.last_tick = Some(now);

        self.accum += elapsed * self.fps.max(0.001);
        if self.accum < 1.0 {
            return None;
        }

        // Consume whole frames from the accumulator (usually exactly one).
        let steps = self.accum.floor() as i32;
        self.accum -= steps as f32;

        let before = self.pos;
        self.advance(steps);
        if self.pos != before {
            Some(self.current())
        } else {
            None
        }
    }

    /// Move `steps` present frames in the playing direction: wrap at the end with
    /// `loop_`, else stop playing on the last (or, in reverse, the first) frame.
    fn advance(&mut self, steps: i32) {
        let last_pos = self.indices.len().saturating_sub(1);
        let (end, start) = if self.reverse { (0, last_pos) } else { (last_pos, 0) };
        for _ in 0..steps {
            if self.pos == end {
                if self.loop_ {
                    self.pos = start;
                } else {
                    self.playing = false; // reached the end, stop
                    break;
                }
            } else if self.reverse {
                self.pos -= 1;
            } else {
                self.pos += 1;
            }
            self.stepped = self.stepped.saturating_add(1);
        }
    }
}

/// What playback really shows, for the transport bar's fps readout: the frames
/// presented during the last second and how many the clock passed unshown
/// before each. "Playing at 30 fps" is then measured, not assumed: a frame the
/// viewport could not present in time is a skip here, whatever the clock did.
#[derive(Debug, Default)]
pub struct ShowMeter {
    /// `(when presented, frames skipped just before it)`, oldest first, pruned
    /// to [`ShowMeter::WINDOW`].
    shown: VecDeque<(Instant, u32)>,
}

impl ShowMeter {
    /// The measuring window.
    pub const WINDOW: Duration = Duration::from_secs(1);

    /// A frame was presented at `at`, `skipped` frames after the previous one.
    pub fn record(&mut self, at: Instant, skipped: u32) {
        self.shown.push_back((at, skipped));
        self.prune(at);
    }

    /// Forget everything (playback stopped or the sequence changed).
    pub fn clear(&mut self) {
        self.shown.clear();
    }

    /// Frames presented per second in the window ending at `now`: the frame
    /// intervals over their span. `None` until two frames 100 ms apart exist
    /// (a shorter span is too noisy to show).
    pub fn fps(&mut self, now: Instant) -> Option<f32> {
        self.prune(now);
        let (first, last) = (self.shown.front()?.0, self.shown.back()?.0);
        let span = last.duration_since(first).as_secs_f32();
        (span >= 0.1).then(|| (self.shown.len() - 1) as f32 / span)
    }

    /// Frames skipped in the window ending at `now`.
    pub fn skipped(&mut self, now: Instant) -> u32 {
        self.prune(now);
        self.shown.iter().map(|&(_, s)| s).sum()
    }

    fn prune(&mut self, now: Instant) {
        while self.shown.front().is_some_and(|&(t, _)| now.duration_since(t) > Self::WINDOW) {
            self.shown.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A contiguous present set 1..=5 for the basic clock tests.
    fn contiguous() -> Vec<i32> {
        vec![1, 2, 3, 4, 5]
    }

    /// Every clock step is counted, and reading the count resets it.
    #[test]
    fn stepped_counts_clock_moves_and_resets_on_read() {
        let mut pb = Playback::new(contiguous(), 24.0);
        pb.advance(3);
        assert_eq!(pb.take_stepped(), 3);
        assert_eq!(pb.take_stepped(), 0);
        pb.loop_ = false;
        pb.advance(10);
        assert_eq!(pb.take_stepped(), 1, "a stopped clock (end, no loop) is not a step");
    }

    /// 30 frames over one second with no skips read as 30 fps and 0 skipped;
    /// skips are summed; events older than the window drop out.
    #[test]
    fn show_meter_measures_shown_rate_and_skips() {
        let t0 = Instant::now();
        let mut m = ShowMeter::default();
        assert_eq!(m.fps(t0), None);
        for i in 0..=30u32 {
            m.record(t0 + Duration::from_millis(u64::from(i) * 1000 / 30), 0);
        }
        let now = t0 + Duration::from_millis(1000);
        let fps = m.fps(now).unwrap();
        assert!((fps - 30.0).abs() < 0.5, "{fps}");
        assert_eq!(m.skipped(now), 0);
        // Half rate with one skip before each shown frame: 15 fps, 15 skipped.
        let mut half = ShowMeter::default();
        for i in 0..=15u32 {
            half.record(t0 + Duration::from_millis(u64::from(i) * 1000 / 15), u32::from(i > 0));
        }
        let fps = half.fps(now).unwrap();
        assert!((fps - 15.0).abs() < 0.5, "{fps}");
        assert_eq!(half.skipped(now), 15);
        // Two seconds later everything has left the window.
        let later = now + Duration::from_secs(2);
        assert_eq!(half.fps(later), None);
        assert_eq!(half.skipped(later), 0);
    }

    #[test]
    fn paused_does_not_advance() {
        let mut pb = Playback::new(contiguous(), 24.0);
        assert_eq!(pb.tick(), None);
        assert_eq!(pb.current(), 1);
    }

    #[test]
    fn scrub_snaps_to_nearest_present_frame() {
        // Present set with a gap: 1, 2, 10 (3..=9 missing).
        let mut pb = Playback::new(vec![1, 2, 10], 24.0);
        pb.scrub(99);
        assert_eq!(pb.current(), 10, "past the end clamps to last present");
        pb.scrub(-3);
        assert_eq!(pb.current(), 1, "before the start clamps to first present");
        pb.scrub(4);
        assert_eq!(pb.current(), 2, "4 is nearest to present frame 2");
        pb.scrub(8);
        assert_eq!(pb.current(), 10, "8 is nearest to present frame 10");
    }

    #[test]
    fn step_skips_gaps() {
        // Unpadded-style present set with gaps: 0, 1, 6, 10.
        let mut pb = Playback::new(vec![0, 1, 6, 10], 24.0);
        assert_eq!(pb.current(), 0);
        pb.step_next();
        assert_eq!(pb.current(), 1);
        pb.step_next();
        assert_eq!(
            pb.current(),
            6,
            "skips the 2..=5 gap to the next present frame"
        );
        pb.step_next();
        assert_eq!(pb.current(), 10);
        pb.step_next();
        assert_eq!(pb.current(), 0, "wraps to first when looping");
        pb.step_prev();
        assert_eq!(pb.current(), 10, "wraps to last when looping");
    }

    #[test]
    fn step_clamps_when_not_looping() {
        let mut pb = Playback::new(vec![1, 2, 3], 24.0);
        pb.loop_ = false;
        pb.set_pos(2);
        pb.step_next();
        assert_eq!(
            pb.current(),
            3,
            "clamps at last present frame when not looping"
        );
    }

    /// Reverse play walks down the present frames, wraps to the last with
    /// `loop_`, and stops on the first without it; forward is unchanged.
    #[test]
    fn advance_follows_the_direction() {
        let mut pb = Playback::new(vec![1, 2, 6, 10], 24.0);
        pb.play(true);
        assert!(pb.playing && pb.reverse);
        pb.advance(1);
        assert_eq!(pb.current(), 10, "reverse from the first frame wraps to the last");
        pb.advance(2);
        assert_eq!(pb.current(), 2);
        pb.loop_ = false;
        pb.advance(5);
        assert_eq!((pb.current(), pb.playing), (1, false), "stops on the first frame");

        pb.play(false);
        pb.advance(2);
        assert_eq!(pb.current(), 6);
        pb.play(false);
        assert!(!pb.playing, "the same direction again pauses");
        pb.go_last();
        assert_eq!(pb.current(), 10);
        pb.go_first();
        assert_eq!(pb.current(), 1);
    }

    #[test]
    fn set_indices_keeps_current_near() {
        let mut pb = Playback::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 24.0);
        pb.scrub(8);
        assert_eq!(pb.current(), 8);
        // Re-detect to a shorter present set; current snaps to nearest present.
        pb.set_indices(vec![1, 2, 3, 4, 5]);
        assert_eq!(
            pb.current(),
            5,
            "8 snaps onto the nearest still-present frame"
        );
    }
}
