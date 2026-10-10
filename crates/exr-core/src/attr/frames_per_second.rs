//! Port of OpenEXR `ImfFramesPerSecond.h/.cpp` — convenience functions
//! related to the `framesPerSecond` standard attribute.
//!
//! Ten pre-defined exact frame rates plus `guess_exact_fps`, which snaps an
//! inexact rate to the nearest of the four "drop-frame" rates
//! (23.976/29.97/47.952/59.94) when it is close, else returns it unchanged.
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfFramesPerSecond.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfFramesPerSecond.cpp>

use crate::attr::rational::Rational;

/// `inline Rational fps_23_976()` — 23.976023... fps.
pub fn fps_23_976() -> Rational {
    Rational::new(24000, 1001)
}
/// `inline Rational fps_24()` — 24.0 fps (35mm film).
pub fn fps_24() -> Rational {
    Rational::new(24, 1)
}
/// `inline Rational fps_25()` — 25.0 fps (PAL video).
pub fn fps_25() -> Rational {
    Rational::new(25, 1)
}
/// `inline Rational fps_29_97()` — 29.970029... fps (NTSC video).
pub fn fps_29_97() -> Rational {
    Rational::new(30000, 1001)
}
/// `inline Rational fps_30()` — 30.0 fps (60Hz HDTV frames).
pub fn fps_30() -> Rational {
    Rational::new(30, 1)
}
/// `inline Rational fps_47_952()` — 47.952047... fps.
pub fn fps_47_952() -> Rational {
    Rational::new(48000, 1001)
}
/// `inline Rational fps_48()` — 48.0 fps.
pub fn fps_48() -> Rational {
    Rational::new(48, 1)
}
/// `inline Rational fps_50()` — 50.0 fps (PAL video fields).
pub fn fps_50() -> Rational {
    Rational::new(50, 1)
}
/// `inline Rational fps_59_94()` — 59.940059... fps (NTSC video fields).
pub fn fps_59_94() -> Rational {
    Rational::new(60000, 1001)
}
/// `inline Rational fps_60()` — 60.0 fps (60Hz HDTV fields).
pub fn fps_60() -> Rational {
    Rational::new(60, 1)
}

/// `Rational guessExactFps(double fps)` — snap a `double` frame rate to the
/// nearest pre-defined drop-frame rate if it is within `0.002`, else return
/// `Rational(fps)` (the best rational approximation, `Rational::from_f64`).
pub fn guess_exact_fps(fps: f64) -> Rational {
    guess_exact_fps_rational(Rational::from_f64(fps))
}

/// `Rational guessExactFps(const Rational& fps)` — the `Rational` overload:
/// compares `f64` values directly rather than re-deriving a fraction.
pub fn guess_exact_fps_rational(fps: Rational) -> Rational {
    const EPSILON: f64 = 0.002;
    let value = fps.as_f64();

    if (value - fps_23_976().as_f64()).abs() < EPSILON {
        return fps_23_976();
    }
    if (value - fps_29_97().as_f64()).abs() < EPSILON {
        return fps_29_97();
    }
    if (value - fps_47_952().as_f64()).abs() < EPSILON {
        return fps_47_952();
    }
    if (value - fps_59_94().as_f64()).abs() < EPSILON {
        return fps_59_94();
    }
    fps
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predefined_rates() {
        assert_eq!(fps_23_976(), Rational::new(24000, 1001));
        assert_eq!(fps_24(), Rational::new(24, 1));
        assert_eq!(fps_25(), Rational::new(25, 1));
        assert_eq!(fps_29_97(), Rational::new(30000, 1001));
        assert_eq!(fps_30(), Rational::new(30, 1));
        assert_eq!(fps_47_952(), Rational::new(48000, 1001));
        assert_eq!(fps_48(), Rational::new(48, 1));
        assert_eq!(fps_50(), Rational::new(50, 1));
        assert_eq!(fps_59_94(), Rational::new(60000, 1001));
        assert_eq!(fps_60(), Rational::new(60, 1));
    }

    #[test]
    fn guess_exact_fps_snaps_close_rates() {
        assert_eq!(guess_exact_fps(23.976), fps_23_976());
        assert_eq!(guess_exact_fps(29.97), fps_29_97());
        assert_eq!(guess_exact_fps(47.952), fps_47_952());
        assert_eq!(guess_exact_fps(59.94), fps_59_94());
    }

    #[test]
    fn guess_exact_fps_leaves_unrelated_rates_alone() {
        let g = guess_exact_fps(24.0);
        assert_eq!(g.as_f64(), 24.0);
        let g = guess_exact_fps(120.0);
        assert_eq!(g.as_f64(), 120.0);
    }
}
