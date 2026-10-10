//! Plumbing shared by the tools that rewrite pixels (`tile`, `multiview`): C-style argument
//! numbers, the "same file" test, and the bridge between a
//! [`FlatImageLevel`] (typed channel planes in memory) and the byte buffers exr-core's readers and
//! writers move pixels through.
//!
//! The C++ tools hand the library `Slice`s that point into their `Image` classes. exr-core's slices
//! are pointer-free layouts over a caller buffer, so pixels travel through exr-util's bounded strip
//! buffers ([`Strip`]): a band of rows is read into one and copied into the channels, or gathered
//! from them and written. Memory stays at one copy of the image plus one strip.

use std::fs;

use clap::CommandFactory;
use exr_core::attr::{Channel, Compression, Name};
use exr_core::io::OStream;
use exr_core::Result;
use exr_util::flat_image_level::FlatImageLevel;
use exr_util::image_channel::ImageChannel;
use exr_util::image_io::{Strip, scanline_unit, strip_rows};

use super::{Cli, CliError, CliResult};

/// `strtol (tok, 0, 0)` as every tool of the C++ set reads an integer: optional blanks and sign, then
/// decimal, `0x` hex or leading-`0` octal. Stricter than `strtol`, which stops at the first bad
/// character and yields 0 for no digits: the whole token must be a number that fits an `int`.
pub fn c_int(tok: &str) -> std::result::Result<i32, String> {
    let bad = || format!("Expected an integer, got \"{tok}\"");
    let t = tok.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r']);
    let (neg, t) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (radix, digits) = match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(h) => (16, h),
        None if t.len() > 1 && t.starts_with('0') => (8, &t[1..]),
        None => (10, t),
    };
    // `from_str_radix` would also take a sign here.
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return Err(bad());
    }
    let n = i64::from_str_radix(digits, radix).map_err(|_| bad())?;
    i32::try_from(if neg { -n } else { n }).map_err(|_| format!("The integer \"{tok}\" is out of range"))
}

/// A hexadecimal float as `strtod` spells it after the `0x`: hex digits, an optional point and more
/// digits, an optional binary exponent `p[+-]digits`. The whole text must be one (`None` otherwise).
fn hex_float(hex: &str) -> Option<f64> {
    let (mantissa, exp) = match hex.find(['p', 'P']) {
        Some(at) => (&hex[..at], Some(&hex[at + 1..])),
        None => (hex, None),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if int.len() + frac.len() == 0 || !int.bytes().chain(frac.bytes()).all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let digit = |c: u8| f64::from(char::from(c).to_digit(16).unwrap_or(0));
    let value = int.bytes().chain(frac.bytes()).fold(0.0f64, |v, c| v * 16.0 + digit(c));
    let exp = match exp {
        Some(e) => e.parse::<i32>().ok().filter(|_| e.bytes().all(|c| c.is_ascii_digit() || c == b'+' || c == b'-'))?,
        None => 0,
    };
    Some(value * 2f64.powi(exp.saturating_sub(4 * frac.len() as i32)))
}

/// `strtod (tok, 0)` as every tool of the C++ set reads a real number: blanks, a sign, then decimal
/// digits with an optional point and exponent, a hexadecimal float (`0x1`, `0x1p3`), `inf`,
/// `infinity` or `nan`. Stricter than `strtod`, which takes the longest numeric prefix and yields 0
/// for none (so `abc` is 0 there): the whole token must be a number.
pub fn c_f64(tok: &str) -> std::result::Result<f64, String> {
    let bad = || format!("Expected a number, got \"{tok}\"");
    let t = tok.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r']);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        return hex_float(hex).map(|v| if neg { -v } else { v }).ok_or_else(bad);
    }
    t.parse::<f64>().map_err(|_| bad())
}

/// `static_cast<float>(strtod (tok, 0))`: [`c_f64`] stored in a `float`, as the tools keep every real
/// option value (exposure, radius, padding, compression level, attribute floats).
pub fn c_f32(tok: &str) -> std::result::Result<f32, String> {
    c_f64(tok).map(|f| f as f32)
}

/// The first `N` tokens of `v` through `parse` (`c_int` / `c_f32`): an option that takes a fixed
/// run of numbers (`chromaticities` 8 floats, `rational` 7 ints). Missing tokens stay `0`; the
/// callers check the count first.
pub fn c_arr<T: Copy + Default, const N: usize>(
    v: &[String],
    parse: fn(&str) -> std::result::Result<T, String>,
) -> std::result::Result<[T; N], String> {
    let mut out = [T::default(); N];
    for (slot, tok) in out.iter_mut().zip(v) {
        *slot = parse(tok)?;
    }
    Ok(out)
}

/// The `n` value tokens after option `i` of a hand-parsed command line, or the error the C++ tool
/// throws for a missing one.
pub fn values<'a>(argv: &'a [String], i: usize, n: usize, missing: &str) -> std::result::Result<&'a [String], String> {
    argv.get(i + 1..i + 1 + n).ok_or_else(|| missing.to_string())
}

/// `getCompression`: a codec by its name (case-insensitive), as `-z` takes it.
pub fn get_compression(name: &str) -> std::result::Result<Compression, String> {
    Compression::from_name(name).ok_or_else(|| format!("Unknown compression method \"{name}\""))
}

/// Whether both names reach one file: equal text, or (both existing) equal canonical paths. The
/// C++ tools compare the strings only, so `x.exr ./x.exr` would overwrite the input.
pub fn same_file(a: &str, b: &str) -> bool {
    // A path that cannot be resolved (the output not created yet) is simply not the same file.
    a == b || matches!((fs::canonicalize(a), fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y)
}

/// Print the long help of the subcommand `sub` (what `-h` does in the C++ tools, which have their
/// own option grammar: clap shows the generated reference as the help of the raw arguments).
pub fn print_help(sub: &str) -> CliResult {
    let mut root = Cli::command();
    root.build();
    let help = root
        .find_subcommand_mut(sub)
        .ok_or_else(|| CliError::Other(format!("the {sub} command is not registered")))?
        .render_long_help();
    out!("{help}");
    Ok(())
}

/// `DummyOStream` of the C++ tools: writes nothing, knows how far it got.
pub struct CountOut(pub u64);

impl OStream for CountOut {
    fn file_name(&self) -> &str {
        "<dummy>"
    }
    fn write(&mut self, buf: &[u8]) -> Result<()> {
        self.0 += buf.len() as u64;
        Ok(())
    }
    fn tellp(&mut self) -> Result<u64> {
        Ok(self.0)
    }
    fn seekp(&mut self, pos: u64) -> Result<()> {
        self.0 = pos;
        Ok(())
    }
}

/// The channels of a level as exr-util's strip layouts want them.
fn chans(level: &FlatImageLevel) -> impl Iterator<Item = (&Name, Channel)> {
    level.iter().map(|(n, c)| (n, c.channel()))
}

/// The strip layout of the window `[x0, x1] x [y0, y1]` of every channel of `level`.
pub fn strip_of(level: &FlatImageLevel, xs: (i32, i32), ys: (i32, i32)) -> Result<Strip> {
    Strip::new(chans(level), xs, ys)
}

/// [`strip_of`] for the channels `names` of `level` only (a file that holds some of the channels).
pub fn strip_of_some<'a>(level: &FlatImageLevel, names: impl Iterator<Item = &'a Name>, xs: (i32, i32), ys: (i32, i32)) -> Result<Strip> {
    let chans = names.map(|n| Ok((n, level.channel(n)?.channel()))).collect::<Result<Vec<_>>>()?;
    Strip::new(chans.into_iter(), xs, ys)
}

/// The alignment of the strips of a scanline file with codec `comp` holding `level`'s channels.
pub fn unit_of(level: &FlatImageLevel, comp: Compression) -> i64 {
    scanline_unit(chans(level), comp)
}

/// Rows per strip for a band `width` pixels wide of `level`, a multiple of `unit` (see
/// [`strip_rows`]).
pub fn rows_of(level: &FlatImageLevel, width: i64, unit: i64) -> i64 {
    strip_rows(chans(level), width, unit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use exr_util::flat_image_io::{pack, unpack};
    use exr_core::attr::{LevelMode, LevelRoundingMode, PixelType};
    use exr_util::flat_image::FlatImage;
    use imath_rs::{Box2i, Half, V2i};

    #[test]
    fn c_int_reads_like_strtol_base_0() {
        assert_eq!(c_int("64"), Ok(64));
        assert_eq!(c_int("-5"), Ok(-5));
        assert_eq!(c_int("  +7"), Ok(7));
        assert_eq!(c_int("0x10"), Ok(16));
        assert_eq!(c_int("010"), Ok(8));
        assert_eq!(c_int("0"), Ok(0));
        assert_eq!(c_int("-2147483648"), Ok(i32::MIN));
        for bad in ["", "abc", "12abc", "0x", "08", "--1", "2147483648", "99999999999999999999"] {
            assert!(c_int(bad).is_err(), "{bad:?}");
        }
        assert_eq!(c_int("x"), Err("Expected an integer, got \"x\"".to_string()));
    }

    #[test]
    fn c_f64_reads_the_whole_token() {
        assert_eq!(c_f64("1.5"), Ok(1.5));
        assert_eq!(c_f64("  -2e2"), Ok(-200.0));
        assert_eq!(c_f64(".5"), Ok(0.5));
        assert_eq!(c_f64("inf"), Ok(f64::INFINITY));
        assert!(c_f64("nan").is_ok_and(f64::is_nan));
        for bad in ["", "x", "1.5x", "1 2", "--1"] {
            assert!(c_f64(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn c_f64_takes_every_strtod_form_and_nothing_else() {
        assert_eq!((c_f64("0x1"), c_f64("0x10"), c_f64("0x1p3"), c_f64("0x.8"), c_f64("-0x2"), c_f64("0x1.8p1"), c_f64("0X1P-2")), (Ok(1.0), Ok(16.0), Ok(8.0), Ok(0.5), Ok(-2.0), Ok(3.0), Ok(0.25)));
        assert_eq!((c_f64("5."), c_f64("+3"), c_f64("1e3"), c_f64("7e-1")), (Ok(5.0), Ok(3.0), Ok(1000.0), Ok(0.7)));
        assert_eq!((c_f64("Infinity"), c_f64("-INF")), (Ok(f64::INFINITY), Ok(f64::NEG_INFINITY)));
        assert_eq!(c_f64("1e999"), Ok(f64::INFINITY));
        for bad in ["abc", "5x", "0x", "0xg", "0x1p", "0x1z", "0x.", "e5", ".", "-", "1e", "2e3.5"] {
            assert!(c_f64(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn compressions_by_name() {
        assert_eq!(get_compression("PIZ"), Ok(Compression::Piz));
        assert_eq!(get_compression("lzw"), Err("Unknown compression method \"lzw\"".to_string()));
    }

    /// A strip packed from one level unpacks into another unchanged, at an offset window.
    #[test]
    fn pack_unpack_round_trips_a_sub_rectangle() {
        let dw = Box2i { min: V2i { x: -2, y: 3 }, max: V2i { x: 5, y: 9 } };
        let mut img = FlatImage::new(&dw, LevelMode::OneLevel, LevelRoundingMode::RoundDown).unwrap();
        img.insert_channel("R", &Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        img.insert_channel("Z", &Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        let src = img.level_mut(0).unwrap();
        for y in 3..=9 {
            for x in -2..=5 {
                *src.typed_channel_mut::<Half>("R").unwrap().at_mut(x, y).unwrap() = Half::from_f32((x * 10 + y) as f32);
                *src.typed_channel_mut::<f32>("Z").unwrap().at_mut(x, y).unwrap() = (x * 100 + y) as f32;
            }
        }
        let src = img.level(0).unwrap();
        let strip = strip_of(src, (-1, 4), (5, 7)).unwrap();
        let mut buf = vec![0u8; strip.bytes];
        pack(src, &strip, &mut buf).unwrap();

        let mut dst = FlatImage::new(&dw, LevelMode::OneLevel, LevelRoundingMode::RoundDown).unwrap();
        dst.insert_channel("R", &Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        dst.insert_channel("Z", &Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        let dlv = dst.level_mut(0).unwrap();
        unpack(dlv, &strip, &buf).unwrap();
        let dlv = dst.level(0).unwrap();
        assert_eq!(dlv.typed_channel::<Half>("R").unwrap().at(4, 7).unwrap().to_f32(), 47.0);
        assert_eq!(*dlv.typed_channel::<f32>("Z").unwrap().at(-1, 5).unwrap(), -95.0);
        assert_eq!(dlv.typed_channel::<Half>("R").unwrap().at(5, 7).unwrap().to_f32(), 0.0, "outside the strip");
        assert!(unpack(dst.level_mut(0).unwrap(), &strip, &buf[..buf.len() - 1]).is_err(), "short buffer");
    }

    #[test]
    fn strips_hold_whole_units() {
        let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 99, y: 99 } };
        let mut img = FlatImage::new(&dw, LevelMode::OneLevel, LevelRoundingMode::RoundDown).unwrap();
        img.insert_channel("R", &Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        let lv = img.level(0).unwrap();
        assert_eq!(rows_of(lv, 100, 16) % 16, 0);
        assert_eq!(rows_of(lv, 1 << 30, 16), 16, "at least one unit");
    }

    #[test]
    fn same_file_sees_through_spelling() {
        let dir = std::env::temp_dir().join(format!("exr-rs-toolio-same-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("a.exr");
        fs::write(&f, b"x").unwrap();
        let f = f.to_string_lossy().into_owned();
        let dotted = dir.join(".").join("a.exr").to_string_lossy().into_owned();
        assert!(same_file(&f, &f));
        assert!(same_file(&f, &dotted));
        assert!(!same_file(&f, &dir.join("b.exr").to_string_lossy()));
        fs::remove_dir_all(&dir).unwrap();
    }
}
