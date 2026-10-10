//! The `exrstdattr` command line (`openexr/src/bin/exrstdattr/main.cpp:611-791`): one option
//! table drives the parser, the help text and the tests, so they cannot drift apart.
//!
//! The options are parsed here rather than by clap because the tool's grammar is not clap's:
//! the C++ spellings are single-dash words (`-chromaticities`, `-owner`) that clap would split
//! into short flags, and `-part` is order-sensitive (it applies to the attribute options that
//! FOLLOW it). Both the C++ single-dash and the `--long` spelling are accepted.
//!
//! Number formats follow the C++ (`strtod`, `strtol(.., 0)` with hex / octal prefixes,
//! `strtoul(.., 16)` for time codes) with one deliberate change: text that is not entirely a
//! number is an error, where `strtod`/`strtol` silently take the numeric prefix or 0
//! (`main.cpp:343`, `:420`, `:568`, `:592`), and a value that does not fit its attribute field
//! (a negative denominator, a part number beyond `int`) is an error, not a wrapped value.
//!
//! Range: `-whiteLuminance` must be `>= 0`. The C++ usage text promises it (`main.cpp:88-90`) but its
//! parse site stores any float (`main.cpp:646-649`, `getFloat` without a check); owner decision: we
//! reject a negative value like every other range error. It is the only option whose documented range
//! the C++ code skips (the others are checked, `xDensity` `expTime` `aperture` `isoSpeed`
//! `pixelAspectRatio` even more strictly, `> 0`; `focus` and `screenWindowWidth` as documented).
//! Pinned in `white_luminance_negative_is_refused` (parity) and `checks` (unit).
//!
//! NaN: `nan` / `-nan` store the canonical quiet NaN (`0x7fc00000` / `0xffc00000`). The MSVC `strtod`
//! the C++ oracle runs gives `0x7fffffff` / `0xffffffff` (and accepts `nan(n-char-seq)`, which we
//! reject); glibc's `strtod` gives the canonical one, so the payload of a parsed NaN is a C-runtime
//! property, not something the format defines. Pinned in `nan_attribute_values_differ_only_in_payload`.

use exr_core::{
    Attribute, Chromaticities, ChromaticitiesAttribute, Envmap, EnvmapAttribute, FloatAttribute, IntAttribute, KeyCode,
    KeyCodeAttribute, Rational, RationalAttribute, StringAttribute, TimeCode, TimeCodeAttribute, TimeCodePacking,
};
use imath_rs::V2f;

use crate::cli::tool_io::{c_arr, c_f32 as float, c_int as int};

/// A range check applied to a float value before it is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    None,
    /// `>= 0` (`isNonNegative`, `main.cpp:241`).
    NonNegative,
    /// `> 0` (`isPositive`, `main.cpp:253`).
    Positive,
}

/// What an option's values are, which fixes how many tokens it takes and the attribute it makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `-part i|any`: selects the headers the following options apply to.
    Part,
    /// 8 floats: the `chromaticities` attribute.
    Chromaticities,
    /// 1 float, checked.
    Float(Check),
    /// `-focus`: a positive float or `inf` / `infinity`.
    FloatOrInf,
    /// 2 floats.
    V2f,
    /// 1 string.
    Str,
    /// `-capDate`: a string that must be `YYYY:MM:DD hh:mm:ss`.
    Date,
    /// `latlong` / `LATLONG` / `cube` / `CUBE`.
    Envmap,
    /// Two integers: numerator, denominator.
    Rational,
    /// Seven integers.
    KeyCode,
    /// Two base-16 words.
    TimeCode,
    /// `-string name value`.
    NameStr,
    /// `-float name value`.
    NameFloat,
    /// `-int name value`.
    NameInt,
    /// `-erase name`.
    Erase,
}

impl Kind {
    /// How many value tokens follow the option.
    pub fn arity(self) -> usize {
        match self {
            Kind::Chromaticities => 8,
            Kind::KeyCode => 7,
            Kind::V2f | Kind::Rational | Kind::TimeCode | Kind::NameStr | Kind::NameFloat | Kind::NameInt => 2,
            _ => 1,
        }
    }

    /// The error for too few value tokens (`main.cpp`'s `Expected ..` messages; `-erase`'s C++
    /// text, "a name and an integer", is a copy-paste slip, `main.cpp:486`).
    fn expected(self) -> &'static str {
        match self {
            Kind::Part => "Expected a part number (or \"any\")",
            Kind::Chromaticities => "Expected 8 chromaticity values",
            Kind::Float(_) => "Expected a float",
            Kind::FloatOrInf => "Expected a positive float (or \"infinity\")",
            Kind::V2f => "Expected two floats",
            Kind::Str | Kind::Date => "Expected a string",
            Kind::Envmap => "Expected an env map",
            Kind::Rational => "Expected a rational",
            Kind::KeyCode => "Expected a key code",
            Kind::TimeCode => "Expected a time code",
            Kind::NameStr => "Expected a name and string",
            Kind::NameFloat => "Expected a name and a float",
            Kind::NameInt => "Expected a name and an integer",
            Kind::Erase => "Expected an attribute name",
        }
    }
}

/// One option of the tool.
pub struct Opt {
    /// The option without its dash(es); for an attribute option also the attribute's name.
    pub name: &'static str,
    pub kind: Kind,
    /// The value placeholders shown in the help (`"f f"`).
    pub args: &'static str,
    /// The help text (`\n` separates lines).
    pub help: &'static str,
}

const fn opt(name: &'static str, kind: Kind, args: &'static str, help: &'static str) -> Opt {
    Opt { name, kind, args, help }
}

/// Every option, in the order of the C++ usage text (one row each: kept unwrapped as a table).
#[rustfmt::skip]
pub const OPTS: &[Opt] = &[
    opt("part", Kind::Part, "i|any", "the header of part i becomes \"current\" (any or -1: all headers); later\noptions affect only the current headers. All are current before the first -part.\nExample: -focus 3 -part 2 -aperture 8 -part any -owner luke"),
    opt("chromaticities", Kind::Chromaticities, "f f f f f f f f", "CIE xy chromaticities of the red, green and blue primaries and the white point"),
    opt("whiteLuminance", Kind::Float(Check::NonNegative), "f", "white luminance, in candelas per square meter (float, >= 0.0)"),
    opt("adoptedNeutral", Kind::V2f, "f f", "CIE xy coordinates that should be considered \"neutral\" during color rendering"),
    opt("renderingTransform", Kind::Str, "s", "name of the CTL rendering transform for this image (deprecated)"),
    opt("lookModTransform", Kind::Str, "s", "name of the CTL look modification transform for this image (deprecated)"),
    opt("xDensity", Kind::Float(Check::Positive), "f", "horizontal output density, in pixels per inch (float, > 0.0)"),
    opt("owner", Kind::Str, "s", "name of the owner of the image"),
    opt("comments", Kind::Str, "s", "additional information about the image"),
    opt("capDate", Kind::Date, "s", "date the image was created or captured, in local time (YYYY:MM:DD hh:mm:ss)"),
    opt("utcOffset", Kind::Float(Check::None), "f", "offset of local time at capDate from UTC, in seconds (UTC == local time + x)"),
    opt("longitude", Kind::Float(Check::None), "f", "degrees east of Greenwich where the image was recorded"),
    opt("latitude", Kind::Float(Check::None), "f", "degrees north of the equator where the image was recorded"),
    opt("altitude", Kind::Float(Check::None), "f", "meters above sea level where the image was recorded"),
    opt("focus", Kind::FloatOrInf, "f", "the camera's focus distance, in meters (float > 0, or \"infinity\")"),
    opt("expTime", Kind::Float(Check::Positive), "f", "exposure time, in seconds (float > 0)"),
    opt("aperture", Kind::Float(Check::Positive), "f", "lens aperture, in f-stops (float > 0)"),
    opt("isoSpeed", Kind::Float(Check::Positive), "f", "effective speed of the film or image sensor used to record the image (float > 0)"),
    opt("envmap", Kind::Envmap, "s", "the image is an environment map (LATLONG or CUBE)"),
    opt("framesPerSecond", Kind::Rational, "i i", "playback frame rate as a ratio n/d of two integers"),
    opt("keyCode", Kind::KeyCode, "i i i i i i i", "motion picture film frame key code: film manufacturer code (0-99), film type code\n(0-99), film roll prefix (0-999999), count (0-9999), offset of frame in perforations\n(0-119), perforations per frame (1-15), perforations per count (20-120)"),
    opt("timeCode", Kind::TimeCode, "i i", "SMPTE time and control code: two 8-digit base-16 integers (time address and\nflags; user data and control codes)"),
    opt("wrapmodes", Kind::Str, "s", "how a texture map is extrapolated outside 0..1 (e.g. \"clamp\" or \"periodic,clamp\")"),
    opt("pixelAspectRatio", Kind::Float(Check::Positive), "f", "width divided by height of a pixel (float > 0)"),
    opt("screenWindowWidth", Kind::Float(Check::NonNegative), "f", "width of the screen window (float >= 0)"),
    opt("screenWindowCenter", Kind::V2f, "f f", "center of the screen window"),
    opt("string", Kind::NameStr, "s s", "custom string attribute: name and value"),
    opt("float", Kind::NameFloat, "s f", "custom float attribute: name and value"),
    opt("int", Kind::NameInt, "s i", "custom integer attribute: name and value"),
    opt("erase", Kind::Erase, "s", "remove the attribute with the given name"),
];

/// A parsed attribute to set.
pub struct Set {
    pub name: String,
    /// The part it applies to, -1 for all.
    pub part: i32,
    pub attr: Box<dyn Attribute>,
}

/// A parsed attribute to erase.
pub struct Erase {
    pub name: String,
    /// The part it applies to, -1 for all.
    pub part: i32,
}

/// What the command line asks for.
pub enum Cmd {
    Help,
    Version,
    Edit(Job),
}

/// One edit job: copy `input` to `output` with the attribute changes applied.
pub struct Job {
    pub input: String,
    pub output: String,
    pub sets: Vec<Set>,
    pub erases: Vec<Erase>,
}

/// What one option produced.
enum Made {
    Part(i32),
    Set(String, Box<dyn Attribute>),
    Erase(String),
}

/// Parse the tool's arguments (everything after `exr-rs attr`), left to right like the C++: the
/// first error wins, `-h` / `--version` act when reached. Tokens starting with `-` that are not
/// options are errors (the C++ takes them for file names, so a mistyped option turns into the
/// confusing "too many filenames"); `--` ends the options.
pub fn parse(argv: &[String]) -> Result<Cmd, String> {
    let (mut input, mut output): (Option<&str>, Option<&str>) = (None, None);
    let (mut sets, mut erases) = (Vec::new(), Vec::new());
    let mut part = -1;
    let mut only_files = false;
    let mut i = 0;
    while i < argv.len() {
        let tok = argv[i].as_str();
        if !only_files {
            if tok == "--" {
                only_files = true;
                i += 1;
                continue;
            }
            let name = tok.strip_prefix("--").or_else(|| tok.strip_prefix('-')).filter(|n| !n.is_empty());
            if let Some(name) = name {
                match name {
                    "h" | "help" => return Ok(Cmd::Help),
                    "version" => return Ok(Cmd::Version),
                    _ => {}
                }
                let opt = OPTS.iter().find(|o| o.name == name).ok_or_else(|| format!("Unknown option {tok}"))?;
                let values = argv.get(i + 1..i + 1 + opt.kind.arity()).ok_or(opt.kind.expected())?;
                match make(opt, values)? {
                    Made::Part(p) => part = p,
                    Made::Set(name, attr) => sets.push(Set { name, part, attr }),
                    Made::Erase(name) => erases.push(Erase { name, part }),
                }
                i += 1 + opt.kind.arity();
                continue;
            }
        }
        if input.is_none() {
            input = Some(tok);
        } else if output.is_none() {
            output = Some(tok);
        } else {
            return Err("Invalid arguments or too many filenames specified".into());
        }
        i += 1;
    }
    Ok(Cmd::Edit(Job {
        input: input.ok_or("Missing input filename")?.into(),
        // `main.cpp:791` says "Missing input filename" for this case too.
        output: output.ok_or("Missing output filename")?.into(),
        sets,
        erases,
    }))
}

/// Turn one option's value tokens into its effect (the C++ `getFloat` / `getV2f` / ... family).
fn make(opt: &Opt, v: &[String]) -> Result<Made, String> {
    let name = opt.name;
    let set = |a: Box<dyn Attribute>| Ok(Made::Set(name.to_string(), a));
    match opt.kind {
        Kind::Part => Ok(Made::Part(if v[0] == "any" { -1 } else { int(&v[0])? })),
        Kind::Chromaticities => {
            let f = c_arr::<f32, 8>(v, float)?;
            set(Box::new(ChromaticitiesAttribute::new(Chromaticities {
                red: [f[0], f[1]],
                green: [f[2], f[3]],
                blue: [f[4], f[5]],
                white: [f[6], f[7]],
            })))
        }
        Kind::Float(check) => {
            let f = float(&v[0])?;
            match check {
                Check::NonNegative if f < 0.0 => {
                    return Err(format!("The value for the {name} attribute must not be less than zero"));
                }
                Check::Positive if f <= 0.0 => {
                    return Err(format!("The value for the {name} attribute must be greater than zero"));
                }
                _ => {}
            }
            set(Box::new(FloatAttribute::new(f)))
        }
        Kind::FloatOrInf => {
            let f = if v[0] == "inf" || v[0] == "infinity" {
                f32::INFINITY
            } else {
                let f = float(&v[0])?;
                if f <= 0.0 {
                    return Err(format!(
                        "The value for the {name} attribute must be greater than zero, or \"infinity\""
                    ));
                }
                f
            };
            set(Box::new(FloatAttribute::new(f)))
        }
        Kind::V2f => {
            let f = c_arr::<f32, 2>(v, float)?;
            set(Box::new(exr_core::V2fAttribute::new(V2f { x: f[0], y: f[1] })))
        }
        Kind::Str => set(Box::new(StringAttribute::new(v[0].as_str().into()))),
        Kind::Date => {
            check_date(name, &v[0])?;
            set(Box::new(StringAttribute::new(v[0].as_str().into())))
        }
        Kind::Envmap => {
            let map = match v[0].as_str() {
                "latlong" | "LATLONG" => Envmap::LatLong,
                "cube" | "CUBE" => Envmap::Cube,
                _ => return Err(format!("The value for the {name} attribute must be either LATLONG or CUBE")),
            };
            set(Box::new(EnvmapAttribute::new(map)))
        }
        Kind::Rational => {
            // The C++ passes `strtol` results to `Rational(int, unsigned)`: a negative
            // denominator silently wraps to ~4 billion (`main.cpp:420`).
            let n = int(&v[0])?;
            let d =
                u32::try_from(int(&v[1])?).map_err(|_| format!("The denominator of {name} must not be negative"))?;
            set(Box::new(RationalAttribute::new(Rational::new(n, d))))
        }
        Kind::KeyCode => {
            let n = c_arr::<i32, 7>(v, int)?;
            let mut k = KeyCode::default();
            // Setter order as in `getKeyCode` (`main.cpp:568-574`): the first invalid field reports.
            k.set_film_mfc_code(n[0]).map_err(|e| e.to_string())?;
            k.set_film_type(n[1]).map_err(|e| e.to_string())?;
            k.set_prefix(n[2]).map_err(|e| e.to_string())?;
            k.set_count(n[3]).map_err(|e| e.to_string())?;
            k.set_perf_offset(n[4]).map_err(|e| e.to_string())?;
            k.set_perfs_per_frame(n[5]).map_err(|e| e.to_string())?;
            k.set_perfs_per_count(n[6]).map_err(|e| e.to_string())?;
            set(Box::new(KeyCodeAttribute::new(k)))
        }
        Kind::TimeCode => {
            let mut t = TimeCode::default();
            t.set_time_and_flags(hex(&v[0])?, TimeCodePacking::Tv60);
            t.set_user_data(hex(&v[1])?);
            set(Box::new(TimeCodeAttribute::new(t)))
        }
        // The attribute name is the first value, not the option.
        Kind::NameStr => Ok(Made::Set(v[0].clone(), Box::new(StringAttribute::new(v[1].as_str().into())))),
        Kind::NameFloat => Ok(Made::Set(v[0].clone(), Box::new(FloatAttribute::new(float(&v[1])?)))),
        Kind::NameInt => Ok(Made::Set(v[0].clone(), Box::new(IntAttribute::new(int(&v[1])?)))),
        Kind::Erase => Ok(Made::Erase(v[0].clone())),
    }
}

/// `strtoul(tok, 0, 16)` (optional `0x`): a base-16 word that fits 32 bits.
fn hex(tok: &str) -> Result<u32, String> {
    let t = tok.trim();
    let digits = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("Expected a base-16 integer, got \"{tok}\""));
    }
    u32::from_str_radix(digits, 16).map_err(|_| format!("The base-16 value \"{tok}\" does not fit 32 bits"))
}

/// `isDate` (`main.cpp:289-329`): exactly `YYYY:MM:DD hh:mm:ss` with a real calendar date.
fn check_date(name: &str, s: &str) -> Result<(), String> {
    let bad = || format!("The value for the {name} attribute is not a valid date of the form \"YYYY:MM:DD hh:mm:ss\"");
    let b = s.as_bytes();
    if b.len() != 19 || b[4] != b':' || b[7] != b':' || b[10] != b' ' || b[13] != b':' || b[16] != b':' {
        return Err(bad());
    }
    // `strToInt`: -1 when any character is not a digit.
    let num = |from: usize, len: usize| {
        b[from..from + len]
            .iter()
            .try_fold(0i32, |x, &c| c.is_ascii_digit().then(|| x * 10 + i32::from(c - b'0')))
            .unwrap_or(-1)
    };
    let (y, mo, d) = (num(0, 4), num(5, 2), num(8, 2));
    let (h, mi, sec) = (num(11, 2), num(14, 2), num(17, 2));
    if y < 0
        || !(1..=12).contains(&mo)
        || d < 1
        || !(0..=23).contains(&h)
        || !(0..=59).contains(&mi)
        || !(0..=59).contains(&sec)
    {
        return Err(bad());
    }
    let days = match mo {
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if d > days { Err(bad()) } else { Ok(()) }
}

/// The help for the positional arguments of the command: what the tool does, then one entry per
/// option, generated from [`OPTS`].
pub fn help() -> String {
    let mut s = String::from(
        "Read infile, set the values of attributes in its headers and save the result in outfile\n\
         (they must not be the same file: no editing in place). Options take the single-dash\n\
         spelling of OpenEXR's exrstdattr (-owner luke) or the double-dash one (--owner luke).\n\
         The pixels are copied as they are, never decoded.\n",
    );
    for o in OPTS {
        s.push_str(&format!("\n  -{} {}\n", o.name, o.args));
        for line in o.help.lines() {
            s.push_str(&format!("        {line}\n"));
        }
    }
    s.push_str("\n  -h, --help\n        print this message\n      --version\n        print version information\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    fn job(line: &str) -> Job {
        match parse(&args(line)) {
            Ok(Cmd::Edit(j)) => j,
            Ok(_) => panic!("not an edit: {line}"),
            Err(e) => panic!("{line}: {e}"),
        }
    }

    fn err(line: &str) -> String {
        match parse(&args(line)) {
            Err(e) => e,
            Ok(_) => panic!("{line} parsed"),
        }
    }

    /// The table is well formed: unique names, nonempty help, the `args` placeholders agree with
    /// the arity (-part and -erase aside, which have one value).
    #[test]
    fn the_table_is_consistent() {
        for (i, o) in OPTS.iter().enumerate() {
            assert!(OPTS[..i].iter().all(|p| p.name != o.name), "duplicate {}", o.name);
            assert!(!o.help.is_empty(), "{}", o.name);
            assert_eq!(o.args.split(' ').count(), o.kind.arity(), "{}: `{}`", o.name, o.args);
        }
        let help = help();
        for o in OPTS {
            assert!(help.contains(&format!("  -{} {}\n", o.name, o.args)), "help lacks {}", o.name);
        }
    }

    /// Both spellings of every option parse and make the attribute of the option's own name.
    #[test]
    fn every_option_parses_in_both_spellings() {
        let samples: [(&str, &[&str]); 25] = [
            ("chromaticities", &["0.64", "0.33", "0.3", "0.6", "0.15", "0.06", "0.3127", "0.329"]),
            ("whiteLuminance", &["100"]),
            ("adoptedNeutral", &["0.3", "0.3"]),
            ("renderingTransform", &["rt"]),
            ("lookModTransform", &["lm"]),
            ("xDensity", &["72"]),
            ("owner", &["luke"]),
            ("comments", &["two words"]),
            ("capDate", &["2026:09:29 12:30:59"]),
            ("utcOffset", &["-7200"]),
            ("longitude", &["10.5"]),
            ("latitude", &["-3"]),
            ("altitude", &["500"]),
            ("focus", &["infinity"]),
            ("expTime", &["0.01"]),
            ("aperture", &["5.6"]),
            ("isoSpeed", &["400"]),
            ("envmap", &["CUBE"]),
            ("framesPerSecond", &["24000", "1001"]),
            ("keyCode", &["1", "2", "3", "4", "5", "6", "70"]),
            ("timeCode", &["0x1234", "ffff"]),
            ("wrapmodes", &["clamp"]),
            ("pixelAspectRatio", &["1.5"]),
            ("screenWindowWidth", &["2"]),
            ("screenWindowCenter", &["0", "0.5"]),
        ];
        for o in OPTS {
            if matches!(o.kind, Kind::Part | Kind::NameStr | Kind::NameFloat | Kind::NameInt | Kind::Erase) {
                continue;
            }
            let (_, values) =
                samples.iter().find(|(n, _)| *n == o.name).unwrap_or_else(|| panic!("no sample for {}", o.name));
            for dashes in ["-", "--"] {
                let mut argv = vec![format!("{dashes}{}", o.name)];
                argv.extend(values.iter().map(|v| v.to_string()));
                argv.extend(["in".to_string(), "out".to_string()]);
                let Ok(Cmd::Edit(j)) = parse(&argv) else { panic!("{argv:?} did not parse") };
                assert_eq!((j.sets.len(), j.sets[0].name.as_str()), (1, o.name));
                assert_eq!((j.input.as_str(), j.output.as_str()), ("in", "out"));
            }
        }
    }

    /// The values land in the attribute: read them back through `Attribute::as_any`.
    #[test]
    fn values_are_stored() {
        let j = job("-chromaticities 1 2 3 4 5 6 7 8 -keyCode 1 2 3 4 5 6 70 -framesPerSecond 24 1 a b");
        let get = |i: usize| j.sets[i].attr.as_any();
        assert_eq!(
            get(0).downcast_ref::<ChromaticitiesAttribute>().map(|a| a.value),
            Some(Chromaticities { red: [1.0, 2.0], green: [3.0, 4.0], blue: [5.0, 6.0], white: [7.0, 8.0] })
        );
        let k = &get(1).downcast_ref::<KeyCodeAttribute>().unwrap().value;
        assert_eq!((k.prefix(), k.count(), k.perfs_per_count()), (3, 4, 70));
        assert_eq!(get(2).downcast_ref::<RationalAttribute>().map(|a| a.value), Some(Rational::new(24, 1)));
    }

    /// `-part` applies to the options after it; `any` and `-1` select all; the erase list has its own.
    #[test]
    fn part_is_order_sensitive() {
        let j = job("-focus 3 -part 2 -aperture 8 -expTime 0.01 -erase x -part any -owner luke a b");
        let parts: Vec<(&str, i32)> = j.sets.iter().map(|s| (s.name.as_str(), s.part)).collect();
        assert_eq!(parts, [("focus", -1), ("aperture", 2), ("expTime", 2), ("owner", -1)]);
        assert_eq!((j.erases[0].name.as_str(), j.erases[0].part), ("x", 2));
        assert_eq!(job("-part 1 -part -1 -owner o a b").sets[0].part, -1);
    }

    #[test]
    fn custom_attributes_take_a_name_and_a_value() {
        let j = job("-string s v -float f 1.5 -int i 0x10 a b");
        let names: Vec<&str> = j.sets.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["s", "f", "i"]);
        assert_eq!(j.sets[2].attr.as_any().downcast_ref::<IntAttribute>().map(|a| a.value), Some(16));
    }

    /// `strtol(.., 0)` base prefixes; whole-token strictness; range.
    #[test]
    fn integer_formats() {
        for (tok, n) in
            [("12", 12), ("-12", -12), ("+7", 7), ("0x1f", 31), ("0X1F", 31), ("010", 8), ("0", 0), ("  5", 5)]
        {
            assert_eq!(int(tok), Ok(n), "{tok}");
        }
        for tok in ["", "abc", "12abc", "0x", "08", "1.5", "0x-1", "--1", "99999999999"] {
            assert!(int(tok).is_err(), "{tok}");
        }
        assert_eq!(int("2147483647"), Ok(i32::MAX));
        assert!(int("2147483648").is_err());
        assert_eq!(int("-2147483648"), Ok(i32::MIN));
    }

    #[test]
    fn float_formats() {
        for (tok, f) in [("1", 1.0), ("-2.5", -2.5), ("1e3", 1000.0), (".5", 0.5), ("+3", 3.0)] {
            assert_eq!(float(tok), Ok(f), "{tok}");
        }
        assert_eq!(float("inf"), Ok(f32::INFINITY));
        assert!(float("nan").unwrap().is_nan());
        assert!(float("1.5x").is_err() && float("").is_err());
    }

    #[test]
    fn time_code_words_are_base_16() {
        assert_eq!(hex("ff"), Ok(255));
        assert_eq!(hex("0xFF"), Ok(255));
        assert_eq!(hex("ffffffff"), Ok(u32::MAX));
        assert!(hex("100000000").is_err() && hex("").is_err() && hex("-1").is_err() && hex("g").is_err());
    }

    /// `isDate`: format, ranges, month lengths, leap years.
    #[test]
    fn dates() {
        for ok in [
            "2026:09:29 12:30:59",
            "2024:02:29 00:00:00",
            "2000:02:29 23:59:59",
            "1999:12:31 00:00:00",
            "0000:01:01 00:00:00",
        ] {
            assert!(check_date("capDate", ok).is_ok(), "{ok}");
        }
        for bad in [
            "2026:09:29 12:30:5",
            "2026-09-29 12:30:59",
            "2026:13:01 00:00:00",
            "2026:00:01 00:00:00",
            "2026:01:00 00:00:00",
            "2026:02:29 00:00:00",
            "1900:02:29 00:00:00",
            "2026:04:31 00:00:00",
            "2026:01:32 00:00:00",
            "2026:01:01 24:00:00",
            "2026:01:01 00:60:00",
            "2026:01:01 00:00:60",
            "20x6:01:01 00:00:00",
            "",
        ] {
            assert!(check_date("capDate", bad).is_err(), "{bad}");
        }
    }

    /// The range checks name the option and use the C++ wording.
    #[test]
    fn checks() {
        assert!(err("-whiteLuminance -1 a b").contains("whiteLuminance attribute must not be less than zero"));
        assert!(job("-whiteLuminance 0 a b").sets.len() == 1);
        assert!(err("-xDensity 0 a b").contains("xDensity attribute must be greater than zero"));
        assert!(err("-expTime -1 a b").contains("greater than zero"));
        assert!(err("-screenWindowWidth -1 a b").contains("must not be less than zero"));
        assert!(job("-screenWindowWidth 0 a b").sets.len() == 1);
        assert!(err("-focus 0 a b").contains("or \"infinity\""));
        assert!(job("-focus inf a b").sets.len() == 1);
        assert!(err("-envmap Cube a b").contains("LATLONG or CUBE"));
        assert!(err("-keyCode 100 0 0 0 0 4 64 a b").contains("manufacturer"));
        assert!(err("-framesPerSecond 24 -1 a b").contains("denominator"));
        assert!(err("-capDate 2026 a b").contains("YYYY:MM:DD"));
        // The C++ takes the numeric prefix (`-float x 1.5x` -> 1.5) or 0; we refuse.
        assert!(err("-float x abc a b").contains("Expected a number"));
    }

    /// Too few values, file-name problems, unknown options, `--`.
    #[test]
    fn grammar_errors() {
        assert!(err("-chromaticities 1 2 3 a b").contains("8 chromaticity"));
        assert_eq!(err("a -owner"), "Expected a string");
        assert_eq!(err("a -int x"), "Expected a name and an integer");
        assert_eq!(err("a b c"), "Invalid arguments or too many filenames specified");
        assert_eq!(err("a"), "Missing output filename");
        assert_eq!(err("-owner x"), "Missing input filename");
        assert_eq!(err("-ownr x a b"), "Unknown option -ownr");
        // A value may look like an option; `--` makes the rest file names.
        assert_eq!(job("-owner -h a b").sets.len(), 1);
        let j = job("-- -a -b");
        assert_eq!((j.input.as_str(), j.output.as_str()), ("-a", "-b"));
        // Errors surface in argument order: the bad option comes before the file-name problem.
        assert!(err("-xDensity -1 a b c").contains("xDensity"));
    }

    #[test]
    fn help_and_version_act_when_reached() {
        assert!(matches!(parse(&args("-h")), Ok(Cmd::Help)));
        assert!(matches!(parse(&args("--help")), Ok(Cmd::Help)));
        assert!(matches!(parse(&args("a --version")), Ok(Cmd::Version)));
        // `-h` after a failing option is never reached.
        assert!(parse(&args("-xDensity 0 -h")).is_err());
        // ... and as a value it is just a value.
        assert!(matches!(parse(&args("-owner -h a b")), Ok(Cmd::Edit(_))));
    }
}
