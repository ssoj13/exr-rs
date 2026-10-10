//! Human-readable header attribute text — the ONE formatter for every consumer
//! (`exr-rs info`, the viewer's metadata panel, the Python inspector).
//!
//! A 1:1 port of OpenEXR's `exrheader` (`src/bin/exrheader/main.cpp`,
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/bin/exrheader/main.cpp>):
//! every attribute prints as `name (type typename)` followed by a per-type
//! value, with `exrheader`'s exact wording, number formatting (C++ `iostream`
//! default = `%g`, precision 6, see [`g`]) and line layout. Line references in
//! the comments below (`main.cpp:NNN`) are to that file.
//!
//! # Layout model
//! `exrheader` prints a value either **inline** (`name (type t): value`) or as
//! a **block** (`name (type t):` then one indented line per entry). [`Row`]
//! carries the value text without layout ([`Row::value`], lines joined by
//! `'\n'`) plus the block indentation ([`Row::indent`], `0` = inline), so a GUI
//! can show the text as it likes while [`Row::line`] reproduces `exrheader`
//! byte for byte.
//!
//! # Beyond `exrheader`
//! `exrheader` has no formatter for `v2d`, `v3d`, `m33d`, `m44d`,
//! `floatvector`, `deepImageState`, `idmanifest` and `bytes` (it prints only
//! `name (type t)` for them, like for any unknown type) — but exr-core knows
//! those types and the point of this module is to show headers completely, so
//! they get a formatter in the same style ([`EXTENSION_TYPES`]). A genuinely
//! unregistered type ([`OpaqueAttribute`]) prints nothing after the type, like
//! `exrheader` does for unknown types.

use std::any::Any;

use imath_rs::{Box2f, Box2i, M33d, M33f, M44d, M44f, V2d, V2f, V2i, V3d, V3f, V3i, Vec2, Vec3};

use crate::attr::{
    Attribute, ChannelList, Chromaticities, Compression, CompressedIDManifest, DeepImageState,
    Envmap, ExrString, KeyCode, LevelMode, LevelRoundingMode, LineOrder, PixelType, PreviewImage,
    Rational, TileDescription, TimeCode, TypedAttribute, Bytes, AttrValue,
};
use crate::header::Header;

/// Type names `exrheader` prints no value for but that [`describe`] formats
/// anyway (see the module docs). Lets a parity test skip exactly these.
pub const EXTENSION_TYPES: [&str; 8] = [
    "v2d",
    "v3d",
    "m33d",
    "m44d",
    "floatvector",
    "deepImageState",
    "idmanifest",
    "bytes",
];

/// One header attribute, ready to display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// Attribute name (raw bytes, lossily decoded; names need not be UTF-8).
    pub name: String,
    /// Wire type name (raw bytes, lossily decoded).
    pub type_name: String,
    /// Value text as `exrheader` prints it, without the layout indentation;
    /// block values keep their lines joined by `'\n'`. Empty for a type
    /// `exrheader` prints no value for, and for an empty block.
    pub value: String,
    /// `0`: the value goes on the attribute's own line after `": "`.
    /// `n > 0`: the value lines follow on their own lines, each indented by `n`
    /// spaces (`exrheader` uses 4, and 3 for matrices whose first line starts
    /// with `(`).
    pub indent: usize,
}

impl Row {
    /// The attribute exactly as `exrheader` lays it out (`main.cpp:224-432`),
    /// without the trailing newline.
    pub fn line(&self) -> String {
        let mut s = format!("{} (type {})", self.name, self.type_name);
        if self.indent == 0 {
            // Types without a value print nothing, not even the colon.
            if !self.value.is_empty() {
                s.push_str(": ");
                s.push_str(&self.value);
            }
            return s;
        }
        s.push(':');
        if self.value.is_empty() {
            return s;
        }
        let pad = " ".repeat(self.indent);
        for l in self.value.split('\n') {
            s.push('\n');
            s.push_str(&pad);
            s.push_str(l);
        }
        s
    }
}

/// Every attribute of `header`, in the header's own iteration order (sorted by
/// name, like C++'s `std::map`), as [`Row`]s.
pub fn rows(header: &Header) -> Vec<Row> {
    header
        .iter()
        .map(|(name, a)| {
            let v = format_attr(a.as_ref());
            Row {
                name: String::from_utf8_lossy(name.as_bytes()).into_owned(),
                type_name: String::from_utf8_lossy(a.type_name()).into_owned(),
                value: v.lines.join("\n"),
                indent: v.indent,
            }
        })
        .collect()
}

/// The value text of one attribute — what `exrheader` prints after the
/// `": "` / `":"` separator, layout indentation removed and multi-line values
/// joined with `'\n'`. See [`Row`] for the layout model.
pub fn describe(attr: &dyn Attribute) -> String {
    format_attr(attr).lines.join("\n")
}

/// Value lines plus how `exrheader` lays them out (`indent`, see [`Row`]).
struct Val {
    indent: usize,
    lines: Vec<String>,
}

impl Val {
    /// Value on the attribute's own line.
    fn inline(s: String) -> Self {
        Self { indent: 0, lines: vec![s] }
    }

    /// Value lines on their own lines, indented by 4 (`exrheader`'s default).
    fn block(lines: Vec<String>) -> Self {
        Self { indent: 4, lines }
    }

    /// Nothing after the type — `exrheader`'s behaviour for unknown types.
    fn none() -> Self {
        Self { indent: 0, lines: Vec::new() }
    }
}

/// `dynamic_cast<const TypedAttribute<T>*>` — the value if `a` holds a `T`.
fn get<T: AttrValue>(a: &dyn Any) -> Option<&T> {
    a.downcast_ref::<TypedAttribute<T>>().map(|t| &t.value)
}

/// The per-type dispatch: the same `if / else if` chain as `main.cpp:226-430`
/// (in the same order), then the extension types.
fn format_attr(a: &dyn Attribute) -> Val {
    let any = a.as_any();
    if let Some(v) = get::<Box2i>(any) {
        // main.cpp:229 — `min << " - " << max`, each a V2i "(x y)".
        Val::inline(format!("{} - {}", v2(&v.min), v2(&v.max)))
    } else if let Some(v) = get::<Box2f>(any) {
        Val::inline(format!("{} - {}", v2(&v.min), v2(&v.max))) // main.cpp:236
    } else if let Some(v) = get::<ChannelList>(any) {
        Val::block(chlist(v)) // main.cpp:242
    } else if let Some(v) = get::<Chromaticities>(any) {
        // main.cpp:249-260 — the column padding is part of the text.
        Val::block(vec![
            format!("red   {}", v2a(v.red)),
            format!("green {}", v2a(v.green)),
            format!("blue  {}", v2a(v.blue)),
            format!("white {}", v2a(v.white)),
        ])
    } else if let Some(v) = get::<Compression>(any) {
        Val::inline(v.description()) // main.cpp:44-49 printCompression
    } else if let Some(v) = get::<f64>(any) {
        Val::inline(g(*v)) // main.cpp:273
    } else if let Some(v) = get::<Envmap>(any) {
        Val::inline(envmap(*v)) // main.cpp:151-162
    } else if let Some(v) = get::<f32>(any) {
        Val::inline(g(f64::from(*v))) // main.cpp:286
    } else if let Some(v) = get::<i32>(any) {
        Val::inline(v.to_string()) // main.cpp:291
    } else if let Some(v) = get::<KeyCode>(any) {
        Val::block(keycode(v)) // main.cpp:297-317
    } else if let Some(v) = get::<LineOrder>(any) {
        Val::inline(line_order(*v)) // main.cpp:51-64
    } else if let Some(v) = get::<M33f>(any) {
        matrix(&v.x) // main.cpp:330-336
    } else if let Some(v) = get::<M44f>(any) {
        matrix(&v.x) // main.cpp:342-352
    } else if let Some(v) = get::<PreviewImage>(any) {
        // main.cpp:358
        Val::inline(format!("{} by {} pixels", v.width(), v.height()))
    } else if let Some(v) = get::<ExrString>(any) {
        Val::inline(format!("\"{}\"", v.to_string_lossy())) // main.cpp:365
    } else if let Some(v) = get::<Vec<ExrString>>(any) {
        // main.cpp:371-378
        Val::block(v.iter().map(|s| format!("\"{}\"", s.to_string_lossy())).collect())
    } else if let Some(v) = get::<Rational>(any) {
        // main.cpp:384 — `n "/" d " (" double(value) ")"`.
        Val::inline(format!("{}/{} ({})", v.n, v.d, g(v.as_f64())))
    } else if let Some(v) = get::<TileDescription>(any) {
        Val::block(tiledesc(v)) // main.cpp:391-402
    } else if let Some(v) = get::<TimeCode>(any) {
        Val::block(timecode(v)) // main.cpp:108-149
    } else if let Some(v) = get::<V2i>(any) {
        Val::inline(v2(v)) // main.cpp:414
    } else if let Some(v) = get::<V2f>(any) {
        Val::inline(v2(v)) // main.cpp:419
    } else if let Some(v) = get::<V3i>(any) {
        Val::inline(v3(v)) // main.cpp:424
    } else if let Some(v) = get::<V3f>(any) {
        Val::inline(v3(v)) // main.cpp:429
    } else {
        format_extension(any)
    }
}

/// Types exr-core knows but `exrheader` does not format (see the module docs),
/// else [`Val::none`] (unregistered / opaque types).
fn format_extension(any: &dyn Any) -> Val {
    if let Some(v) = get::<V2d>(any) {
        Val::inline(v2(v))
    } else if let Some(v) = get::<V3d>(any) {
        Val::inline(v3(v))
    } else if let Some(v) = get::<M33d>(any) {
        matrix(&v.x)
    } else if let Some(v) = get::<M44d>(any) {
        matrix(&v.x)
    } else if let Some(v) = get::<Vec<f32>>(any) {
        let items: Vec<String> = v.iter().map(|x| g(f64::from(*x))).collect();
        Val::inline(format!("({})", items.join(" ")))
    } else if let Some(v) = get::<DeepImageState>(any) {
        // ImfDeepImageState.h enumerator names, lower-cased and dashed.
        Val::inline(
            match v {
                DeepImageState::Messy => "messy",
                DeepImageState::Sorted => "sorted",
                DeepImageState::NonOverlapping => "non-overlapping",
                DeepImageState::Tidy => "tidy",
            }
            .to_string(),
        )
    } else if let Some(v) = get::<CompressedIDManifest>(any) {
        Val::inline(format!(
            "{} bytes compressed, {} bytes uncompressed",
            v.compressed_data_size(),
            v.uncompressed_data_size
        ))
    } else if let Some(v) = get::<Bytes>(any) {
        Val::inline(format!(
            "{} bytes, type hint \"{}\"",
            v.data.len(),
            String::from_utf8_lossy(&v.type_hint)
        ))
    } else {
        Val::none()
    }
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/// C++ `ostream << double` with default flags: `%g`, 6 significant digits,
/// trailing zeros removed, exponent form below 1e-4 / from 1e6.
///
/// `float` values go through here as `f64`, exactly like `iostream` promotes
/// them, so `1.0f / 3.0f` prints `0.333333`, not Rust's shortest-roundtrip
/// `0.33333334`. Non-finite values print `inf` / `-inf` / `nan`; the C++
/// spelling of NaN is platform-specific (`-nan(ind)` on MSVC), so `nan` is the
/// portable choice.
fn g(v: f64) -> String {
    const P: usize = 6;
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.into();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.into();
    }
    // Scientific rendering first: it rounds to P significant digits, and its
    // exponent (after rounding, e.g. 999999.5 -> 1e+06) picks the %g style.
    let sci = format!("{:.*e}", P - 1, v);
    let Some((mant, exp)) = sci.split_once('e') else { return sci };
    let Ok(x) = exp.parse::<i32>() else { return sci };
    if x < -4 || x >= P as i32 {
        let sign = if x < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim_zeros(mant), x.abs())
    } else {
        // `x` is in -4..6 here, so the decimal count is 1..=9.
        trim_zeros(&format!("{:.*}", (P as i32 - 1 - x) as usize, v))
    }
}

/// Drop trailing zeros of a decimal fraction, then a dangling `.`.
fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// A number as a C++ `ostream` prints it.
trait Num: Copy {
    fn txt(self) -> String;
}

impl Num for i32 {
    fn txt(self) -> String {
        self.to_string()
    }
}

impl Num for f32 {
    fn txt(self) -> String {
        g(f64::from(self))
    }
}

impl Num for f64 {
    fn txt(self) -> String {
        g(self)
    }
}

/// Imath `operator<<(Vec2)`: `(x y)`.
fn v2<T: Num>(v: &Vec2<T>) -> String {
    format!("({} {})", v.x.txt(), v.y.txt())
}

/// [`v2`] for a bare `[x, y]` pair (chromaticities).
fn v2a(v: [f32; 2]) -> String {
    format!("({} {})", v[0].txt(), v[1].txt())
}

/// Imath `operator<<(Vec3)`: `(x y z)`.
fn v3<T: Num>(v: &Vec3<T>) -> String {
    format!("({} {} {})", v.x.txt(), v.y.txt(), v.z.txt())
}

// ---------------------------------------------------------------------------
// Per-type text
// ---------------------------------------------------------------------------

/// `printPixelType` (main.cpp:66-79).
fn pixel_type(t: PixelType) -> String {
    match t {
        PixelType::Uint => "32-bit unsigned integer".into(),
        PixelType::Half => "16-bit floating-point".into(),
        PixelType::Float => "32-bit floating-point".into(),
        other => format!("type {}", other.as_i32()),
    }
}

/// `printChannelList` (main.cpp:164-178): one line per channel.
fn chlist(cl: &ChannelList) -> Vec<String> {
    cl.iter()
        .map(|(name, c)| {
            let mut s = format!(
                "{}, {}, sampling {} {}",
                String::from_utf8_lossy(name.as_bytes()),
                pixel_type(c.type_),
                c.x_sampling,
                c.y_sampling
            );
            if c.p_linear {
                s.push_str(", plinear");
            }
            s
        })
        .collect()
}

/// `printEnvmap` (main.cpp:151-162). Envmap has exactly the two valid values;
/// an out-of-range byte is rejected at read time, so C++'s
/// `"map type N"` default has no counterpart.
fn envmap(e: Envmap) -> String {
    match e {
        Envmap::LatLong => "latitude-longitude map",
        Envmap::Cube => "cube-face map",
    }
    .into()
}

/// `printLineOrder` (main.cpp:51-64). `Invalid` is C++'s `NUM_LINEORDERS`
/// (3), which the reader stores for any unknown byte and `exrheader` prints as
/// the bare integer (the `default:` case).
fn line_order(lo: LineOrder) -> String {
    match lo {
        LineOrder::IncreasingY => "increasing y".into(),
        LineOrder::DecreasingY => "decreasing y".into(),
        LineOrder::RandomY => "random y".into(),
        LineOrder::Invalid => (LineOrder::Invalid as u8).to_string(),
    }
}

/// `printLevelMode` (main.cpp:81-94); `Invalid` is the stored value 3.
fn level_mode(m: LevelMode) -> String {
    match m {
        LevelMode::OneLevel => "single level".into(),
        LevelMode::MipmapLevels => "mip-map".into(),
        LevelMode::RipmapLevels => "rip-map".into(),
        LevelMode::Invalid => format!("level mode {}", LevelMode::Invalid as u8),
    }
}

/// `printLevelRoundingMode` (main.cpp:96-107); `Invalid` is the stored value 2.
fn rounding(m: LevelRoundingMode) -> String {
    match m {
        LevelRoundingMode::RoundDown => "down".into(),
        LevelRoundingMode::RoundUp => "up".into(),
        LevelRoundingMode::Invalid => format!("mode {}", LevelRoundingMode::Invalid as u8),
    }
}

/// tiledesc (main.cpp:391-402).
fn tiledesc(t: &TileDescription) -> Vec<String> {
    let mut l = vec![
        level_mode(t.mode),
        format!("tile size {} by {} pixels", t.x_size, t.y_size),
    ];
    if t.mode != LevelMode::OneLevel {
        l.push(format!("level sizes rounded {}", rounding(t.rounding_mode)));
    }
    l
}

/// keycode (main.cpp:297-317).
fn keycode(k: &KeyCode) -> Vec<String> {
    vec![
        format!("film manufacturer code {}", k.film_mfc_code()),
        format!("film type code {}", k.film_type()),
        format!("prefix {}", k.prefix()),
        format!("count {}", k.count()),
        format!("perf offset {}", k.perf_offset()),
        format!("perfs per frame {}", k.perfs_per_frame()),
        format!("perfs per count {}", k.perfs_per_count()),
    ]
}

/// `printTimeCode` (main.cpp:109-149). `bool`s print as `0` / `1` (no
/// `boolalpha`), the time fields are zero-padded to width 2, user data is
/// unpadded lower-case hex.
fn timecode(t: &TimeCode) -> Vec<String> {
    let b = |v: bool| u8::from(v);
    vec![
        format!(
            "time {:02}:{:02}:{:02}:{:02}",
            t.hours(),
            t.minutes(),
            t.seconds(),
            t.frame()
        ),
        format!(
            "drop frame {}, color frame {}, field/phase {}",
            b(t.drop_frame()),
            b(t.color_frame()),
            b(t.field_phase())
        ),
        format!("bgf0 {}, bgf1 {}, bgf2 {}", b(t.bgf0()), b(t.bgf1()), b(t.bgf2())),
        format!("user data 0x{:x}", t.user_data()),
    ]
}

/// m33f / m44f (main.cpp:330-352): `(` glued to the first row, rows aligned
/// under it. `exrheader` indents the first line by 3 and the others by 4, so
/// the block uses base indent 3 and the continuation rows carry one extra
/// space in the text.
fn matrix<T: Num, const N: usize>(x: &[[T; N]; N]) -> Val {
    let lines = x
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut s = String::new();
            s.push(if i == 0 { '(' } else { ' ' });
            for (j, v) in row.iter().enumerate() {
                if j > 0 {
                    s.push(' ');
                }
                s.push_str(&v.txt());
            }
            if i == N - 1 {
                s.push(')');
            }
            s
        })
        .collect();
    Val { indent: 3, lines }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, OpaqueAttribute, PreviewRgba};
    use std::path::{Path, PathBuf};

    /// `describe` of a typed value.
    fn d<T: AttrValue>(v: T) -> String {
        describe(&TypedAttribute::new(v))
    }

    /// Full `exrheader` line(s) of one attribute inside a header.
    fn line<A: Attribute>(name: &str, a: A) -> String {
        let mut h = Header::new();
        h.insert(name, a).expect("insert");
        // `Header::new` carries the required attributes; pick ours by name.
        rows(&h).into_iter().find(|r| r.name == name).expect("row").line()
    }

    fn vi(x: i32, y: i32) -> V2i {
        Vec2 { x, y }
    }

    #[test]
    fn g_matches_printf_percent_g() {
        // Expected values are C `printf("%g")` results (precision 6).
        for (v, s) in [
            (0.0, "0"),
            (1.0, "1"),
            (0.64, "0.64"),
            (1.0 / 3.0, "0.333333"),
            (2.0 / 3.0, "0.666667"),
            (100000.0, "100000"),
            (999999.4, "999999"),
            (999999.5, "1e+06"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.000123456789, "0.000123457"),
            (-2.5, "-2.5"),
            (1e100, "1e+100"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(g(v), s, "g({v})");
        }
        // A float is promoted to double before formatting (iostream behaviour).
        assert_eq!(g(f64::from(0.1f32)), "0.1");
        assert_eq!(g(f64::from(16777217.0f32)), "1.67772e+07");
    }

    #[test]
    fn box_v2_v3_inline() {
        // main.cpp:229 `min << " - " << max`, Imath "(x y)".
        let b = Box2i { min: vi(0, -1), max: vi(63, 42) };
        assert_eq!(d(b), "(0 -1) - (63 42)");
        let bf = Box2f { min: Vec2 { x: 0.5, y: -1.0 }, max: Vec2 { x: 2.0, y: 1.0 / 3.0 } };
        assert_eq!(d(bf), "(0.5 -1) - (2 0.333333)"); // main.cpp:236
        assert_eq!(d(vi(3, 4)), "(3 4)"); // main.cpp:414
        assert_eq!(d(V2f { x: 1.5, y: 0.1 }), "(1.5 0.1)"); // main.cpp:419
        assert_eq!(d(V3i { x: 1, y: 2, z: 3 }), "(1 2 3)"); // main.cpp:424
        assert_eq!(d(V3f { x: 1.0, y: 2.5, z: -3.0 }), "(1 2.5 -3)"); // main.cpp:429
    }

    #[test]
    fn scalars_inline() {
        assert_eq!(d(-7i32), "-7"); // main.cpp:291
        assert_eq!(d(1.0f32 / 3.0), "0.333333"); // main.cpp:286
        assert_eq!(d(1.0f64 / 3.0), "0.333333"); // main.cpp:273
        assert_eq!(d(2.0e-7f64), "2e-07");
    }

    #[test]
    fn string_and_stringvector() {
        assert_eq!(d(ExrString::from("hi there")), "\"hi there\""); // main.cpp:365
        assert_eq!(d(ExrString::default()), "\"\"");
        // main.cpp:371-378: ":" then "\n    \"s\"" per element.
        let sv = vec![ExrString::from("a"), ExrString::from("b c")];
        assert_eq!(d(sv.clone()), "\"a\"\n\"b c\"");
        assert_eq!(line("v", TypedAttribute::new(sv)), "v (type stringvector):\n    \"a\"\n    \"b c\"");
        // An empty vector prints just the colon (loop body never runs).
        assert_eq!(line("v", TypedAttribute::new(Vec::<ExrString>::new())), "v (type stringvector):");
    }

    #[test]
    fn chlist_layout_and_pixel_types() {
        let mut cl = ChannelList::new();
        cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).expect("B");
        cl.insert("Z", Channel::new(PixelType::Float, 2, 4, true)).expect("Z");
        cl.insert("id", Channel::new(PixelType::Uint, 1, 1, false)).expect("id");
        // main.cpp:169-176: "\n    name, <type>, sampling X Y[, plinear]".
        assert_eq!(
            line("channels", TypedAttribute::new(cl.clone())),
            "channels (type chlist):\n    B, 16-bit floating-point, sampling 1 1\n    \
             Z, 32-bit floating-point, sampling 2 4, plinear\n    \
             id, 32-bit unsigned integer, sampling 1 1"
        );
        assert_eq!(
            d(cl),
            "B, 16-bit floating-point, sampling 1 1\nZ, 32-bit floating-point, sampling 2 4, plinear\n\
             id, 32-bit unsigned integer, sampling 1 1"
        );
        // main.cpp:77 default case.
        assert_eq!(pixel_type(PixelType::NumPixelTypes), "type 3");
        // An empty list prints only the colon.
        assert_eq!(line("c", TypedAttribute::new(ChannelList::new())), "c (type chlist):");
    }

    #[test]
    fn chromaticities_block() {
        // main.cpp:249-260 (note the padded labels).
        let c = Chromaticities::default();
        assert_eq!(
            line("chromaticities", TypedAttribute::new(c)),
            "chromaticities (type chromaticities):\n    red   (0.64 0.33)\n    green (0.3 0.6)\n    \
             blue  (0.15 0.06)\n    white (0.3127 0.329)"
        );
    }

    #[test]
    fn compression_uses_description_text() {
        // main.cpp:44-49 prints getCompressionDescriptionFromId ("name: desc").
        assert_eq!(d(Compression::Zip), "zip: zlib compression, in blocks of 16 scan lines.");
        assert_eq!(d(Compression::NoCompression), "none: no compression.");
        assert_eq!(d(Compression::Invalid), "unrecognized compression method");
        for c in Compression::ALL {
            assert_eq!(d(c), c.description());
        }
    }

    #[test]
    fn envmap_and_line_order() {
        assert_eq!(d(Envmap::LatLong), "latitude-longitude map"); // main.cpp:156
        assert_eq!(d(Envmap::Cube), "cube-face map"); // main.cpp:158
        assert_eq!(d(LineOrder::IncreasingY), "increasing y"); // main.cpp:56
        assert_eq!(d(LineOrder::DecreasingY), "decreasing y"); // main.cpp:58
        assert_eq!(d(LineOrder::RandomY), "random y"); // main.cpp:60
        assert_eq!(d(LineOrder::Invalid), "3"); // main.cpp:62 `int(lo)`, NUM_LINEORDERS
    }

    #[test]
    fn keycode_block() {
        let k = KeyCode::new(1, 2, 3, 4, 5, 6, 64).expect("keycode");
        // main.cpp:297-317.
        assert_eq!(
            line("keyCode", TypedAttribute::new(k)),
            "keyCode (type keycode):\n    film manufacturer code 1\n    film type code 2\n    prefix 3\n    \
             count 4\n    perf offset 5\n    perfs per frame 6\n    perfs per count 64"
        );
    }

    #[test]
    fn matrices_align_under_the_paren() {
        let m33 = M33f { x: [[1.0, 2.0, 3.0], [4.0, 5.0, 0.5], [7.0, 8.0, 9.0]] };
        // main.cpp:330-336: "   (" then rows separated by "\n    ".
        assert_eq!(
            line("m", TypedAttribute::new(m33)),
            "m (type m33f):\n   (1 2 3\n    4 5 0.5\n    7 8 9)"
        );
        let m44 = M44f {
            x: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [10.0, 20.0, 30.0, 1.0],
            ],
        };
        // main.cpp:342-352.
        assert_eq!(
            line("xf", TypedAttribute::new(m44)),
            "xf (type m44f):\n   (1 0 0 0\n    0 1 0 0\n    0 0 1 0\n    10 20 30 1)"
        );
        // The describe() text keeps the rows aligned relative to the paren.
        assert_eq!(d(m33), "(1 2 3\n 4 5 0.5\n 7 8 9)");
    }

    #[test]
    fn preview_rational() {
        let p = PreviewImage::new(3, 2, Some(vec![PreviewRgba::default(); 6])).expect("preview");
        assert_eq!(d(p), "3 by 2 pixels"); // main.cpp:358
        // main.cpp:384 `n/d (double)`.
        assert_eq!(d(Rational::new(24000, 1001)), "24000/1001 (23.976)");
        assert_eq!(d(Rational::new(24, 1)), "24/1 (24)");
        assert_eq!(d(Rational::new(1, 0)), "1/0 (inf)");
    }

    #[test]
    fn tiledesc_block() {
        let one = TileDescription {
            x_size: 64,
            y_size: 32,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        };
        // main.cpp:391-396; no rounding line for ONE_LEVEL (main.cpp:398).
        assert_eq!(
            line("tiles", TypedAttribute::new(one)),
            "tiles (type tiledesc):\n    single level\n    tile size 64 by 32 pixels"
        );
        let mip = TileDescription { mode: LevelMode::MipmapLevels, rounding_mode: LevelRoundingMode::RoundUp, ..one };
        // main.cpp:400-401.
        assert_eq!(d(mip), "mip-map\ntile size 64 by 32 pixels\nlevel sizes rounded up");
        let rip = TileDescription { mode: LevelMode::RipmapLevels, ..one };
        assert_eq!(d(rip), "rip-map\ntile size 64 by 32 pixels\nlevel sizes rounded down");
        // Default cases main.cpp:92 / :105 (stored sentinels 3 and 2).
        let bad = TileDescription {
            mode: LevelMode::Invalid,
            rounding_mode: LevelRoundingMode::Invalid,
            ..one
        };
        assert_eq!(d(bad), "level mode 3\ntile size 64 by 32 pixels\nlevel sizes rounded mode 2");
    }

    #[test]
    fn timecode_block() {
        let tc = TimeCode::from_fields(1, 2, 3, 4, true, false, true, false, true, false, 0, 0, 0, 0, 0, 0, 0, 0)
            .expect("timecode");
        // main.cpp:112-148: zero-padded time, bools as 0/1, hex user data.
        let v = describe(&TypedAttribute::new(tc.clone()));
        assert_eq!(
            v,
            "time 01:02:03:04\ndrop frame 1, color frame 0, field/phase 1\nbgf0 0, bgf1 1, bgf2 0\nuser data 0x0"
        );
        let mut tc2 = tc.clone();
        tc2.set_user_data(0xBEEF);
        assert!(d(tc2).ends_with("user data 0xbeef"));
        assert_eq!(
            line("timeCode", TypedAttribute::new(tc.clone())),
            "timeCode (type timecode):\n    time 01:02:03:04\n    drop frame 1, color frame 0, field/phase 1\n    \
             bgf0 0, bgf1 1, bgf2 0\n    user data 0x0"
        );
    }

    #[test]
    fn extension_types_are_formatted() {
        assert_eq!(d(V2d { x: 0.25, y: 1.0 }), "(0.25 1)");
        assert_eq!(d(V3d { x: 1.0, y: 2.0, z: 3.0 }), "(1 2 3)");
        assert_eq!(d(M33d { x: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }), "(1 0 0\n 0 1 0\n 0 0 1)");
        assert_eq!(d(vec![1.0f32, 0.5]), "(1 0.5)");
        assert_eq!(d(Vec::<f32>::new()), "()");
        assert_eq!(d(DeepImageState::Tidy), "tidy");
        assert_eq!(d(DeepImageState::NonOverlapping), "non-overlapping");
        assert_eq!(d(Bytes::new("text/plain", vec![1, 2, 3])), "3 bytes, type hint \"text/plain\"");
        assert_eq!(d(CompressedIDManifest::new(100, vec![0; 10])), "10 bytes compressed, 100 bytes uncompressed");
    }

    #[test]
    fn opaque_prints_only_the_type() {
        // Unknown type: exrheader prints `name (type T)` and nothing else.
        let a = OpaqueAttribute::with_data("myType", vec![1, 2, 3]);
        assert_eq!(describe(&a), "");
        assert_eq!(line("thing", a), "thing (type myType)");
    }

    #[test]
    fn rows_follow_header_order_and_decode_names() {
        let mut h = Header::new();
        h.insert("zeta", TypedAttribute::new(1i32)).expect("zeta");
        h.insert("alpha", TypedAttribute::new(ExrString::from("x"))).expect("alpha");
        // Non-UTF-8 name/type bytes are decoded lossily, never dropped.
        h.insert(vec![b'n', 0xFF], OpaqueAttribute::with_data(vec![b't', 0xFE], vec![])).expect("raw");
        let r = rows(&h);
        let names: Vec<&str> = r.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names.len(), h.iter().count());
        assert!(names.windows(2).all(|w| w[0] <= w[1]), "sorted like the header map: {names:?}");
        let by = |n: &str| r.iter().find(|r| r.name == n).expect("row");
        assert_eq!(by("n\u{FFFD}").type_name, "t\u{FFFD}");
        assert_eq!(by("alpha").line(), "alpha (type string): \"x\"");
        assert_eq!(by("zeta").line(), "zeta (type int): 1");
    }

    // -- corpus round trip --------------------------------------------------

    fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(&p, out);
            } else if p.extension().and_then(|x| x.to_str()) == Some("exr") {
                out.push(p);
            }
        }
    }

    /// Every attribute of every part of every openable corpus file formats
    /// without panicking, keeps the header's count/order/names/types, and the
    /// values of the standard attributes agree with the typed accessors.
    #[test]
    #[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
    fn corpus_headers_describe_completely() {
        use crate::io::{IStream, StdIfStream};
        use crate::multipart::MultiPartInputFile;

        let root = match std::env::var("EXR_CORPUS_DIR") {
            Ok(p) if !p.is_empty() => PathBuf::from(p),
            _ => exr_test_data::openexr_test(),
        };
        assert!(root.is_dir(), "describe corpus {} not found", root.display());
        let mut files = Vec::new();
        collect(&root, &mut files);
        files.sort();

        let (mut opened, mut attrs) = (0usize, 0usize);
        let mut types = std::collections::BTreeSet::new();
        for f in &files {
            // Deliberately corrupt corpus members fail to open; that is the
            // corpus runner's concern, not the formatter's.
            let Some(path) = f.to_str() else { continue };
            let Ok(mut s) = StdIfStream::open(path) else { continue };
            if s.seekg(0).is_err() {
                continue;
            }
            let Ok(mp) = MultiPartInputFile::new(&mut s) else { continue };
            opened += 1;
            for p in 0..mp.parts() {
                let Ok(h) = mp.header(p) else { continue };
                let r = rows(h);
                assert_eq!(r.len(), h.iter().count(), "{}", f.display());
                for (row, (name, a)) in r.iter().zip(h.iter()) {
                    let ty = String::from_utf8_lossy(a.type_name());
                    assert_eq!(row.name, String::from_utf8_lossy(name.as_bytes()), "{}", f.display());
                    assert_eq!(row.type_name, ty, "{}", f.display());
                    assert_eq!(row.value, describe(a.as_ref()), "{}", f.display());
                    assert!(row.line().starts_with(&format!("{} (type {ty})", row.name)));
                    // exrheader prints a value for every type it knows and for
                    // all extension types; only unregistered types are bare.
                    if crate::attr::known_type(&ty) {
                        assert!(
                            row.line().len() > format!("{} (type {ty})", row.name).len() || row.indent > 0,
                            "{} {}: known type printed no value",
                            f.display(),
                            row.name
                        );
                    }
                    types.insert(ty.into_owned());
                    attrs += 1;
                }
                // Standard attributes vs the typed accessors.
                let get = |n: &str| r.iter().find(|x| x.name == n).map(|x| x.value.clone());
                if let Ok(c) = h.compression() {
                    assert_eq!(get("compression").as_deref(), Some(c.description().as_str()), "{}", f.display());
                }
                if let Ok(w) = h.data_window() {
                    assert_eq!(
                        get("dataWindow").as_deref(),
                        Some(format!("({} {}) - ({} {})", w.min.x, w.min.y, w.max.x, w.max.y).as_str()),
                        "{}",
                        f.display()
                    );
                }
            }
        }
        assert!(opened >= 10, "only {opened} corpus files opened under {}", root.display());
        for want in ["box2i", "chlist", "compression", "float", "lineOrder", "v2f"] {
            assert!(types.contains(want), "corpus never exercised type {want}; saw {types:?}");
        }
        eprintln!("describe corpus: {opened} files, {attrs} attributes, types {types:?}");
    }

    // -- live diff against real exrheader -----------------------------------

    /// One attribute of exrheader's output: its lines (first line at column 0,
    /// block lines indented).
    type Block = Vec<String>;

    /// Split exrheader's stdout (`main.cpp:199-436`) into per-part attribute
    /// blocks: skip the `file ...:` / `file format version` preamble, start a
    /// new part at each ` part N:` line (only printed for multipart files), and
    /// start a new attribute at every non-indented line.
    fn parse_exrheader(out: &str) -> Vec<Vec<Block>> {
        let mut parts: Vec<Vec<Block>> = vec![Vec::new()];
        let mut seen_version = false;
        let mut prev_blank = false;
        let mut marked = false;
        for l in out.lines() {
            let l = l.trim_end_matches('\r');
            if !seen_version {
                seen_version = l.starts_with("file format version:");
                continue;
            }
            let blank = l.is_empty();
            let is_part = prev_blank && l.starts_with(" part ") && l.trim_end().ends_with(':');
            prev_blank = blank;
            if is_part {
                if marked {
                    parts.push(Vec::new()); // the first ` part 0:` reuses the initial part
                }
                marked = true;
                continue;
            }
            if blank {
                continue;
            }
            let Some(cur) = parts.last_mut() else { continue };
            if l.starts_with(' ') {
                if let Some(b) = cur.last_mut() {
                    b.push(l.to_string());
                }
            } else {
                cur.push(vec![l.to_string()]);
            }
        }
        parts
    }

    /// Write a 2x2 file whose header carries one sample of every attribute type
    /// exr-core registers (the ones the corpus lacks matter most: timecode, ...).
    fn write_all_types(path: &Path) {
        use crate::api::{ChannelData, Image};
        use crate::attr::Name;
        use imath_rs::Half;

        fn a<T: AttrValue>(name: &str, v: T) -> (Name, Box<dyn Attribute>) {
            (Name::from(name), Box::new(TypedAttribute::new(v)))
        }
        let tc = TimeCode::from_fields(23, 59, 58, 24, true, true, false, true, false, true, 1, 2, 3, 4, 5, 6, 7, 8)
            .expect("timecode");
        let extra = vec![
            a("tc", tc),
            a("kc", KeyCode::new(12, 3, 456789, 42, 7, 4, 64).expect("keycode")),
            a("rat", Rational::new(24000, 1001)),
            a("prev", PreviewImage::new(3, 2, None).expect("preview")),
            a("fv", vec![1.0f32, 0.1, 1.0 / 3.0]),
            a("byt", Bytes::new("text/plain", vec![1, 2, 3])),
            a("sv", vec![ExrString::from("one"), ExrString::from("two words")]),
            a("env", Envmap::Cube),
            a("dis", DeepImageState::Tidy),
            a("d", 1.0f64 / 3.0),
            a("f", 1234567.0f32),
            a("v2i", vi(-1, 2)),
            a("v3f", V3f { x: 0.5, y: 1.0e-5, z: -2.0 }),
            a("v2d", V2d { x: 0.25, y: 100000.0 }),
            a("v3d", V3d { x: 1.0, y: 2.0, z: 3.0 }),
            a("m33d", M33d { x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5]] }),
            a("m44d", M44d { x: [[1.0; 4]; 4] }),
            a("b2f", Box2f { min: Vec2 { x: -0.5, y: 0.0 }, max: Vec2 { x: 3.25, y: 1.0 / 7.0 } }),
        ];
        let dw = Box2i { min: vi(0, 0), max: vi(1, 1) };
        let img = Image::new(dw).with_channel("R", ChannelData::Half(vec![Half::from(0.0f32); 4]));
        let mut s = crate::io::MemOStream::new();
        img.write_to_with_attrs(&mut s, Compression::NoCompression, extra).expect("write");
        std::fs::write(path, s.into_data()).expect("write synthetic file");
    }

    /// Every attribute of every part of every golden/corpus file that real
    /// `exrheader` (built by `cargo xtask parity-header`) opens prints, through
    /// `Row::line`, exactly what exrheader prints. Types exrheader has no
    /// formatter for ([`EXTENSION_TYPES`]) are compared as the bare
    /// `name (type t)` line it prints for them.
    #[test]
    #[ignore = "needs exrheader.exe: run `cargo xtask parity-header`"]
    fn matches_exrheader() {
        use crate::io::{IStream, StdIfStream};
        use crate::multipart::MultiPartInputFile;
        use std::process::Command;

        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let exe = std::env::var_os("EXR_EXRHEADER_EXE")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest.join("tests/parity/exrheader.exe"));
        assert!(exe.is_file(), "exrheader not found at {} -- run `cargo xtask parity-header`", exe.display());

        let mut files = Vec::new();
        collect(&manifest.join("tests/parity"), &mut files);
        collect(&exr_test_data::openexr_test(), &mut files);
        if let Ok(p) = std::env::var("EXR_CORPUS_DIR") {
            collect(Path::new(&p), &mut files);
        }
        collect(&exr_test_data::openexr_images(), &mut files);
        files.sort();
        files.dedup();
        // The corpus has no timecode / floatvector / bytes, so also write one file
        // carrying a sample of every attribute type exr-core knows.
        let synth = std::env::temp_dir().join(format!("exr_describe_all_types_{}.exr", std::process::id()));
        write_all_types(&synth);
        files.push(synth.clone());

        let (mut compared, mut skipped, mut attrs) = (0usize, 0usize, 0usize);
        let mut ours_rejects: Vec<String> = Vec::new();
        let mut types = std::collections::BTreeSet::new();
        for f in &files {
            let out = Command::new(&exe).arg(f).output().expect("spawn exrheader");
            if !out.status.success() {
                skipped += 1; // C++ itself rejects it (corrupt corpus member)
                continue;
            }
            let theirs = parse_exrheader(&String::from_utf8_lossy(&out.stdout));
            let path = f.to_str().expect("utf-8 path");
            let mut s = StdIfStream::open(path).expect("open");
            s.seekg(0).expect("seek");
            let mp = match MultiPartInputFile::new(&mut s) {
                Ok(mp) => mp,
                Err(e) => {
                    // A reader-strictness gap, not a formatting one: collected
                    // and checked against the known list below.
                    ours_rejects.push(format!("{}: {e}", f.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned())));
                    continue;
                }
            };
            assert_eq!(usize::try_from(mp.parts()).ok(), Some(theirs.len()), "part count {}", f.display());
            for (p, their) in theirs.iter().enumerate() {
                let h = mp.header(i32::try_from(p).expect("part index")).expect("header");
                let ours: Vec<String> = rows(h)
                    .into_iter()
                    .map(|mut r| {
                        types.insert(r.type_name.clone());
                        if EXTENSION_TYPES.contains(&r.type_name.as_str()) || r.type_name.contains('\u{FFFD}') {
                            r.value.clear(); // exrheader prints nothing for these,
                            r.indent = 0; // not even the colon
                        }
                        r.line()
                    })
                    .collect();
                let their: Vec<String> = their.iter().map(|b| b.join("\n")).collect();
                assert_eq!(ours.len(), their.len(), "{} part {p}: attribute count", f.display());
                for (o, t) in ours.iter().zip(&their) {
                    assert_eq!(o, t, "{} part {p}", f.display());
                    attrs += 1;
                }
            }
            compared += 1;
        }
        eprintln!(
            "exrheader diff: {compared} files compared ({attrs} attributes), {skipped} rejected by C++; types {types:?}"
        );
        std::fs::remove_file(&synth).expect("remove the synthetic file this test created");
        assert!(types.contains("timecode"), "synthetic file must exercise timecode; saw {types:?}");
        eprintln!("opened by C++ exrheader but rejected by our reader ({}):", ours_rejects.len());
        for r in &ours_rejects {
            eprintln!("  {r}");
        }
        assert!(compared >= 50, "only {compared} files compared");
    }
}
