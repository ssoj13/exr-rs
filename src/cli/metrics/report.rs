//! The reports of `exrmetrics` (`main.cpp`: `printTiming`, `printPartStats`, `jsonStats`, `csvStats`).
//!
//! The text follows the C++ to the space: it is what scripts read. Numbers are formatted as a C++
//! `ostream` prints a `double` (six significant digits, `%g` style), see [`g`].
//!
//! Deliberate differences, for a file with no timing samples (`--passes 0`): the CSV report prints
//! `---` for the missing median (the C++ takes the median of an empty vector, which reads past it and
//! crashes, `main.cpp:154`), and the JSON report `null` where the C++ prints nothing at all
//! (`"write time": ,`, not valid JSON; `main.cpp:161`).

use exr_core::attr::Compression;

use super::exrmetrics::{FileMetrics, PartStats, PixelMode};
use super::opts::{TIME_READ, TIME_REREAD, TIME_WRITE};

/// One measured combination of input file, compression and pixel mode (`runData`).
pub struct Run {
    /// Which input file (the C++ compares the `argv` pointers: the same name given twice is two files).
    pub file: usize,
    pub name: String,
    pub mode: PixelMode,
    /// `None`: the original method.
    pub compression: Option<Compression>,
    pub metrics: FileMetrics,
}

/// A `double` as `ostream << double` prints it by default: `%g` with six significant digits.
pub fn g(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.into();
    }
    // The decimal exponent after rounding to six digits decides fixed or exponent notation.
    let sci = format!("{x:.5e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let strip = |s: &str| if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s.to_string() };
    if !(-4..6).contains(&exp) {
        format!("{}e{}{:02}", strip(mantissa), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        strip(&format!("{x:.*}", (5 - exp) as usize))
    }
}

/// `median`; `None` for no samples.
fn median(perf: &[f64]) -> Option<f64> {
    match perf {
        [] => None,
        [one] => Some(*one),
        _ => {
            let mut d = perf.to_vec();
            d.sort_by(f64::total_cmp);
            Some((d[d.len() / 2] + d[(d.len() - 1) / 2]) / 2.0)
        }
    }
}

/// `printTiming`: one sample is its number, several are an object with the values and statistics.
fn print_timing(perf: &[f64], out: &mut String, raw: bool, stats: bool) {
    match perf {
        [] => out.push_str("null"),
        [one] => out.push_str(&g(*one)),
        _ => {
            out.push('{');
            if raw {
                out.push_str("\"values\": [ ");
            }
            let n = perf.len() as f64;
            let k = perf[0];
            let (mut sum, mut x, mut x2) = (0.0, 0.0, 0.0);
            let (mut minimum, mut maximum) = (f64::MAX, -f64::MAX);
            for (i, &v) in perf.iter().enumerate() {
                if raw {
                    if i > 0 {
                        out.push_str(" , ");
                    }
                    out.push_str(&g(v));
                }
                sum += v;
                x += v - k;
                x2 += (v - k) * (v - k);
                maximum = maximum.max(v);
                minimum = minimum.min(v);
            }
            if raw {
                out.push_str(" ] , ");
            }
            if stats {
                out.push_str(&format!(
                    "\"min\": {}, \"max\": {}, \"mean\": {}, \"median\": {}, \"std dev\": {}",
                    g(minimum),
                    g(maximum),
                    g(sum / n),
                    g(median(perf).unwrap_or(f64::NAN)),
                    g(((x2 - x * x / n) / n).sqrt())
                ));
            }
            out.push('}');
        }
    }
}

/// `printPartStats`: the timings the run asked for, one per line, `indent`ed, joined by `,\n`.
fn print_part_stats(out: &mut String, data: &PartStats, indent: &str, timing: u8, part_size: bool, raw: bool, stats: bool) {
    let mut output = false;
    if timing & TIME_READ != 0 {
        if data.size_data.is_deep {
            out.push_str(&format!("{indent}\"count read time\": "));
            print_timing(&data.count_read, out, raw, stats);
            output = true;
        }
        if output {
            out.push_str(",\n");
        }
        out.push_str(&format!("{indent}\"read time\": "));
        print_timing(&data.read, out, raw, stats);
        output = true;
    }
    if timing & TIME_WRITE != 0 {
        if output {
            out.push_str(",\n");
        }
        output = true;
        out.push_str(&format!("{indent}\"write time\": "));
        print_timing(&data.write, out, raw, stats);
    }
    if timing & TIME_REREAD != 0 {
        if data.size_data.is_deep {
            if output {
                out.push_str(",\n");
            }
            output = true;
            out.push_str(&format!("{indent}\"count re-read time\": "));
            print_timing(&data.count_reread, out, raw, stats);
        }
        if output {
            out.push_str(",\n");
        }
        output = true;
        out.push_str(&format!("{indent}\"re-read time\": "));
        print_timing(&data.reread, out, raw, stats);
    }
    if part_size {
        if output {
            out.push_str(",\n");
        }
        out.push_str(&format!("{indent}\"size on disk\": {}\n", data.size_on_disk));
    }
}

/// `std::quoted`: in double quotes, `"` and `\` escaped by a backslash.
fn quoted(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The name `getCompressionNameFromId` gives: the codec's, and for none recorded the empty string
/// that the C++ reads from beyond its table.
fn comp_name(c: Option<Compression>) -> &'static str {
    c.map_or("", |c| c.name())
}

/// `jsonStats`.
pub fn json_stats(runs: &[Run], output_size_data: bool, timing: u8, part_size: bool, raw: bool, stats: bool) -> String {
    let mut out = String::from("[\n");
    let mut last: Option<usize> = None;
    let mut first_entry = true;
    for run in runs {
        let m = &run.metrics;
        let total = &m.total.size_data;
        if last != Some(run.file) {
            if last.is_some() {
                out.push_str("\n   ]\n");
                out.push_str(" },\n");
            }
            out.push_str(&format!(" {{\n  \"file\":{},\n", quoted(&run.name)));
            last = Some(run.file);
            if output_size_data {
                out.push_str(&format!("  \"input file size\": {},\n", m.input_size));
                out.push_str(&format!("  \"pixels\": {},\n", total.pixel_count));
                out.push_str(&format!("  \"channels\": {},\n", total.channel_count));
                out.push_str(&format!("  \"total raw size\": {},\n", total.raw_size));
                let comp = total.compression.map_or("mixed", |c| c.name());
                out.push_str(&format!("  \"compression\": \"{comp}\",\n"));
                let ty = if total.part_type.is_empty() { "mixed" } else { &total.part_type };
                out.push_str(&format!("  \"part type\": \"{ty}\",\n"));
                if total.is_tiled {
                    out.push_str(&format!("  \"tile count\": {},\n", total.tile_count));
                }
                if m.stats.len() > 1 {
                    out.push_str("  \"parts\":\n   [\n");
                    for (part, s) in m.stats.iter().enumerate() {
                        let d = &s.size_data;
                        out.push_str("    {\n");
                        out.push_str(&format!("      \"part\": {part},\n"));
                        out.push_str(&format!("      \"pixels\": {},\n", d.pixel_count));
                        out.push_str(&format!("      \"channels\": {},\n", d.channel_count));
                        out.push_str(&format!("      \"compression\": \"{}\",\n", comp_name(d.compression)));
                        out.push_str(&format!("      \"part type\": \"{}\",\n", d.part_type));
                        if d.is_tiled {
                            out.push_str(&format!("      \"tile count\": {},\n", d.tile_count));
                        }
                        out.push_str(&format!("      \"total raw size\": {}\n", d.raw_size));
                        out.push_str("    }");
                        if part + 1 < m.stats.len() {
                            out.push(',');
                        }
                        out.push('\n');
                    }
                    out.push_str("   ],\n");
                }
            }
            out.push_str("  \"metrics\":\n   [");
            first_entry = true;
        }

        let comp = run.compression.map_or("original", |c| c.name());
        if !first_entry {
            out.push(',');
        }
        out.push('\n');
        out.push_str("    {\n");
        out.push_str(&format!("      \"compression\": \"{comp}\",\n"));
        out.push_str(&format!("      \"pixel mode\": \"{}\"", run.mode.name()));
        if output_size_data {
            out.push_str(",\n");
            out.push_str(&format!("      \"output size\": {}", m.output_size));
        }
        if timing != 0 {
            out.push_str(",\n");
            print_part_stats(&mut out, &m.total, "      ", timing, false, raw, stats);
        }
        if timing != 0 && m.stats.len() > 1 {
            out.push_str(",\n");
            out.push_str("      \"parts\":\n");
            out.push_str("       [\n");
            for (part, s) in m.stats.iter().enumerate() {
                out.push_str("        {\n");
                out.push_str(&format!("          \"part\": {part},\n"));
                print_part_stats(&mut out, s, "          ", timing, part_size, raw, stats);
                out.push_str("\n        }");
                if part + 1 < m.stats.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str("       ]\n");
        } else {
            out.push('\n');
        }
        out.push_str("    }");
        first_entry = false;
    }
    if last.is_some() {
        out.push_str("\n   ]\n");
        out.push_str(" }\n");
    }
    out.push_str("]\n");
    out
}

/// A median for a CSV cell: the number, or `---` for no samples.
fn cell(perf: &[f64]) -> String {
    median(perf).map_or_else(|| "---".to_string(), g)
}

/// `csvStats`.
pub fn csv_stats(runs: &[Run], output_size_data: bool, timing: u8) -> String {
    let mut out = String::from("file name");
    if output_size_data {
        out.push_str(",input size,pixel count,channel count,tile count,raw size");
    }
    out.push_str(",compression,pixel mode");
    if output_size_data {
        out.push_str(",output size");
    }
    if timing & TIME_READ != 0 {
        out.push_str(",count read time,read time");
    }
    if timing & TIME_WRITE != 0 {
        out.push_str(",write time");
    }
    if timing & TIME_REREAD != 0 {
        out.push_str(",count reread time,reread time");
    }
    out.push('\n');
    for run in runs {
        let (m, t) = (&run.metrics, &run.metrics.total);
        out.push_str(&run.name);
        if output_size_data {
            out.push_str(&format!(",{},{},{}", m.input_size, t.size_data.pixel_count, t.size_data.channel_count));
            if t.size_data.is_tiled {
                out.push_str(&format!(",{}", t.size_data.tile_count));
            } else {
                out.push_str(",---");
            }
            out.push_str(&format!(",{}", t.size_data.raw_size));
        }
        out.push_str(&format!(",{},{}", run.compression.map_or("original", |c| c.name()), run.mode.name()));
        if output_size_data {
            out.push_str(&format!(",{}", m.output_size));
        }
        if timing & TIME_READ != 0 {
            out.push(',');
            out.push_str(&if t.size_data.is_deep { cell(&t.count_read) } else { "---".into() });
            out.push(',');
            out.push_str(&cell(&t.read));
        }
        if timing & TIME_WRITE != 0 {
            out.push(',');
            out.push_str(&cell(&t.write));
        }
        if timing & TIME_REREAD != 0 {
            out.push(',');
            out.push_str(&if t.size_data.is_deep { cell(&t.count_reread) } else { "---".into() });
            out.push(',');
            out.push_str(&cell(&t.reread));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_prints_like_an_ostream() {
        for (x, want) in [
            (0.0, "0"),
            (1.0, "1"),
            (0.5, "0.5"),
            (0.0001547, "0.0001547"),
            (8.91e-05, "8.91e-05"),
            (1.0e-5, "1e-05"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (0.000392533333, "0.000392533"),
            (2.1853e-06, "2.1853e-06"),
            (100000.0, "100000"),
            (999999.5, "1e+06"),
            (-2.5, "-2.5"),
            (1e100, "1e+100"),
            (0.1 + 0.2, "0.3"),
        ] {
            assert_eq!(g(x), want, "{x}");
        }
        assert_eq!((g(f64::INFINITY), g(f64::NEG_INFINITY), g(f64::NAN)), ("inf".to_string(), "-inf".to_string(), "nan".to_string()));
    }

    #[test]
    fn timing_objects() {
        let mut s = String::new();
        print_timing(&[0.5], &mut s, true, true);
        assert_eq!(s, "0.5");
        s.clear();
        print_timing(&[1.0, 3.0, 2.0], &mut s, true, true);
        assert_eq!(s, "{\"values\": [ 1 , 3 , 2 ] , \"min\": 1, \"max\": 3, \"mean\": 2, \"median\": 2, \"std dev\": 0.816497}");
        s.clear();
        print_timing(&[1.0, 2.0], &mut s, false, false);
        assert_eq!(s, "{}");
        s.clear();
        print_timing(&[], &mut s, true, true);
        assert_eq!(s, "null");
    }

    #[test]
    fn medians() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[4.0]), Some(4.0));
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }

    #[test]
    fn quoting_escapes_backslashes_and_quotes() {
        assert_eq!(quoted("a\\b\"c"), "\"a\\\\b\\\"c\"");
    }
}
