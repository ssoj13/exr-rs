//! The command line of `exrmetrics` (`main.cpp`: `usageMessage` and `options::parse`).
//!
//! The grammar is the tool's own and is read left to right: an option that takes a value takes the
//! next word whatever it looks like, every other word (an unknown `-x` included) is an input file, and
//! `-h` / `--version` act when they are reached. Numbers are read by the shared strict parsers
//! (`c_int`, `c_f64`): a word that is not a number is an error, where the C++ `atoi` / `atof` take
//! the digits it starts with and 0 for none (`-t x` is 0 threads, `-l abc` level 0).

use exr_core::attr::Compression;

use super::exrmetrics::PixelMode;
use crate::cli::tool_io::{c_f32, c_int};

/// `--time` bits: which operations are timed.
pub const TIME_READ: u8 = 1;
pub const TIME_WRITE: u8 = 2;
pub const TIME_REREAD: u8 = 4;

/// Everything `main` needs to run: the files and the measurements to make.
#[derive(Debug)]
pub struct Opts {
    pub out_file: Option<String>,
    pub in_files: Vec<String>,
    /// The part to copy, -1 for all.
    pub part: i32,
    /// Worker threads: 0 is serial, -1 lets the system choose (`-m`).
    pub threads: i32,
    /// ZIP / DWA level; infinite when `-l` was not given.
    pub level: f32,
    pub passes: i32,
    pub timing: u8,
    pub output_size_data: bool,
    pub output_part_size_on_disk: bool,
    pub verbose: bool,
    pub csv: bool,
    pub pixel_modes: Vec<PixelMode>,
    /// `None` is the original method (`NUM_COMPRESSION_METHODS` in the C++).
    pub compressions: Vec<Option<Compression>>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            out_file: None,
            in_files: Vec::new(),
            part: -1,
            threads: 0,
            level: f32::INFINITY,
            passes: 1,
            timing: TIME_READ | TIME_WRITE | TIME_REREAD,
            output_size_data: true,
            output_part_size_on_disk: false,
            verbose: false,
            csv: false,
            pixel_modes: Vec::new(),
            compressions: Vec::new(),
        }
    }
}

/// What the command line asks for.
pub enum Cmd {
    Help,
    Version,
    Run(Opts),
}

/// The verbose `usageMessage` text, for the help of the arguments.
pub fn help() -> String {
    format!(
        "Read an OpenEXR image from infile, write an identical copy to outfile\n\
         reporting time taken to read/write and file sizes.\n\
         \n\
         Options:\n\
         \n  -o file                     file to write to. If no file specified, uses a memory buffer\n\
         \x20                             note: file may be overwritten multiple times during tests\n\
         \n  -p n                        part number to copy, or \"all\" for all parts\n\
         \x20                             default is \"all\" \n\
         \n  -m                          set to multi-threaded (system selected thread count)\n\
         \x20 -t n                        Use a pool of n worker threads for processing files.\n\
         \x20                             Default is single threaded (no thread pool)\n\
         \n  -l level                    set DWA or ZIP compression level\n\
         \n  -z,--compression list       list of compression methods to test\n\
         \x20                             ({names},orig,all\n\
         \x20                             default orig: retains original method)\n\
         \n  --convert                   shorthand options for writing a new file with no metrics:\n\
         \x20                             -p all --time none --pixelmode orig --no-size --passes 1\n\
         \x20                             change pixel data type/compression by specifying --pixelmode/-z after --convert\n\
         \n  --bench                     shorthand options for robust performance benchmarking:\n\
         \x20                             -p all --compression all --time write,reread --passes 10 --type half,float --no-size --csv\n\
         \n  -16 rgba|all                [DEPRECATED] force 16 bit half float: either just RGBA, or all channels\n\
         \x20                             Use --pixelmode half or --pixelmode mixed instead\n\
         \n  --pixelmode list            list of pixel data types to use (float,half,mixed,orig)\n\
         \x20                             mixed uses half for RGBA, float for others. Default is 'orig'\n\
         \n  --time list                 comma separated list of operations to report timing for.\n\
         \x20                             operations can be any of read,write,reread (use --time none for no timing)\n\
         \n  --no-size                   don't output size data\n\
         \n  --part-disk-size            Output the on-disk size of the data portion of each part in a multipart EXR file\n\
         \x20                             (including chunk headers). File output (-o) is not supported when using this option\n\
         \n  --json                      print output as JSON dictionary (Default mode)\n\
         \n  --csv                       print output in csv mode. If passes>1, show median timing\n\
         \x20                             default is JSON mode\n\
         \n  --passes num                write and re-read file num times (default 1)\n\
         \n  -h, --help                  print this message\n\
         \n  -v                          output progress messages\n\
         \n  --version                   print version information\n",
        names = Compression::names_string("/")
    )
}

/// The short `usageMessage`, printed with "Missing input file".
pub fn usage() -> &'static str {
    "Usage: exr-rs metrics [options] infile [infile2...] [-o outfile]\n"
}

/// `split (str, ',')`: what `std::getline` makes of a comma separated list. The empty list has no
/// items, a trailing comma adds none, a doubled comma makes an empty one.
fn split(list: &str) -> Vec<&str> {
    let mut items: Vec<&str> = list.split(',').collect();
    if items.last().is_some_and(|last| last.is_empty()) {
        items.pop();
    }
    items
}

/// The value word of the option at `argv[i]`, or the C++ message when it is the last word.
fn value<'a>(argv: &'a [String], i: usize, missing: &str) -> Result<&'a str, String> {
    argv.get(i + 1).map(String::as_str).ok_or_else(|| missing.to_string())
}

/// `options::parse`: the first error wins; `-h` and `--version` return when reached (before any
/// default is applied).
pub fn parse(argv: &[String]) -> Result<Cmd, String> {
    let mut o = Opts::default();
    let mut i = 0;
    while i < argv.len() {
        let tok = argv[i].as_str();
        match tok {
            "-h" | "--help" => return Ok(Cmd::Help),
            "--version" => return Ok(Cmd::Version),
            "-m" => {
                o.threads = -1;
                i += 1;
            }
            "-t" => {
                let v = value(argv, i, "Missing thread count value with -t option")?;
                o.threads = c_int(v)?;
                if o.threads < 0 {
                    return Err(format!("bad thread count {v} specified to -t option"));
                }
                i += 2;
            }
            "--bench" => {
                o.compressions = Compression::ALL.iter().map(|c| Some(*c)).collect();
                o.passes = 10;
                o.output_size_data = false;
                o.csv = true;
                o.part = -1;
                o.timing = TIME_WRITE | TIME_REREAD;
                o.pixel_modes = vec![PixelMode::AllHalf, PixelMode::AllFloat];
                i += 1;
            }
            "--convert" => {
                o.pixel_modes = vec![PixelMode::Original];
                o.passes = 1;
                o.output_size_data = false;
                o.timing = 0;
                o.part = -1;
                i += 1;
            }
            "--passes" => {
                let v = value(argv, i, "Missing pass count value with --passes option")?;
                o.passes = c_int(v)?;
                if o.passes < 0 {
                    return Err(format!("bad value for passes {v} specified to --passes option"));
                }
                i += 2;
            }
            "-v" => {
                o.verbose = true;
                i += 1;
            }
            "--csv" => {
                o.csv = true;
                i += 1;
            }
            "--json" => {
                o.csv = false;
                i += 1;
            }
            "-z" | "--compression" => {
                o.compressions.clear();
                let v = value(argv, i, &format!("Missing compression value with {tok} option"))?;
                for name in split(v) {
                    match name {
                        "orig" => o.compressions.push(None),
                        "all" => o.compressions.extend(Compression::ALL.iter().map(|c| Some(*c))),
                        _ => o
                            .compressions
                            .push(Some(Compression::from_name(name).ok_or_else(|| format!("unknown compression type {name}"))?)),
                    }
                }
                i += 2;
            }
            "-p" => {
                let v = value(argv, i, "Missing part number with -p option")?;
                if v == "all" {
                    o.part = -1;
                } else {
                    o.part = c_int(v)?;
                    if o.part < -1 {
                        return Err(format!("bad part {} specified to -p option", o.part));
                    }
                }
                i += 2;
            }
            "-l" => {
                let v = value(argv, i, "Missing compression level number with -l option")?;
                o.level = c_f32(v)?;
                if o.level < 0.0 {
                    return Err(format!("bad level {} specified to -l option", o.level));
                }
                i += 2;
            }
            "-o" => {
                let v = value(argv, i, "Missing filename specified with -o")?;
                if o.out_file.is_some() {
                    return Err("-o output filename can only be specified once".into());
                }
                if o.output_part_size_on_disk {
                    return Err("--part-disk-size cannot be used with file output (-o)".into());
                }
                o.out_file = Some(v.to_string());
                i += 2;
            }
            // Deprecated spelling of --pixelmode, kept by the C++ for old scripts.
            "-16" => {
                match value(argv, i, "Missing mode with -16 option")? {
                    "all" => o.pixel_modes.push(PixelMode::AllHalf),
                    "rgba" => o.pixel_modes.push(PixelMode::MixedHalfFloat),
                    _ => return Err(" bad mode for -16 option: must be 'all' or 'rgba'".into()),
                }
                i += 2;
            }
            "--pixelmode" => {
                let v = value(argv, i, "Missing type list with  with --pixelmode option")?;
                o.pixel_modes.clear();
                for item in split(v) {
                    o.pixel_modes.push(match item {
                        "half" => PixelMode::AllHalf,
                        "float" => PixelMode::AllFloat,
                        "rgba" | "mixed" => PixelMode::MixedHalfFloat,
                        "orig" => PixelMode::Original,
                        _ => return Err(format!("bad pixel type {item} for --pixelmode: must be half,float,rgba,mixed or orig")),
                    });
                }
                i += 2;
            }
            "--time" => {
                let v = value(argv, i, "Missing value list with --time option")?;
                o.timing = 0;
                if v != "none" {
                    for item in split(v) {
                        o.timing |= match item {
                            "read" => TIME_READ,
                            "reread" => TIME_REREAD,
                            "write" => TIME_WRITE,
                            _ => return Err("bad value in timing list. Options are read,write,reread".into()),
                        };
                    }
                }
                i += 2;
            }
            "--no-size" => {
                o.output_size_data = false;
                i += 1;
            }
            "--part-disk-size" => {
                if o.out_file.is_some() {
                    return Err("--part-disk-size cannot be used with file output (-o)".into());
                }
                o.output_part_size_on_disk = true;
                i += 1;
            }
            "-i" => {
                o.in_files.push(value(argv, i, "Missing filename with -i option")?.to_string());
                i += 2;
            }
            file => {
                o.in_files.push(file.to_string());
                i += 1;
            }
        }
    }
    if o.in_files.is_empty() {
        return Err(format!("Missing input file\n{}", usage().trim_end()));
    }
    if !o.output_size_data && o.timing == 0 && o.out_file.is_none() {
        return Err("Nothing to do: no output file specified, and all performance/size data disabled\nUse -o to specify output image filename".into());
    }
    if o.pixel_modes.is_empty() {
        o.pixel_modes.push(PixelMode::Original);
    }
    if o.compressions.is_empty() {
        o.compressions.push(None);
    }
    Ok(Cmd::Run(o))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    fn opts(line: &str) -> Opts {
        match parse(&args(line)) {
            Ok(Cmd::Run(o)) => o,
            Ok(_) => panic!("not a run: {line}"),
            Err(e) => panic!("{line}: {e}"),
        }
    }

    fn err(line: &str) -> String {
        match parse(&args(line)) {
            Err(e) => e,
            Ok(_) => panic!("{line} parsed"),
        }
    }

    #[test]
    fn defaults() {
        let o = opts("a.exr");
        assert_eq!((o.in_files, o.part, o.threads, o.passes, o.timing), (vec!["a.exr".to_string()], -1, 0, 1, 7));
        assert!(o.level.is_infinite() && o.output_size_data && !o.csv && !o.verbose && o.out_file.is_none());
        assert_eq!((o.pixel_modes, o.compressions), (vec![PixelMode::Original], vec![None]));
    }

    #[test]
    fn every_value_option() {
        let o = opts("-t 4 --passes 3 -p 2 -l 1.5 -o out.exr -z zip,piz --pixelmode half,mixed --time read,write a.exr b.exr");
        assert_eq!((o.threads, o.passes, o.part, o.level, o.out_file.as_deref()), (4, 3, 2, 1.5, Some("out.exr")));
        assert_eq!(o.compressions, vec![Some(Compression::Zip), Some(Compression::Piz)]);
        assert_eq!(o.pixel_modes, vec![PixelMode::AllHalf, PixelMode::MixedHalfFloat]);
        assert_eq!((o.timing, o.in_files.len()), (TIME_READ | TIME_WRITE, 2));
        assert_eq!(opts("-m a").threads, -1);
        assert_eq!(opts("-p all a").part, -1);
        assert_eq!(opts("-p -1 a").part, -1);
        assert_eq!(opts("-i x.exr y.exr").in_files, ["x.exr", "y.exr"]);
        assert_eq!(opts("--compression orig,all a").compressions.len(), 13);
    }

    /// Numbers must be numbers (the C++ takes `atoi`'s prefix and 0 for none): all forms of
    /// `strtol` / `strtod` are accepted, nothing else.
    #[test]
    fn numbers_must_be_numbers() {
        for bad in ["-t x a", "--passes 2x a", "-p 3abc a", "-l abc -z zip a", "-l 5x -z zip a", "-t 1.5 a"] {
            assert!(err(bad).starts_with("Expected "), "{bad}: {}", err(bad));
        }
        assert_eq!((opts("-t 0x4 a").threads, opts("-p 010 a").part), (4, 8));
        assert_eq!((opts("-l 0x1 a").level, opts("-l 1e0 a").level, opts("-l 0x1p3 a").level), (1.0, 1.0, 8.0));
        assert!(opts("-l nan a").level.is_nan() && opts("-l inf a").level.is_infinite());
    }

    #[test]
    fn shorthands_and_their_order() {
        let o = opts("--bench a");
        assert_eq!((o.passes, o.csv, o.output_size_data, o.timing), (10, true, false, TIME_WRITE | TIME_REREAD));
        assert_eq!((o.compressions.len(), o.pixel_modes), (12, vec![PixelMode::AllHalf, PixelMode::AllFloat]));
        let o = opts("--convert -o out.exr a");
        assert_eq!((o.timing, o.output_size_data, o.passes, o.pixel_modes), (0, false, 1, vec![PixelMode::Original]));
        // A later option overrides what a shorthand set, an earlier one is overridden by it.
        assert_eq!(opts("--convert --pixelmode half -o o a").pixel_modes, vec![PixelMode::AllHalf]);
        assert_eq!(opts("--pixelmode half --convert -o o a").pixel_modes, vec![PixelMode::Original]);
        assert!(!opts("--bench --json a").csv);
        assert_eq!(opts("--bench -z zip a").compressions, vec![Some(Compression::Zip)]);
    }

    #[test]
    fn deprecated_16_appends() {
        assert_eq!(opts("-16 all -16 rgba a").pixel_modes, vec![PixelMode::AllHalf, PixelMode::MixedHalfFloat]);
        assert_eq!(opts("-16 all --pixelmode float a").pixel_modes, vec![PixelMode::AllFloat]);
        assert_eq!(err("-16 x a"), " bad mode for -16 option: must be 'all' or 'rgba'");
    }

    /// `std::getline` semantics of the lists.
    #[test]
    fn lists_split_like_getline() {
        assert_eq!(split(""), Vec::<&str>::new());
        assert_eq!(split("a"), ["a"]);
        assert_eq!(split("a,"), ["a"]);
        assert_eq!(split(",a"), ["", "a"]);
        assert_eq!(split("a,,b"), ["a", "", "b"]);
        assert_eq!(split(","), [""]);
    }

    #[test]
    fn the_cpp_error_messages() {
        assert_eq!(err("a -t"), "Missing thread count value with -t option");
        assert_eq!(err("-t -1 a"), "bad thread count -1 specified to -t option");
        assert_eq!(err("a --passes"), "Missing pass count value with --passes option");
        assert_eq!(err("--passes -2 a"), "bad value for passes -2 specified to --passes option");
        assert_eq!(err("a -z"), "Missing compression value with -z option");
        assert_eq!(err("a --compression"), "Missing compression value with --compression option");
        assert_eq!(err("-z lzw a"), "unknown compression type lzw");
        assert_eq!(err("-z zip,,piz a"), "unknown compression type ");
        assert_eq!(err("a -p"), "Missing part number with -p option");
        assert_eq!(err("-p -2 a"), "bad part -2 specified to -p option");
        assert_eq!(err("a -l"), "Missing compression level number with -l option");
        assert_eq!(err("-l -1 a"), "bad level -1 specified to -l option");
        assert_eq!(err("a -o"), "Missing filename specified with -o");
        assert_eq!(err("-o a -o b c"), "-o output filename can only be specified once");
        assert_eq!(err("--part-disk-size -o a c"), "--part-disk-size cannot be used with file output (-o)");
        assert_eq!(err("-o a --part-disk-size c"), "--part-disk-size cannot be used with file output (-o)");
        assert_eq!(err("a -16"), "Missing mode with -16 option");
        assert_eq!(err("a --pixelmode"), "Missing type list with  with --pixelmode option");
        assert_eq!(err("--pixelmode half,x a"), "bad pixel type x for --pixelmode: must be half,float,rgba,mixed or orig");
        assert_eq!(err("a --time"), "Missing value list with --time option");
        assert_eq!(err("--time read,x a"), "bad value in timing list. Options are read,write,reread");
        assert_eq!(err("a -i"), "Missing filename with -i option");
        assert!(err("-v").starts_with("Missing input file\nUsage: "));
        assert!(err("--no-size --time none a").starts_with("Nothing to do: "));
    }

    #[test]
    fn time_lists() {
        assert_eq!(opts("--time none a").timing, 0);
        assert_eq!(opts("--time read,reread a").timing, TIME_READ | TIME_REREAD);
        assert_eq!(err("--time , a"), "bad value in timing list. Options are read,write,reread");
        // The empty list times nothing (it is not "none", but it has no items).
        let words: Vec<String> = ["--time", "", "a"].map(String::from).into();
        assert!(matches!(parse(&words), Ok(Cmd::Run(o)) if o.timing == 0));
    }

    #[test]
    fn nothing_to_do_needs_an_output_or_a_report() {
        assert!(matches!(parse(&args("--no-size --time none -o out a")), Ok(Cmd::Run(_))));
        assert!(matches!(parse(&args("--no-size --time read a")), Ok(Cmd::Run(_))));
        assert!(matches!(parse(&args("--time none a")), Ok(Cmd::Run(_))), "sizes are still reported");
    }

    #[test]
    fn unknown_options_are_files_and_help_acts_when_reached() {
        assert_eq!(opts("-x --y a").in_files, ["-x", "--y", "a"]);
        assert!(matches!(parse(&args("a -h")), Ok(Cmd::Help)));
        assert!(matches!(parse(&args("--version a")), Ok(Cmd::Version)));
        assert_eq!(err("-t -1 -h"), "bad thread count -1 specified to -t option", "the first error wins");
    }

    #[test]
    fn the_help_lists_every_option_and_codec() {
        let h = help();
        for opt in ["-o file", "-p n", "-m ", "-t n", "-l level", "-z,--compression list", "--convert", "--bench", "-16 rgba|all", "--pixelmode list", "--time list", "--no-size", "--part-disk-size", "--json", "--csv", "--passes num", "-h, --help", "-v ", "--version"] {
            assert!(h.contains(opt), "help lacks {opt}");
        }
        for c in Compression::ALL {
            assert!(h.contains(c.name()), "help lacks {}", c.name());
        }
    }
}
