//! The part-name plumbing of `exrmultipart.cpp`: how an input argument
//! `<file>[:<partnum>][::<partname>]` is taken apart, what a part is called when the argument does
//! not say, and how clashing part names are made unique.
//!
//! The functions follow the C++ character for character (the names they produce end up in the
//! output file, which is compared byte for byte), quirks included; each quirk is marked.

use std::collections::BTreeSet;
use std::path::MAIN_SEPARATOR;

use exr_core::attr::{ExrString, StringAttribute};
use exr_core::header::Header;


/// One parsed input argument (`parse_filename`).
#[derive(Debug, PartialEq, Eq)]
pub struct Spec {
    /// The file, with the `:<partnum>` and `::<partname>` suffixes removed.
    pub file: String,
    /// The part name: the `::<partname>` text, else the file name stripped of its directory,
    /// `.exr` and frame number.
    pub name: String,
    /// The name came from `::<partname>`: it overrides a name the part already has.
    pub forced: bool,
    /// The one part asked for with `:<partnum>`; `None` takes every part.
    pub part: Option<i32>,
}

/// `is_number`: a non-empty run of ASCII digits.
fn is_number(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `parse_partname`: the default part name of a file name. The directory goes (the C++ splits at
/// its platform's separator only, `\` on Windows, so `dir/a.exr` keeps `dir/a` there), then
/// `.exr` and the frame number of `name.0001.exr`. A name without `.exr` keeps any `:` text.
fn default_name(file: &str) -> String {
    let mut name = file.rsplit(MAIN_SEPARATOR).next().unwrap_or(file).to_string();
    if let Some(ext) = name.rfind(".exr") {
        name.truncate(ext);
        if let Some(dot) = name.rfind('.')
            && is_number(&name[dot + 1..])
        {
            name.truncate(dot);
        }
    }
    name
}

/// `parse_filename`: split `<file>[:<partnum>][::<partname>]`. The last `::` separates the name,
/// then the last `:` of what is left the part number, which must be all digits (so a Windows drive
/// letter, `C:/x.exr`, is refused as the C++ refuses it).
pub fn parse_filename(arg: &str) -> Result<Spec, String> {
    let mut file = arg.to_string();
    let (name, forced) = match file.rfind("::") {
        Some(at) => {
            let name = file[at + 2..].to_string();
            file.truncate(at);
            (name, true)
        }
        None => (default_name(&file), false),
    };
    let part = match file.rfind(':') {
        None => None,
        Some(at) if is_number(&file[at + 1..]) => {
            // Digits only (`is_number`): out of the `int` range is the C++'s saturated value, past any part.
            let n = file[at + 1..].parse::<i32>().unwrap_or(i32::MAX);
            file.truncate(at);
            Some(n)
        }
        Some(_) => return Err("part number must be a number".into()),
    };
    Ok(Spec { file, name, forced, part })
}

/// The bytes of the header's `name` attribute; `None` when it has none.
pub fn name_of(h: &Header) -> Option<&[u8]> {
    h.find_typed_attribute::<ExrString>("name").map(|a| a.value.as_bytes())
}

/// Set a string attribute from raw bytes: the names and views in a header are not necessarily UTF-8,
/// and the C++ `std::string` does not mind.
pub fn set_str(h: &mut Header, attr: &str, bytes: &[u8]) -> exr_core::Result<()> {
    h.insert(attr, StringAttribute::new(ExrString::from_bytes(bytes.to_vec())))
}

/// `make_unique_names`: a part whose name an earlier part already has becomes
/// `<name>_<index>_<n>` with the first `n` not taken yet. The C++ builds that text in one
/// `ostringstream` and calls `s.clear ()` between attempts, which resets the stream's error flags
/// and not its text, so a second attempt appends to the first (`a_2_1a_2_2`); that is kept.
pub fn make_unique_names(headers: &mut [Header]) -> Result<(), String> {
    let mut taken: BTreeSet<Vec<u8>> = BTreeSet::new();
    for (i, h) in headers.iter_mut().enumerate() {
        let base = name_of(h).ok_or("Software Error: header does not have a valid name")?.to_vec();
        if taken.contains(&base) {
            let mut text = Vec::new();
            for n in 1usize.. {
                text.extend_from_slice(&base);
                text.extend_from_slice(format!("_{i}_{n}").as_bytes());
                if !taken.contains(&text) {
                    break;
                }
            }
            set_str(h, "name", &text).map_err(|e| e.to_string())?;
        }
        taken.insert(name_of(h).map(<[u8]>::to_vec).unwrap_or_default());
    }
    Ok(())
}

/// `filename_check`: no input may be the output. The C++ compares the spellings only; `same`
/// also sees two paths to one file (writing over an input that is still being read).
pub fn filename_check<'a>(inputs: impl IntoIterator<Item = &'a str>, output: &str, same: impl Fn(&str, &str) -> bool) -> Result<(), String> {
    if inputs.into_iter().any(|f| same(f, output)) {
        return Err("input and output file names cannot be the same".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(arg: &str) -> Spec {
        parse_filename(arg).unwrap_or_else(|e| panic!("{arg}: {e}"))
    }

    #[test]
    fn a_bare_file_takes_every_part_named_after_itself() {
        let s = spec("beauty.exr");
        assert_eq!((s.file.as_str(), s.name.as_str(), s.forced, s.part), ("beauty.exr", "beauty", false, None));
    }

    #[test]
    fn the_frame_number_and_the_directory_leave_the_name() {
        assert_eq!(spec("beauty.0042.exr").name, "beauty");
        assert_eq!(spec("beauty.v2.exr").name, "beauty.v2", "not a number");
        assert_eq!(spec("a.exr.exr").name, "a.exr", "the last .exr only");
        assert_eq!(spec("noext").name, "noext");
        assert_eq!(spec("x.EXR").name, "x.EXR", "the extension is case sensitive");
        let dir = format!("shots{MAIN_SEPARATOR}beauty.0001.exr");
        assert_eq!(spec(&dir).name, "beauty");
        // The other separator is not one on Windows, and is the only one elsewhere.
        let other = if MAIN_SEPARATOR == '\\' { "shots/beauty.exr" } else { "shots\\beauty.exr" };
        assert_eq!(spec(other).name, other.trim_end_matches(".exr"));
    }

    #[test]
    fn part_number_and_forced_name() {
        let s = spec("a.exr:2");
        assert_eq!((s.file.as_str(), s.name.as_str(), s.forced, s.part), ("a.exr", "a", false, Some(2)));
        let s = spec("a.exr::diffuse");
        assert_eq!((s.file.as_str(), s.name.as_str(), s.forced, s.part), ("a.exr", "diffuse", true, None));
        let s = spec("a.exr:10::spec");
        assert_eq!((s.file.as_str(), s.name.as_str(), s.forced, s.part), ("a.exr", "spec", true, Some(10)));
        // Without `.exr` the `:n` stays in the default name.
        assert_eq!(spec("a:3").name, "a:3");
        assert_eq!(spec("a:3").part, Some(3));
        // An empty forced name is still forced.
        let s = spec("a.exr::");
        assert_eq!((s.name.as_str(), s.forced), ("", true));
    }

    #[test]
    fn a_bad_part_number_is_refused() {
        for bad in ["a.exr:x", "a.exr:", "a:::b", "C:/x.exr", "a.exr:-1", "a.exr:1x::n"] {
            assert_eq!(parse_filename(bad), Err("part number must be a number".to_string()), "{bad}");
        }
    }

    fn named(names: &[&str]) -> Vec<Header> {
        names
            .iter()
            .map(|n| {
                let mut h = Header::new();
                h.set_name(n).expect("name");
                h
            })
            .collect()
    }

    fn names_of(hs: &[Header]) -> Vec<String> {
        hs.iter().map(|h| String::from_utf8_lossy(name_of(h).unwrap_or_default()).into_owned()).collect()
    }

    #[test]
    fn a_repeated_name_gets_its_index_and_a_counter() {
        let mut hs = named(&["a", "b", "a", "a"]);
        make_unique_names(&mut hs).expect("unique");
        assert_eq!(names_of(&hs), ["a", "b", "a_2_1", "a_3_1"]);
    }

    /// `a_2_1` already taken: the C++ stream keeps its text, so the retry is `a_2_1a_2_2`.
    #[test]
    fn a_clash_with_the_made_up_name_appends_rather_than_restarts() {
        let mut hs = named(&["a", "a_2_1", "a"]);
        make_unique_names(&mut hs).expect("unique");
        assert_eq!(names_of(&hs), ["a", "a_2_1", "a_2_1a_2_2"]);
    }

    #[test]
    fn empty_names_clash_like_any_other() {
        let mut hs = named(&["", "", ""]);
        make_unique_names(&mut hs).expect("unique");
        assert_eq!(names_of(&hs), ["", "_1_1", "_2_1"]);
    }

    #[test]
    fn a_header_without_a_name_is_a_software_error() {
        let mut hs = vec![Header::new()];
        assert_eq!(make_unique_names(&mut hs), Err("Software Error: header does not have a valid name".to_string()));
    }

    #[test]
    fn the_output_may_not_be_an_input() {
        let eq = |a: &str, b: &str| a == b;
        assert!(filename_check(["a.exr", "b.exr"], "b.exr", eq).is_err());
        assert!(filename_check(["a.exr", "b.exr"], "c.exr", eq).is_ok());
    }
}
