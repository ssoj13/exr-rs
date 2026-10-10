//! Source guard (test only): no production code in the workspace creates a file with
//! `StdOfStream::create` outside `exr_core::io`.
//!
//! Every file a tool or library writes goes through `exr_core::io::write_file`, which writes a
//! temporary file, checks the flush and renames it, so a failing run leaves no half-written output
//! (outputs are all-or-nothing). A direct `StdOfStream::create` elsewhere
//! would silently bypass that. The scan is lexical (comments, strings and `#[cfg(test)]` items are
//! skipped), so it needs no parser dependency and fails with `file:line` for each offender.

use std::fs;
use std::path::{Path, PathBuf};

/// `true` for every byte of `src` that is code: not inside a comment, a string or a char literal.
fn code_mask(src: &str) -> Vec<bool> {
    let mut code = vec![true; src.len()];
    let mut i = 0;
    while i < src.len() {
        match non_code_end(src, i) {
            Some(end) => {
                code[i..end].fill(false);
                i = end;
            }
            None => i += 1,
        }
    }
    code
}

/// If a comment, string or char literal starts at byte `i`, the index just after it.
fn non_code_end(src: &str, i: usize) -> Option<usize> {
    let b = src.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    match b[i] {
        b'/' if b.get(i + 1) == Some(&b'/') => Some(src[i..].find('\n').map_or(b.len(), |n| i + n)),
        b'/' if b.get(i + 1) == Some(&b'*') => {
            let (mut depth, mut j) = (0, i);
            while j < b.len() {
                if b[j..].starts_with(b"/*") {
                    depth += 1;
                    j += 2;
                } else if b[j..].starts_with(b"*/") {
                    depth -= 1;
                    j += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            Some(j)
        }
        b'"' => Some(skip_string(b, i + 1, None)),
        // A raw string: `r` or `br`, `#`s, a quote (and `r` not the tail of an identifier).
        b'r' if i == 0 || !ident(b[i - 1]) || (b[i - 1] == b'b' && (i == 1 || !ident(b[i - 2]))) => {
            let hashes = b[i + 1..].iter().take_while(|&&c| c == b'#').count();
            (b.get(i + 1 + hashes) == Some(&b'"')).then(|| skip_string(b, i + 2 + hashes, Some(hashes)))
        }
        // A char literal is `'x'` or `'\..'`; anything else is a lifetime.
        b'\'' => {
            let mut chars = src[i + 1..].chars();
            match chars.next() {
                // The closing quote is searched after the escaped char (`'\''` escapes a quote).
                Some('\\') => Some(src[i + 3..].find('\'').map_or(b.len(), |n| i + 3 + n + 1)),
                Some(c) if chars.next() == Some('\'') => Some(i + 1 + c.len_utf8() + 1),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The index after the string whose body starts at `i`: a plain string ends at an unescaped quote,
/// a raw one (`raw` = its `#` count) at a quote followed by that many `#`.
fn skip_string(b: &[u8], mut i: usize, raw: Option<usize>) -> usize {
    while i < b.len() {
        match (raw, b[i]) {
            (None, b'\\') => i += 1,
            (None, b'"') => return i + 1,
            (Some(n), b'"') if b[i + 1..].iter().take_while(|&&c| c == b'#').count() >= n => return i + 1 + n,
            _ => {}
        }
        i += 1;
    }
    b.len()
}

/// The end (exclusive) of the `{ .. }` block that opens at `open`, counting only code braces.
fn block_end(b: &[u8], code: &[bool], open: usize) -> usize {
    let mut depth = 0;
    for i in open..b.len() {
        if !code[i] {
            continue;
        }
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    b.len()
}

/// What the scan of one file found.
#[derive(Debug, Default, PartialEq)]
struct Scan {
    /// 1-based lines of production uses of `StdOfStream::create` (or an alias of the type).
    hits: Vec<usize>,
    /// Names of `#[cfg(test)] mod name;` files: their code is test code too.
    test_mods: Vec<String>,
    /// `#![cfg(test)]`: the whole file is test code.
    all_test: bool,
}

/// Scan one source text.
fn scan(src: &str) -> Scan {
    let b = src.as_bytes();
    let code = code_mask(src);
    let mut out = Scan::default();
    // Byte ranges that are test code.
    let mut test: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if code[i] && b[i..].starts_with(b"#![cfg(test)]") {
            out.all_test = true;
        }
        if code[i] && b[i..].starts_with(b"#[cfg(test)]") {
            let mut j = i + b"#[cfg(test)]".len();
            while j < b.len() && !(code[j] && (b[j] == b'{' || b[j] == b';')) {
                j += 1;
            }
            let end = if b.get(j) == Some(&b'{') { block_end(b, &code, j) } else { (j + 1).min(b.len()) };
            if b.get(j) == Some(&b';') {
                let item = src[i + b"#[cfg(test)]".len()..j].split_whitespace().collect::<Vec<_>>();
                if let Some(at) = item.iter().position(|w| *w == "mod") {
                    out.test_mods.extend(item.get(at + 1).map(|n| n.to_string()));
                }
            }
            test.push((i, end));
            i = end;
            continue;
        }
        i += 1;
    }
    let mut at = 0;
    while let Some(p) = src[at..].find("StdOfStream") {
        let s = at + p;
        at = s + "StdOfStream".len();
        let word_start = s == 0 || !(b[s - 1].is_ascii_alphanumeric() || b[s - 1] == b'_');
        if !code[s] || !word_start || test.iter().any(|&(lo, hi)| (lo..hi).contains(&s)) {
            continue;
        }
        let tail: String = src[at..].split_whitespace().collect::<Vec<_>>().join(" ");
        let tail = tail.replace(" ::", "::").replace(":: ", "::");
        if tail.starts_with("::create") || tail.starts_with("as ") {
            out.hits.push(src[..s].matches('\n').count() + 1);
        }
    }
    out
}

/// Every `.rs` file under `dir`, `target` and hidden directories excluded.
fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rs_files(&p, out);
            }
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A call in production code is a hit; the same call in a comment, a string, a raw string or a
    /// `#[cfg(test)]` block is not; an alias of the type is a hit (it would dodge the call check).
    #[test]
    fn scan_finds_production_calls_only() {
        let prod = "fn f() {\n    let s = StdOfStream::create(p)?;\n}\n";
        assert_eq!(scan(prod).hits, [2]);
        assert_eq!(scan("fn f() { exr_core::io::StdOfStream :: create(p) }").hits, [1]);
        assert_eq!(scan("use exr_core::io::StdOfStream as W;").hits, [1]);
        for quiet in [
            "// StdOfStream::create(p)\n",
            "/* a /* StdOfStream::create(p) */ b */\n",
            "const S: &str = \"StdOfStream::create(p)\";\n",
            "const S: &str = r#\"StdOfStream::create(\"p\")\"#;\n",
            "/// doc: StdOfStream::create(p)\nfn f() {}\n",
            "#[cfg(test)]\nmod tests {\n    fn t() { StdOfStream::create(p).unwrap(); }\n}\n",
            "pub use io::{StdIfStream, StdOfStream};\n",
            "struct MyStdOfStream; fn f() { MyStdOfStream::create(); }\n",
        ] {
            assert_eq!(scan(quiet).hits, Vec::<usize>::new(), "{quiet}");
        }
        // Production code after a closed test block counts again; lifetimes and chars do not
        // confuse the block matching.
        let after = "#[cfg(test)]\nmod t {\n    fn f<'a>(c: char) -> &'a str { if c == '}' { \"}\" } else { \"{\" } }\n}\nfn g() { StdOfStream::create(p); }\n";
        assert_eq!(scan(after).hits, [5]);
    }

    #[test]
    fn scan_reports_test_only_files() {
        assert_eq!(scan("#[cfg(test)]\nmod tests;\nmod other;\n").test_mods, ["tests"]);
        assert!(scan("#![cfg(test)]\n").all_test);
    }

    /// The guard itself: every `.rs` file of the workspace (its crates' `src`, the root `src`, the
    /// `xtask`), the test-only code skipped, has no `StdOfStream::create` outside
    /// `exr_core::io`.
    #[test]
    fn no_production_stdofstream_create_outside_exr_core_io() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut roots = vec![root.join("src"), root.join("xtask").join("src")];
        if let Ok(rd) = fs::read_dir(root.join("crates")) {
            roots.extend(rd.flatten().map(|e| e.path().join("src")));
        }
        let mut files = Vec::new();
        for r in &roots {
            rs_files(r, &mut files);
        }
        assert!(files.len() > 100, "the scan found only {} files: wrong roots?", files.len());

        let allowed = root.join("crates").join("exr-core").join("src").join("io");
        let mut test_files = Vec::new();
        let mut hits = Vec::new();
        for f in files.iter().filter(|f| !f.starts_with(&allowed)) {
            let s = scan(&fs::read_to_string(f).unwrap_or_default());
            let dir = f.parent().unwrap_or(root);
            for m in &s.test_mods {
                test_files.push(dir.join(format!("{m}.rs")));
                test_files.push(dir.join(m));
            }
            if s.all_test {
                test_files.push(f.clone());
            }
            hits.extend(s.hits.iter().map(|l| (f.clone(), *l)));
        }
        let bad: Vec<String> = hits
            .into_iter()
            .filter(|(f, _)| !test_files.iter().any(|t| f.starts_with(t)))
            .map(|(f, l)| format!("{}:{l}", f.strip_prefix(root).unwrap_or(&f).display()))
            .collect();
        assert!(bad.is_empty(), "StdOfStream::create outside exr_core::io (use io::write_file): {bad:?}");
    }
}
