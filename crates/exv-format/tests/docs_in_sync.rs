//! `docs/FORMAT.md` vs the constants of this crate: the byte tables and enumerations of the spec must say what the
//! code writes, so a changed discriminant, offset or limit fails a test instead of silently outdating the spec.
//!
//! Plain `std` string parsing of the markdown (no dependency). Tables are located by their header cells, so a
//! renamed or reformatted table is reported as "table not found" (never silently skipped); every check names the
//! doc row and the code value that disagree. `drift_is_detected` mutates the real document in memory and demands a
//! failure for each class of check, so the checks themselves cannot rot into no-ops.

use exv_format::header::FRAME_COUNT_OFF;
use exv_format::limits::DEFAULT_MAX_ALLOC;
use exv_format::{
    Channel, CodecId, Dtype, Header, MAX_TILE_AREA, MAX_TILE_DIM, Meta, TAG_BC6H, TAG_DEFLATE, TAG_EXV_RT, TAG_HTJ2K, TAG_RANS, TAG_RAW,
    VERSION, classify_blob,
};

const MD: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/FORMAT.md"));

/// Every `Dtype`; the exhaustive match below stops compiling when a variant is added, forcing this list (and the doc) up to date.
const DTYPES: [Dtype; 6] = [Dtype::F16, Dtype::F32, Dtype::Bf16, Dtype::U8, Dtype::U16, Dtype::U32];

fn dtypes_are_exhaustive(d: Dtype) {
    match d {
        Dtype::F16 | Dtype::F32 | Dtype::Bf16 | Dtype::U8 | Dtype::U16 | Dtype::U32 => {}
    }
}

/// Cells of one markdown table row, trimmed and without backticks.
fn cells(line: &str) -> Vec<String> {
    line.trim().trim_matches('|').split('|').map(|c| c.trim().trim_matches('`').trim().to_string()).collect()
}

/// Body rows of the first table whose leading header cells equal `head` (separator row skipped).
fn rows(md: &str, head: &[&str]) -> Result<Vec<Vec<String>>, String> {
    let mut lines = md.lines();
    while let Some(l) = lines.next() {
        if !l.trim_start().starts_with('|') {
            continue;
        }
        let c = cells(l);
        if c.len() < head.len() || c.iter().zip(head).any(|(a, b)| a != b) {
            continue;
        }
        return Ok(lines
            .by_ref()
            .skip(1)
            .take_while(|l| l.trim_start().starts_with('|'))
            .map(cells)
            .collect());
    }
    Err(format!("FORMAT.md: table with header {head:?} not found"))
}

/// Unsigned decimal right after the first occurrence of `needle`.
fn num_after(md: &str, needle: &str) -> Result<u64, String> {
    let at = md.find(needle).ok_or_else(|| format!("FORMAT.md: `{needle}` not found"))? + needle.len();
    let digits: String = md[at..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().map_err(|_| format!("FORMAT.md: no number after `{needle}`"))
}

/// Compare a doc number with the code value, naming both on mismatch.
fn eq(errs: &mut Vec<String>, what: &str, doc: u64, code: u64) {
    if doc != code {
        errs.push(format!("{what}: FORMAT.md says {doc}, code says {code}"));
    }
}

/// §3 Dtype table: code, element size, float flag.
fn check_dtypes(md: &str, errs: &mut Vec<String>) -> Result<(), String> {
    let t = rows(md, &["Name", "Code", "Element size", "Float"])?;
    if t.len() != DTYPES.len() {
        errs.push(format!("Dtype table has {} rows, code has {} dtypes", t.len(), DTYPES.len()));
    }
    for d in DTYPES {
        let name = format!("{d:?}");
        let Some(r) = t.iter().find(|r| r[0] == name) else {
            errs.push(format!("Dtype {name} missing from the FORMAT.md §3 table"));
            continue;
        };
        eq(errs, &format!("Dtype {name} code"), r[1].parse::<u64>().unwrap_or(u64::MAX), u64::from(d.code()));
        eq(errs, &format!("Dtype {name} size"), r[2].parse::<u64>().unwrap_or(u64::MAX), d.size() as u64);
        if (r[3] == "yes") != d.is_float() {
            errs.push(format!("Dtype {name} float: FORMAT.md says {}, code says {}", r[3], d.is_float()));
        }
        if Dtype::from_code(d.code()) != Some(d) {
            errs.push(format!("Dtype {name}: from_code(code) does not round-trip"));
        }
    }
    if Dtype::from_code(DTYPES.len() as u8).is_some() {
        errs.push("a dtype code past the table decodes (the doc says any other code is corrupt)".into());
    }
    Ok(())
}

/// §4.1 blob tag table: tag value per `TAG_*` const, and that the doc lists exactly the tags `classify_blob` accepts.
fn check_tags(md: &str, errs: &mut Vec<String>) -> Result<(), String> {
    let t = rows(md, &["Tag", "Const", "Payload"])?;
    let consts = [
        ("TAG_RAW", TAG_RAW),
        ("TAG_RANS", TAG_RANS),
        ("TAG_DEFLATE", TAG_DEFLATE),
        ("TAG_HTJ2K", TAG_HTJ2K),
        ("TAG_EXV_RT", TAG_EXV_RT),
        ("TAG_BC6H", TAG_BC6H),
    ];
    for (name, v) in consts {
        match t.iter().find(|r| r[1] == name) {
            Some(r) => eq(errs, name, r[0].parse::<u64>().unwrap_or(u64::MAX), u64::from(v)),
            None => errs.push(format!("{name} missing from the FORMAT.md §4.1 table")),
        }
    }
    for tag in 0..=u8::MAX {
        let doc = t.iter().any(|r| r[0].parse::<u8>() == Ok(tag));
        if doc != classify_blob(&[tag]).is_some() {
            errs.push(format!("blob tag {tag}: documented = {doc}, classify_blob accepts = {}", !doc));
        }
    }
    Ok(())
}

/// §7 codec id table: id per `CodecId` const (+ the reserved 5), and that the doc lists exactly the ids `spec_name` knows.
fn check_codecs(md: &str, errs: &mut Vec<String>) -> Result<(), String> {
    let t = rows(md, &["Codec", "Id", "codec_params", "Profile"])?;
    let consts = [
        ("RAW", CodecId::RAW),
        ("LOSSLESS_FAST", CodecId::LOSSLESS_FAST),
        ("WAVELET53", CodecId::WAVELET53),
        ("WAVELET53_LOSSY", CodecId::WAVELET53_LOSSY),
        ("EXV_RT", CodecId::EXV_RT),
        ("HTJ2K", CodecId::HTJ2K),
        ("BC6H", CodecId::BC6H),
    ];
    for (name, c) in consts {
        match t.iter().find(|r| r[0] == name) {
            Some(r) => eq(errs, &format!("CodecId::{name}"), r[1].parse::<u64>().unwrap_or(u64::MAX), u64::from(c.0)),
            None => errs.push(format!("CodecId::{name} missing from the FORMAT.md §7 table")),
        }
    }
    // The reserved id is documented as `(5)` with no constant.
    if !t.iter().any(|r| r[0] == "(5)" && r[1] == "5") {
        errs.push("FORMAT.md §7 lost the reserved `(5)` row".into());
    }
    for id in 0..=16u32 {
        let doc = t.iter().any(|r| r[0] != "(5)" && r[1].parse::<u32>() == Ok(id));
        if doc != CodecId(id).spec_name().is_some() {
            errs.push(format!("codec id {id}: documented = {doc}, CodecId::spec_name knows it = {}", !doc));
        }
    }
    Ok(())
}

/// A header with distinct field values and an empty-or-meta block, as bytes.
fn header_bytes(meta: Option<&Meta>) -> Vec<u8> {
    let h = Header {
        width: 1000,
        height: 2000,
        frame_count: 3,
        tile_w: 64,
        tile_h: 32,
        codec_id: CodecId::HTJ2K.0,
        codec_params: vec![7, 8],
        channels: vec![Channel { name: "r".into(), dtype: Dtype::F16 }],
    };
    let mut out = Vec::new();
    let res = match meta {
        Some(m) => h.write_with_meta(m, &mut out),
        None => h.write(&mut out),
    };
    res.expect("header writes");
    out
}

/// §2.1 byte layout: offsets/sizes of the fixed fields against a really serialized header, `VERSION`,
/// `FRAME_COUNT_OFF` and the flags bit.
fn check_layout(md: &str, errs: &mut Vec<String>) -> Result<(), String> {
    let sec = md.split("### 2.1").nth(1).ok_or("FORMAT.md: §2.1 not found")?;
    let block = sec.split("```").nth(1).ok_or("FORMAT.md: §2.1 has no code block")?;
    // `offset size name : type ...`; the `..` (variable offset) rows do not parse and are skipped.
    let mut fields = Vec::new();
    for l in block.lines() {
        let t: Vec<&str> = l.split_whitespace().collect();
        let [off, size, name, ":", ty, ..] = t[..] else { continue };
        let (Ok(off), Ok(size)) = (off.parse::<usize>(), size.parse::<usize>()) else { continue };
        let want = match ty {
            "u8" => 1,
            "u16" => 2,
            "u32" => 4,
            _ => 0,
        };
        eq(errs, &format!("layout field {name} size vs type {ty}"), size as u64, want);
        fields.push((name, off, size, l));
    }
    let get = |n: &str| fields.iter().find(|f| f.0 == n).copied().ok_or_else(|| format!("FORMAT.md §2.1: field `{n}` missing"));
    let plain = header_bytes(None);
    let with_meta = header_bytes(Some(&Meta { fps: Some((24, 1)), ..Meta::default() }));

    let (_, off, _, line) = get("version")?;
    eq(errs, "version offset", off as u64, 0);
    if plain[off] != VERSION {
        errs.push(format!("version byte at offset {off} is {}, VERSION is {VERSION}", plain[off]));
    }
    let doc_ver = line.split("= 0x").nth(1).and_then(|s| u8::from_str_radix(s.get(..2)?, 16).ok());
    if doc_ver != Some(VERSION) {
        errs.push(format!("FORMAT.md says header version {doc_ver:?}, VERSION is {VERSION}"));
    }

    let (_, off, _, line) = get("flags")?;
    if plain[off] != 0 || with_meta[off] & 1 == 0 || !line.contains("bit0 = a Meta TLV") {
        errs.push(format!("flags byte (offset {off}): bit0 must be set exactly when a Meta block follows (doc: bit0 = Meta TLV)"));
    }

    for (name, want) in [("width", 1000u32), ("height", 2000), ("frame_count", 3), ("tile_w", 64), ("tile_h", 32), ("codec_id", 6)] {
        let (_, off, size, _) = get(name)?;
        let got = plain.get(off..off + size).and_then(|b| <[u8; 4]>::try_from(b).ok()).map(u32::from_le_bytes);
        if got != Some(want) {
            errs.push(format!("layout field {name}: bytes at offset {off} read {got:?}, header holds {want}"));
        }
    }
    let (_, off, size, _) = get("cp_len")?;
    if plain.get(off..off + size) != Some(&2u16.to_le_bytes()[..]) {
        errs.push(format!("layout field cp_len: bytes at offset {off} are not the u16 params length"));
    }
    eq(errs, "FRAME_COUNT_OFF", get("frame_count")?.1 as u64, FRAME_COUNT_OFF as u64);
    Ok(())
}

/// §2.2 Meta TLV table: tag and value length of each record, against a really serialized block.
fn check_meta(md: &str, errs: &mut Vec<String>) -> Result<(), String> {
    let t = rows(md, &["Tag", "Len", "Value"])?;
    let base = header_bytes(None).len();
    let one = |m: Meta| header_bytes(Some(&m))[base..].to_vec();
    let metas = [
        Meta { fps: Some((1, 1)), ..Meta::default() },
        Meta { primaries: 1, ..Meta::default() },
        Meta { alpha: 1, ..Meta::default() },
        Meta { pixel_aspect: Some((1, 1)), ..Meta::default() },
        Meta { timecode: Some(1), ..Meta::default() },
        Meta { exr_headers: vec![vec![0u8; 8]], ..Meta::default() },
    ];
    if t.len() != metas.len() {
        errs.push(format!("Meta table has {} rows, code writes {} record kinds", t.len(), metas.len()));
    }
    for (i, m) in metas.into_iter().enumerate() {
        // `meta_len:u32` then one `[tag:u8][len:u16][value]`.
        let b = one(m);
        let (tag, len) = (b.get(4).copied().unwrap_or(0), b.get(5).map_or(0, |&l| u16::from(l)));
        let Some(r) = t.get(i) else { continue };
        eq(errs, &format!("Meta record #{} tag", i + 1), r[0].parse::<u64>().unwrap_or(u64::MAX), u64::from(tag));
        eq(errs, &format!("Meta record #{} length", i + 1), r[1].parse::<u64>().unwrap_or(u64::MAX), u64::from(len));
    }
    Ok(())
}

/// Numeric limits quoted in the prose: tile edge / area caps and the default decode budget.
fn check_limits(md: &str, errs: &mut Vec<String>) -> Result<(), String> {
    eq(errs, "MAX_TILE_DIM", num_after(md, "MAX_TILE_DIM (")?, u64::from(MAX_TILE_DIM));
    eq(errs, "MAX_TILE_AREA", 1u64 << num_after(md, "MAX_TILE_AREA = 2^")?, MAX_TILE_AREA);
    eq(errs, "DEFAULT_MAX_ALLOC (GiB)", num_after(md, "`DEFAULT_MAX_ALLOC` = ")?, DEFAULT_MAX_ALLOC >> 30);
    Ok(())
}

/// One check: pushes mismatches, `Err` = the part of the doc it reads is missing.
type Step = fn(&str, &mut Vec<String>) -> Result<(), String>;

/// All checks over `md`; empty = in sync. A structural problem (table not found) is reported like a mismatch.
fn check(md: &str) -> Vec<String> {
    let mut errs = Vec::new();
    let steps: [Step; 6] = [check_dtypes, check_tags, check_codecs, check_layout, check_meta, check_limits];
    for step in steps {
        if let Err(e) = step(md, &mut errs) {
            errs.push(e);
        }
    }
    errs
}

#[test]
fn docs_in_sync() {
    DTYPES.into_iter().for_each(dtypes_are_exhaustive);
    let errs = check(MD);
    assert!(errs.is_empty(), "docs/FORMAT.md disagrees with exv-format:\n  {}", errs.join("\n  "));
}

/// Every class of check must fire on a doctored document (else it could be a silent no-op).
#[test]
fn drift_is_detected() {
    let mutations = [
        ("| `U32`  | 5 |", "| `U32`  | 6 |"),
        ("| `F16`  | 0 | 2 | yes |", "| `F16`  | 0 | 4 | yes |"),
        ("| 4 | `TAG_EXV_RT`", "| 5 | `TAG_EXV_RT`"),
        ("| `HTJ2K` | 6 |", "| `HTJ2K` | 7 |"),
        ("| `LOSSLESS_FAST` | 1 |", "| `LOSSLESS_FAST` | 9 |"),
        ("= 0x01 (VERSION)", "= 0x02 (VERSION)"),
        ("frame_count : u32", "frame_count : u16"),
        ("| 5 | 4 | `timecode:u32`", "| 5 | 8 | `timecode:u32`"),
        ("MAX_TILE_DIM (65536)", "MAX_TILE_DIM (65535)"),
        ("MAX_TILE_AREA = 2^30", "MAX_TILE_AREA = 2^29"),
    ];
    for (from, to) in mutations {
        assert!(MD.contains(from), "mutation anchor `{from}` no longer in FORMAT.md: update this test with the doc");
        assert!(!check(&MD.replacen(from, to, 1)).is_empty(), "doctoring `{from}` -> `{to}` was not detected");
    }
}
