//! Header attributes whose malformed VALUES OpenEXRCore 3.4.15 accepts or drops rather than
//! rejects (non-strict mode, what every C++ reader class selects): exr-core must open the file
//! the same way. Each test flips one byte of a committed golden and names the C++ verdicts it
//! was derived from, reproducible with
//! `target/parity-tools/exrcheck.exe [-c] <file>` and `crates/exr-core/tests/parity/exrheader.exe <file>`
//! (built by `cargo run -p xtask -- parity-exr` / `parity-tools`).
//!
//! * `lineOrder` byte >= 3: `extract_attr_uint8` fails the attribute, the loop drops it, the
//!   required-attribute pass defaults INCREASING_Y (`parse_header.c:1237-1245`, `validation.c:118-139`).
//! * `type` with any bytes: a string like any other, unknown values give storage UNKNOWN
//!   (`parse_header.c:1789-1819`); a plain part's flags win and the string is rewritten
//!   (`validation.c:513-547`).
//! * deep `version` != 1: stored as is; only decode-init refuses it (`decoding.c:307-318`).

use std::path::Path;

use exr_core::attr::LineOrder;
use exr_core::header::Header;
use exr_core::io::MemIStream;
use exr_core::scanline::InputFile;
use exr_core::MultiPartInputFile;

fn golden(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("parity").join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// `(value offset, declared size)` of the first attribute called `name` in a single-part header.
fn find_attr(bytes: &[u8], name: &str) -> (usize, usize) {
    let cstr = |from: usize| from + bytes[from..].iter().position(|&b| b == 0).expect("unterminated name");
    let mut p = 8;
    while bytes[p] != 0 {
        let name_end = cstr(p);
        let type_end = cstr(name_end + 1);
        let size = i32::from_le_bytes(bytes[type_end + 1..type_end + 5].try_into().expect("size")) as usize;
        if &bytes[p..name_end] == name.as_bytes() {
            return (type_end + 5, size);
        }
        p = type_end + 5 + size;
    }
    panic!("no attribute {name}");
}

/// `bytes` with `name`/`typ`/`value` appended to the header, the chunk offset table shifted to match.
fn add_attr(bytes: &[u8], name: &str, typ: &str, value: &[u8]) -> Vec<u8> {
    let mut end = 8;
    while bytes[end] != 0 {
        let name_end = end + bytes[end..].iter().position(|&b| b == 0).expect("name");
        let type_end = name_end + 1 + bytes[name_end + 1..].iter().position(|&b| b == 0).expect("type");
        let size = i32::from_le_bytes(bytes[type_end + 1..type_end + 5].try_into().expect("size")) as usize;
        end = type_end + 5 + size;
    }
    let mut attr = [name.as_bytes(), b"\0", typ.as_bytes(), b"\0"].concat();
    attr.extend_from_slice(&(value.len() as i32).to_le_bytes());
    attr.extend_from_slice(value);

    let table = end + 1;
    let first = u64::from_le_bytes(bytes[table..table + 8].try_into().expect("offset"));
    let chunks = (first as usize - table) / 8;
    let mut out = [&bytes[..end], &attr, &bytes[end..]].concat();
    for i in 0..chunks {
        let at = table + attr.len() + 8 * i;
        let off = u64::from_le_bytes(out[at..at + 8].try_into().expect("offset")) + attr.len() as u64;
        out[at..at + 8].copy_from_slice(&off.to_le_bytes());
    }
    out
}

/// Open `bytes` the way `MultiPartInputFile` does (autoAddType, every part validated).
fn open_with<R>(bytes: Vec<u8>, f: impl FnOnce(&mut MultiPartInputFile<'_>) -> R) -> exr_core::Result<R> {
    let mut s = MemIStream::new(bytes);
    let mut mp = MultiPartInputFile::new(&mut s)?;
    Ok(f(&mut mp))
}

fn line_order_of(bytes: Vec<u8>) -> LineOrder {
    open_with(bytes, |mp| mp.header(0).and_then(Header::line_order)).expect("opens").expect("lineOrder")
}

/// `golden_tiled_zip_one_decreasing.exr` carries DECREASING_Y. Oracle, with its `lineOrder` byte
/// set to 255 or 3: `exrcheck` and `exrcheck -c` say OK, `exrheader` prints `lineOrder (type
/// lineOrder): increasing y` (the intact file prints `decreasing y`).
#[test]
fn out_of_range_line_order_is_dropped_and_defaulted() {
    let good = golden("golden_tiled_zip_one_decreasing.exr");
    let (at, size) = find_attr(&good, "lineOrder");
    assert_eq!(size, 1);
    assert_eq!(line_order_of(good.clone()), LineOrder::DecreasingY, "the golden itself");
    for bad_value in [3u8, 99, 255] {
        let mut bytes = good.clone();
        bytes[at] = bad_value;
        assert_eq!(line_order_of(bytes), LineOrder::IncreasingY, "lineOrder byte {bad_value}");
    }
}

/// Same for a deep part (`golden_deep_composite_a.exr`, byte flipped to 0xff): `exrcheck -c` OK,
/// `exrcheck` OK, `exrheader` `lineOrder (type lineOrder): increasing y`.
#[test]
fn out_of_range_line_order_of_a_deep_part_is_dropped() {
    let mut bytes = golden("golden_deep_composite_a.exr");
    let (at, _) = find_attr(&bytes, "lineOrder");
    bytes[at] = 0xff;
    assert_eq!(line_order_of(bytes), LineOrder::IncreasingY);
}

/// Any attribute of type `lineOrder` is range-checked, not only the required one
/// (`parse_header.c:2162-2167`). Oracle, `golden_rgba.exr` plus a `lineOrder`-typed attribute:
/// value 9 is absent from `exrheader`'s listing, value 2 is listed as `random y`.
#[test]
fn out_of_range_custom_line_order_attribute_is_dropped() {
    let base = golden("golden_rgba.exr");
    let has = |value: u8| {
        let bytes = add_attr(&base, "myOrder", "lineOrder", &[value]);
        open_with(bytes, |mp| mp.header(0).map(|h| h.contains("myOrder"))).expect("opens").expect("header")
    };
    assert!(has(2));
    assert!(!has(9));
}

/// `golden_deep_composite_a.exr` with a byte of the `deepscanline` string set to 0xff (and 0x80 in
/// the string's first byte): `exrcheck -c` OK, `exrheader` prints the string verbatim; plain
/// `exrcheck` says bad because the API path cannot open an unknown type. So the file opens, the
/// part is of UNKNOWN storage, the bytes stay as they were, and only asking for a deep reader fails.
#[test]
fn non_utf8_type_of_a_deep_part_is_unknown_storage() {
    let good = golden("golden_deep_composite_a.exr");
    let (at, size) = find_attr(&good, "type");
    assert_eq!(&good[at..at + size], b"deepscanline");
    for (byte, mask) in [(3usize, 0xffu8), (0, 0x80)] {
        let mut bytes = good.clone();
        bytes[at + byte] ^= mask;
        let want = bytes[at..at + size].to_vec();
        let (raw, deep) = open_with(bytes, |mp| {
            let raw = mp.header(0).expect("header").type_bytes().expect("type").to_vec();
            (raw, mp.deep_scanline_input_part(0).is_err())
        })
        .expect("opens like C++");
        assert_eq!(raw, want);
        assert!(deep, "a deep reader must be refused for the unknown type");
    }
}

/// A plain single-part file whose header carries a `type` (the flags say scanline): C++ rewrites a
/// `type` that disagrees with the flags, whatever its bytes (`validation.c:522-547`). Oracle,
/// `golden_rgba.exr` plus `type = "scanlineimage"` with its third byte xor 0xff / 0x80 / 0x01:
/// `exrcheck` and `exrcheck -c` OK, `exrheader` prints `type (type string): "scanlineimage"`.
#[test]
fn non_utf8_type_of_a_plain_part_is_rewritten_from_the_flags() {
    let base = add_attr(&golden("golden_rgba.exr"), "type", "string", b"scanlineimage");
    let (at, _) = find_attr(&base, "type");
    for mask in [0xffu8, 0x80, 0x01] {
        let mut bytes = base.clone();
        bytes[at + 2] ^= mask;
        let mut s = MemIStream::new(bytes);
        let inf = InputFile::new(&mut s).expect("opens like C++");
        assert_eq!(inf.header().type_().expect("utf-8 again"), "scanlineimage", "^{mask:#x}");
    }
}

/// `golden_deep_composite_a.exr` with its deep `version` attribute changed (byte 0 of the value
/// xor 0x01 gives 0): the header parse stores it as it is, `exrcheck -c` OK. The C++ refuses it
/// where pixels are decoded (`decoding.c:307-318`: "Version 0 not supported for deepscanline
/// images"), so plain `exrcheck` says bad, and so does asking for our deep reader.
#[test]
fn deep_version_other_than_one_opens_and_the_reader_refuses_it() {
    let mut bytes = golden("golden_deep_composite_a.exr");
    let (at, size) = find_attr(&bytes, "version");
    assert_eq!(size, 4);
    bytes[at] ^= 0x01;
    let (version, refused) = open_with(bytes, |mp| {
        let v = mp.header(0).expect("header").version().expect("version");
        (v, mp.deep_scanline_input_part(0).err().map(|e| e.to_string()))
    })
    .expect("opens like C++");
    assert_eq!(version, 0);
    let msg = refused.expect("the reader must refuse");
    assert!(msg.contains("Version 0 not supported"), "{msg}");
}

/// `compression` byte >= 12: dropped, the required pass defaults ZIP (`parse_header.c:1030-1038`,
/// `validation.c:24-39`). Oracle, `golden_tiled_piz_one.exr` with the byte set to 200: `exrheader`
/// prints `compression (type compression): zip: zlib compression, in blocks of 16 scan lines.`
/// (the intact file prints `piz`); `exrcheck -c` on `golden_rgba.exr` with it at 0xff/0x80: OK.
#[test]
fn out_of_range_compression_is_dropped_and_defaulted() {
    use exr_core::attr::Compression;
    let good = golden("golden_tiled_piz_one.exr");
    let (at, size) = find_attr(&good, "compression");
    assert_eq!(size, 1);
    let comp = |bytes: Vec<u8>| open_with(bytes, |mp| mp.header(0).and_then(Header::compression)).expect("opens").expect("compression");
    assert_eq!(comp(good.clone()), Compression::Piz);
    for bad_value in [12u8, 200, 255] {
        let mut bytes = good.clone();
        bytes[at] = bad_value;
        assert_eq!(comp(bytes), Compression::Zip, "compression byte {bad_value}");
    }
}

/// A type name that is not UTF-8 is an unregistered type like any other: its absurd declared size
/// drops the attribute without consuming the value and parsing resumes there. Oracle,
/// `golden_tiled_zip_one_randomy.exr` with the NUL that ends the `owner` attribute's type name
/// (`string`) xor 0xff: `exrcheck` and `exrcheck -c` OK.
#[test]
fn non_utf8_type_name_with_a_bad_size_resumes_parsing() {
    let good = golden("golden_tiled_zip_one_randomy.exr");
    let (value, _) = find_attr(&good, "owner");
    let nul = value - 5; // the byte before the 4-byte size
    assert_eq!(good[nul], 0);
    let mut bytes = good;
    bytes[nul] ^= 0xff;
    let parts = open_with(bytes, |mp| (mp.parts(), mp.header(0).map(|h| h.contains("tiles")))).expect("opens like C++");
    assert_eq!(parts.0, 1);
    assert!(parts.1.expect("header"));
}
