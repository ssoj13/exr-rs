//! Tests of [`super`]: goldens of every storage kind, damaged input, limit restoration, the C++
//! quirks that are reproduced on purpose, and (behind `test-data`) the whole private image set.
//! Agreement with the C++ `exrcheck` itself is `tests/parity_tool_check.rs` at the workspace root.

use super::*;
use std::path::{Path, PathBuf};

/// A committed golden (written by, and cross-read against, C++ OpenEXR 3.4.15).
fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../exr-core/tests/parity").join(name)
}

fn opts(m: bool, t: bool, c: bool) -> CheckOpts {
    CheckOpts { reduce_memory: m, reduce_time: t, core: c }
}

/// Every flag combination.
fn all_opts() -> Vec<CheckOpts> {
    (0..8).map(|i| opts(i & 1 != 0, i & 2 != 0, i & 4 != 0)).collect()
}

fn path_str(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

/// Goldens of every storage kind read clean through every mode, from the file and from memory.
#[test]
fn goldens_are_ok_in_every_mode() {
    for name in [
        "golden.exr",
        "golden_rgba.exr",
        "golden_piz.exr",
        "golden_dwab.exr",
        "golden_htj2k256.exr",
        "golden_sub_dwaa_x2y2.exr",
        "golden_sub_b44_x1y2.exr",
        "golden_deep_tiled_zips_mipmap.exr",
        "golden_deep_tiled_none_one.exr",
        "golden_deep_composite_a.exr",
        "golden_deep_composite_b.exr",
    ] {
        let path = golden(name);
        let bytes = std::fs::read(&path).expect("golden");
        for o in all_opts() {
            assert!(!check_file(path_str(&path), o), "{name} {o:?} (file)");
            assert!(!check_bytes(&bytes, o), "{name} {o:?} (memory)");
        }
    }
}

/// `exrcheck.exe` (C++ 3.4.15) calls the mixed multipart golden bad (`tests/parity_tool_check.rs` compares every
/// golden with the C++), and so do we.
#[test]
fn goldens_the_cpp_calls_bad_are_bad() {
    let path = golden("golden_multipart.exr");
    assert!(check_file(path_str(&path), opts(false, false, false)));
}

/// Better than the C++: the flat readers of `exrcheck` composite a deep scanline part
/// and throw when it has no `Z` or no `A` channel (`ImfCompositeDeepScanLine.cpp:103-123`), which
/// `ImfCheckFile.cpp:898/1176/1189` take for damage, so the C++ calls these valid goldens bad. Here they
/// are OK in every mode, while damage to one of them is still found (`flipped_payload_bytes`).
#[test]
fn deep_scanline_files_without_z_or_alpha_are_ok() {
    for name in ["golden_deep_scanline_none.exr", "golden_deep_scanline_zips.exr", "golden_deep_scanline_rle.exr"] {
        let path = golden(name);
        for o in all_opts() {
            assert!(!check_file(path_str(&path), o), "{name} {o:?}");
        }
    }
}

/// Damaged files: empty, garbage, wrong magic, cut in the pixel data, cut in the header, missing.
#[test]
fn damaged_files_are_bad() {
    let good = std::fs::read(golden("golden_piz.exr")).expect("golden");
    let mut bad_magic = good.clone();
    bad_magic[0] ^= 0xff;
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("garbage", (0..4096u32).map(|i| (i * 31 % 251) as u8).collect()),
        ("magic", bad_magic),
        ("cut pixel data", good[..good.len() - 40].to_vec()),
        ("cut header", good[..100].to_vec()),
    ];
    for (what, data) in cases {
        for o in all_opts() {
            assert!(check_bytes(&data, o), "{what} {o:?}");
        }
    }
    assert!(check_file("no/such/file.exr", opts(false, false, false)));
}

/// 20 flipped bytes inside the compressed pixel data. Verdicts taken from `exrcheck.exe`: codecs whose
/// decoder validates its stream (PXR24, DWAB, HTJ2K, deep ZIPS) make the file bad in every mode, while
/// PIZ, RLE and B44 decode the damaged bytes without noticing.
#[test]
fn flipped_payload_bytes() {
    let flip = |name: &str| {
        let mut d = std::fs::read(golden(name)).expect("golden");
        let n = d.len();
        d[n - 120..n - 100].iter_mut().for_each(|b| *b ^= 0xff);
        d
    };
    for name in [
        "golden_pxr24.exr",
        "golden_dwab.exr",
        "golden_htj2k256.exr",
        "golden_deep_tiled_zips_one.exr",
        "golden_deep_scanline_zips.exr",
    ] {
        let data = flip(name);
        for o in all_opts() {
            assert!(check_bytes(&data, o), "{name} {o:?}");
        }
    }
    for name in ["golden_piz.exr", "golden_rle.exr", "golden_b44.exr"] {
        let data = flip(name);
        for o in all_opts() {
            assert!(!check_bytes(&data, o), "{name} {o:?}");
        }
    }
}

/// Damage only in the deep sample *data* (the last 4 bytes of a deep file are the Adler-32 of its last
/// data stream; the count table is a separate stream before it): only the data pass of the deep engines
/// can notice. Each engine is run directly, so no other read path can mask a missing data pass.
#[test]
fn deep_engines_read_the_sample_data() {
    let data = |name: &str, damage: bool| {
        let mut d = std::fs::read(golden(name)).expect("golden");
        if damage {
            let n = d.len();
            d[n - 4..].iter_mut().for_each(|b| *b ^= 0xff);
        }
        d
    };
    for guard in [Guard::Api, Guard::Core] {
        for o in [opts(false, false, false), opts(true, false, false)] {
            for (name, tiled) in [("golden_deep_tiled_zips_one.exr", true), ("golden_deep_scanline_zips.exr", false)] {
                let verdicts: Vec<bool> = [false, true]
                    .into_iter()
                    .map(|damage| {
                        let bytes = data(name, damage);
                        let mut s = PtrIStream::new(&bytes);
                        if tiled {
                            read_deep_tile(&mut DeepTiledInputFile::new(&mut s).expect("open"), o, guard)
                        } else {
                            read_deep_scanline(&mut DeepScanLineInputFile::new(&mut s).expect("open"), o, guard)
                        }
                    })
                    .collect();
                assert_eq!(verdicts, [false, true], "{name} {o:?}");
            }
        }
    }
}

/// The path and memory variants give one verdict, also on damaged input.
#[test]
fn path_and_memory_agree_on_truncations() {
    let good = std::fs::read(golden("golden_multipart.exr")).expect("golden");
    let dir = std::env::temp_dir().join(format!("exr-util-check-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    for cut in (0..good.len()).step_by(good.len() / 23 + 1) {
        let file = dir.join("cut.exr");
        std::fs::write(&file, &good[..cut]).expect("write");
        for o in all_opts() {
            assert_eq!(check_file(path_str(&file), o), check_bytes(&good[..cut], o), "cut at {cut} {o:?}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// `runChecks` restores the process-global limits it changes.
#[test]
fn limits_are_restored() {
    let read = || (get_maximum_sample_count(), Header::max_image_size(), Header::max_tile_size());
    let lock = CHECK_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let before = read();
    drop(lock);
    let path = golden("golden.exr");
    for o in all_opts() {
        check_file(path_str(&path), o);
        let _lock = CHECK_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(before, read(), "{o:?}");
    }
}

/// Reduce mode caps the image, tile and deep sample sizes exactly as `runChecks` does.
#[test]
fn reduce_mode_sets_the_caps() {
    let _lock = CHECK_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let saved = Saved::apply(opts(true, false, false));
    assert_eq!(Header::max_image_size(), (2048, 2048));
    assert_eq!(Header::max_tile_size(), (512, 512));
    assert_eq!(get_maximum_sample_count(), 1 << 20);
    drop(saved);
    let saved = Saved::apply(opts(false, false, false));
    assert_eq!(Header::max_tile_size(), (16384, 16384));
    drop(saved);
}

/// `readTile`'s missing `break` (ImfCheckFile.cpp:346-357): channels `i % 3 == 1` are UINT, not FLOAT.
#[test]
fn tile_frame_buffer_reproduces_the_fallthrough() {
    let bytes = std::fs::read(golden("golden_rgba.exr")).expect("golden");
    let mut s = PtrIStream::new(&bytes);
    let multi = MultiPartInputFile::new(&mut s).expect("open");
    let h = multi.header(0).expect("header");
    let types = |quirk: bool| -> Vec<PixelType> {
        let (fb, _) = flat_fb(h, 8, 0, quirk).expect("fb");
        fb.iter().map(|(_, s)| s.type_).collect()
    };
    let n = types(false).len();
    assert!(n >= 2, "{n} channels");
    for (i, (plain, quirk)) in types(false).into_iter().zip(types(true)).enumerate() {
        assert_eq!(plain == PixelType::Float, i % 3 == 1, "channel {i}");
        assert_ne!(quirk, PixelType::Float, "channel {i}");
        assert_eq!(quirk == PixelType::Uint, i % 3 != 0, "channel {i}");
    }
}

/// The in-memory stream refuses reads and seeks past the end like `PtrIStream`.
#[test]
fn ptr_stream_bounds() {
    let data = [1u8, 2, 3, 4];
    let mut s = PtrIStream::new(&data);
    let mut b = [0u8; 3];
    assert!(s.read(&mut b).is_ok());
    assert!(s.read(&mut b).is_err());
    assert!(s.seekg(4).is_ok());
    assert!(s.seekg(5).is_err());
    assert_eq!(s.size(), Some(4));
}

#[test]
fn extent_sign_extends_like_c() {
    assert_eq!(extent(-3, 4), 8);
    assert_eq!(extent(0, -1), 0);
}

/// The private image set: file and memory verdicts agree, and the well-formed openexr-images
/// directories are OK in the default mode (the reduce modes cap the image size at 2048, which some
/// of them exceed).
#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn image_set() {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).expect("dir").flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "exr") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&exr_test_data::openexr_images(), &mut files);
    walk(&exr_test_data::openexr_test(), &mut files);
    assert!(files.len() > 100, "{} files", files.len());
    let clean = ["ScanLines", "Tiles", "MultiResolution", "Chromaticities", "DisplayWindow"];
    for f in &files {
        let bytes = std::fs::read(f).expect("read");
        for o in [opts(false, false, false), opts(true, true, false), opts(false, false, true)] {
            let by_path = check_file(path_str(f), o);
            assert_eq!(by_path, check_bytes(&bytes, o), "{} {o:?}", f.display());
            let in_clean_dir = f.components().any(|c| clean.iter().any(|d| c.as_os_str() == *d));
            if in_clean_dir && !o.core && !o.reduce_memory && !o.reduce_time {
                assert!(!by_path, "{} {o:?}", f.display());
            }
        }
    }
}

/// The deep limits of `ImfCheckFile.cpp` at their boundaries. Their effect on a verdict cannot be shown
/// with a file (the flat read of a deep scanline part sees the same damage, and the core path decodes either
/// way), so the predicates are pinned directly.
#[test]
fn deep_limits_sit_at_their_c_values() {
    assert!(core_decodes(1) && core_decodes(4095));
    assert!(!core_decodes(0) && !core_decodes(4096), "no samples or >= 4096 bytes: decoded into nowhere");
    assert!(api_line_fits(2047, 2048) && !api_line_fits(2048, 2048), "`fileBufferSize + bufferSize < 4096`");
    let bought = |c: u32, limit, strict| Unit::of([c].into_iter(), 1, 10, true, strict, limit).buf;
    // tiles (`:722`): a pixel is buffered while `count * bytesPerSample < 4096`
    assert_eq!((bought(409, MAX_BYTES_PER_DEEP_SCANLINE, true), bought(410, MAX_BYTES_PER_DEEP_SCANLINE, true)), (409, 0));
    // scanlines (`:521`): while `count * bytesPerSample <= 1000`
    assert_eq!((bought(100, MAX_BYTES_PER_DEEP_PIXEL, false), bought(101, MAX_BYTES_PER_DEEP_PIXEL, false)), (100, 0));
}

/// Better than the C++: `readDeepTile` asks for the tiles
/// `[min (x, y), max (x, y)]` squared (`ImfCheckFile.cpp:698`) and so calls a valid deep tiled file with more
/// tile columns than rows bad; here the single tile is requested and the file is OK, in the API and the
/// core path. Pinned on a saved file: 40 x 8 pixels in 8 x 4 tiles is 5 x 2 tiles.
#[test]
fn a_valid_wide_deep_tiled_file_is_ok_to_the_api_path() {
    use crate::deep_image::DeepImage;
    use crate::deep_image_io::save_deep_image;
    use crate::image_data_window::DataWindowSource;
    use exr_core::attr::{Compression, LevelRoundingMode, PixelType, TileDescription};
    use imath_rs::V2i;
    let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 39, y: 7 } };
    let mut img = DeepImage::new(&dw, LevelMode::OneLevel, LevelRoundingMode::RoundDown).unwrap();
    img.insert_type("Z", PixelType::Float).unwrap();
    img.level_mut(0).unwrap().edit_counts(|c| c.fill(1)).unwrap();
    let mut hdr = Header::new();
    *hdr.compression_mut().unwrap() = Compression::Zips;
    hdr.set_tile_description(TileDescription { x_size: 8, y_size: 4, ..Default::default() }).unwrap();
    let path = std::env::temp_dir().join(format!("exr-util-wide-deep-{}.exr", std::process::id()));
    save_deep_image(&path, Some(&hdr), &img, DataWindowSource::Image).unwrap();
    let (api, core) = (check_file(path_str(&path), opts(false, false, false)), check_file(path_str(&path), opts(false, false, true)));
    std::fs::remove_file(&path).unwrap();
    assert_eq!((api, core), (false, false));
}

/// Byte offset of the value of attribute `name` in the first header of `d`.
fn attr_at(d: &[u8], name: &str) -> usize {
    let size = |i: usize| i32::from_le_bytes(d[i..i + 4].try_into().unwrap()) as usize;
    let mut i = 8; // after magic and version word
    while d[i] != 0 {
        let n = d[i..].iter().position(|&b| b == 0).unwrap();
        let t = d[i + n + 1..].iter().position(|&b| b == 0).unwrap();
        let at = i + n + 1 + t + 1;
        if &d[i..i + n] == name.as_bytes() {
            return at + 4;
        }
        i = at + 4 + size(at);
    }
    panic!("no attribute {name}");
}

/// Damage only the `type` or `version` attribute of a golden. OpenEXRCore takes any byte string as a
/// part type (an unknown one is an unknown storage, `parse_header.c:1789-1819`), so a non-UTF-8 `type` must
/// not fail the core check; a deep scanline part with `version != 1` is ignored by it (`decoding.c:310`
/// refuses it in `exr_decoding_initialize`, `ImfCheckFile.cpp:1385-1386` ignores that), a deep tiled one
/// is bad (`:1554-1558`). Verdicts `(default, core)`, `true` = bad, taken from `exrcheck.exe`
/// (`tests/parity_tool_check.rs::type_and_version_damage_matches_exrcheck` compares them on every run).
#[test]
fn type_and_version_damage_verdicts() {
    let cases: [(&str, &str, u8, bool, bool); 8] = [
        ("golden_deep_scanline_zips.exr", "type", 0xff, true, false),
        ("golden_deep_scanline_zips.exr", "type", b'x', true, false),
        ("golden_deep_scanline_zips.exr", "version", 2, true, false),
        ("golden_deep_tiled_zips_one.exr", "type", 0xff, true, false),
        ("golden_deep_tiled_zips_one.exr", "type", b'x', true, false),
        ("golden_deep_tiled_zips_one.exr", "version", 2, true, true),
        ("golden_multipart.exr", "type", 0xff, true, false),
        ("golden_multipart.exr", "type", b'x', true, false),
    ];
    for (name, attr, byte, default_bad, core_bad) in cases {
        let mut d = std::fs::read(golden(name)).expect("golden");
        let at = attr_at(&d, attr);
        d[at] = byte;
        let got = (check_bytes(&d, opts(false, false, false)), check_bytes(&d, opts(false, false, true)));
        assert_eq!(got, (default_bad, core_bad), "{name} {attr}={byte:#x}");
    }
}

// Small real files keep guard mutations safe while measuring work before allocation.
fn flat_fixture(w: i32, h: i32, channels: usize, compression: exr_core::attr::Compression) -> Vec<u8> {
    use exr_core::api::{ChannelData, Image};
    let mut image = Image::new(Box2i {
        min: imath_rs::V2i { x: 0, y: 0 },
        max: imath_rs::V2i { x: w - 1, y: h - 1 },
    });
    for c in 0..channels {
        image.insert_channel(format!("C{c}"), ChannelData::Half(vec![imath_rs::Half::from_f32(0.5); (w * h) as usize])).unwrap();
    }
    let mut stream = exr_core::io::MemOStream::new();
    image.write_to(&mut stream, compression).unwrap();
    stream.data().to_vec()
}

fn flat_table_start(d: &[u8]) -> usize {
    let mut i = 8;
    while d[i] != 0 {
        let n = d[i..].iter().position(|&b| b == 0).unwrap();
        let t = d[i + n + 1..].iter().position(|&b| b == 0).unwrap();
        let at = i + n + 1 + t + 1;
        let size = i32::from_le_bytes(d[at..at + 4].try_into().unwrap()) as usize;
        i = at + 4 + size;
    }
    i + 1
}

fn flat_complete(d: &[u8]) -> bool {
    let mut stream = PtrIStream::new(d);
    MultiPartInputFile::new(&mut stream).unwrap().part_complete(0).unwrap()
}

fn reset_flat_work() {
    FLAT_READER_ATTEMPTS.with(|n| n.set(0));
    FALLBACK_READER_ATTEMPTS.with(|n| n.set(0));
    CORE_CHUNK_ATTEMPTS.with(|n| n.set(0));
}

#[test]
fn incomplete_flat_table_stops_before_reader_allocation() {
    let mut d = flat_fixture(8, 3, 1, exr_core::attr::Compression::NoCompression);
    let table = flat_table_start(&d);
    d[table..table + 3 * 8].fill(0);
    d.truncate(table + 3 * 8);
    assert!(!flat_complete(&d));
    for core in [false, true] {
        reset_flat_work();
        assert!(check_bytes(&d, opts(false, false, core)));
        assert_eq!(FLAT_READER_ATTEMPTS.with(|n| n.get()), 0, "core={core}: no per-height reader metadata");
        assert_eq!(FALLBACK_READER_ATTEMPTS.with(|n| n.get()), 0, "bad verdict cannot reopen a reader");
    }
}

#[test]
fn recovered_flat_table_is_checked_normally() {
    let mut d = flat_fixture(8, 3, 1, exr_core::attr::Compression::NoCompression);
    let table = flat_table_start(&d);
    d[table..table + 3 * 8].fill(0);
    assert!(flat_complete(&d), "all chunks reconstruct the complete table");
    for core in [false, true] {
        reset_flat_work();
        assert!(!check_bytes(&d, opts(false, false, core)));
        assert!(FLAT_READER_ATTEMPTS.with(|n| n.get()) > 0);
    }
}

#[test]
fn reduce_memory_still_skips_wide_incomplete_flat_parts() {
    let mut d = flat_fixture(1, 1, 64, exr_core::attr::Compression::Piz);
    for name in ["dataWindow", "displayWindow"] {
        let at = attr_at(&d, name);
        d[at + 8..at + 12].copy_from_slice(&2047i32.to_le_bytes());
    }
    let table = flat_table_start(&d);
    d[table..table + 8].fill(0);
    d.truncate(table + 8);
    assert!(!flat_complete(&d));
    reset_flat_work();
    assert!(!check_bytes(&d, opts(true, false, false)), "API -m intentionally skips an 8 MiB row buffer");
    assert_eq!(FLAT_READER_ATTEMPTS.with(|n| n.get()), 0);
    assert!(check_bytes(&d, opts(true, false, true)), "Core still checks chunk info");
}

#[test]
fn core_flat_stops_after_the_first_bad_chunk() {
    let mut d = flat_fixture(8, 3, 1, exr_core::attr::Compression::NoCompression);
    let table = flat_table_start(&d);
    let first = u64::from_le_bytes(d[table..table + 8].try_into().unwrap()) as usize;
    d[first + 4..first + 8].copy_from_slice(&(-1i32).to_le_bytes());
    assert!(flat_complete(&d));
    reset_flat_work();
    assert!(check_bytes(&d, opts(false, false, true)));
    assert_eq!(CORE_CHUNK_ATTEMPTS.with(|n| n.get()), 1, "a bad verdict cannot improve on later chunks");
}

struct StreamingDeepScan {
    header: Header,
    counts_calls: usize,
    events: Vec<char>,
    fail_data: bool,
}

impl DeepScanIn for StreamingDeepScan {
    fn hdr(&self) -> &Header { &self.header }
    fn set_fb(&mut self, _: &DeepFrameBuffer) -> Result<()> { Ok(()) }
    fn counts(&mut self, _: i32, _: i32, sc: &mut [u8]) -> Result<()> {
        self.counts_calls += 1;
        self.events.push('c');
        assert_eq!(sc.len(), 8 * 4, "only one row of counts is buffered");
        for c in sc.as_chunks_mut::<4>().0 { c.copy_from_slice(&1u32.to_ne_bytes()); }
        Ok(())
    }
    fn data(&mut self, _: i32, _: i32, _: &[u8], data: &mut [u8]) -> Result<()> {
        self.events.push('d');
        assert_eq!(data.len(), 8 * 4, "only one row of samples is buffered");
        if self.fail_data { Err(ExrError::Input("test data failure".into())) } else { Ok(()) }
    }
    fn raw(&mut self, _: i32) -> Result<()> { Ok(()) }
    fn payload(&mut self, _: i32) -> Result<()> { Ok(()) }
}

fn streaming_header(w: i32, h: i32) -> Header {
    let mut header = Header::with_dimensions(w, h, 1.0, imath_rs::V2f { x: 0.0, y: 0.0 },
        1.0, exr_core::attr::LineOrder::IncreasingY, exr_core::attr::Compression::NoCompression).unwrap();
    let mut channels = exr_core::attr::ChannelList::new();
    channels.insert("Z", exr_core::attr::Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    header.insert("channels", exr_core::attr::ChannelListAttribute::new(channels)).unwrap();
    header
}

/// An image whose full count raster exceeds 4 GiB still begins checking rows.
#[test]
fn deep_scanline_checks_stream_counts_then_data_and_ignore_full_raster_size() {
    for guard in [Guard::Api, Guard::Core] {
        let mut small = StreamingDeepScan { header: streaming_header(8, 3),
            counts_calls: 0, events: Vec::new(), fail_data: false };
        assert!(!read_deep_scanline(&mut small, opts(false, false, false), guard));
        assert_eq!(small.events, ['c', 'd', 'c', 'd', 'c', 'd']);

        let mut large = StreamingDeepScan { header: streaming_header(8, 200_000_000),
            counts_calls: 0, events: Vec::new(), fail_data: true };
        assert!(read_deep_scanline(&mut large, opts(false, true, false), guard));
        assert_eq!(large.counts_calls, 1, "must read the first row, not reject the full count raster");
    }
}

struct StreamingDeepTile {
    header: Header,
    counts_calls: usize,
}

impl DeepTileIn for StreamingDeepTile {
    fn hdr(&self) -> &Header { &self.header }
    fn levels(&self) -> (i32, i32) { (1, 1) }
    fn ntiles(&self, _: i32, _: i32) -> Result<(i32, i32)> { Ok((40_000, 40_000)) }
    fn level_dw(&self, _: i32, _: i32) -> Result<Box2i> { self.header.data_window() }
    fn set_fb(&mut self, _: &DeepFrameBuffer) -> Result<()> { Ok(()) }
    fn counts(&mut self, _: i32, _: i32, _: i32, _: i32, sc: &mut [u8]) -> Result<()> {
        self.counts_calls += 1;
        assert_eq!(sc.len(), 2 * 2 * 4, "only one tile of counts is buffered");
        for c in sc.as_chunks_mut::<4>().0 { c.copy_from_slice(&1u32.to_ne_bytes()); }
        Ok(())
    }
    fn data(&mut self, _: i32, _: i32, _: i32, _: i32, _: &[u8], data: &mut [u8]) -> Result<()> {
        assert_eq!(data.len(), 2 * 2 * 4, "only one tile of samples is buffered");
        Err(ExrError::Input("test data failure".into()))
    }
    fn raw(&mut self, _: i32, _: i32, _: i32, _: i32) -> Result<()> { Ok(()) }
    fn payload(&mut self, _: i32, _: i32, _: i32, _: i32) -> Result<()> { Ok(()) }
}

#[test]
fn deep_tile_checks_stream_a_single_tile_in_a_large_level() {
    for guard in [Guard::Api, Guard::Core] {
        let mut header = streaming_header(80_000, 80_000);
        header.insert("tiles", exr_core::attr::TileDescriptionAttribute::new(exr_core::attr::TileDescription {
            x_size: 2, y_size: 2, mode: LevelMode::OneLevel,
            rounding_mode: exr_core::attr::LevelRoundingMode::RoundDown,
        })).unwrap();
        let mut input = StreamingDeepTile { header, counts_calls: 0 };
        assert!(read_deep_tile(&mut input, opts(false, true, false), guard));
        assert_eq!(input.counts_calls, 1, "must read the first tile before a data failure");
    }
}

#[test]
fn deep_unit_scratch_retains_capacity_and_clears_counts() {
    let rect = Box2i { min: imath_rs::V2i { x: -9, y: -12 }, max: imath_rs::V2i { x: -3, y: -7 } };
    let mut sc = CountRaster::new(&rect).unwrap();
    sc.bytes.fill(0xff);
    let capacity = sc.bytes.capacity();
    let address = sc.bytes.as_ptr();
    let smaller = Box2i { min: imath_rs::V2i { x: 21, y: 13 }, max: imath_rs::V2i { x: 23, y: 14 } };
    sc.set_window(&smaller).unwrap();
    assert_eq!(sc.bytes.len(), 3 * 2 * 4);
    assert_eq!(sc.bytes.capacity(), capacity);
    assert_eq!(sc.bytes.as_ptr(), address);
    assert_eq!(sc.total(), 0, "reused counts cannot contain the previous tile's values");
    sc.set_window(&rect).unwrap();
    assert_eq!(sc.bytes.capacity(), capacity);
    assert_eq!(sc.bytes.as_ptr(), address);
}
struct CountingDeepScan {
    inner: StreamingDeepScan,
    setters: usize,
}
impl DeepScanIn for CountingDeepScan {
    fn hdr(&self) -> &Header {
        self.inner.hdr()
    }
    fn set_fb(&mut self, fb: &DeepFrameBuffer) -> Result<()> {
        self.setters += 1;
        assert_eq!(fb.get_sample_count_slice().y_stride, 0);
        self.inner.set_fb(fb)
    }
    fn counts(&mut self, a: i32, b: i32, sc: &mut [u8]) -> Result<()> {
        self.inner.counts(a, b, sc)
    }
    fn data(&mut self, a: i32, b: i32, sc: &[u8], heap: &mut [u8]) -> Result<()> {
        self.inner.data(a, b, sc, heap)
    }
    fn raw(&mut self, y: i32) -> Result<()> {
        self.inner.raw(y)
    }
    fn payload(&mut self, y: i32) -> Result<()> {
        self.inner.payload(y)
    }
}
struct CountingDeepTile {
    inner: StreamingDeepTile,
    setters: usize,
}
impl DeepTileIn for CountingDeepTile {
    fn hdr(&self) -> &Header {
        self.inner.hdr()
    }
    fn levels(&self) -> (i32, i32) {
        self.inner.levels()
    }
    fn ntiles(&self, x: i32, y: i32) -> Result<(i32, i32)> {
        self.inner.ntiles(x, y)
    }
    fn level_dw(&self, x: i32, y: i32) -> Result<Box2i> {
        self.inner.level_dw(x, y)
    }
    fn set_fb(&mut self, fb: &DeepFrameBuffer) -> Result<()> {
        self.setters += 1;
        let sc = fb.get_sample_count_slice();
        assert!(sc.x_tile_coords && sc.y_tile_coords);
        assert_eq!(sc.base_offset, 0);
        self.inner.set_fb(fb)
    }
    fn counts(&mut self, x: i32, y: i32, lx: i32, ly: i32, sc: &mut [u8]) -> Result<()> {
        self.inner.counts(x, y, lx, ly, sc)
    }
    fn data(&mut self, x: i32, y: i32, lx: i32, ly: i32, sc: &[u8], heap: &mut [u8]) -> Result<()> {
        self.inner.data(x, y, lx, ly, sc, heap)
    }
    fn raw(&mut self, x: i32, y: i32, lx: i32, ly: i32) -> Result<()> {
        self.inner.raw(x, y, lx, ly)
    }
    fn payload(&mut self, x: i32, y: i32, lx: i32, ly: i32) -> Result<()> {
        self.inner.payload(x, y, lx, ly)
    }
}

#[test]
fn deep_unit_layout_is_set_once_and_clipped_counts_ignore_padding() {
    let mut scan = CountingDeepScan {
        inner: StreamingDeepScan {
            header: streaming_header(8, 3),
            counts_calls: 0,
            events: Vec::new(),
            fail_data: false,
        },
        setters: 0,
    };
    assert!(!read_deep_scanline(
        &mut scan,
        opts(false, false, false),
        Guard::Api
    ));
    assert_eq!(scan.setters, 1);
    assert_eq!(scan.inner.counts_calls, 3);
    let mut header = streaming_header(80_000, 80_000);
    header
        .insert(
            "tiles",
            exr_core::attr::TileDescriptionAttribute::new(exr_core::attr::TileDescription {
                x_size: 2,
                y_size: 2,
                mode: LevelMode::OneLevel,
                rounding_mode: exr_core::attr::LevelRoundingMode::RoundDown,
            }),
        )
        .unwrap();
    let mut tile = CountingDeepTile {
        inner: StreamingDeepTile {
            header,
            counts_calls: 0,
        },
        setters: 0,
    };
    assert!(read_deep_tile(
        &mut tile,
        opts(false, true, false),
        Guard::Api
    ));
    assert_eq!(tile.setters, 1);
    let rect = Box2i {
        min: imath_rs::V2i { x: -9, y: -12 },
        max: imath_rs::V2i { x: -7, y: -11 },
    };
    let mut sc = CountRaster::new(&rect).unwrap();
    sc.set_active_window(&Box2i {
        min: rect.min,
        max: imath_rs::V2i { x: -9, y: -11 },
    })
    .unwrap();
    for (i, bytes) in sc.bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        bytes.copy_from_slice(&(if i % 3 == 0 { 1u32 } else { u32::MAX }).to_ne_bytes());
    }
    assert_eq!(
        sc.total(),
        2,
        "clipped cells only, independently of stale padding"
    );
    let names = vec![
        (Name::from("A"), PixelType::Half, 1, 1),
        (Name::from("Z"), PixelType::Float, 1, 1),
    ];
    let fb = deep_unit_fb(&names, &sc.slice).unwrap();
    assert_eq!(fb.get("A").unwrap().slice.base_offset, 0);
    assert_eq!(fb.get("Z").unwrap().slice.base_offset, 4);
    assert_eq!(fb.get("A").unwrap().sample_stride, 8);
    assert_eq!(fb.get("Z").unwrap().sample_stride, 8);
}

struct ClippedDeepTiles {
    header: Header,
    setters: usize,
    count_calls: Vec<(i32, i32, i32, i32)>,
    data_calls: Vec<(i32, i32, i32, i32, usize)>,
}

impl ClippedDeepTiles {
    fn dimensions(level: i32) -> i32 {
        [5, 2, 1][level as usize]
    }

    fn count(x: i32, y: i32, lx: i32, ly: i32) -> u32 {
        (x + 2 * y + 3 * lx + ly).rem_euclid(5) as u32 + 1
    }

    fn valid_cells(&self, dx: i32, dy: i32, lx: i32, ly: i32) -> Vec<(usize, u32)> {
        let mut cells = Vec::new();
        for y in dy * 2..((dy + 1) * 2).min(Self::dimensions(ly)) {
            for x in dx * 3..((dx + 1) * 3).min(Self::dimensions(lx)) {
                let offset = ((y - dy * 2) * 3 + x - dx * 3) as usize * 4;
                cells.push((offset, Self::count(x - 9, y - 12, lx, ly)));
            }
        }
        cells
    }
}

impl DeepTileIn for ClippedDeepTiles {
    fn hdr(&self) -> &Header {
        &self.header
    }
    fn levels(&self) -> (i32, i32) {
        (3, 3)
    }
    fn ntiles(&self, lx: i32, ly: i32) -> Result<(i32, i32)> {
        Ok((
            (Self::dimensions(lx) + 2) / 3,
            (Self::dimensions(ly) + 1) / 2,
        ))
    }
    fn level_dw(&self, lx: i32, ly: i32) -> Result<Box2i> {
        Ok(Box2i {
            min: imath_rs::V2i { x: -9, y: -12 },
            max: imath_rs::V2i {
                x: -9 + Self::dimensions(lx) - 1,
                y: -12 + Self::dimensions(ly) - 1,
            },
        })
    }
    fn set_fb(&mut self, fb: &DeepFrameBuffer) -> Result<()> {
        self.setters += 1;
        let counts = fb.get_sample_count_slice();
        assert_eq!(
            (counts.base_offset, counts.x_stride, counts.y_stride),
            (0, 4, 12)
        );
        assert!(counts.x_tile_coords && counts.y_tile_coords);
        assert_eq!(fb.get("A").unwrap().slice.base_offset, 0);
        assert_eq!(fb.get("Z").unwrap().slice.base_offset, 4);
        assert_eq!(fb.get("A").unwrap().sample_stride, 8);
        assert_eq!(fb.get("Z").unwrap().sample_stride, 8);
        Ok(())
    }
    fn counts(&mut self, dx: i32, dy: i32, lx: i32, ly: i32, counts: &mut [u8]) -> Result<()> {
        assert_eq!(counts.len(), 3 * 2 * 4);
        // The decoder writes only valid cells. Padding is deliberately nonzero so
        // the checker must update the active rectangle before summing this tile.
        for cell in counts.as_chunks_mut::<4>().0 {
            cell.copy_from_slice(&1000u32.to_ne_bytes());
        }
        for (offset, count) in self.valid_cells(dx, dy, lx, ly) {
            counts[offset..offset + 4].copy_from_slice(&count.to_ne_bytes());
        }
        self.count_calls.push((dx, dy, lx, ly));
        Ok(())
    }
    fn data(
        &mut self,
        dx: i32,
        dy: i32,
        lx: i32,
        ly: i32,
        counts: &[u8],
        heap: &mut [u8],
    ) -> Result<()> {
        let cells = self.valid_cells(dx, dy, lx, ly);
        let samples: usize = cells.iter().map(|(_, count)| *count as usize).sum();
        assert_eq!(
            heap.len(),
            samples * 2 * 4,
            "valid cells only at tile ({dx},{dy}), level ({lx},{ly})"
        );
        for (offset, count) in cells {
            assert_eq!(
                u32::from_ne_bytes(counts[offset..offset + 4].try_into().unwrap()),
                count
            );
        }
        for sample in heap.as_chunks_mut::<8>().0 {
            sample[..4].copy_from_slice(&0.5f32.to_ne_bytes());
            sample[4..].copy_from_slice(&2.0f32.to_ne_bytes());
        }
        self.data_calls.push((dx, dy, lx, ly, heap.len()));
        Ok(())
    }
    fn raw(&mut self, _: i32, _: i32, _: i32, _: i32) -> Result<()> {
        Ok(())
    }
    fn payload(&mut self, _: i32, _: i32, _: i32, _: i32) -> Result<()> {
        panic!("every valid nonempty tile must decode into the sample heap")
    }
}

#[test]
fn deep_tile_checker_updates_clipped_cells_across_tiles_and_ripmap_levels() {
    for guard in [Guard::Api, Guard::Core] {
        for reduce_memory in [false, true] {
            let mut header = streaming_header(5, 5);
            *header.data_window_mut().unwrap() = Box2i {
                min: imath_rs::V2i { x: -9, y: -12 },
                max: imath_rs::V2i { x: -5, y: -8 },
            };
            header
                .channels_mut()
                .unwrap()
                .insert(
                    "A",
                    exr_core::attr::Channel::new(PixelType::Half, 1, 1, false),
                )
                .unwrap();
            header
                .set_tile_description(exr_core::attr::TileDescription {
                    x_size: 3,
                    y_size: 2,
                    mode: LevelMode::RipmapLevels,
                    rounding_mode: exr_core::attr::LevelRoundingMode::RoundDown,
                })
                .unwrap();
            let mut input = ClippedDeepTiles {
                header,
                setters: 0,
                count_calls: Vec::new(),
                data_calls: Vec::new(),
            };
            assert!(!read_deep_tile(
                &mut input,
                opts(reduce_memory, false, false),
                guard
            ));
            assert_eq!(input.setters, 1);
            assert_eq!(input.count_calls.len(), 20);
            let decoded: Vec<_> = input
                .data_calls
                .iter()
                .map(|&(x, y, lx, ly, _)| (x, y, lx, ly))
                .collect();
            assert_eq!(
                decoded, input.count_calls,
                "all clipped tiles and levels must decode successfully"
            );
            assert!(
                input
                    .data_calls
                    .iter()
                    .any(|&(x, y, lx, ly, _)| x == 1 && y == 2 && lx == 0 && ly == 0)
            );
            assert!(
                input
                    .data_calls
                    .iter()
                    .any(|&(_, _, lx, ly, _)| lx == 2 && ly == 2)
            );
        }
    }
}
