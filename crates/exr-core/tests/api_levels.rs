//! Gates for the `api::Image` facade's tiled write, MIP/RIP levels and subsampled
//! channels.
//!
//! Strongest first:
//! * `resaving_cpp_goldens_is_byte_identical` — `read_all` + `write` of files written
//!   by real C++ OpenEXR 3.4.15 (mip / rip / round-up / offset window / subsampled)
//!   reproduces them byte for byte: header carrying, level order, tile order and the
//!   subsampled plane layout, against the oracle's own output.
//! * `cpp_oracle_reads_a_facade_mip_file` — the live oracle reads what the facade wrote.
//! * per-level pixels checked against an INDEPENDENT low-level reader / a closed-form
//!   pattern, never against the facade reading its own output.
//! * byte identity against a hand-written low-level `TiledOutputFile` path.

mod support;

use std::path::{Path, PathBuf};
use std::process::Command;

use exr_core::api::planes::planes;
use exr_core::api::{ChannelData, Image};
use exr_core::attr::{
    Attribute, Channel, ChannelList, Compression, LevelMode, LevelRoundingMode, Name, PixelType,
    TileDescription, TileDescriptionAttribute,
};
use exr_core::error::ExrError;
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::header::Header;
use exr_core::io::{MemIStream, MemOStream};
use exr_core::scanline::InputFile;
use exr_core::tiled::{TiledInputFile, TiledOutputFile};
use exr_core::Unsupported;
use imath_rs::{Box2i, Half, V2i};

fn golden(name: &str) -> Option<PathBuf> {
    support::fixture_or_skip(
        name,
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("parity").join(name),
        "Committed golden; restore it from git.",
    )
}

fn bx(x0: i32, y0: i32, x1: i32, y1: i32) -> Box2i {
    Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x1, y: y1 } }
}

fn td(size: u32, mode: LevelMode, rounding_mode: LevelRoundingMode) -> TileDescription {
    TileDescription { x_size: size, y_size: size, mode, rounding_mode }
}

fn write_bytes(img: &Image, comp: Compression) -> Vec<u8> {
    let mut out = MemOStream::new();
    img.write_to(&mut out, comp).expect("facade write");
    out.into_data()
}

// ---------------------------------------------------------------------------
// C++ goldens: read_all + write reproduces them
// ---------------------------------------------------------------------------

/// `read_all` then `write` of a C++-written lossless golden is the same file.
fn resave_identical(name: &str) {
    let Some(path) = golden(name) else { return };
    let original = std::fs::read(&path).expect("read golden");
    let img = Image::read_all(&path).unwrap_or_else(|e| panic!("{name}: read_all: {e}"));
    let again = write_bytes(&img, img.compression);
    assert!(again == original, "{name}: re-saved bytes differ ({} vs {} bytes)", again.len(), original.len());
}

#[test]
fn resaving_cpp_goldens_is_byte_identical() {
    for name in [
        "golden_tiled_none_one.exr",
        "golden_tiled_none_one_offset.exr",
        "golden_tiled_zip_one.exr",
        "golden_tiled_piz_one.exr",
        "golden_tiled_none_mipmap.exr",
        "golden_tiled_zip_mipmap.exr",
        "golden_tiled_piz_mipmap.exr",
        "golden_tiled_zip_mipmap_roundup.exr",
        "golden_tiled_zip_ripmap.exr",
        "golden_tiled_zip_one_decreasing.exr",
    ] {
        resave_identical(name);
    }
}

/// Subsampled scanline goldens (every sampling the oracle wrote, lossless codecs).
#[test]
fn resaving_cpp_subsampled_goldens_is_byte_identical() {
    for codec in ["none", "zip", "zips", "rle", "piz"] {
        for xy in ["x1y1", "x1y2", "x2y1", "x2y2"] {
            resave_identical(&format!("golden_sub_{codec}_{xy}.exr"));
        }
    }
}

// ---------------------------------------------------------------------------
// Independent low-level level reader
// ---------------------------------------------------------------------------

/// Level `(lx, ly)` of a tiled file through the raw `TiledInputFile` with a hand-built
/// frame buffer: `(window, [(channel, type, native bytes)])`. Shares no code with the
/// facade's `ReadPlan`.
fn low_level(bytes: &[u8], lx: i32, ly: i32) -> (Box2i, Vec<(String, PixelType, Vec<u8>)>) {
    let mut is = MemIStream::new(bytes.to_vec());
    let mut input = TiledInputFile::new(&mut is).expect("open");
    let win = input.data_window_for_level(lx, ly).expect("level window");
    let (w, h) = ((win.max.x - win.min.x + 1) as usize, (win.max.y - win.min.y + 1) as usize);
    let chans: Vec<(String, PixelType)> =
        input.header().channels().expect("channels").iter().map(|(n, c)| (n.to_string(), c.type_)).collect();
    let mut fb = FrameBuffer::new();
    let mut off = 0usize;
    let mut spans = Vec::new();
    for (name, ty) in &chans {
        let size = if *ty == PixelType::Half { 2 } else { 4 };
        fb.insert(name.as_str(), Slice::make(*ty, off, win.min, w as i64, size, size * w, 1, 1, 0.0, false, false).unwrap())
            .unwrap();
        spans.push((off, w * h * size));
        off += w * h * size;
    }
    let mut buf = vec![0u8; off];
    input.set_frame_buffer(&fb).expect("fb");
    let (nx, ny) = (input.num_x_tiles(lx).unwrap(), input.num_y_tiles(ly).unwrap());
    input.read_tiles(0, nx - 1, 0, ny - 1, lx, ly, &mut buf).expect("read_tiles");
    let out = chans.into_iter().zip(spans).map(|((n, t), (o, l))| (n, t, buf[o..o + l].to_vec())).collect();
    (win, out)
}

/// Every level of a C++ golden, as the facade loads it, equals the raw reader's.
#[test]
fn read_all_levels_equal_the_low_level_reader() {
    for name in [
        "golden_tiled_zip_mipmap.exr",
        "golden_tiled_zip_mipmap_roundup.exr",
        "golden_tiled_zip_ripmap.exr",
        "golden_tiled_none_one_offset.exr",
    ] {
        let Some(path) = golden(name) else { return };
        let bytes = std::fs::read(&path).unwrap();
        let img = Image::read_all(&path).unwrap();
        let tiling = img.tiling().expect("read_all keeps the tiling");
        let (nx, ny) = img.num_levels().unwrap();
        let mut seen = 0;
        for ly in 0..ny {
            for lx in 0..nx {
                if tiling.mode == LevelMode::MipmapLevels && lx != ly {
                    assert!(img.level(lx, ly).is_none(), "{name}: ({lx},{ly}) is not a MIP level");
                    continue;
                }
                let lv = img.level(lx, ly).unwrap_or_else(|| panic!("{name}: level ({lx},{ly}) missing"));
                let (win, raw) = low_level(&bytes, lx, ly);
                assert_eq!(lv.window(), win, "{name} ({lx},{ly}) window");
                assert_eq!(lv.channels().count(), raw.len(), "{name} ({lx},{ly}) channel count");
                for ((n, d), (rn, rt, rb)) in lv.channels().zip(&raw) {
                    assert_eq!((n, d.pixel_type()), (rn.as_str(), *rt), "{name} ({lx},{ly})");
                    assert!(&d.to_ne_bytes() == rb, "{name} level ({lx},{ly}) channel {n} differs");
                }
                seen += 1;
            }
        }
        assert!(seen > 1 || tiling.mode == LevelMode::OneLevel, "{name}: no levels beyond 0 were exercised");
    }
}

// ---------------------------------------------------------------------------
// Closed-form pattern images
// ---------------------------------------------------------------------------

/// Sample value of channel `c` at absolute pixel `(x, y)` of level `(lx, ly)`.
fn val(lx: i32, ly: i32, c: usize, x: i32, y: i32) -> f32 {
    (((lx * 7 + ly * 13 + c as i32 * 29 + x * 3 - y * 5).rem_euclid(1000)) as f32) * 0.5
}

/// Planes `R` (half), `U` (uint), `Z` (float) of `window` at level `(lx, ly)`.
fn pattern(lx: i32, ly: i32, window: Box2i) -> Vec<(String, ChannelData)> {
    let pts = || (window.min.y..=window.max.y).flat_map(move |y| (window.min.x..=window.max.x).map(move |x| (x, y)));
    vec![
        ("R".into(), ChannelData::Half(pts().map(|(x, y)| Half::from(val(lx, ly, 0, x, y))).collect())),
        ("U".into(), ChannelData::Uint(pts().map(|(x, y)| val(lx, ly, 1, x, y) as u32).collect())),
        ("Z".into(), ChannelData::Float(pts().map(|(x, y)| val(lx, ly, 2, x, y)).collect())),
    ]
}

/// A tiled image whose every level follows [`pattern`].
fn pyramid(dw: Box2i, t: TileDescription) -> Image {
    let mut img = Image::new(dw).with_tiling(t);
    for (n, d) in pattern(0, 0, dw) {
        img.insert_channel(n, d).unwrap();
    }
    let (nx, ny) = img.num_levels().unwrap();
    for ly in 0..ny {
        for lx in 0..nx {
            if (lx, ly) == (0, 0) || (t.mode == LevelMode::MipmapLevels && lx != ly) {
                continue;
            }
            let w = img.level_window(lx, ly).unwrap();
            img.insert_level(lx, ly, pattern(lx, ly, w)).unwrap();
        }
    }
    img
}

/// Facade-written pyramid -> raw reader, per level, against the closed form.
fn pyramid_reads_back(dw: Box2i, t: TileDescription, comp: Compression) {
    let img = pyramid(dw, t);
    let bytes = write_bytes(&img, comp);
    let (nx, ny) = img.num_levels().unwrap();
    let mut checked = 0;
    for ly in 0..ny {
        for lx in 0..nx {
            if t.mode == LevelMode::MipmapLevels && lx != ly {
                continue;
            }
            let (win, raw) = low_level(&bytes, lx, ly);
            assert_eq!(win, img.level_window(lx, ly).unwrap(), "({lx},{ly}) window");
            for ((name, _, got), (en, expected)) in raw.iter().zip(pattern(lx, ly, win)) {
                assert_eq!(name, &en);
                assert!(&expected.to_ne_bytes() == got, "{t:?} {comp:?} level ({lx},{ly}) channel {name}");
            }
            checked += 1;
        }
    }
    assert!(checked > 1 || t.mode == LevelMode::OneLevel, "only level 0 was checked");
    // And the whole file loads back to the same Image, with every level.
    let back = Image::read_all_from(&mut MemIStream::new(bytes)).unwrap();
    assert_eq!(back.tiling(), Some(t));
    assert_eq!(back.data_window, img.data_window);
    for ly in 0..ny {
        for lx in 0..nx {
            if let (Some(a), Some(b)) = (img.level(lx, ly), back.level(lx, ly)) {
                assert!(a.channels().eq(b.channels()), "round trip level ({lx},{ly})");
            }
        }
    }
}

#[test]
fn facade_mip_write_reads_back_per_level() {
    for (dw, round) in [
        (bx(0, 0, 36, 22), LevelRoundingMode::RoundDown),
        (bx(-5, -3, 31, 19), LevelRoundingMode::RoundUp),
    ] {
        for comp in [Compression::NoCompression, Compression::Zip, Compression::Piz] {
            pyramid_reads_back(dw, td(8, LevelMode::MipmapLevels, round), comp);
        }
    }
}

#[test]
fn facade_rip_write_reads_back_per_level() {
    pyramid_reads_back(bx(0, 0, 36, 22), td(8, LevelMode::RipmapLevels, LevelRoundingMode::RoundDown), Compression::Zip);
    pyramid_reads_back(bx(-5, -3, 20, 11), td(4, LevelMode::RipmapLevels, LevelRoundingMode::RoundUp), Compression::NoCompression);
}

#[test]
fn facade_one_level_tiled_write_reads_back() {
    pyramid_reads_back(bx(-5, -3, 36, 22), td(16, LevelMode::OneLevel, LevelRoundingMode::RoundDown), Compression::Zip);
}

/// `read_level` returns one level as a plain image over that level's window.
#[test]
fn read_level_returns_a_plain_image_per_level() {
    let t = td(8, LevelMode::RipmapLevels, LevelRoundingMode::RoundDown);
    let bytes = write_bytes(&pyramid(bx(0, 0, 36, 22), t), Compression::Zip);
    let lvl = Image::read_level_from(&mut MemIStream::new(bytes.clone()), 2, 1).unwrap();
    assert_eq!(lvl.tiling(), None);
    assert_eq!(lvl.data_window, bx(0, 0, 8, 10), "level (2,1) of 37x23 rounded down");
    for (n, d) in pattern(2, 1, lvl.data_window) {
        assert_eq!(lvl.channel(&n), Some(&d), "channel {n}");
    }
    assert!(matches!(Image::read_level_from(&mut MemIStream::new(bytes), 9, 0), Err(ExrError::Arg(_))));
}

// ---------------------------------------------------------------------------
// Byte identity against an independent low-level writer
// ---------------------------------------------------------------------------

/// The facade's tiled bytes equal a hand-written `TiledOutputFile` path (own header,
/// own planar buffers, own frame buffers, one `write_tile` per tile in file order).
fn low_level_bytes(dw: Box2i, t: TileDescription, comp: Compression) -> Vec<u8> {
    let mut header = Header::new();
    *header.data_window_mut().unwrap() = dw;
    *header.display_window_mut().unwrap() = dw;
    *header.compression_mut().unwrap() = comp;
    let mut list = ChannelList::new();
    list.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    list.insert("U", Channel::new(PixelType::Uint, 1, 1, false)).unwrap();
    list.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    *header.channels_mut().unwrap() = list;
    header.set_tile_description(t).unwrap();

    let mut out = MemOStream::new();
    {
        let mut of = TiledOutputFile::new(&mut out, &header).unwrap();
        let (nxl, nyl) = (of.num_x_levels(), of.num_y_levels());
        for ly in 0..nyl {
            for lx in 0..nxl {
                if t.mode == LevelMode::MipmapLevels && lx != ly {
                    continue;
                }
                let win = of.data_window_for_level(lx, ly).unwrap();
                let w = (win.max.x - win.min.x + 1) as usize;
                let mut buf = Vec::new();
                let mut fb = FrameBuffer::new();
                for (name, d) in pattern(lx, ly, win) {
                    let size = d.pixel_type().size_hint();
                    fb.insert(name.as_str(), Slice::make(d.pixel_type(), buf.len(), win.min, w as i64, size, size * w, 1, 1, 0.0, false, false).unwrap())
                        .unwrap();
                    buf.extend(d.to_ne_bytes());
                }
                of.set_frame_buffer(&fb).unwrap();
                for dy in 0..of.num_y_tiles(ly).unwrap() {
                    for dx in 0..of.num_x_tiles(lx).unwrap() {
                        of.write_tile(&buf, dx, dy, lx, ly).unwrap();
                    }
                }
            }
        }
    }
    out.into_data()
}

trait SizeHint {
    fn size_hint(self) -> usize;
}
impl SizeHint for PixelType {
    fn size_hint(self) -> usize {
        if self == PixelType::Half { 2 } else { 4 }
    }
}

#[test]
fn facade_tiled_write_matches_the_low_level_writer_byte_for_byte() {
    for (dw, t, comp) in [
        (bx(0, 0, 36, 22), td(8, LevelMode::OneLevel, LevelRoundingMode::RoundDown), Compression::Zip),
        (bx(-5, -3, 36, 22), td(16, LevelMode::OneLevel, LevelRoundingMode::RoundDown), Compression::NoCompression),
        (bx(0, 0, 36, 22), td(8, LevelMode::MipmapLevels, LevelRoundingMode::RoundDown), Compression::Zip),
        (bx(-5, -3, 31, 19), td(8, LevelMode::MipmapLevels, LevelRoundingMode::RoundUp), Compression::Piz),
        (bx(0, 0, 36, 22), td(8, LevelMode::RipmapLevels, LevelRoundingMode::RoundDown), Compression::Zip),
    ] {
        let facade = write_bytes(&pyramid(dw, t), comp);
        assert!(facade == low_level_bytes(dw, t, comp), "{t:?} {comp:?} {dw:?}: facade diverged from the low-level writer");
    }
}

// ---------------------------------------------------------------------------
// Live C++ oracle
// ---------------------------------------------------------------------------

/// `oracle read-tiled` reads a facade-written MIP file and every TPIX record of every
/// level equals the closed form (the oracle's `read-tiled` is ONE/MIP only).
#[test]
fn cpp_oracle_reads_a_facade_mip_file() {
    let Some(oracle) = support::oracle_or_skip("cpp_oracle_reads_a_facade_mip_file") else { return };
    let t = td(8, LevelMode::MipmapLevels, LevelRoundingMode::RoundDown);
    let dw = bx(0, 0, 36, 22);
    let dir = std::env::temp_dir().join("exr_api_levels");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("facade_mip.exr");
    pyramid(dw, t).write(&path, Compression::Zip).unwrap();

    let out = Command::new(&oracle).arg("read-tiled").arg(&path).output().expect("run oracle");
    assert!(out.status.success(), "oracle read-tiled failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let (mut pix, mut levels) = (0usize, std::collections::BTreeSet::new());
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"TPIX") {
            continue;
        }
        let (l, chan): (i32, &str) = (f[1].parse().unwrap(), f[2]);
        let (x, y): (i32, i32) = (f[3].parse().unwrap(), f[4].parse().unwrap());
        let bits = u32::from_str_radix(f[5], 16).unwrap();
        let v = val(l, l, ["R", "U", "Z"].iter().position(|c| *c == chan).unwrap(), x, y);
        let want = match chan {
            "R" => u32::from(Half::from(v).bits()),
            "U" => v as u32,
            _ => v.to_bits(),
        };
        assert_eq!(bits, want, "level {l} {chan} ({x},{y})");
        levels.insert(l);
        pix += 1;
    }
    assert_eq!(levels.len(), 6, "37x23 round-down has 6 MIP levels");
    assert!(pix > 37 * 23 * 3, "oracle printed too few pixels: {pix}");
}

// ---------------------------------------------------------------------------
// Subsampled
// ---------------------------------------------------------------------------

/// Y full-res float, RY/BY at 2x2 half, A at 2x1 uint over a window with a negative
/// origin (multiples of the sampling, as EXR requires).
fn sub_image(dw: Box2i) -> Image {
    let (w, h) = ((dw.max.x - dw.min.x + 1) as usize, (dw.max.y - dw.min.y + 1) as usize);
    let mut img = Image::new(dw);
    img.insert_channel("Y", ChannelData::Float((0..w * h).map(|i| i as f32 * 0.25).collect())).unwrap();
    for (n, s) in [("BY", 3u32), ("RY", 5)] {
        img.insert_sub(n, ChannelData::Half((0..(w / 2) * (h / 2)).map(|i| Half::from((i as u32 * s % 97) as f32)).collect()), 2, 2).unwrap();
    }
    img.insert_sub("A", ChannelData::Uint((0..(w / 2) * h).map(|i| i as u32 * 11).collect()), 2, 1).unwrap();
    img
}

#[test]
fn subsampled_round_trip_matches_the_native_plane_reader() {
    for (dw, comp) in [
        (bx(-4, -2, 11, 7), Compression::Zip),
        (bx(0, 0, 15, 9), Compression::NoCompression),
        (bx(-8, 4, 7, 13), Compression::Piz),
    ] {
        let img = sub_image(dw);
        let bytes = write_bytes(&img, comp);

        let back = Image::read_all_from(&mut MemIStream::new(bytes.clone())).unwrap();
        assert_eq!(back.sampling("RY"), Some((2, 2)));
        assert_eq!(back.sampling("A"), Some((2, 1)));
        assert_eq!(back.sampling("Y"), Some((1, 1)));
        assert_eq!(back.channels().collect::<Vec<_>>(), img.channels().collect::<Vec<_>>(), "{dw:?} {comp:?}");

        // Independent reader of the same file: exr-core's native plane reader.
        let mut is = MemIStream::new(bytes.clone());
        let ps = planes(&mut InputFile::new(&mut is).unwrap()).unwrap();
        assert_eq!(ps.len(), 4);
        for p in ps {
            assert_eq!(Some(p.sampling), img.sampling(&p.name), "{}", p.name);
            assert_eq!(Some(&p.data), img.channel(&p.name), "{} {dw:?}", p.name);
            assert_eq!(Some((p.width as usize, p.height as usize)), img.samples(&p.name));
        }

        // The strict reader still refuses, by kind.
        let err = Image::read_from(&mut MemIStream::new(bytes)).unwrap_err();
        assert!(matches!(err, ExrError::Unsupported(Unsupported::Subsampled)), "{err:?}");
    }
}

#[test]
fn a_subsampled_channel_cannot_be_written_tiled_or_with_an_odd_window() {
    let img = sub_image(bx(-4, -2, 11, 7)).with_tiling(td(8, LevelMode::OneLevel, LevelRoundingMode::RoundDown));
    let mut out = MemOStream::new();
    let err = img.write_to(&mut out, Compression::Zip).unwrap_err();
    assert!(err.to_string().contains("subsampling"), "tiled + subsampled: {err}");

    // Odd width with 2x sampling: the header sanity check (C++ too) rejects it.
    let mut odd = Image::new(bx(0, 0, 6, 5));
    odd.insert_sub("C", ChannelData::Float(vec![0.0; 4 * 3]), 2, 2).unwrap();
    let err = odd.write_to(&mut MemOStream::new(), Compression::Zip).unwrap_err();
    assert!(err.to_string().contains("multiple"), "odd window: {err}");
}

#[test]
fn insert_sub_validates_the_sample_grid() {
    let mut img = Image::new(bx(0, 0, 7, 5));
    assert!(matches!(img.insert_sub("C", ChannelData::Float(vec![0.0; 8 * 6]), 2, 2), Err(ExrError::Arg(_))), "full-size plane at 2x2");
    assert!(matches!(img.insert_sub("C", ChannelData::Float(vec![0.0; 12]), 0, 1), Err(ExrError::Arg(_))), "sampling 0");
    img.insert_sub("C", ChannelData::Float(vec![0.0; 12]), 2, 2).unwrap();
    assert_eq!(img.samples("C"), Some((4, 3)));
    // Replacing with insert_channel resets the sampling in the same record.
    img.insert_channel("C", ChannelData::Float(vec![0.0; 8 * 6])).unwrap();
    assert_eq!(img.sampling("C"), Some((1, 1)));
}

#[test]
fn into_channels_refuses_to_drop_sampling() {
    let err = sub_image(bx(0, 0, 7, 5)).into_channels().map(|_| ()).unwrap_err();
    assert!(matches!(&err, ExrError::Arg(m) if m.contains("\"A\"") || m.contains("\"BY\"")), "{err:?}");
    let plain = Image::new(bx(0, 0, 1, 1)).with_channel("R", ChannelData::Float(vec![0.0; 4]));
    assert_eq!(plain.into_channels().unwrap().count(), 1);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

fn arg(r: Result<(), ExrError>, what: &str) -> String {
    match r {
        Err(ExrError::Arg(m)) => m,
        other => panic!("{what}: expected Arg, got {other:?}"),
    }
}

#[test]
fn level_errors_are_specific() {
    let dw = bx(0, 0, 36, 22);
    let mip = td(8, LevelMode::MipmapLevels, LevelRoundingMode::RoundDown);

    // A pyramid with a level missing is refused BEFORE any byte is written.
    let mut img = Image::new(dw).with_tiling(mip);
    for (n, d) in pattern(0, 0, dw) {
        img.insert_channel(n, d).unwrap();
    }
    let mut out = MemOStream::new();
    let m = arg(img.write_to(&mut out, Compression::Zip), "missing level");
    assert!(m.contains("Level (1, 1)"), "{m}");
    assert!(out.into_data().is_empty(), "nothing may be written before validation");

    // insert_level: wrong index for the mode, level 0, no tiling, wrong length, wrong channels.
    assert!(arg(img.insert_level(1, 0, pattern(1, 0, dw)), "MIP off-diagonal").contains("does not exist"));
    assert!(arg(img.insert_level(0, 0, pattern(0, 0, dw)), "level 0").contains("image itself"));
    let w1 = img.level_window(1, 1).unwrap();
    assert!(arg(img.insert_level(1, 1, pattern(1, 1, dw)), "wrong window").contains("samples"));
    let mut short = pattern(1, 1, w1);
    short.pop();
    assert!(arg(img.insert_level(1, 1, short), "missing channel").contains("level 0's channels"));
    let mut retyped = pattern(1, 1, w1);
    retyped[0].1 = ChannelData::Float(vec![0.0; 18 * 11]);
    assert!(arg(img.insert_level(1, 1, retyped), "retyped channel").contains("level 0's channels"));
    let mut untiled = Image::new(dw);
    assert!(arg(untiled.insert_level(1, 1, pattern(1, 1, w1)), "no tiling").contains("needs a tiling"));

    // A level that the ONE_LEVEL tiling of an overlay does not have.
    let mut full = pyramid(dw, mip);
    let overlay: Vec<(Name, Box<dyn Attribute>)> = vec![(
        "tiles".into(),
        Box::new(TileDescriptionAttribute::new(td(8, LevelMode::OneLevel, LevelRoundingMode::RoundDown))),
    )];
    let m = arg(full.write_to_with_attrs(&mut MemOStream::new(), Compression::Zip, overlay), "extra level");
    assert!(m.contains("does not have"), "{m}");

    // set_tiling drops the levels (their geometry changed); levels without tiling don't write.
    full.set_tiling(Some(mip));
    assert!(full.level(1, 1).is_none());
    // read_level on a scanline file: only (0, 0).
    let scan = write_bytes(&Image::new(bx(0, 0, 3, 3)).with_channel("R", ChannelData::Float(vec![0.0; 16])), Compression::Zip);
    assert!(Image::read_level_from(&mut MemIStream::new(scan.clone()), 0, 0).is_ok());
    assert!(matches!(Image::read_level_from(&mut MemIStream::new(scan), 1, 1), Err(ExrError::Arg(_))));
}

#[test]
fn a_tiled_image_is_not_a_multipart_scanline_part() {
    let img = pyramid(bx(0, 0, 36, 22), td(8, LevelMode::MipmapLevels, LevelRoundingMode::RoundDown));
    let path = std::env::temp_dir().join("exr_api_levels").join("mp.exr");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let err = exr_core::api::write_multipart_scanline(&path, vec![(img, Compression::Zip, "a".into())]).unwrap_err();
    assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
}
