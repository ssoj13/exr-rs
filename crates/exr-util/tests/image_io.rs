//! Whole-file image I/O of exr-util: save then load must give back every sample of every level, and
//! the C++-written goldens must load to what exr-core's own readers see.

use std::path::{Path, PathBuf};

use exr_core::attr::{Channel, Compression, LevelMode, LevelRoundingMode, LineOrder, Name, PixelType, TileDescription};
use exr_core::header::Header;
use exr_util::deep_image::DeepImage;
use exr_util::deep_image_io::{load_deep_image, save_deep_image};
use exr_util::flat_image::FlatImage;
use exr_util::flat_image_io::{load_flat_image, save_flat_image};
use exr_util::image::Image;
use exr_util::image_channel::ImageChannel;
use exr_util::image_data_window::DataWindowSource;
use exr_util::image_io::{load_image, save_image, AnyImage};
use exr_util::image_level::ImageLevel;
use imath_rs::{Box2i, Half, V2i};

const DOWN: LevelRoundingMode = LevelRoundingMode::RoundDown;
const UP: LevelRoundingMode = LevelRoundingMode::RoundUp;

fn bx(x0: i32, y0: i32, x1: i32, y1: i32) -> Box2i {
    Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x1, y: y1 } }
}

/// A scratch file that is removed on drop.
struct Tmp(PathBuf);

impl Tmp {
    fn new(name: &str) -> Self {
        // A counter too: tests run in parallel and several build the same named file.
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Tmp(std::env::temp_dir().join(format!("exr-util-{}-{n}-{name}.exr", std::process::id())))
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../exr-core/tests/parity").join(name)
}

/// A deterministic hash, the source of every test sample.
fn hash(a: u32, b: u32, c: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA6B) ^ c.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^ (h >> 13)
}

fn half(h: u32) -> Half {
    Half::from_bits((h & 0x7BFF) as u16) // finite: exponent below 31
}

fn float(h: u32) -> f32 {
    f32::from_bits(h & 0x7F7F_FFFF)
}

/// The channel set of the flat tests; `sub` adds a 2x2 subsampled one (scanline files only).
fn flat_image(dw: Box2i, mode: LevelMode, rounding: LevelRoundingMode, sub: bool) -> FlatImage {
    let mut img = FlatImage::new(&dw, mode, rounding).unwrap();
    img.insert_type("R", PixelType::Half).unwrap();
    img.insert_type("G", PixelType::Half).unwrap();
    img.insert_type("Z", PixelType::Float).unwrap();
    img.insert_type("id", PixelType::Uint).unwrap();
    if sub {
        img.insert_channel("BY", &Channel::new(PixelType::Half, 2, 2, false)).unwrap();
        img.insert_channel("C3", &Channel::new(PixelType::Float, 3, 3, false)).unwrap();
    }
    for (lx, ly) in levels(&img) {
        let lv = img.level_at_mut(lx, ly).unwrap();
        let seed = (lx * 16 + ly) as u32;
        for (k, (_, ch)) in lv.iter_mut().enumerate() {
            let k = k as u32;
            use exr_util::flat_image_channel::FlatImageChannel as C;
            match ch {
                C::Half(c) => c.pixels_mut().iter_mut().enumerate().for_each(|(i, p)| *p = half(hash(seed, k, i as u32))),
                C::Float(c) => c.pixels_mut().iter_mut().enumerate().for_each(|(i, p)| *p = float(hash(seed, k, i as u32))),
                C::Uint(c) => c.pixels_mut().iter_mut().enumerate().for_each(|(i, p)| *p = hash(seed, k, i as u32)),
            }
        }
    }
    img
}

fn levels<L: ImageLevel>(img: &Image<L>) -> Vec<(i32, i32)> {
    let mut v = Vec::new();
    for ly in 0..img.num_y_levels() {
        for lx in 0..img.num_x_levels() {
            if img.level_at(lx, ly).is_ok() {
                v.push((lx, ly));
            }
        }
    }
    v
}

/// Same levels, windows, channels and bit-identical samples.
fn assert_flat_eq(a: &FlatImage, b: &FlatImage) {
    use exr_util::flat_image_channel::FlatImageChannel as C;
    assert_eq!((a.level_mode(), a.level_rounding_mode()), (b.level_mode(), b.level_rounding_mode()));
    assert_eq!(levels(a), levels(b));
    assert_eq!(a.data_window(), b.data_window());
    for (lx, ly) in levels(a) {
        let (la, lb) = (a.level_at(lx, ly).unwrap(), b.level_at(lx, ly).unwrap());
        assert_eq!(la.data_window(), lb.data_window(), "level ({lx},{ly}) window");
        let names: Vec<_> = la.iter().map(|(n, c)| (n.clone(), c.channel())).collect();
        assert_eq!(names, lb.iter().map(|(n, c)| (n.clone(), c.channel())).collect::<Vec<_>>());
        for ((n, ca), (_, cb)) in la.iter().zip(lb.iter()) {
            let same = match (ca, cb) {
                (C::Half(x), C::Half(y)) => x.pixels().iter().map(|h| h.bits() as u32).eq(y.pixels().iter().map(|h| h.bits() as u32)),
                (C::Float(x), C::Float(y)) => x.pixels().iter().map(|f| f.to_bits()).eq(y.pixels().iter().map(|f| f.to_bits())),
                (C::Uint(x), C::Uint(y)) => x.pixels() == y.pixels(),
                _ => false,
            };
            assert!(same, "level ({lx},{ly}) channel {n}");
        }
    }
}

fn header(comp: Compression, tiles: Option<(u32, u32)>, order: LineOrder) -> Header {
    let mut h = Header::new();
    *h.compression_mut().unwrap() = comp;
    *h.line_order_mut().unwrap() = order;
    *h.display_window_mut().unwrap() = bx(-100, -100, 100, 100);
    if let Some((x_size, y_size)) = tiles {
        h.set_tile_description(TileDescription { x_size, y_size, ..Default::default() }).unwrap();
    }
    h
}

#[test]
fn flat_scanline_round_trips_with_every_lossless_codec() {
    for comp in [Compression::NoCompression, Compression::Rle, Compression::Zips, Compression::Zip, Compression::Piz] {
        let img = flat_image(bx(-6, -6, 23, 41), LevelMode::OneLevel, DOWN, true);
        let tmp = Tmp::new(&format!("flat-scan-{comp:?}"));
        let hdr = header(comp, None, LineOrder::IncreasingY);
        save_flat_image(&tmp.0, Some(&hdr), &img, DataWindowSource::Image).unwrap();
        let mut back = FlatImage::empty().unwrap();
        let got = load_flat_image(&tmp.0, &mut back).unwrap();
        assert_flat_eq(&img, &back);
        assert_eq!(got.compression().unwrap(), comp);
        assert_eq!(got.display_window().unwrap(), bx(-100, -100, 100, 100));
        assert!(!got.has_tile_description(), "a scanline load leaves out `tiles`");
    }
}

#[test]
fn the_files_channels_and_window_come_from_the_image_not_the_header() {
    let img = flat_image(bx(0, 0, 9, 9), LevelMode::OneLevel, DOWN, false);
    let mut hdr = header(Compression::Zip, None, LineOrder::IncreasingY);
    hdr.channels_mut().unwrap().insert("STALE", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    *hdr.data_window_mut().unwrap() = bx(5, 5, 500, 500);
    let tmp = Tmp::new("flat-stale-header");
    save_flat_image(&tmp.0, Some(&hdr), &img, DataWindowSource::Image).unwrap();
    let mut back = FlatImage::empty().unwrap();
    load_flat_image(&tmp.0, &mut back).unwrap();
    assert_flat_eq(&img, &back);
    assert!(back.level(0).unwrap().find_channel("STALE").is_none());
}

#[test]
fn flat_scanline_decreasing_y_and_strips_round_trip() {
    // 300 rows of 2000 pixels: several strips, and DECREASING_Y writes them bottom-up.
    let img = flat_image(bx(3, -7, 2002, 292), LevelMode::OneLevel, DOWN, false);
    for order in [LineOrder::IncreasingY, LineOrder::DecreasingY] {
        let tmp = Tmp::new(&format!("flat-strips-{order:?}"));
        save_flat_image(&tmp.0, Some(&header(Compression::Zip, None, order)), &img, DataWindowSource::Image).unwrap();
        let mut back = FlatImage::empty().unwrap();
        let got = load_flat_image(&tmp.0, &mut back).unwrap();
        assert_eq!(got.line_order().unwrap(), order);
        assert_flat_eq(&img, &back);
    }
}

#[test]
fn flat_tiled_round_trips_every_level_mode() {
    for (mode, rounding) in [
        (LevelMode::OneLevel, DOWN),
        (LevelMode::MipmapLevels, DOWN),
        (LevelMode::MipmapLevels, UP),
        (LevelMode::RipmapLevels, DOWN),
        (LevelMode::RipmapLevels, UP),
    ] {
        for order in [LineOrder::IncreasingY, LineOrder::DecreasingY, LineOrder::RandomY] {
            let img = flat_image(bx(-5, 3, 58, 44), mode, rounding, false);
            let tmp = Tmp::new(&format!("flat-tiled-{mode:?}-{rounding:?}-{order:?}"));
            save_flat_image(&tmp.0, Some(&header(Compression::Zip, Some((16, 8)), order)), &img, DataWindowSource::Image).unwrap();
            let mut back = FlatImage::empty().unwrap();
            let got = load_flat_image(&tmp.0, &mut back).unwrap();
            assert_flat_eq(&img, &back);
            let td = got.tile_description().unwrap();
            assert_eq!((td.x_size, td.y_size, td.mode, td.rounding_mode), (16, 8, mode, rounding));
        }
    }
}

#[test]
fn a_multi_level_image_without_a_header_is_saved_tiled_64() {
    let img = flat_image(bx(0, 0, 99, 69), LevelMode::MipmapLevels, DOWN, false);
    let tmp = Tmp::new("flat-default-header");
    save_flat_image(&tmp.0, None, &img, DataWindowSource::Image).unwrap();
    let mut back = FlatImage::empty().unwrap();
    let got = load_flat_image(&tmp.0, &mut back).unwrap();
    assert_flat_eq(&img, &back);
    let td = got.tile_description().unwrap();
    assert_eq!((td.x_size, td.y_size), (64, 64));
    assert_eq!(got.display_window().unwrap(), bx(0, 0, 99, 69), "default header: display window = data window");
}

#[test]
fn saving_with_the_header_window_crops() {
    let img = flat_image(bx(0, 0, 15, 11), LevelMode::OneLevel, DOWN, false);
    let mut hdr = header(Compression::Zip, None, LineOrder::IncreasingY);
    *hdr.data_window_mut().unwrap() = bx(4, 2, 40, 7); // partly outside the image: intersected
    let tmp = Tmp::new("flat-crop");
    save_flat_image(&tmp.0, Some(&hdr), &img, DataWindowSource::Header).unwrap();
    let mut back = FlatImage::empty().unwrap();
    load_flat_image(&tmp.0, &mut back).unwrap();
    assert_eq!(*back.data_window(), bx(4, 2, 15, 7));
    let (full, cut) = (img.level(0).unwrap(), back.level(0).unwrap());
    for c in ["R", "id"] {
        for y in 2..=7 {
            for x in 4..=15 {
                match c {
                    "R" => assert_eq!(
                        cut.typed_channel::<Half>(c).unwrap().at(x, y).unwrap().bits(),
                        full.typed_channel::<Half>(c).unwrap().at(x, y).unwrap().bits()
                    ),
                    _ => assert_eq!(cut.typed_channel::<u32>(c).unwrap().at(x, y), full.typed_channel::<u32>(c).unwrap().at(x, y)),
                }
            }
        }
    }
    let mip = flat_image(bx(0, 0, 15, 11), LevelMode::MipmapLevels, DOWN, false);
    assert!(save_flat_image(&tmp.0, Some(&hdr), &mip, DataWindowSource::Header).is_err(), "no cropping of mip maps");
}

/// A deep image with zero, small and large counts, all three sample types.
fn deep_image(dw: Box2i, mode: LevelMode, rounding: LevelRoundingMode) -> DeepImage {
    let mut img = DeepImage::new(&dw, mode, rounding).unwrap();
    img.insert_type("R", PixelType::Half).unwrap();
    img.insert_type("Z", PixelType::Float).unwrap();
    img.insert_type("id", PixelType::Uint).unwrap();
    for (lx, ly) in levels(&img) {
        let lv = img.level_at_mut(lx, ly).unwrap();
        let seed = (lx * 16 + ly) as u32;
        lv.edit_counts(|c| c.iter_mut().enumerate().for_each(|(i, n)| *n = [0, 1, 3, 0, 9, 2][hash(seed, 7, i as u32) as usize % 6])).unwrap();
        let (w, dw) = (lv.sample_counts().pixels_per_row(), *lv.data_window());
        for i in 0..lv.sample_counts().num_pixels() as i32 {
            let (x, y) = (dw.min.x + i % w, dw.min.y + i / w);
            lv.typed_channel_mut::<Half>("R").unwrap().at_mut(x, y).unwrap().iter_mut().enumerate().for_each(|(k, p)| *p = half(hash(seed, i as u32, k as u32)));
            lv.typed_channel_mut::<f32>("Z").unwrap().at_mut(x, y).unwrap().iter_mut().enumerate().for_each(|(k, p)| *p = float(hash(seed + 1, i as u32, k as u32)));
            lv.typed_channel_mut::<u32>("id").unwrap().at_mut(x, y).unwrap().iter_mut().enumerate().for_each(|(k, p)| *p = hash(seed + 2, i as u32, k as u32));
        }
    }
    img
}

fn assert_deep_eq(a: &DeepImage, b: &DeepImage) {
    assert_eq!((a.level_mode(), a.level_rounding_mode()), (b.level_mode(), b.level_rounding_mode()));
    assert_eq!(levels(a), levels(b));
    for (lx, ly) in levels(a) {
        let (la, lb) = (a.level_at(lx, ly).unwrap(), b.level_at(lx, ly).unwrap());
        assert_eq!(la.data_window(), lb.data_window());
        assert_eq!(la.sample_counts().num_samples(), lb.sample_counts().num_samples(), "level ({lx},{ly}) counts");
        let dw = *la.data_window();
        for y in dw.min.y..=dw.max.y {
            for x in dw.min.x..=dw.max.x {
                let bits = |l: &exr_util::deep_image_level::DeepImageLevel| {
                    (
                        l.typed_channel::<Half>("R").unwrap().at(x, y).unwrap().iter().map(|h| h.bits()).collect::<Vec<_>>(),
                        l.typed_channel::<f32>("Z").unwrap().at(x, y).unwrap().iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                        l.typed_channel::<u32>("id").unwrap().at(x, y).unwrap().to_vec(),
                    )
                };
                assert_eq!(bits(la), bits(lb), "level ({lx},{ly}) pixel ({x},{y})");
            }
        }
    }
}

#[test]
fn deep_scanline_round_trips() {
    for dw in [bx(0, 0, 10, 6), bx(-7, 4, 20, 30)] {
        let img = deep_image(dw, LevelMode::OneLevel, DOWN);
        let tmp = Tmp::new("deep-scan");
        let hdr = header(Compression::Zips, None, LineOrder::IncreasingY);
        save_deep_image(&tmp.0, Some(&hdr), &img, DataWindowSource::Image).unwrap();
        let mut back = DeepImage::empty().unwrap();
        let got = load_deep_image(&tmp.0, &mut back).unwrap();
        assert_deep_eq(&img, &back);
        assert_eq!(got.compression().unwrap(), Compression::Zips);
    }
}

#[test]
fn deep_compression_is_forced_to_zips() {
    let img = deep_image(bx(0, 0, 5, 5), LevelMode::OneLevel, DOWN);
    let tmp = Tmp::new("deep-zips");
    save_deep_image(&tmp.0, Some(&header(Compression::Piz, None, LineOrder::IncreasingY)), &img, DataWindowSource::Image).unwrap();
    let mut back = DeepImage::empty().unwrap();
    assert_eq!(load_deep_image(&tmp.0, &mut back).unwrap().compression().unwrap(), Compression::Zips);
}

#[test]
fn deep_tiled_round_trips_every_level_mode() {
    for (mode, rounding) in [
        (LevelMode::OneLevel, DOWN),
        (LevelMode::MipmapLevels, DOWN),
        (LevelMode::MipmapLevels, UP),
        (LevelMode::RipmapLevels, DOWN),
    ] {
        eprintln!("deep tiled {mode:?} {rounding:?}");
        let img = deep_image(bx(-3, 2, 18, 13), mode, rounding);
        let tmp = Tmp::new(&format!("deep-tiled-{mode:?}-{rounding:?}"));
        save_deep_image(&tmp.0, Some(&header(Compression::Zips, Some((8, 4)), LineOrder::IncreasingY)), &img, DataWindowSource::Image).unwrap();
        let mut back = DeepImage::empty().unwrap();
        let got = load_deep_image(&tmp.0, &mut back).unwrap();
        assert_deep_eq(&img, &back);
        assert_eq!(got.tile_description().unwrap().mode, mode);
    }
}

#[test]
fn deep_crop_keeps_the_covered_pixels() {
    let img = deep_image(bx(0, 0, 11, 8), LevelMode::OneLevel, DOWN);
    let mut hdr = header(Compression::Zips, None, LineOrder::IncreasingY);
    *hdr.data_window_mut().unwrap() = bx(3, 2, 8, 5);
    let tmp = Tmp::new("deep-crop");
    save_deep_image(&tmp.0, Some(&hdr), &img, DataWindowSource::Header).unwrap();
    let mut back = DeepImage::empty().unwrap();
    load_deep_image(&tmp.0, &mut back).unwrap();
    assert_eq!(*back.data_window(), bx(3, 2, 8, 5));
    let (full, cut) = (img.level(0).unwrap(), back.level(0).unwrap());
    for y in 2..=5 {
        for x in 3..=8 {
            assert_eq!(cut.sample_counts().at(x, y).unwrap(), full.sample_counts().at(x, y).unwrap());
            assert_eq!(
                cut.typed_channel::<u32>("id").unwrap().at(x, y).unwrap(),
                full.typed_channel::<u32>("id").unwrap().at(x, y).unwrap()
            );
        }
    }
}

#[test]
fn load_image_picks_the_kind_and_rejects_the_rest() {
    let flat = flat_image(bx(0, 0, 9, 9), LevelMode::OneLevel, DOWN, false);
    let deep = deep_image(bx(0, 0, 9, 9), LevelMode::OneLevel, DOWN);
    let (tf, td) = (Tmp::new("any-flat"), Tmp::new("any-deep"));
    save_image(&tf.0, None, &AnyImage::Flat(flat), DataWindowSource::Image).unwrap();
    save_image(&td.0, None, &AnyImage::Deep(deep), DataWindowSource::Image).unwrap();
    assert!(matches!(load_image(&tf.0).unwrap().0, AnyImage::Flat(_)));
    assert!(matches!(load_image(&td.0).unwrap().0, AnyImage::Deep(_)));
    let msg = |r: exr_core::Result<Header>| r.err().unwrap().to_string();
    assert!(msg(load_flat_image(&td.0, &mut FlatImage::empty().unwrap())).contains("as a flat image"));
    assert!(msg(load_deep_image(&tf.0, &mut DeepImage::empty().unwrap())).contains("as a deep image"));
    let junk = Tmp::new("junk");
    std::fs::write(&junk.0, b"not an exr file at all").unwrap();
    assert!(load_image(&junk.0).err().unwrap().to_string().contains("not an OpenEXR file"));
    assert!(load_image(golden("golden_multipart.exr")).err().unwrap().to_string().contains("Multi-part"));
}

/// Level 0 of a flat image as f32 planes, by channel name.
fn level0_planes(img: &FlatImage) -> Vec<(Name, Vec<f32>)> {
    use exr_util::flat_image_channel::FlatImageChannel as C;
    img.level(0)
        .unwrap()
        .iter()
        .map(|(n, c)| {
            let v = match c {
                C::Half(c) => c.pixels().iter().map(|h| h.to_f32()).collect(),
                C::Float(c) => c.pixels().to_vec(),
                C::Uint(c) => c.pixels().iter().map(|&u| u as f32).collect(),
            };
            (n.clone(), v)
        })
        .collect()
}

#[test]
fn flat_goldens_load_to_what_exr_core_reads() {
    let mut checked = 0;
    for entry in std::fs::read_dir(golden("")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".exr") || name.contains("deep") || name.contains("multipart") || name.contains("sub_") {
            continue;
        }
        // Files the facade cannot open (luminance-chroma, non-UTF-8 names, ...) are not comparable.
        let Ok(img) = exr_core::read(&path) else { continue };
        let mut mine = FlatImage::empty().unwrap();
        load_flat_image(&path, &mut mine).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut planes = level0_planes(&mine);
        let mut theirs: Vec<_> = img.channels().map(|(n, c)| (Name::from(n), c.to_f32())).collect();
        planes.sort_by(|a, b| a.0.cmp(&b.0));
        theirs.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(planes.len(), theirs.len(), "{name}: channel count");
        for ((n, a), (m, b)) in planes.iter().zip(&theirs) {
            assert_eq!(n, m, "{name}");
            assert!(a.iter().map(|f| f.to_bits()).eq(b.iter().map(|f| f.to_bits())), "{name}: channel {n}");
        }
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} goldens compared");
}

#[test]
fn subsampled_goldens_load_and_keep_their_grids() {
    for name in ["golden_sub_none_x2y2.exr", "golden_sub_zip_x2y1.exr", "golden_sub_piz_x1y2.exr", "golden_sub_rle_x2y2.exr"] {
        let mut img = FlatImage::empty().unwrap();
        load_flat_image(golden(name), &mut img).unwrap_or_else(|e| panic!("{name}: {e}"));
        let lv = img.level(0).unwrap();
        let dw = *lv.data_window();
        for (n, c) in lv.iter() {
            let (w, h) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1);
            assert_eq!((c.pixels_per_row(), c.pixels_per_column()), (w / c.x_sampling(), h / c.y_sampling()), "{name} {n}");
        }
        // and back out again, through the same strip code
        let tmp = Tmp::new("sub-golden");
        save_flat_image(&tmp.0, None, &img, DataWindowSource::Image).unwrap();
        let mut back = FlatImage::empty().unwrap();
        load_flat_image(&tmp.0, &mut back).unwrap();
        assert_flat_eq(&img, &back);
    }
}

#[test]
fn tiled_goldens_have_the_levels_the_header_says() {
    for (name, mode, rounding, nx, ny) in [
        ("golden_tiled_zip_mipmap.exr", LevelMode::MipmapLevels, DOWN, None, None),
        ("golden_tiled_zip_mipmap_roundup.exr", LevelMode::MipmapLevels, UP, None, None),
        ("golden_tiled_zip_ripmap.exr", LevelMode::RipmapLevels, DOWN, None, None),
        ("golden_tiled_zip_one_decreasing.exr", LevelMode::OneLevel, DOWN, Some(1), Some(1)),
        ("golden_tiled_zip_one_randomy.exr", LevelMode::OneLevel, DOWN, Some(1), Some(1)),
    ] {
        let mut img = FlatImage::empty().unwrap();
        let hdr = load_flat_image(golden(name), &mut img).unwrap_or_else(|e| panic!("{name}: {e}"));
        let td = hdr.tile_description().unwrap();
        assert_eq!((td.mode, td.rounding_mode), (mode, rounding), "{name}");
        assert_eq!((img.level_mode(), img.level_rounding_mode()), (mode, rounding), "{name}");
        // The level count the file itself reports.
        let mut stream = exr_core::io::StdIfStream::open(golden(name).to_str().unwrap()).unwrap();
        let f = exr_core::tiled::TiledInputFile::new(&mut stream).unwrap();
        assert_eq!((img.num_x_levels(), img.num_y_levels()), (f.num_x_levels(), f.num_y_levels()), "{name}");
        if let (Some(x), Some(y)) = (nx, ny) {
            assert_eq!((img.num_x_levels(), img.num_y_levels()), (x, y));
        }
        for (lx, ly) in levels(&img) {
            assert_eq!(*img.level_at(lx, ly).unwrap().data_window(), f.data_window_for_level(lx, ly).unwrap(), "{name} ({lx},{ly})");
        }
        // Loading what we saved changes nothing.
        let tmp = Tmp::new("tiled-golden");
        save_flat_image(&tmp.0, Some(&hdr), &img, DataWindowSource::Image).unwrap();
        let mut back = FlatImage::empty().unwrap();
        load_flat_image(&tmp.0, &mut back).unwrap();
        assert_flat_eq(&img, &back);
    }
}

/// Per pixel `(count, sum of the samples as FLOAT)` of channel `k` of a deep scanline file, read with
/// exr-core's raw API: counts raster first, then the flat heap (one FLOAT region per channel).
fn core_deep_digest(path: &Path, k: usize) -> (Box2i, Vec<(u32, f64)>, Vec<Name>) {
    use exr_core::deep::{DeepFrameBuffer, DeepScanLineInputFile, DeepSlice};
    use exr_core::frame_buffer::Slice;
    let mut f = DeepScanLineInputFile::open(path).unwrap();
    let hdr = f.header().clone();
    let dw = hdr.data_window().unwrap();
    let (w, h) = ((dw.max.x - dw.min.x + 1) as usize, (dw.max.y - dw.min.y + 1) as usize);
    let names: Vec<Name> = hdr.channels().unwrap().iter().map(|(n, _)| n.clone()).collect();
    let ys = w as i64 * 4;
    let count = Slice::new(PixelType::Uint, -(dw.min.y as i64 * ys + dw.min.x as i64 * 4), 4, ys as usize, 1, 1, 0.0, false, false).unwrap();
    let fb = |heap: i64| {
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(count.clone()).unwrap();
        for (k, n) in names.iter().enumerate() {
            fb.insert(n.clone(), DeepSlice::new(PixelType::Float, heap * k as i64, 8, 0, 4, 1, 1, 0.0, false, false).unwrap()).unwrap();
        }
        fb
    };
    let mut counts = vec![0u8; w * h * 4];
    f.set_frame_buffer(&fb(0)).unwrap();
    f.read_pixel_sample_counts(dw.min.y, dw.max.y, &mut counts).unwrap();
    let n: Vec<u32> = counts.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let total: usize = n.iter().map(|&c| c as usize).sum();
    let mut heap = vec![0u8; total * 4 * names.len()];
    f.set_frame_buffer(&fb((total * 4) as i64)).unwrap();
    f.read_pixels(dw.min.y, dw.max.y, &counts, &mut heap).unwrap();
    let region = total * 4 * k;
    let mut start = 0usize;
    let digest = n
        .iter()
        .map(|&c| {
            let s: f64 = (0..c as usize)
                .map(|j| f32::from_ne_bytes(heap[region + (start + j) * 4..region + (start + j) * 4 + 4].try_into().unwrap()) as f64)
                .sum();
            start += c as usize;
            (c, s)
        })
        .collect();
    (dw, digest, names)
}

#[test]
fn deep_scanline_goldens_match_exr_cores_raw_reader() {
    for name in ["golden_deep_scanline_none.exr", "golden_deep_scanline_zips.exr", "golden_deep_scanline_rle.exr", "golden_deep_scanline_none_offset.exr"] {
        let mut img = DeepImage::empty().unwrap();
        load_deep_image(golden(name), &mut img).unwrap_or_else(|e| panic!("{name}: {e}"));
        let lv = img.level(0).unwrap();
        // Compare every channel that converts to FLOAT exactly (not the 32-bit integers).
        let mut compared = 0;
        for (k, (cname, ch)) in lv.iter().enumerate() {
            if ch.pixel_type() == PixelType::Uint {
                continue;
            }
            let (dw, digest, names) = core_deep_digest(&golden(name), k);
            assert_eq!(names[k], *cname, "{name}: channel order");
            assert_eq!(*lv.data_window(), dw);
            let mut i = 0;
            for y in dw.min.y..=dw.max.y {
                for x in dw.min.x..=dw.max.x {
                    let (n, sum) = match ch.pixel_type() {
                        PixelType::Half => {
                            let s = lv.typed_channel::<Half>(cname).unwrap().at(x, y).unwrap();
                            (s.len(), s.iter().map(|h| h.to_f32() as f64).sum::<f64>())
                        }
                        _ => {
                            let s = lv.typed_channel::<f32>(cname).unwrap().at(x, y).unwrap();
                            (s.len(), s.iter().map(|&f| f as f64).sum::<f64>())
                        }
                    };
                    assert_eq!((n as u32, sum), digest[i], "{name} channel {cname} pixel ({x},{y})");
                    i += 1;
                }
            }
            compared += 1;
        }
        assert!(compared > 0, "{name}: nothing compared");
    }
}

#[test]
fn deep_tiled_goldens_round_trip_with_their_levels() {
    for name in [
        "golden_deep_tiled_none_mipmap.exr",
        "golden_deep_tiled_zips_mipmap.exr",
        "golden_deep_tiled_none_one.exr",
        "golden_deep_tiled_rle_one.exr",
        "golden_deep_tiled_zips_one.exr",
    ] {
        let mut img = DeepImage::empty().unwrap();
        let hdr = load_deep_image(golden(name), &mut img).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(levels(&img).iter().any(|&(x, y)| img.level_at(x, y).unwrap().sample_counts().total_samples() > 0), "{name}: has samples");
        let tmp = Tmp::new("deep-tiled-golden");
        save_deep_image(&tmp.0, Some(&hdr), &img, DataWindowSource::Image).unwrap();
        let mut back = DeepImage::empty().unwrap();
        load_deep_image(&tmp.0, &mut back).unwrap();
        let (a, b) = (levels(&img), levels(&back));
        assert_eq!(a, b, "{name}");
        for (x, y) in a {
            let (la, lb) = (img.level_at(x, y).unwrap(), back.level_at(x, y).unwrap());
            assert_eq!(la.sample_counts().num_samples(), lb.sample_counts().num_samples(), "{name} ({x},{y})");
            for (n, ca) in la.iter() {
                let dw = *la.data_window();
                let first = |l: &exr_util::deep_image_level::DeepImageLevel| -> Vec<u64> {
                    let mut v = Vec::new();
                    for yy in dw.min.y..=dw.max.y {
                        for xx in dw.min.x..=dw.max.x {
                            match ca.pixel_type() {
                                PixelType::Half => v.extend(l.typed_channel::<Half>(n).unwrap().at(xx, yy).unwrap().iter().map(|h| h.bits() as u64)),
                                PixelType::Float => v.extend(l.typed_channel::<f32>(n).unwrap().at(xx, yy).unwrap().iter().map(|f| f.to_bits() as u64)),
                                _ => v.extend(l.typed_channel::<u32>(n).unwrap().at(xx, yy).unwrap().iter().map(|&u| u as u64)),
                            }
                        }
                    }
                    v
                };
                assert_eq!(first(la), first(lb), "{name} ({x},{y}) channel {n}");
            }
        }
    }
}

/// The deep goldens come from the C++ oracle with a known formula (`oracle.cpp:818`): the pixel
/// `(lx, ly)` of a level, counted from the level's corner, has `(lx + ly) % 4` samples whose values
/// depend on `pidx = ly * levelWidth + lx`. This is the independent check of every level.
#[test]
fn deep_goldens_hold_the_cpp_oracles_formula_on_every_level() {
    for name in [
        "golden_deep_scanline_none.exr",
        "golden_deep_scanline_none_offset.exr",
        "golden_deep_scanline_zips.exr",
        "golden_deep_tiled_none_mipmap.exr",
        "golden_deep_tiled_zips_mipmap.exr",
        "golden_deep_tiled_none_one.exr",
        "golden_deep_tiled_zips_one.exr",
    ] {
        let mut img = DeepImage::empty().unwrap();
        load_deep_image(golden(name), &mut img).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (lx, ly) in levels(&img) {
            let lv = img.level_at(lx, ly).unwrap();
            let dw = *lv.data_window();
            let w = dw.max.x - dw.min.x + 1;
            for y in 0..=(dw.max.y - dw.min.y) {
                for x in 0..w {
                    let (px, py, pidx) = (dw.min.x + x, dw.min.y + y, (y * w + x) as f32);
                    let n = ((x + y) % 4) as usize;
                    let at = |c: &str| lv.typed_channel::<Half>(c).unwrap().at(px, py).unwrap().to_vec();
                    let r = at("R");
                    let z = lv.typed_channel::<f32>("Z").unwrap().at(px, py).unwrap();
                    assert_eq!((r.len(), z.len()), (n, n), "{name} level ({lx},{ly}) pixel ({x},{y}) count");
                    for s in 0..n {
                        let sf = s as f32;
                        assert_eq!(r[s].bits(), Half::from_f32(1.0 + pidx * 0.5 + sf).bits(), "{name} ({lx},{ly}) R");
                        assert_eq!(at("G")[s].bits(), Half::from_f32(100.0 + pidx * 0.25 + sf * 2.0).bits(), "{name} G");
                        assert_eq!(at("B")[s].bits(), Half::from_f32(-50.0 + pidx - sf).bits(), "{name} B");
                        assert_eq!(z[s], pidx * 100.0 + sf, "{name} ({lx},{ly}) Z");
                    }
                }
            }
        }
    }
}

/// The tiled flat goldens come from the C++ oracle too (`oracle.cpp:584`): level `(lx, ly)` holds,
/// at level pixel index `tidx = y * levelWidth + x` and `bias = lx * 7 + ly * 13`, the values below.
#[test]
fn tiled_goldens_hold_the_cpp_oracles_formula_on_every_level() {
    for name in [
        "golden_tiled_zip_mipmap.exr",
        "golden_tiled_zip_mipmap_roundup.exr",
        "golden_tiled_zip_ripmap.exr",
        "golden_tiled_none_mipmap.exr",
        "golden_tiled_piz_mipmap.exr",
        "golden_tiled_none_one_offset.exr",
        "golden_tiled_zip_one_decreasing.exr",
    ] {
        let mut img = FlatImage::empty().unwrap();
        load_flat_image(golden(name), &mut img).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut levels_seen = 0;
        for (lx, ly) in levels(&img) {
            let lv = img.level_at(lx, ly).unwrap();
            let dw = *lv.data_window();
            let w = dw.max.x - dw.min.x + 1;
            let bias = (lx * 7 + ly * 13) as f32;
            for y in 0..=(dw.max.y - dw.min.y) {
                for x in 0..w {
                    let (px, py, t) = (dw.min.x + x, dw.min.y + y, (y * w + x) as f32);
                    let half_at = |c: &str| lv.typed_channel::<Half>(c).unwrap().at(px, py).unwrap().bits();
                    let ctx = format!("{name} level ({lx},{ly}) pixel ({x},{y})");
                    assert_eq!(half_at("R"), Half::from_f32(1.0 + (t + bias) * 0.5).bits(), "{ctx} R");
                    assert_eq!(half_at("G"), Half::from_f32(100.0 + t * 0.25 + bias).bits(), "{ctx} G");
                    assert_eq!(half_at("B"), Half::from_f32(-50.0 + t - bias).bits(), "{ctx} B");
                    assert_eq!(*lv.typed_channel::<f32>("A").unwrap().at(px, py).unwrap(), (((t + bias) as i32) % 16) as f32 * 0.013 + 0.25, "{ctx} A");
                    assert_eq!(*lv.typed_channel::<f32>("Z").unwrap().at(px, py).unwrap(), (t + bias) * 4.0, "{ctx} Z");
                    assert_eq!(*lv.typed_channel::<u32>("U").unwrap().at(px, py).unwrap(), (t as u32) * 3 + bias as u32 + 11, "{ctx} U");
                }
            }
            levels_seen += 1;
        }
        assert!(levels_seen >= 1, "{name}");
    }
}

/// Real C++ OpenEXR (`exrcheck`, which reads every part through the library) accepts every file
/// `save*Image` writes. Run by `cargo xtask parity-tools`, which builds the tool and sets
/// `EXR_TOOLS_DIR`.
///
/// Deep RIPMAP and deep scanline files are checked with `-c` (the OpenEXRCore path) only: the
/// library path of `exrcheck` calls deep scanline files without `Z` or `A` bad (not reproduced, see `check_file::cannot_flatten`) and has a bug for deep tiled
/// ones (`ImfCheckFile.cpp:698`, not reproduced, see `check_accepts_valid_wide_deep_tiled_and_ripmap_files`).
#[test]
#[ignore = "needs the C++ tools: cargo xtask parity-tools (EXR_TOOLS_DIR)"]
fn cpp_exrcheck_accepts_what_we_save() {
    let mut files: Vec<(Tmp, &str)> = Vec::new();
    let mut add = |name: String, arg: &'static str, save: &dyn Fn(&Path)| {
        let t = Tmp::new(&name);
        save(&t.0);
        files.push((t, arg));
    };
    for (mode, rounding) in [(LevelMode::MipmapLevels, DOWN), (LevelMode::MipmapLevels, UP), (LevelMode::RipmapLevels, DOWN)] {
        let flat = flat_image(bx(-5, 3, 58, 44), mode, rounding, false);
        add(format!("chk-flat-{mode:?}-{rounding:?}"), "-m", &|p| {
            save_flat_image(p, Some(&header(Compression::Zip, Some((16, 8)), LineOrder::IncreasingY)), &flat, DataWindowSource::Image).unwrap();
        });
        let deep = deep_image(bx(-3, 2, 18, 13), mode, rounding);
        let arg = if mode == LevelMode::RipmapLevels { "-c" } else { "-t" };
        add(format!("chk-deep-{mode:?}-{rounding:?}"), arg, &|p| {
            save_deep_image(p, Some(&header(Compression::Zips, Some((8, 4)), LineOrder::IncreasingY)), &deep, DataWindowSource::Image).unwrap();
        });
    }
    for order in [LineOrder::IncreasingY, LineOrder::DecreasingY] {
        let img = flat_image(bx(-6, -6, 23, 41), LevelMode::OneLevel, DOWN, true);
        add(format!("chk-scan-{order:?}"), "-t", &|p| {
            save_flat_image(p, Some(&header(Compression::Piz, None, order)), &img, DataWindowSource::Image).unwrap();
        });
    }
    let deep = deep_image(bx(0, 0, 10, 6), LevelMode::OneLevel, DOWN);
    add("chk-deep-scan".into(), "-c", &|p| save_deep_image(p, None, &deep, DataWindowSource::Image).unwrap());
    for (f, arg) in &files {
        let out = cpp_exrcheck().arg(arg).arg(&f.0).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success() && text.trim_end().ends_with("OK"), "{} ({arg}): {text}", f.0.display());
    }
}

fn cpp_exrcheck() -> std::process::Command {
    let dir = std::env::var_os("EXR_TOOLS_DIR").map(PathBuf::from).expect("EXR_TOOLS_DIR");
    std::process::Command::new(dir.join(if cfg!(windows) { "exrcheck.exe" } else { "exrcheck" }))
}

/// A valid deep tiled file with more tile columns than tile rows: the library path of C++
/// `exrcheck` calls it bad (`ImfCheckFile.cpp:698`), the OpenEXRCore path `-c` does not, and
/// `exr-rs check` calls it OK in both (the deliberate difference, see `check_file`).
#[test]
#[ignore = "needs the C++ tools: cargo xtask parity-tools (EXR_TOOLS_DIR)"]
fn cpp_exrcheck_rejects_the_wide_deep_file_we_accept() {
    let t = wide_deep_file("chk-deep-wide-cpp");
    let cpp = |arg: &str| String::from_utf8_lossy(&cpp_exrcheck().arg(arg).arg(&t.0).output().unwrap().stdout).trim_end().ends_with("OK");
    let ours = |core: bool| !exr_util::check_file::check_file(t.0.to_str().unwrap(), exr_util::check_file::CheckOpts { core, ..Default::default() });
    assert_eq!((cpp("-t"), cpp("-c")), (false, true));
    assert_eq!((ours(false), ours(true)), (true, true));
}

fn wide_deep_file(name: &str) -> Tmp {
    // 40 x 8 pixels in 8 x 4 tiles: 5 tile columns, 2 tile rows.
    let img = deep_image(bx(0, 0, 39, 7), LevelMode::OneLevel, DOWN);
    let t = Tmp::new(name);
    save_deep_image(&t.0, Some(&header(Compression::Zips, Some((8, 4)), LineOrder::IncreasingY)), &img, DataWindowSource::Image).unwrap();
    t
}

/// Better than the C++: `readDeepTile` asks for the tiles `[min (x, y), max (x, y)]` squared instead of
/// tile `(x, y)` (`ImfCheckFile.cpp:698`), which makes `exrcheck` reject valid deep tiled files with more
/// tile columns than rows, and deep RIPMAP files with a wide-short level. The port asks for the tile it
/// means to, so these files are OK; real damage is still found.
#[test]
fn check_accepts_valid_wide_deep_tiled_and_ripmap_files() {
    use exr_util::check_file::{check_bytes, check_file, CheckOpts};
    let p = |t: &Tmp| t.0.to_str().unwrap().to_string();
    let modes = [CheckOpts::default(), CheckOpts { core: true, ..Default::default() }, CheckOpts { reduce_memory: true, ..Default::default() }];
    let wide = wide_deep_file("chk-wide");
    let rip = Tmp::new("chk-rip");
    save_deep_image(&rip.0, Some(&header(Compression::Zips, Some((8, 4)), LineOrder::IncreasingY)), &deep_image(bx(-3, 2, 18, 13), LevelMode::RipmapLevels, DOWN), DataWindowSource::Image).unwrap();
    for t in [&wide, &rip] {
        for o in modes {
            assert!(!check_file(&p(t), o), "{} {o:?}", t.0.display());
        }
    }
    // a flipped byte in the first tile's count table (leader: 4 coordinates and 3 sizes, 40 bytes) is still found
    let mut bytes = std::fs::read(&wide.0).unwrap();
    let at = deep_chunks(&bytes)[0].0 + 43;
    bytes[at] ^= 0xff;
    assert!(check_bytes(&bytes, CheckOpts::default()));
}

/// The strongest check there is: the image loaded from a file the C++ library wrote, saved again
/// with its own header, is the same file byte for byte (level order, tile order, chunking, header
/// attribute order, every codec). Exceptions, each for a reason the C++ savers share:
/// deep files are always saved as ZIPS (`ImfDeepImageIO.cpp:155`), DWA re-encodes decoded (lossy)
/// data, and the ZIP level is not stored in the file.
#[test]
fn resaving_the_cpp_goldens_reproduces_them_byte_for_byte() {
    let (mut same, mut skipped) = (0, Vec::new());
    for entry in std::fs::read_dir(golden("")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".exr") || name.contains("multipart") {
            continue;
        }
        let (img, hdr) = load_image(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let comp = hdr.compression().unwrap();
        let deep = matches!(img, AnyImage::Deep(_));
        if (deep && comp != Compression::Zips) || matches!(comp, Compression::Dwaa | Compression::Dwab) || name.contains("_level") {
            skipped.push(name);
            continue;
        }
        let t = Tmp::new("resave");
        save_image(&t.0, Some(&hdr), &img, DataWindowSource::Image).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (orig, out) = (std::fs::read(&path).unwrap(), std::fs::read(&t.0).unwrap());
        let first = orig.iter().zip(&out).position(|(a, b)| a != b);
        assert!(first.is_none() && orig.len() == out.len(), "{name}: {} vs {} bytes, first difference at {first:?}", orig.len(), out.len());
        same += 1;
    }
    assert!(same >= 75, "only {same} goldens compared, skipped {skipped:?}");
    assert!(skipped.len() <= 30, "{skipped:?}");
}

/// A strip frame buffer with renamed slices reads the file channel `R` into the image channel `right.R`
/// (what a multi-view tool needs), and `unpack` still finds the planes by the image names.
#[test]
fn a_renamed_frame_buffer_reads_file_channel_r_into_right_r() {
    use exr_core::scanline::InputFile;
    use exr_util::flat_image_io::unpack;
    use exr_util::image_io::Strip;
    let src = flat_image(bx(-3, 2, 12, 9), LevelMode::OneLevel, DOWN, false);
    let tmp = Tmp::new("renamed-fb");
    save_flat_image(&tmp.0, None, &src, DataWindowSource::Image).unwrap();

    let dw = bx(-3, 2, 12, 9);
    let mut dst = FlatImage::new(&dw, LevelMode::OneLevel, DOWN).unwrap();
    dst.insert_type("right.R", PixelType::Half).unwrap();
    let lv = dst.level_mut(0).unwrap();
    let chans: Vec<_> = lv.iter().map(|(n, c)| (n.clone(), c.channel())).collect();
    let strip = Strip::new(chans.iter().map(|(n, c)| (n, *c)), (-3, 12), (2, 9)).unwrap();
    let fb = strip.frame_buffer_with(|n| Name::from(n.to_string().trim_start_matches("right."))).unwrap();
    assert!(fb.get("R").is_ok() && fb.get("right.R").is_err());
    let mut inp = InputFile::open(&tmp.0).unwrap();
    inp.set_frame_buffer(&fb).unwrap();
    let mut buf = vec![0u8; strip.bytes];
    inp.read_pixels(2, 9, &mut buf).unwrap();
    unpack(lv, &strip, &buf).unwrap();
    let (got, want) = (lv.typed_channel::<Half>("right.R").unwrap(), src.level(0).unwrap().typed_channel::<Half>("R").unwrap());
    assert!(got.pixels().iter().map(|h| h.bits()).eq(want.pixels().iter().map(|h| h.bits())));
    assert!(want.pixels().iter().any(|h| h.bits() != 0));
}

// ---- crafted files for the `-m` limits of `exrcheck` ---------------------------------------------

/// Bytes of a saved flat gradient of `n` FLOAT channels, ZIP compressed: small on disk, large in memory.
fn gradient_file(w: i32, h: i32, n: usize, tiles: Option<(u32, u32)>) -> Vec<u8> {
    let mut img = FlatImage::new(&bx(0, 0, w - 1, h - 1), LevelMode::OneLevel, DOWN).unwrap();
    for k in 0..n {
        img.insert_type(format!("c{k:02}"), PixelType::Float).unwrap();
    }
    for (k, (_, ch)) in img.level_mut(0).unwrap().iter_mut().enumerate() {
        if let exr_util::flat_image_channel::FlatImageChannel::Float(c) = ch {
            c.pixels_mut().iter_mut().enumerate().for_each(|(i, p)| *p = (i % 4096) as f32 * 0.25 + k as f32);
        }
    }
    let t = Tmp::new(&format!("gradient-{w}x{h}x{n}-{}", tiles.is_some()));
    save_flat_image(&t.0, Some(&header(Compression::Zip, tiles, LineOrder::IncreasingY)), &img, DataWindowSource::Image).unwrap();
    std::fs::read(&t.0).unwrap()
}

/// Bytes of a saved deep file with `n` samples in every pixel (channels half, float, uint), scanline
/// or (with `tiles`) tiled.
fn deep_file(w: i32, h: i32, n: u32, tiles: Option<(u32, u32)>) -> Vec<u8> {
    let mut img = DeepImage::new(&bx(0, 0, w - 1, h - 1), LevelMode::OneLevel, DOWN).unwrap();
    img.insert_type("R", PixelType::Half).unwrap();
    img.insert_type("Z", PixelType::Float).unwrap();
    img.insert_type("id", PixelType::Uint).unwrap();
    let lv = img.level_mut(0).unwrap();
    lv.edit_counts(|c| c.iter_mut().for_each(|x| *x = n)).unwrap();
    for y in 0..h {
        for x in 0..w {
            let mut view = lv.typed_channel_mut::<f32>("Z").unwrap();
            let z = view.at_mut(x, y).unwrap();
            z.iter_mut().enumerate().for_each(|(k, p)| *p = (hash(x as u32, y as u32, k as u32) & 0xffff) as f32);
        }
    }
    let t = Tmp::new(&format!("deep-{w}x{h}x{n}-{}", tiles.is_some()));
    save_deep_image(&t.0, Some(&header(Compression::Zips, tiles, LineOrder::IncreasingY)), &img, DataWindowSource::Image).unwrap();
    std::fs::read(&t.0).unwrap()
}

/// Every chunk of a deep scanline file: the offset of its leader and the byte range of its packed
/// sample data.
fn deep_chunks(d: &[u8]) -> Vec<(usize, std::ops::Range<usize>)> {
    let u32_at = |i: usize| u32::from_le_bytes(d[i..i + 4].try_into().unwrap()) as usize;
    let u64_at = |i: usize| u64::from_le_bytes(d[i..i + 8].try_into().unwrap()) as usize;
    let mut i = 8; // after magic and version: attributes up to the empty name
    while d[i] != 0 {
        let name = d[i..].iter().position(|&b| b == 0).unwrap();
        let ty = d[i + name + 1..].iter().position(|&b| b == 0).unwrap();
        i += name + 1 + ty + 1;
        i += 4 + u32_at(i);
    }
    i += 1;
    // One offset per scanline (a deep chunk is one line): entries until one stops pointing past the table.
    let mut offs = Vec::new();
    while offs.len() < 4096 {
        let o = u64_at(i + offs.len() * 8);
        if o < i + (offs.len() + 1) * 8 || o >= d.len() {
            break;
        }
        offs.push(o);
    }
    // chunk: y (4), packed count table size, packed data size, unpacked size (3 x 8), table, data
    offs.iter().map(|&o| (o, o + 28 + u64_at(o + 4)..o + 28 + u64_at(o + 4) + u64_at(o + 12))).collect()
}

fn flipped(d: &[u8], pos: usize) -> Vec<u8> {
    let mut v = d.to_vec();
    v[pos] ^= 0xff;
    v
}

/// The crafted cases: `(name, damaged bytes)`. Each one sits between a limit of `ImfCheckFile.cpp` and
/// half (or double) of it, so a limit changed by a factor 2 changes a verdict below. (The deep limits
/// cannot be shown this way: the flat (compositing) read of a deep scanline part sees the same damage
/// whatever the limit says, and for deep tiles the `1000` bound decides first; `check_file`'s unit tests pin those predicates.)
fn crafted() -> Vec<(&'static str, Vec<u8>)> {
    // `-m` limits images to 2048 x 2048 (`:1101`), so a part is "wide" (`:866`) through its channels:
    // 2048 x 200 bytes x 16 lines = 6.5e6, below 8e6 (`:45`) and above 4e6
    let wide = gradient_file(2048, 16, 50, None);
    // 4 tiles of 512 x 512 x 8 bytes per row (`gMaxTileBytes` is 1e6): large and, by width, also "wide"
    let large = gradient_file(2048, 512, 2, Some((512, 512)));
    // One 256 x 256 deep tile of 4 FLOAT channels, no samples: `tileSize * max (bytesPerPixel, 4)` is
    // 1.05e6, a "large tile" (`:893`), but the tile check's own `tileSize * 4` (`:605`) is 262144: only
    // the outer `!reduceMemory || !largeTiles` gate keeps `-m` away from the damaged count table.
    let (deep_tile, tile_at) = {
        let mut img = DeepImage::new(&bx(0, 0, 255, 255), LevelMode::OneLevel, DOWN).unwrap();
        for c in ["A", "B", "C", "D"] {
            img.insert_type(c, PixelType::Float).unwrap();
        }
        let t = Tmp::new("deep-tile-zero");
        save_deep_image(&t.0, Some(&header(Compression::Zips, Some((256, 256)), LineOrder::IncreasingY)), &img, DataWindowSource::Image).unwrap();
        let d = std::fs::read(&t.0).unwrap();
        let o = deep_chunks(&d)[0].0; // the leader of the only tile: 4 coordinates, 3 sizes, then the table
        (d, o + 40)
    };
    vec![
        ("wide scanline part", flipped(&wide, wide.len() - 50)),
        ("large tile", flipped(&large, large.len() - 50)),
        ("large deep tile, damaged count table", flipped(&deep_tile, tile_at + 43)),
    ]
}

/// Verdicts of `exrcheck` (`-m` and default) on the crafted files, found with the C++ tool (see
/// `crafted_cases_match_exrcheck`, which checks them again when the tools are built): `(-m, default)`,
/// `true` = bad.
const CRAFTED_VERDICTS: [(bool, bool); 3] = [(true, true), (false, true), (false, true)];

#[test]
fn reduce_memory_limits_decide_what_is_read() {
    use exr_util::check_file::{check_bytes, CheckOpts};
    for ((name, bytes), (m, plain)) in crafted().iter().zip(CRAFTED_VERDICTS) {
        let got = (check_bytes(bytes, CheckOpts { reduce_memory: true, ..Default::default() }), check_bytes(bytes, CheckOpts::default()));
        assert_eq!(got, (m, plain), "{name}");
    }
}

#[test]
#[ignore = "needs the C++ tools: cargo xtask parity-tools (EXR_TOOLS_DIR)"]
fn crafted_cases_match_exrcheck() {
    for ((name, bytes), (m, plain)) in crafted().iter().zip(CRAFTED_VERDICTS) {
        let t = Tmp::new(&format!("crafted-{}", name.replace(' ', "-")));
        std::fs::write(&t.0, bytes).unwrap();
        for (args, want) in [(&["-m"][..], m), (&[][..], plain)] {
            let out = String::from_utf8_lossy(&cpp_exrcheck().args(args).arg(&t.0).output().unwrap().stdout).trim_end().to_string();
            assert_eq!(out.ends_with("bad"), want, "{name} {args:?}: {out}");
        }
    }
}

/// Crafted deep cases for the core check `(name, damaged bytes, (-c, -c -m) is bad)`:
/// - a line of 4800 sample bytes, above `gMaxBytesPerDeepScanline`: not decoded into memory, but the
///   payload is still decompressed, so a flipped payload byte is found;
/// - one 320 x 320 tile of 1 sample per pixel (1 MB of samples, above `gMaxTileBytes`): with `-m` the
///   decoder is not run at all, so the same flip is not found.
fn crafted_deep() -> Vec<(&'static str, Vec<u8>, (bool, bool))> {
    let high = deep_file(20, 6, 40, None);
    let at = deep_chunks(&high)[2].1.start + 20;
    let huge = deep_file(320, 320, 1, Some((320, 320)));
    let empty = deep_file(20, 6, 0, None);
    // byte `k` of the unpacked-size field (leader offset 20) of line 3
    let size_at = |d: &[u8], k: usize| deep_chunks(d)[3].0 + 20 + k;
    vec![
        ("deep line above the decode limit", flipped(&high, at), (true, true)),
        ("deep line without samples, payload size wrong", flipped(&empty, size_at(&empty, 0)), (true, true)),
        ("deep line, unpacked size too large", flipped(&high, size_at(&high, 2)), (true, true)),
        ("deep tile above the memory limit", flipped(&huge, huge.len() - 50), (true, false)),
    ]
}

#[test]
fn the_core_check_decodes_what_it_does_not_keep_and_skips_what_reduce_memory_says() {
    use exr_util::check_file::{check_bytes, CheckOpts};
    for (name, bytes, (plain, reduced)) in crafted_deep() {
        let core = |m| check_bytes(&bytes, CheckOpts { reduce_memory: m, core: true, ..Default::default() });
        assert_eq!((core(false), core(true)), (plain, reduced), "{name}");
    }
}

#[test]
#[ignore = "needs the C++ tools: cargo xtask parity-tools (EXR_TOOLS_DIR)"]
fn crafted_deep_cases_match_exrcheck() {
    for (name, bytes, (plain, reduced)) in crafted_deep() {
        let t = Tmp::new(&format!("crafted-{}", name.replace(' ', "-")));
        std::fs::write(&t.0, bytes).unwrap();
        for (args, want) in [(&["-c"][..], plain), (&["-c", "-m"][..], reduced)] {
            let out = String::from_utf8_lossy(&cpp_exrcheck().args(args).arg(&t.0).output().unwrap().stdout).trim_end().to_string();
            assert_eq!(out.ends_with("bad"), want, "{name} {args:?}: {out}");
        }
    }
}


/// A failing save leaves an existing file byte-identical and no temporary file behind (`io::write_file`:
/// the bytes go to a temporary file next to the target that is renamed over it only when complete).
#[test]
fn a_failing_save_leaves_the_existing_file_and_no_temp_file() {
    let dir = std::env::temp_dir().join(format!("exr-util-atomic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let flat = flat_image(bx(0, 0, 7, 7), LevelMode::OneLevel, DOWN, false);
    let deep = deep_image(bx(0, 0, 7, 7), LevelMode::OneLevel, DOWN);
    let scan_bad = header(Compression::Zip, None, LineOrder::RandomY); // RANDOM_Y is tiled only
    let tiled_bad = {
        let mut h = header(Compression::Zip, Some((8, 8)), LineOrder::IncreasingY);
        h.set_tile_description(TileDescription { x_size: 0, ..Default::default() }).unwrap();
        h
    };
    type Save<'a> = Box<dyn Fn(&Path) -> exr_core::Result<()> + 'a>;
    let cases: Vec<(&str, Save)> = vec![
        ("flat scanline", Box::new(|p| save_flat_image(p, Some(&scan_bad), &flat, DataWindowSource::Image))),
        ("flat tiled", Box::new(|p| save_flat_image(p, Some(&tiled_bad), &flat, DataWindowSource::Image))),
        ("deep scanline", Box::new(|p| save_deep_image(p, Some(&scan_bad), &deep, DataWindowSource::Image))),
        ("deep tiled", Box::new(|p| save_deep_image(p, Some(&tiled_bad), &deep, DataWindowSource::Image))),
    ];
    for (what, save) in cases {
        let target = dir.join(format!("{}.exr", what.replace(' ', "-")));
        std::fs::write(&target, b"the previous file").unwrap();
        assert!(save(&target).is_err(), "{what}: the save must fail");
        assert_eq!(std::fs::read(&target).unwrap(), b"the previous file", "{what}: target changed");
        // no target at all when there was none
        let fresh = dir.join(format!("fresh-{}.exr", what.replace(' ', "-")));
        assert!(save(&fresh).is_err());
        assert!(!fresh.exists(), "{what}: a failed save created the target");
    }
    let left: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".tmp")).collect();
    assert!(left.is_empty(), "temporary files left: {left:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}
