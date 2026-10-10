//! Subsampled files through `GpuLoader` (CPU only, no GPU):
//!
//! - `open` keeps every channel at its NATIVE sampling (low-level `FrameBuffer`
//!   path) — checked byte for byte against the C++ `InputFile` decode dumps of
//!   the `golden_sub_*` matrix (`oracle.cpp` `dump-sub`), all four samplings;
//! - `open_rgba` reconstructs luminance-chroma to RGBA — checked bit for bit
//!   against the C++ `RgbaInputFile` dump of `golden_yc.exr` (`rgba-read`).
//!
//! Goldens come from `cargo run -p xtask -- parity-exr`.

use std::path::PathBuf;

// SSOT: one skip helper for the whole workspace. Cargo does not share `tests/`
// modules across crates, so the file is included by path rather than copied.
#[path = "../../exr-core/tests/support/mod.rs"]
mod support;

const HOW_TO_GET: &str = "Run `cargo run -p xtask -- parity-exr` to regenerate the C++ goldens.";

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../exr-core/tests/parity").join(name)
}

fn half_words(bytes: &[u8], le: bool) -> Vec<u16> {
    bytes
        .as_chunks::<2>().0.iter()
        .map(|b| if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_ne_bytes([b[0], b[1]]) })
        .collect()
}

/// Every `golden_sub_<codec>_x<xs>y<ys>.exr` that has a C++ decode dump: the
/// native planes (`A` full-res, then `S` at its sampling, alphabetical like
/// the dump) equal the C++ planes exactly, with the reduced grid sizes.
#[test]
fn open_keeps_native_sampling_and_matches_cpp() {
    let name = "open_keeps_native_sampling_and_matches_cpp";
    let mut checked = 0;
    for codec in ["b44", "b44a", "pxr24", "dwaa", "dwab", "htj2k256", "htj2k32", "none", "zip", "piz"] {
        for (xs, ys) in [(1u32, 1u32), (2, 1), (1, 2), (2, 2)] {
            let exr = golden(&format!("golden_sub_{codec}_x{xs}y{ys}.exr"));
            let dump = PathBuf::from(format!("{}.dump", exr.display()));
            if !dump.exists() {
                continue; // lossless cells have no dump; the lossy ones do
            }
            let Some(exr) = support::fixture_or_skip(name, exr, HOW_TO_GET) else { return };
            let loader = exr_gpu::GpuLoader::open(&exr).expect("open subsampled golden");
            let part = loader.default_part();
            let (w, h) = (part.width(), part.height());
            let (a, aw, ah) = part.channel_bytes("A").expect("channel A");
            let (s, sw, sh) = part.channel_bytes("S").expect("channel S");
            assert_eq!((aw, ah), (w, h), "{codec} x{xs}y{ys}: A is full resolution");
            assert_eq!((sw, sh), (w / xs, h / ys), "{codec} x{xs}y{ys}: S at its own sampling");
            let want = half_words(&std::fs::read(&dump).expect("read dump"), true);
            let mut got = half_words(a, false);
            got.extend(half_words(s, false));
            assert_eq!(got, want, "{codec} x{xs}y{ys}: planes differ from the C++ InputFile decode");
            checked += 1;
        }
    }
    assert!(checked >= 8, "only {checked} subsampled dumps found; {HOW_TO_GET}");
}

/// A luminance-chroma file opens as what it stores (`RgbaOutputFile`
/// `WRITE_YCA`): `Y` and `A` at full resolution, `RY`/`BY` at 2x2, as two
/// layers with their own texture sizes.
#[test]
fn open_splits_luminance_chroma_by_sampling() {
    let Some(path) = support::fixture_or_skip("open_splits_luminance_chroma_by_sampling", golden("golden_yc.exr"), HOW_TO_GET)
    else {
        return;
    };
    let loader = exr_gpu::GpuLoader::open(&path).expect("open luminance-chroma file");
    let part = loader.default_part();
    let (w, h) = (part.width(), part.height());
    let layers: Vec<(String, usize, (u32, u32))> =
        part.layers().iter().map(|l| (l.label(), l.len(), (l.width, l.height))).collect();
    assert_eq!(
        layers,
        [("<main>".to_string(), 2, (w, h)), ("<main> 2x2".to_string(), 2, (w.div_ceil(2), h.div_ceil(2)))]
    );
    for (ch, dims) in [("A", (w, h)), ("Y", (w, h)), ("RY", (w / 2, h / 2)), ("BY", (w / 2, h / 2))] {
        let (_, cw, chh) = part.channel_bytes(ch).unwrap_or_else(|| panic!("channel {ch}"));
        assert_eq!((cw, chh), dims, "{ch}");
    }
    let chroma = &part.layers()[1];
    assert_eq!(part.vram_estimate(chroma).unwrap(), u64::from(chroma.width) * u64::from(chroma.height) * 4, "Rg16Float");
}

/// `open_rgba` is C++'s `RgbaInputFile` decode, bit for bit.
#[test]
fn open_rgba_matches_cpp_rgba_input_file() {
    let name = "open_rgba_matches_cpp_rgba_input_file";
    let Some(path) = support::fixture_or_skip(name, golden("golden_yc.exr"), HOW_TO_GET) else { return };
    let Some(dump) = support::fixture_or_skip(name, golden("golden_yc.exr.dump"), HOW_TO_GET) else { return };
    let bytes = std::fs::read(dump).expect("read dump");
    let (dw, dh) = (
        u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    );
    let pixels = half_words(&bytes[8..], true); // r, g, b, a per pixel

    let loader = exr_gpu::GpuLoader::open_rgba(&path).expect("open_rgba");
    let part = loader.default_part();
    assert_eq!((part.width(), part.height()), (dw, dh));
    let layer = part.default_layer().expect("a layer");
    assert_eq!((layer.label(), layer.len()), ("<main>".to_string(), 4));
    for (c, ch) in ["R", "G", "B", "A"].iter().enumerate() {
        let (plane, _, _) = part.channel_bytes(ch).expect("rgba channel");
        let got = half_words(plane, false);
        let want: Vec<u16> = pixels.iter().skip(c).step_by(4).copied().collect();
        assert_eq!(got, want, "channel {ch} differs from the C++ RgbaInputFile decode");
    }
}

/// A data window that does not start at the origin: the native reader must
/// offset every plane by `divp(min, sampling)`. The fixture is written by
/// exr-core's own writer (checked against C++ elsewhere) with values that are a
/// formula of the ABSOLUTE sample coordinates, so the expectation needs no reader.
#[test]
fn open_offsets_planes_for_a_non_origin_data_window() {
    use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
    use exr_core::frame_buffer::{FrameBuffer, Slice};
    use exr_core::header::{Header, SCANLINE_IMAGE};
    use exr_core::io::StdOfStream;
    use exr_core::scanline::OutputFile;
    use imath_rs::{Half, V2f};

    // Data window [4, 11] x [6, 11]: 8 x 6, origin a multiple of the 2x2 sampling.
    let (x0, y0, w, h) = (4i32, 6i32, 8usize, 6usize);
    let (sw, sh) = (w / 2, h / 2);
    let a_val = |x: i32, y: i32| (x + 10 * y) as f32;
    let s_val = |sx: i32, sy: i32| (100 + sx + 10 * sy) as f32;

    let mut header =
        Header::with_dimensions(16, 16, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::Zip)
            .unwrap();
    {
        let dw = header.data_window_mut().unwrap();
        dw.min.x = x0;
        dw.min.y = y0;
        dw.max.x = x0 + w as i32 - 1;
        dw.max.y = y0 + h as i32 - 1;
    }
    let mut cl = ChannelList::new();
    cl.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("S", Channel::new(PixelType::Half, 2, 2, false)).unwrap();
    header.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    header.set_type(SCANLINE_IMAGE).unwrap();

    let mut px = Vec::new();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            px.extend_from_slice(&Half::from(a_val(x0 + x, y0 + y)).bits().to_ne_bytes());
        }
    }
    let s_at = px.len();
    for sy in 0..sh as i32 {
        for sx in 0..sw as i32 {
            px.extend_from_slice(&Half::from(s_val(x0 / 2 + sx, y0 / 2 + sy)).bits().to_ne_bytes());
        }
    }
    let slice = |base: i64, row: usize, s: i32| Slice {
        type_: PixelType::Half,
        base_offset: base,
        x_stride: 2,
        y_stride: row,
        x_sampling: s,
        y_sampling: s,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    };
    let mut fb = FrameBuffer::new();
    fb.insert("A", slice(-(i64::from(x0) * 2 + i64::from(y0) * (w as i64 * 2)), w * 2, 1)).unwrap();
    fb.insert(
        "S",
        slice(s_at as i64 - (i64::from(x0 / 2) * 2 + i64::from(y0 / 2) * (sw as i64 * 2)), sw * 2, 2),
    )
    .unwrap();
    let dir = std::env::temp_dir().join(format!("exr-gpu-sub-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("offset.exr");
    {
        let mut os = StdOfStream::create(path.to_str().unwrap()).unwrap();
        let mut out = OutputFile::new(&mut os, &header).unwrap();
        out.set_frame_buffer(&fb).unwrap();
        out.write_pixels(&px, h as i32).unwrap();
    }

    let loader = exr_gpu::GpuLoader::open(&path).expect("open offset subsampled file");
    let part = loader.default_part();
    let (a, aw, ah) = part.channel_bytes("A").unwrap();
    let (s, ssw, ssh) = part.channel_bytes("S").unwrap();
    assert_eq!((aw, ah, ssw, ssh), (w as u32, h as u32, sw as u32, sh as u32));
    let a = half_words(a, false);
    let s = half_words(s, false);
    for y in 0..h {
        for x in 0..w {
            let want = Half::from(a_val(x0 + x as i32, y0 + y as i32)).bits();
            assert_eq!(a[y * w + x], want, "A at ({x},{y})");
        }
    }
    for sy in 0..sh {
        for sx in 0..sw {
            let want = Half::from(s_val(x0 / 2 + sx as i32, y0 / 2 + sy as i32)).bits();
            assert_eq!(s[sy * sw + sx], want, "S at ({sx},{sy})");
        }
    }
}
