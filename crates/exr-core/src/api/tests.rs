//! Tests for the idiomatic `Image` facade.
//!
//! Covers: round-trip (HALF + FLOAT, NONE + ZIP); the deep / multipart
//! escape-hatch errors; a real-corpus read; and a single-source proof that the
//! facade writer produces the SAME bytes as the equivalent hand-built
//! `scanline::OutputFile` path (so the facade did not fork the writer).

use imath_rs::{Box2i, Half, V2i};

use super::store::sample_dims;
use super::{ChannelData, Image, Unsupported, classify_unsupported, write_rgba_file};
use crate::error::ExrError;
use crate::attr::{
    Channel, ChannelList, Compression, LevelMode, LevelRoundingMode, LineOrder, PixelType,
    TileDescription,
};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::{self, Header};
use crate::io::{MemIStream, MemOStream};
use crate::scanline;

/// Full-resolution sample count of an image's data window, through the one checked
/// layout boundary.
fn pixel_count(img: &Image) -> crate::error::Result<usize> {
    sample_dims(img.data_window, 1, 1).map(|(w, h)| w * h)
}

/// Build a deterministic `Image`: 4x3, an "R" HALF channel and a "Z" FLOAT
/// channel, both full-resolution.
fn sample_image() -> Image {
    let dw = Box2i {
        min: V2i { x: 0, y: 0 },
        max: V2i { x: 3, y: 2 },
    };
    let w = 4usize;
    let h = 3usize;
    let mut r = Vec::with_capacity(w * h);
    let mut z = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            r.push(Half::from((x + y * 10) as f32));
            z.push((x as f32) * 0.25 + (y as f32));
        }
    }
    Image::new(dw)
        .with_channel("R", ChannelData::Half(r))
        .with_channel("Z", ChannelData::Float(z))
}

/// py-parity2 exr-core need (b): `Image::read_header_only` must give the same
/// dimensions/channel-list/attributes as a full [`Image::read_from`], but
/// its channel data must be all-zero (proof it never called
/// `read_pixels`/`read_tiles`) -- and it must still error the same way a
/// full read would on the deep/multipart escape hatches, since it opens the
/// SAME low-level reader constructor, just stops one step earlier.
#[test]
fn read_header_only_gives_dimensions_and_channels_without_decoding_pixels() {
    let img = sample_image();
    let mut out = MemOStream::new();
    img.write_to(&mut out, Compression::Zip).expect("write");
    let bytes = out.into_data();

    let mut is_full = MemIStream::new(bytes.clone());
    let full = Image::read_from(&mut is_full).expect("full read");
    let mut is_ho = MemIStream::new(bytes);
    let header_only = Image::read_header_only_from(&mut is_ho).expect("header-only read");

    assert_eq!(header_only.data_window, full.data_window);
    assert_eq!(header_only.display_window, full.display_window);
    let orig: Vec<_> = full.channels().map(|(n, _)| n.to_string()).collect();
    let got: Vec<_> = header_only.channels().map(|(n, _)| n.to_string()).collect();
    assert_eq!(orig, got, "channel list must match a full read");

    // The real file's "Z" channel is non-constant (see `sample_image`) -- a
    // header-only read must NOT have decoded it.
    match header_only.channel("Z") {
        Some(ChannelData::Float(v)) => {
            assert_eq!(v.len(), pixel_count(&full).unwrap());
            assert!(v.iter().all(|&f| f == 0.0), "header-only pixels must stay zero");
        }
        other => panic!("Z should be FLOAT, got {other:?}"),
    }
}

/// A header-only read of a tiled file works through the SAME dispatch as a
/// full read (`Image::read_header_only_from` -> `TiledInputFile::new`), not
/// a separate, possibly-diverging code path.
#[test]
fn read_header_only_works_for_a_tiled_file() {
    let bytes = write_tiled(LineOrder::IncreasingY);
    let mut is = MemIStream::new(bytes);
    let header_only = Image::read_header_only_from(&mut is).expect("header-only tiled read");
    assert_eq!(header_only.width(), 4);
    assert_eq!(header_only.height(), 3);
    let names: Vec<_> = header_only.channels().map(|(n, _)| n.to_string()).collect();
    assert_eq!(names, vec!["R".to_string(), "Z".to_string()]);
}

/// A header-only read must still hit the multipart escape hatch, exactly
/// like a full read -- it opens the same low-level constructor, so the same
/// version-flag checks apply before anything channel-shaped is even parsed.
#[test]
fn read_header_only_still_rejects_multipart() {
    fn make_part_header(name: &str) -> Header {
        let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 3, y: 3 } };
        let mut h = Header::new();
        *h.data_window_mut().unwrap() = dw;
        *h.display_window_mut().unwrap() = dw;
        *h.compression_mut().unwrap() = Compression::NoCompression;
        *h.line_order_mut().unwrap() = LineOrder::IncreasingY;
        let mut cl = ChannelList::new();
        cl.insert("Z", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        *h.channels_mut().unwrap() = cl;
        h.set_name(name).unwrap();
        h.set_type(header::SCANLINE_IMAGE).unwrap();
        h
    }

    let mut out = MemOStream::new();
    {
        // Two parts, so the version word genuinely carries MULTI_PART_FILE_FLAG
        // (a single-header multipart write has no part prefix).
        let mut mp = crate::multipart::MultiPartOutputFile::new(
            &mut out,
            &[make_part_header("a"), make_part_header("b")],
        )
        .unwrap();
        for n in 0..2 {
            let mut p = mp.output_part(n).unwrap();
            let mut fb = FrameBuffer::new();
            fb.insert(
                "Z",
                Slice {
                    type_: PixelType::Float,
                    base_offset: 0,
                    x_stride: 4,
                    y_stride: 16,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                },
            )
            .unwrap();
            p.set_frame_buffer(&fb).unwrap();
            p.write_pixels(&[0u8; 4 * 4 * 4], 4).unwrap();
        }
    }
    let mut is = MemIStream::new(out.into_data());
    let err = Image::read_header_only_from(&mut is).expect_err("multipart must be rejected");
    assert!(
        matches!(err, ExrError::Unsupported(Unsupported::Multipart)),
        "{err:?}"
    );
}

#[test]
fn round_trip_none() {
    round_trip_with(Compression::NoCompression);
}

#[test]
fn round_trip_zip() {
    round_trip_with(Compression::Zip);
}

fn round_trip_with(compression: Compression) {
    let img = sample_image();

    let mut out = MemOStream::new();
    img.write_to(&mut out, compression).expect("facade write");
    let bytes = out.into_data();

    let mut is = MemIStream::new(bytes);
    let back = Image::read_from(&mut is).expect("facade read");

    // Windows preserved.
    assert_eq!(back.data_window, img.data_window);
    assert_eq!(back.display_window, img.display_window);

    // Channels: same names, same order (alphabetical), same data bit-exact.
    let orig: Vec<_> = img.channels().map(|(n, _)| n.to_string()).collect();
    let got: Vec<_> = back.channels().map(|(n, _)| n.to_string()).collect();
    assert_eq!(orig, got, "channel order/name mismatch");
    assert_eq!(orig, vec!["R".to_string(), "Z".to_string()]);

    match (img.channel("R"), back.channel("R")) {
        (Some(ChannelData::Half(a)), Some(ChannelData::Half(b))) => {
            assert_eq!(
                a.iter().map(|h| h.bits()).collect::<Vec<_>>(),
                b.iter().map(|h| h.bits()).collect::<Vec<_>>(),
                "R HALF bits mismatch"
            );
        }
        other => panic!("R channel type mismatch: {other:?}"),
    }
    match (img.channel("Z"), back.channel("Z")) {
        (Some(ChannelData::Float(a)), Some(ChannelData::Float(b))) => {
            assert_eq!(a, b, "Z FLOAT mismatch");
        }
        other => panic!("Z channel type mismatch: {other:?}"),
    }
}

/// The facade writer must produce byte-for-byte the same file as an equivalent,
/// independently constructed low-level `scanline::OutputFile` path — proving the
/// facade drives the verified writer rather than forking it.
#[test]
fn facade_write_matches_low_level_bytes() {
    let img = sample_image();

    // (1) Facade bytes.
    let mut facade_out = MemOStream::new();
    img.write_to(&mut facade_out, Compression::Zip)
        .expect("facade write");
    let facade_bytes = facade_out.into_data();

    // (2) Independent low-level path: same header + same native channel-planar
    // buffer + same FrameBuffer of Slices, written through OutputFile.
    let dw = img.data_window;
    let w = 4i64;
    let h = 3i32;

    let mut header = Header::new();
    *header.data_window_mut().unwrap() = dw;
    *header.display_window_mut().unwrap() = dw;
    *header.compression_mut().unwrap() = Compression::Zip;
    *header.line_order_mut().unwrap() = LineOrder::IncreasingY;
    let mut channels = ChannelList::new();
    channels.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    channels.insert("Z", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    *header.channels_mut().unwrap() = channels;
    // No `set_type`: a single-part scanline header built with `Header::new` has
    // no `type`, and C++ `OutputFile` only fixes one that is already there
    // (`ImfOutputFile.cpp:775`). The facade must match that byte for byte.

    // Channel-planar native buffer: R plane (HALF) then Z plane (FLOAT).
    let pixels = (w * h as i64) as usize;
    let mut buffer: Vec<u8> = Vec::new();
    if let Some(ChannelData::Half(r)) = img.channel("R") {
        for v in r {
            buffer.extend_from_slice(&v.bits().to_ne_bytes());
        }
    }
    let z_offset = buffer.len();
    if let Some(ChannelData::Float(z)) = img.channel("Z") {
        for v in z {
            buffer.extend_from_slice(&v.to_ne_bytes());
        }
    }
    assert_eq!(buffer.len(), pixels * 2 + pixels * 4);

    let mut fb = FrameBuffer::new();
    fb.insert(
        "R",
        Slice::make(PixelType::Half, 0, dw.min, w, 2, 2 * w as usize, 1, 1, 0.0, false, false)
            .unwrap(),
    )
    .unwrap();
    fb.insert(
        "Z",
        Slice::make(
            PixelType::Float, z_offset, dw.min, w, 4, 4 * w as usize, 1, 1, 0.0, false, false,
        )
        .unwrap(),
    )
    .unwrap();

    let mut low_out = MemOStream::new();
    {
        let mut of = scanline::OutputFile::new(&mut low_out, &header).expect("OutputFile::new");
        of.set_frame_buffer(&fb).expect("set_frame_buffer");
        of.write_pixels(&buffer, h).expect("write_pixels");
    }
    let low_bytes = low_out.into_data();

    assert_eq!(
        facade_bytes, low_bytes,
        "facade writer diverged from the low-level scanline writer"
    );
}

/// Reading a deep file through the facade must return the clear escape-hatch
/// error (never half-decode).
#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn deep_file_returns_escape_hatch() {
    let deep = exr_test_data::openexr_test().join("bin/test_images/11.deep.exr");
    let err = Image::read(&deep).expect_err("deep file must be rejected");
    // The read yields the structured variant (the routing contract), not a
    // stringly-typed error — this is what consumers match on.
    assert!(
        matches!(err, ExrError::Unsupported(Unsupported::Deep)),
        "expected ExrError::Unsupported(Deep), got: {err:?}"
    );
    // Its Display still names the low-level API to use.
    let msg = err.to_string();
    assert!(
        msg.contains("DeepScanLineInputFile"),
        "unexpected deep error message: {msg}"
    );
    // The structured classifier (used by exr-gpu instead of String::contains)
    // must recognise this real escape-hatch error as Deep.
    assert_eq!(classify_unsupported(&err), Some(Unsupported::Deep));
}

/// Reading a multipart file through the facade must return the clear
/// escape-hatch error.
#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn multipart_file_returns_escape_hatch() {
    let mp = exr_test_data::openexr_images().join("Beachball/multipart.0001.exr");
    let err = Image::read(&mp).expect_err("multipart file must be rejected");
    assert!(
        matches!(err, ExrError::Unsupported(Unsupported::Multipart)),
        "expected ExrError::Unsupported(Multipart), got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("MultiPartInputFile"),
        "unexpected multipart error message: {msg}"
    );
    assert_eq!(classify_unsupported(&err), Some(Unsupported::Multipart));
}

/// `classify_unsupported` must return `None` for errors that are NOT facade
/// escape hatches, so a genuine decode failure is never mistaken for a
/// "wrong file shape, fall back" signal.
#[test]
fn classify_unsupported_ignores_unrelated_errors() {
    assert_eq!(classify_unsupported(&ExrError::Io("disk gone".into())), None);
    assert_eq!(
        classify_unsupported(&ExrError::Arg("some unrelated argument error".into())),
        None
    );
    assert_eq!(
        classify_unsupported(&ExrError::Input("truncated chunk".into())),
        None
    );
}

/// `write_rgba_file` must reject a dimension past the i32 pixel-coordinate
/// range with a clean error rather than silently truncating the data window —
/// and it must do so cheaply (validation precedes the buffer allocation), so a
/// 1-row oversized width does not allocate gigabytes first.
#[test]
fn write_rgba_file_rejects_oversized_width() {
    let oversized = i32::MAX as usize + 2;
    let err = write_rgba_file("unused.exr", oversized, 1, |_, _| (0.0, 0.0, 0.0, 0.0))
        .expect_err("oversized width must be rejected");
    assert!(matches!(err, ExrError::Arg(_)), "{err}");
}

/// Zero dimensions are rejected up front (the existing guard).
#[test]
fn write_rgba_file_rejects_zero_dimensions() {
    let err = write_rgba_file("unused.exr", 0, 4, |_, _| (0.0, 0.0, 0.0, 0.0))
        .expect_err("zero width must be rejected");
    assert!(matches!(err, ExrError::Arg(_)), "{err}");
}

/// Read a real single-part scanline corpus file through the facade and check it
/// loads with sensible channels / dimensions.
#[test]
#[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
fn reads_real_corpus_scanline() {
    let file = exr_test_data::openexr_images().join("DisplayWindow/t01.exr");
    let img = Image::read(&file).expect("read real corpus scanline file");

    assert!(img.width() > 0 && img.height() > 0, "empty data window");
    // DisplayWindow test images are RGBA HALF.
    let names: Vec<_> = img.channels().map(|(n, _)| n.to_string()).collect();
    assert!(
        names.iter().any(|n| n == "R"),
        "expected an R channel, got {names:?}"
    );
    for (name, data) in img.channels() {
        assert_eq!(
            data.len(),
            (img.width() * img.height()) as usize,
            "channel \"{name}\" wrong sample count"
        );
    }
}

/// A public, user-built `Image` with an extreme data window must never panic
/// when reporting its extent — the subtraction is widened to i64.
#[test]
fn extreme_window_width_height_no_panic() {
    let dw = Box2i {
        min: V2i { x: i32::MIN, y: i32::MIN },
        max: V2i { x: i32::MAX, y: i32::MAX },
    };
    let img = Image::new(dw);
    // i32::MAX - i32::MIN + 1 == 2^32, computed in i64 without overflow/panic.
    assert_eq!(img.width(), 1i64 << 32);
    assert_eq!(img.height(), 1i64 << 32);
}

/// The window boundary rejects an inverted window (non-positive extent) instead of
/// wrapping `as usize` into a huge allocation.
#[test]
fn pixel_count_rejects_inverted_window() {
    let dw = Box2i {
        min: V2i { x: 10, y: 10 },
        max: V2i { x: 0, y: 0 },
    };
    let img = Image::new(dw);
    assert!(matches!(pixel_count(&img), Err(ExrError::Arg(_))));
}

/// The window boundary rejects a window whose `width * height` overflows `usize`.
#[test]
fn pixel_count_rejects_overflow_window() {
    let dw = Box2i {
        min: V2i { x: i32::MIN, y: i32::MIN },
        max: V2i { x: i32::MAX, y: i32::MAX },
    };
    let img = Image::new(dw);
    // width == height == 2^32, so the product 2^64 overflows usize.
    assert!(matches!(pixel_count(&img), Err(ExrError::Arg(_))));
}

/// A normal image's pixel count is unchanged (`width * height`).
#[test]
fn pixel_count_normal_image_unchanged() {
    let img = sample_image();
    assert_eq!(pixel_count(&img).expect("valid window"), 12);
}

// ---------------------------------------------------------------------------
// Header metadata survives a facade round trip
// ---------------------------------------------------------------------------

/// Write `img` through the facade with `extra` header attributes bolted on,
/// then read it straight back.
fn round_trip_with_header_attrs(
    img: &Image,
    extra: Vec<(crate::attr::Name, Box<dyn crate::attr::Attribute>)>,
) -> Image {
    let mut out = MemOStream::new();
    img.write_to_with_attrs(&mut out, Compression::Zip, extra)
        .expect("facade write");
    let mut is = MemIStream::new(out.into_data());
    Image::read_from(&mut is).expect("facade read")
}

#[test]
fn custom_attributes_survive_a_facade_round_trip() {
    use crate::attr::{Chromaticities, ExrString, StringAttribute, TypedAttribute};

    let chroma = Chromaticities {
        red: [0.64, 0.33],
        green: [0.30, 0.60],
        blue: [0.15, 0.06],
        white: [0.3127, 0.3290],
    };
    let extra: Vec<(crate::attr::Name, Box<dyn crate::attr::Attribute>)> = vec![
        (
            "owner".into(),
            Box::new(StringAttribute::new(ExrString::from("Alex"))),
        ),
        (
            "comments".into(),
            Box::new(StringAttribute::new(ExrString::from("hello"))),
        ),
        (
            "chromaticities".into(),
            Box::new(TypedAttribute::<Chromaticities>::new(chroma)),
        ),
        (
            "pixelAspectRatio".into(),
            Box::new(TypedAttribute::<f32>::new(2.0)),
        ),
    ];

    // First hop: the attributes only exist in `extra`.
    let once = round_trip_with_header_attrs(&sample_image(), extra);
    assert!(once.attributes().contains("owner"), "owner not captured");
    assert!(once.attributes().contains("chromaticities"));

    // Second hop: a PLAIN `write` (no `extra`) must carry them — this is what
    // `exr-rs convert` does, and what used to drop every attribute.
    let mut out = MemOStream::new();
    once.write_to(&mut out, Compression::Zip).expect("plain write");
    let mut is = MemIStream::new(out.into_data());
    let twice = Image::read_from(&mut is).expect("read back");

    assert_eq!(
        once.attributes(),
        twice.attributes(),
        "a plain facade write must preserve every carried attribute"
    );
    let owner = twice
        .attributes()
        .get("owner")
        .and_then(|a| a.as_any().downcast_ref::<StringAttribute>())
        .expect("owner is a string attribute");
    assert_eq!(owner.value.as_str(), Some("Alex"));
    let par = twice
        .attributes()
        .get("pixelAspectRatio")
        .and_then(|a| a.as_any().downcast_ref::<TypedAttribute<f32>>())
        .expect("pixelAspectRatio carried");
    assert_eq!(par.value, 2.0);
}

#[test]
fn derived_attributes_are_never_carried() {
    let img = round_trip_with_header_attrs(&sample_image(), Vec::new());
    for derived in super::ImageAttributes::DERIVED {
        assert!(
            !img.attributes().contains(derived),
            "{derived} must come from the Image's own fields, not the carrier"
        );
    }
    // ...but the windows / compression themselves still round-trip.
    assert_eq!(img.data_window, sample_image().data_window);
    assert_eq!(img.compression, Compression::Zip);
}

#[test]
fn an_unknown_attribute_type_round_trips_as_an_opaque_blob() {
    use crate::attr::OpaqueAttribute;

    let blob = OpaqueAttribute::with_data("vendorThing", vec![1u8, 2, 3, 4, 5]);
    let extra: Vec<(crate::attr::Name, Box<dyn crate::attr::Attribute>)> =
        vec![("vendorAttr".into(), Box::new(blob))];
    let img = round_trip_with_header_attrs(&sample_image(), extra);

    let got = img
        .attributes()
        .get("vendorAttr")
        .expect("opaque attribute carried");
    assert_eq!(got.type_name(), b"vendorThing");

    // And it survives the *second* hop, written from the carrier alone.
    let mut out = MemOStream::new();
    img.write_to(&mut out, Compression::Zip).expect("plain write");
    let mut is = MemIStream::new(out.into_data());
    let twice = Image::read_from(&mut is).expect("read back");
    assert_eq!(img.attributes(), twice.attributes());
}

// ---------------------------------------------------------------------------
// A tiled RANDOM_Y file can be written back through the facade
// ---------------------------------------------------------------------------

/// Write a 4x3 single-level tiled file with the given line order, using the
/// same pixels as [`sample_image`].
fn write_tiled(line_order: LineOrder) -> Vec<u8> {
    use crate::tiled::TiledOutputFile;

    let img = sample_image();
    let mut header = Header::new();
    *header.data_window_mut().expect("dw") = img.data_window;
    *header.display_window_mut().expect("dpw") = img.data_window;
    *header.compression_mut().expect("comp") = Compression::Zip;
    *header.line_order_mut().expect("lo") = line_order;
    let mut channels = ChannelList::new();
    channels
        .insert("R", Channel::new(PixelType::Half, 1, 1, false))
        .expect("R");
    channels
        .insert("Z", Channel::new(PixelType::Float, 1, 1, false))
        .expect("Z");
    *header.channels_mut().expect("channels") = channels;
    header
        .set_tile_description(TileDescription {
            x_size: 4,
            y_size: 3,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        })
        .expect("tiles");
    header.set_type(header::TILED_IMAGE).expect("type");

    // Channel-planar native-endian buffer, like the facade's own layout.
    let mut buf = Vec::new();
    for y in 0..3usize {
        for x in 0..4usize {
            buf.extend_from_slice(&Half::from((x + y * 10) as f32).bits().to_ne_bytes());
        }
    }
    let z_off = buf.len();
    for y in 0..3usize {
        for x in 0..4usize {
            buf.extend_from_slice(&((x as f32) * 0.25 + y as f32).to_ne_bytes());
        }
    }

    let mut out = MemOStream::new();
    {
        let mut of = TiledOutputFile::new(&mut out, &header).expect("tiled writer");
        let mut fb = FrameBuffer::new();
        fb.insert(
            "R",
            Slice::make(
                PixelType::Half,
                0,
                V2i { x: 0, y: 0 },
                4,
                2,
                8,
                1,
                1,
                0.0,
                false,
                false,
            )
            .expect("R slice"),
        )
        .expect("insert R");
        fb.insert(
            "Z",
            Slice::make(
                PixelType::Float,
                z_off,
                V2i { x: 0, y: 0 },
                4,
                4,
                16,
                1,
                1,
                0.0,
                false,
                false,
            )
            .expect("Z slice"),
        )
        .expect("insert Z");
        of.set_frame_buffer(&fb).expect("set_frame_buffer");
        of.write_tiles(&buf, 0, 0, 0, 0, 0, 0).expect("write_tiles");
    }
    out.into_data()
}

#[test]
fn tiled_random_y_round_trips_through_the_facade() {
    let mut is = MemIStream::new(write_tiled(LineOrder::RandomY));
    let img = Image::read_from(&mut is).expect("facade reads a tiled RANDOM_Y file");
    assert_eq!(
        img.line_order,
        LineOrder::RandomY,
        "the read keeps the source order"
    );
    // Structural attributes of the SOURCE (tiled) file must not ride along into
    // the scanline file the facade writes.
    for structural in ["tiles", "type"] {
        assert!(
            !img.attributes().contains(structural),
            "{structural} must not be carried out of a tiled source"
        );
    }

    // The write-back used to fail with "Invalid line order in image header":
    // RANDOM_Y is tiled-only and the facade emits scanlines.
    let mut out = MemOStream::new();
    img.write_to(&mut out, Compression::Zip)
        .expect("write back a tiled RANDOM_Y image as a scanline file");

    let mut is = MemIStream::new(out.into_data());
    let back = Image::read_from(&mut is).expect("read back");
    assert_eq!(
        back.line_order,
        LineOrder::IncreasingY,
        "a scanline file must declare a scanline-legal line order"
    );
    for name in ["R", "Z"] {
        assert_eq!(back.channel(name), img.channel(name), "channel {name}");
    }
}

/// py-parity2 exr-core need (a): a `tiles` overlay attribute (what the
/// Python binding sends for `header["tiles"] = TileDescription(...)`) must
/// make [`Image::write_to_with_attrs`] dispatch to a TILED write, not force
/// scanline — real C++ would open a header with a tile description via
/// `TiledOutputFile`. Round-trips through `TiledInputFile` (not the facade's
/// own scanline reader) to prove the bytes are genuinely tiled, and checks
/// the version word carries the TILED flag but not `MULTI_PART`.
#[test]
fn a_tiles_overlay_attribute_dispatches_to_a_tiled_write() {
    use crate::attr::TileDescriptionAttribute;
    use crate::container::version;
    use crate::tiled::TiledInputFile;

    let img = sample_image();
    let extra: Vec<(crate::attr::Name, Box<dyn crate::attr::Attribute>)> = vec![(
        "tiles".into(),
        Box::new(TileDescriptionAttribute::new(TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        })),
    )];

    let mut out = MemOStream::new();
    img.write_to_with_attrs(&mut out, Compression::Zip, extra)
        .expect("tiled write via the overlay");
    let bytes = out.into_data();

    let version_word = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u32;
    let flags = version::get_flags(version_word);
    assert!(version::is_tiled(flags), "TILED flag must be set");
    assert!(!version::is_multi_part(flags), "still a single-part file");

    let mut is = MemIStream::new(bytes);
    let mut input = TiledInputFile::new(&mut is).expect("open as a real tiled file");
    let back = Image::from_tiled_input_file(&mut input).expect("assemble level 0");
    assert_eq!(back.channel("R"), img.channel("R"), "channel R");
    assert_eq!(back.channel("Z"), img.channel("Z"), "channel Z");
}

/// A `tiles` overlay requesting MIPMAP/RIPMAP levels must be rejected up
/// front -- the facade has no per-level pixel data to write a pyramid from,
/// and silently writing only level 0 under a header that claims more levels
/// exist would produce a structurally-lying file.
#[test]
fn a_mipmap_tiles_overlay_is_rejected_not_silently_truncated() {
    use crate::attr::TileDescriptionAttribute;

    let img = sample_image();
    let extra: Vec<(crate::attr::Name, Box<dyn crate::attr::Attribute>)> = vec![(
        "tiles".into(),
        Box::new(TileDescriptionAttribute::new(TileDescription {
            x_size: 2,
            y_size: 2,
            mode: LevelMode::MipmapLevels,
            rounding_mode: LevelRoundingMode::RoundDown,
        })),
    )];

    let mut out = MemOStream::new();
    let err = img
        .write_to_with_attrs(&mut out, Compression::Zip, extra)
        .expect_err("MIPMAP with no per-level data must be rejected");
    assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
}

#[test]
fn a_legal_scanline_line_order_is_not_rewritten() {
    let mut is = MemIStream::new(write_tiled(LineOrder::DecreasingY));
    let img = Image::read_from(&mut is).expect("read");
    let mut out = MemOStream::new();
    img.write_to(&mut out, Compression::Zip).expect("write");
    let mut is = MemIStream::new(out.into_data());
    let back = Image::read_from(&mut is).expect("read back");
    assert_eq!(
        back.line_order,
        LineOrder::DecreasingY,
        "only RANDOM_Y is normalised; a legal scanline order is preserved"
    );
}

// ---------------------------------------------------------------------------
// Channel mutation — `insert_channel` (checked) is the only mutation path:
// the removed
// `channel_mut` handed out a resizable `&mut ChannelData` with no way to
// re-validate its length on mutation. `build_write_buffer` still re-checks
// every plane as a second line of defense, since the chainable `with_channel`
// builder is deliberately unchecked.
// ---------------------------------------------------------------------------

#[test]
fn insert_channel_edits_survive_a_write() {
    let mut img = sample_image();
    let mut edited = match img.channel("Z").expect("Z exists").clone() {
        ChannelData::Float(v) => v,
        other => panic!("Z should be FLOAT, got {other:?}"),
    };
    edited[5] = 42.0;
    img.insert_channel("Z", ChannelData::Float(edited)).expect("in-place edit");

    let mut out = MemOStream::new();
    img.write_to(&mut out, Compression::Zip).expect("write");
    let mut is = MemIStream::new(out.into_data());
    let back = Image::read_from(&mut is).expect("read back");
    match back.channel("Z").expect("Z") {
        ChannelData::Float(v) => assert_eq!(v[5], 42.0),
        other => panic!("Z should be FLOAT, got {other:?}"),
    }
}

#[test]
fn insert_channel_rejects_a_mis_sized_plane_immediately() {
    // `insert_channel` must reject a mis-sized plane up front, not defer it
    // to a write-time error the way the removed `channel_mut` did.
    let mut img = sample_image();
    let mut short = match img.channel("Z").expect("Z exists").clone() {
        ChannelData::Float(v) => v,
        other => panic!("Z should be FLOAT, got {other:?}"),
    };
    short.truncate(3);
    let err = img
        .insert_channel("Z", ChannelData::Float(short))
        .expect_err("a short plane must be rejected immediately");
    assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
}

#[test]
fn a_resized_channel_is_rejected_at_write() {
    // The chainable `with_channel` builder is deliberately unchecked (unlike
    // `insert_channel`), so `build_write_buffer`'s own re-check is still the
    // ONLY thing catching a mis-sized plane built that way.
    let mut img = sample_image();
    img = img.with_channel("Z", ChannelData::Float(vec![1.0, 2.0, 3.0]));
    let mut out = MemOStream::new();
    let err = img
        .write_to(&mut out, Compression::Zip)
        .expect_err("a short plane must be rejected");
    assert!(matches!(err, ExrError::Arg(_)), "{err:?}");
}

// ---------------------------------------------------------------------------
// A facade-written single-part scanline file carries no `type` attribute —
// C++ only FIXES an existing one (`ImfOutputFile.cpp:775`), it never adds one.
// ---------------------------------------------------------------------------

#[test]
fn a_facade_scanline_file_has_no_type_attribute() {
    let mut out = MemOStream::new();
    sample_image()
        .write_to(&mut out, Compression::Zip)
        .expect("write");
    let bytes = out.into_data();

    let mut is = MemIStream::new(bytes.clone());
    let (header, _) = Header::read_from(&mut is).expect("parse header");
    assert!(
        !header.has_type(),
        "a single-part scanline file must not carry a `type` attribute"
    );
    // Byte-level guard: the attribute name must not appear in the header at all.
    assert!(
        !bytes.windows(4).any(|w| w == b"type"),
        "the string \"type\" appears in the written header"
    );
}

#[test]
fn a_multipart_part_still_carries_its_required_type() {
    // C++ `ImfMultiPartOutputFile.cpp:117` requires a type on every part, so the
    // multipart writer must add what the single-part writer omits.
    let dir = std::env::temp_dir().join(format!("exr-rs-api-mp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("mp.exr");
    super::write_multipart_scanline(
        &path,
        vec![
            (sample_image(), Compression::Zip, "partA".to_string()),
            (sample_image(), Compression::Zip, "partB".to_string()),
        ],
    )
    .expect("multipart write");

    let mut stream = crate::io::StdIfStream::open(path.to_str().expect("utf-8")).expect("open");
    let mp = crate::MultiPartInputFile::new(&mut stream).expect("multipart read");
    assert_eq!(mp.parts(), 2);
    for i in 0..mp.parts() {
        let h = mp.header(i).expect("header");
        assert_eq!(h.type_().expect("type"), header::SCANLINE_IMAGE);
    }
    drop(stream);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// ChannelData single-source-of-truth conversion methods
// ---------------------------------------------------------------------------

#[test]
fn channel_data_to_f32_is_raw_uint_cast() {
    // UINT->f32 is a raw cast (`convert::uint_to_float`), never normalized —
    // that policy choice belongs to the caller (e.g. exr-view's display path).
    let cd = ChannelData::Uint(vec![0, 1000, u32::MAX]);
    assert_eq!(cd.to_f32(), vec![0.0, 1000.0, u32::MAX as f32]);

    let cd = ChannelData::Half(vec![Half::from(1.5), Half::from(-2.0)]);
    assert_eq!(cd.to_f32(), vec![1.5, -2.0]);

    let cd = ChannelData::Float(vec![1.0, 2.0]);
    assert_eq!(cd.to_f32(), vec![1.0, 2.0]);
}

#[test]
fn channel_data_sample_f32_bounds_checked() {
    let cd = ChannelData::Float(vec![1.0, 2.0]);
    assert_eq!(cd.sample_f32(0), Some(1.0));
    assert_eq!(cd.sample_f32(1), Some(2.0));
    assert_eq!(cd.sample_f32(2), None, "out-of-range index must be None, not a silent 0.0");
}

#[test]
fn channel_data_convert_uint_to_half_overflows_above_65504() {
    // Pins the same boundary `half_conversion_boundary.rs` verifies against the C++
    // oracle, through the public `ChannelData::convert` surface this time.
    let cd = ChannelData::Uint(vec![65504, 65505]);
    let converted = cd.convert(PixelType::Half).unwrap();
    match converted {
        ChannelData::Half(v) => {
            assert_eq!(v[0].bits(), 0x7bff);
            assert_eq!(v[1].bits(), 0x7c00, "65505 must overflow to +inf, not round to 65504");
        }
        _ => panic!("expected Half"),
    }
}

#[test]
fn channel_data_convert_float_to_half_overflows_at_65520() {
    let cd = ChannelData::Float(vec![65519.0, 65520.0]);
    let converted = cd.convert(PixelType::Half).unwrap();
    match converted {
        ChannelData::Half(v) => {
            assert_eq!(v[0].bits(), 0x7bff, "65519 rounds to the largest finite half");
            assert_eq!(v[1].bits(), 0x7c00, "65520 overflows to +inf");
        }
        _ => panic!("expected Half"),
    }
}

#[test]
fn channel_data_convert_identity_and_rejects_sentinel() {
    let cd = ChannelData::Float(vec![1.0, 2.0]);
    assert_eq!(cd.convert(PixelType::Float).unwrap(), cd);
    assert!(cd.convert(PixelType::NumPixelTypes).is_err());
}

#[test]
fn channel_data_ne_bytes_round_trip() {
    for cd in [
        ChannelData::Half(vec![Half::from(1.5), Half::from(-2.0), Half::from(0.0)]),
        ChannelData::Float(vec![1.0, -2.5, 3.0]),
        ChannelData::Uint(vec![0, 42, u32::MAX]),
    ] {
        let bytes = cd.to_ne_bytes();
        let back = ChannelData::from_ne_bytes(cd.pixel_type(), &bytes, cd.len()).unwrap();
        assert_eq!(back, cd);
    }
}

#[test]
fn channel_data_ne_bytes_rejects_truncated_buffer() {
    let cd = ChannelData::Float(vec![1.0, 2.0]);
    let bytes = cd.to_ne_bytes();
    assert!(ChannelData::from_ne_bytes(PixelType::Float, &bytes[..bytes.len() - 1], 2).is_err());
}

#[test]
fn channel_data_xdr_bytes_round_trip() {
    for cd in [
        ChannelData::Half(vec![Half::from(1.5), Half::from(-2.0), Half::from(0.0)]),
        ChannelData::Float(vec![1.0, -2.5, 3.0]),
        ChannelData::Uint(vec![0, 42, u32::MAX]),
    ] {
        let bytes = cd.to_xdr_bytes();
        let back = ChannelData::from_xdr_bytes(cd.pixel_type(), &bytes, cd.len()).unwrap();
        assert_eq!(back, cd);
    }
}

#[test]
fn channel_data_xdr_bytes_are_little_endian() {
    // FLOAT bit pattern 1.0f = 0x3f800000; XDR (LE) byte order is 00 00 80 3f.
    let cd = ChannelData::Float(vec![1.0]);
    assert_eq!(cd.to_xdr_bytes(), vec![0x00, 0x00, 0x80, 0x3f]);
}

#[test]
fn channel_data_xdr_bytes_rejects_wrong_length() {
    assert!(ChannelData::from_xdr_bytes(PixelType::Float, &[0u8; 3], 1).is_err());
    assert!(ChannelData::from_xdr_bytes(PixelType::NumPixelTypes, &[], 0).is_err());
}

// ---------------------------------------------------------------------------
// Engagement: the facade feeds the tiled writer in file order
// ---------------------------------------------------------------------------

/// A `size`-tiled image of `dw` with a constant-per-level FLOAT channel on every level.
fn level_image(dw: Box2i, t: TileDescription) -> Image {
    let mut img = Image::new(dw).with_tiling(t);
    let n = img.width() as usize * img.height() as usize;
    img.insert_channel("R", ChannelData::Float(vec![0.0; n])).expect("level 0");
    let (nx, ny) = img.num_levels().expect("levels");
    for ly in 0..ny {
        for lx in 0..nx {
            if (lx, ly) == (0, 0) || (t.mode == LevelMode::MipmapLevels && lx != ly) {
                continue;
            }
            let w = img.level_window(lx, ly).expect("level window");
            let n = (w.max.x - w.min.x + 1) as usize * (w.max.y - w.min.y + 1) as usize;
            img.insert_level(lx, ly, [("R", ChannelData::Float(vec![(lx + ly * 8) as f32; n]))]).expect("level");
        }
    }
    img
}

/// The facade writes whole levels in file order, so the tiled writer never has to
/// buffer a tile that arrived early (`tile_map`): the `TileBuffered` probe stays at 0.
/// The positive control proves the probe can fire: a low-level write with the first
/// tile out of turn does buffer. (Byte identity alone cannot tell: the writer emits
/// the same bytes either way, it just holds more memory.)
#[test]
fn facade_tiled_writes_never_buffer_out_of_order_tiles() {
    use crate::probe::{self, Probe};
    let dw = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 31, y: 19 } };
    for mode in [LevelMode::OneLevel, LevelMode::MipmapLevels, LevelMode::RipmapLevels] {
        let img = level_image(dw, TileDescription { x_size: 4, y_size: 4, mode, rounding_mode: LevelRoundingMode::RoundDown });
        probe::reset(Probe::TileBuffered);
        img.write_to(&mut MemOStream::new(), Compression::Zip).expect("write");
        assert_eq!(probe::get(Probe::TileBuffered), 0, "{mode:?}: the facade fed the writer out of order");
    }

    // Positive control.
    let mut header = Header::new();
    *header.data_window_mut().unwrap() = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 7, y: 7 } };
    *header.display_window_mut().unwrap() = header.data_window().unwrap();
    let mut list = ChannelList::new();
    list.insert("R", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    *header.channels_mut().unwrap() = list;
    header
        .set_tile_description(TileDescription { x_size: 4, y_size: 4, mode: LevelMode::OneLevel, rounding_mode: LevelRoundingMode::RoundDown })
        .unwrap();
    let mut fb = FrameBuffer::new();
    fb.insert("R", Slice::make(PixelType::Float, 0, V2i { x: 0, y: 0 }, 8, 4, 32, 1, 1, 0.0, false, false).unwrap()).unwrap();
    let mut out = MemOStream::new();
    let mut of = crate::tiled::TiledOutputFile::new(&mut out, &header).expect("writer");
    of.set_frame_buffer(&fb).unwrap();
    probe::reset(Probe::TileBuffered);
    for (dx, dy) in [(1, 1), (0, 0), (1, 0), (0, 1)] {
        of.write_tile(&[0u8; 8 * 8 * 4], dx, dy, 0, 0).unwrap();
    }
    assert!(probe::get(Probe::TileBuffered) > 0, "the probe must fire for an out-of-turn tile");
}
