//! Multipart Cryptomatte files: a Cryptomatte type in its
//! own part is found, opened and extracted; its metadata may sit in another
//! part's header; inject re-encodes one part and raw-copies every other.

use std::fs;
use std::path::Path;

use cryptomatte::{
    discover_path, from_named_mattes, CryptomatteFile, EncodeOptions, Error, HeaderAttrs,
    MergeOp, NamedMatte, Windows,
};
use exr_core::attr::{
    Channel, ChannelList, ChannelListAttribute, Compression, ExrString, LevelMode,
    LevelRoundingMode, LineOrder, PixelType, TileDescription,
};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::container::xdr;
use exr_core::deep::frame_buffer::DeepSlice;
use exr_core::deep::DeepFrameBuffer;
use exr_core::header::{Header, DEEP_SCANLINE, DEEP_TILE, SCANLINE_IMAGE, TILED_IMAGE};
use exr_core::io::{MemIStream, MemOStream, StdIfStream, StdOfStream};
use exr_core::{IStream, MultiPartInputFile, MultiPartOutputFile, RawImage, StringAttribute};
use imath_rs::V2f;

const W: usize = 6;
const H: usize = 4;
const N: usize = W * H;

/// One part to write: FLOAT channels (row-major planes) plus header attributes.
struct PartSpec {
    name: &'static str,
    channels: Vec<(String, Vec<f32>)>,
    attrs: HeaderAttrs,
    /// Write the part tiled (4x4 tiles, one level) instead of scanline.
    tiled: bool,
    /// Write a deep part (scanline, or tiled when `tiled`): one FLOAT `Z`, counts `i % 3`.
    deep: bool,
}

/// A scratch directory removed when the guard drops, also when the test fails; the test
/// file is `multi.exr` in it.
fn scratch(name: &str) -> tempfile::TempDir {
    tempfile::Builder::new().prefix(&format!("exr-rs-crypto-{name}-")).tempdir().expect("scratch dir")
}

/// Part `i` as a deep part: sample counts `pixel % 3`, sample value `pixel * 10 + s`.
fn write_deep(mp: &mut MultiPartOutputFile<'_>, i: i32, tiled: bool) {
    let counts: Vec<u32> = (0..N as u32).map(|p| p % 3).collect();
    let mut sc = vec![0u8; N * 4];
    for (k, c) in counts.iter().enumerate() {
        sc[k * 4..k * 4 + 4].copy_from_slice(&c.to_le_bytes());
    }
    let mut heap = Vec::new();
    for (pix, &c) in counts.iter().enumerate() {
        for smp in 0..c {
            heap.extend_from_slice(&((pix * 10) as f32 + smp as f32).to_le_bytes());
        }
    }
    let mut fb = DeepFrameBuffer::new();
    let count_slice = Slice {
        type_: PixelType::Uint,
        base_offset: 0,
        x_stride: 4,
        y_stride: 4 * W,
        x_sampling: 1,
        y_sampling: 1,
        fill_value: 0.0,
        x_tile_coords: false,
        y_tile_coords: false,
    };
    fb.insert_sample_count_slice(count_slice).expect("count slice");
    fb.insert("Z", DeepSlice::new(PixelType::Float, 0, 4, 4 * W, 4, 1, 1, 0.0, false, false).expect("Z"))
        .expect("Z slice");
    if tiled {
        let mut p = mp.deep_tiled_output_part_wrapper(i).expect("deep tiled part");
        p.set_frame_buffer(&fb).expect("fb");
        let (nx, ny) = (p.num_x_tiles(0).expect("nx"), p.num_y_tiles(0).expect("ny"));
        p.write_tiles(&sc, &heap, 0, nx - 1, 0, ny - 1, 0, 0).expect("deep tiles");
    } else {
        let mut p = mp.deep_scanline_output_part_wrapper(i).expect("deep scanline part");
        p.set_frame_buffer(&fb).expect("fb");
        p.write_pixels(&sc, &heap, H as i32).expect("deep scanlines");
    }
}

fn write_multipart(path: &Path, parts: &[PartSpec]) {
    let headers: Vec<Header> = parts
        .iter()
        .map(|p| {
            let mut h = Header::with_dimensions(
                W as i32,
                H as i32,
                1.0,
                V2f { x: 0.0, y: 0.0 },
                1.0,
                LineOrder::IncreasingY,
                // Deep parts allow NONE / RLE / ZIPS only.
                if p.deep { Compression::Zips } else { Compression::Zip },
            )
            .expect("dims");
            h.set_name(p.name).expect("name");
            if p.tiled {
                h.set_type(if p.deep { DEEP_TILE } else { TILED_IMAGE }).expect("type");
                h.set_tile_description(TileDescription {
                    // One tile for a deep part: its heap is laid out in raster order.
                    x_size: if p.deep { 8 } else { 4 },
                    y_size: if p.deep { 8 } else { 4 },
                    mode: LevelMode::OneLevel,
                    rounding_mode: LevelRoundingMode::RoundDown,
                })
                .expect("tiles");
            } else {
                h.set_type(if p.deep { DEEP_SCANLINE } else { SCANLINE_IMAGE }).expect("type");
            }
            let mut cl = ChannelList::new();
            for (c, _) in &p.channels {
                cl.insert(c, Channel::new(PixelType::Float, 1, 1, false)).expect("channel");
            }
            h.insert("channels", ChannelListAttribute::new(cl)).expect("channels");
            for (k, a) in &p.attrs {
                h.insert_boxed(k.clone(), a.clone_attr()).expect("attr");
            }
            h
        })
        .collect();

    let mut os = StdOfStream::create(path.to_str().expect("utf8")).expect("create");
    let mut mp = MultiPartOutputFile::new(&mut os, &headers).expect("multipart out");
    for (i, p) in parts.iter().enumerate() {
        if p.deep {
            write_deep(&mut mp, i as i32, p.tiled);
            continue;
        }
        // Interleave the planes into one native-endian f32 pixel buffer.
        let nch = p.channels.len();
        let mut px = vec![0u8; N * nch * 4];
        for (k, (_, plane)) in p.channels.iter().enumerate() {
            for (j, v) in plane.iter().enumerate() {
                let o = (j * nch + k) * 4;
                px[o..o + 4].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let mut fb = FrameBuffer::new();
        for (k, (c, _)) in p.channels.iter().enumerate() {
            let slice = Slice {
                type_: PixelType::Float,
                base_offset: (k * 4) as i64,
                x_stride: nch * 4,
                y_stride: W * nch * 4,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            };
            fb.insert(c, slice).expect("slice");
        }
        if p.tiled {
            let mut part = mp.tiled_output_part(i as i32).expect("tiled part");
            part.set_frame_buffer(&fb).expect("fb");
            let (nx, ny) = (part.num_x_tiles(0).expect("nx"), part.num_y_tiles(0).expect("ny"));
            part.write_tiles(&px, 0, nx - 1, 0, ny - 1, 0, 0).expect("write tiles");
        } else {
            let mut part = mp.output_part(i as i32).expect("part");
            part.set_frame_buffer(&fb).expect("fb");
            part.write_pixels(&px, H as i32).expect("write");
        }
    }
}

/// `bunny` covering every pixel, as CryptoObject channels + header attributes.
fn bunny_payload() -> (Vec<(String, Vec<f32>)>, HeaderAttrs) {
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    let enc = from_named_mattes(Windows::origin(W, H).unwrap(), &[NamedMatte::new("bunny", &[1.0; N])], &opts)
        .expect("encode");
    let channels = enc
        .image
        .channels()
        .map(|(n, d)| (n.to_string(), cryptomatte::channel_as_f32(d).unwrap().into_owned()))
        .collect();
    let attrs = enc.attrs.iter().map(|(k, a)| (k.clone(), a.clone_attr())).collect();
    (channels, attrs)
}

fn beauty() -> Vec<(String, Vec<f32>)> {
    vec![("R".to_string(), vec![0.5; N])]
}

fn assert_bunny_in_part_1(path: &Path) {
    let layers = discover_path(path).expect("discover");
    assert_eq!(layers.len(), 1, "one CryptoObject entry: {layers:?}");
    assert_eq!((layers[0].name.as_str(), layers[0].part), ("CryptoObject", 1));

    let file = CryptomatteFile::open(path).expect("open");
    assert_eq!(file.part, 1);
    assert!(file.multipart);
    let layer = file.select_layer(None).expect("layer");
    let matte = file.extract_names(layer, &["bunny"]).expect("extract");
    assert_eq!(matte.sum(), N as f32);
}

#[test]
fn type_in_its_own_part_is_found_and_extracted() {
    let tmp = scratch("mp-own-part");
    let path = tmp.path().join("multi.exr");
    let (crypto, attrs) = bunny_payload();
    write_multipart(
        &path,
        &[
            PartSpec { name: "beauty", channels: beauty(), attrs: Vec::new(), tiled: false, deep: false },
            PartSpec { name: "crypto", channels: crypto, attrs, tiled: false, deep: false },
        ],
    );
    assert_bunny_in_part_1(&path);
}

#[test]
fn metadata_in_another_part_follows_the_channels() {
    let tmp = scratch("mp-split");
    let path = tmp.path().join("multi.exr");
    let (crypto, attrs) = bunny_payload();
    write_multipart(
        &path,
        &[
            PartSpec { name: "beauty", channels: beauty(), attrs, tiled: false, deep: false },
            PartSpec { name: "crypto", channels: crypto, attrs: Vec::new(), tiled: false, deep: false },
        ],
    );
    assert_bunny_in_part_1(&path);
}

#[test]
fn metadata_repeated_in_every_part_is_listed_once() {
    let tmp = scratch("mp-repeated");
    let path = tmp.path().join("multi.exr");
    let (crypto, attrs) = bunny_payload();
    let attrs2 = attrs.iter().map(|(k, a)| (k.clone(), a.clone_attr())).collect();
    write_multipart(
        &path,
        &[
            PartSpec { name: "beauty", channels: beauty(), attrs, tiled: false, deep: false },
            PartSpec { name: "crypto", channels: crypto, attrs: attrs2, tiled: false, deep: false },
        ],
    );
    assert_bunny_in_part_1(&path);
}

#[test]
fn unknown_layers_error_on_a_multipart_file() {
    let tmp = scratch("mp-unknown");
    let path = tmp.path().join("multi.exr");
    let (crypto, attrs) = bunny_payload();
    write_multipart(
        &path,
        &[
            PartSpec { name: "beauty", channels: beauty(), attrs: Vec::new(), tiled: false, deep: false },
            PartSpec { name: "crypto", channels: crypto, attrs, tiled: false, deep: false },
        ],
    );
    let file = CryptomatteFile::open(&path).expect("open");
    let err = CryptomatteFile::open_for_layer(&path, Some("Nope")).unwrap_err();
    assert!(matches!(err, Error::Metadata(_)), "{err:?}");
    assert!(file.select_layer(Some("Nope")).is_err());
}

// ---- inject into a multipart file ------------------------------------------

fn string_attr(v: &str) -> Box<dyn exr_core::Attribute> {
    Box::new(StringAttribute::new(ExrString::from(v)))
}

/// beauty (scanline) | crypto + `Z` + `comments` + `view` | tiled AOV | scanline AOV, and with
/// `deep` a deep scanline and a deep tiled part between the last two. `meta_in` is the part
/// whose header carries the `cryptomatte/` keys.
fn mixed_source(path: &Path, meta_in: usize, deep: bool) {
    let (mut crypto, attrs) = bunny_payload();
    crypto.push(("Z".to_string(), (0..N).map(|i| i as f32).collect()));
    let mut attrs = Some(attrs);
    let mut spec = |name, channels, tiled, deep, idx: usize, extra: &[(&str, &str)]| {
        let mut a: HeaderAttrs = if idx == meta_in { attrs.take().unwrap_or_default() } else { Vec::new() };
        for (k, v) in extra {
            a.push(((*k).into(), string_attr(v)));
        }
        PartSpec { name, channels, attrs: a, tiled, deep }
    };
    let z = || vec![("Z".to_string(), Vec::new())];
    let mut parts = vec![
        spec("beauty", beauty(), false, false, 0, &[]),
        spec("crypto", crypto, false, false, 1, &[("comments", "keep me"), ("view", "left")]),
        spec("tiledaov", vec![("T".to_string(), vec![0.25; N])], true, false, 2, &[]),
    ];
    if deep {
        parts.push(spec("deepscan", z(), false, true, 3, &[]));
        parts.push(spec("deeptile", z(), true, true, 4, &[]));
    }
    parts.push(spec("depth", vec![("depth".to_string(), vec![2.0; N])], false, false, 5, &[]));
    write_multipart(path, &parts);
}

fn four_part_source(path: &Path, meta_in: usize) {
    mixed_source(path, meta_in, false);
}

/// Inject `cube` (full coverage, over) into `CryptoObject` of `src`, writing `out`.
fn inject_cube(src: &Path, out: &Path, compression: Option<Compression>) -> Result<(), Error> {
    let file = CryptomatteFile::open(src)?;
    let layer = file.select_layer(Some("CryptoObject"))?;
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    file.inject(layer, "cube", &[1.0; N], MergeOp::Over, &opts)?.write(out, compression)
}

fn raw(path: &Path) -> RawImage {
    RawImage::read(&mut StdIfStream::open(path.to_str().expect("utf8")).expect("open")).expect("raw read")
}

/// A header's attributes serialised, to compare two headers exactly.
fn header_bytes(h: &Header) -> Vec<u8> {
    let mut os = MemOStream::new();
    h.write_attributes_to(&mut os).expect("header bytes");
    os.into_data()
}

#[test]
fn inject_rewrites_one_part_and_raw_copies_the_rest() {
    let tmp = scratch("mp-inject-src");
    let src = tmp.path().join("multi.exr");
    let out = src.with_file_name("out.exr");
    four_part_source(&src, 1);
    inject_cube(&src, &out, None).expect("inject");
    let left = fs::read_dir(src.parent().expect("dir")).expect("ls").count();
    assert_eq!(left, 2, "only the source and the output: no temporary file is left");

    let (a, b) = (raw(&src), raw(&out));
    assert_eq!(a.parts.len(), b.parts.len());
    for (i, (pa, pb)) in a.parts.iter().zip(&b.parts).enumerate() {
        assert_eq!(pa.header.name().expect("name"), pb.header.name().expect("name"), "part {i} name");
        assert_eq!(pa.header.type_().expect("type"), pb.header.type_().expect("type"), "part {i} type");
        if i != 1 {
            assert_eq!(pa.chunks, pb.chunks, "part {i} chunks are byte-identical");
            assert_eq!(header_bytes(&pa.header), header_bytes(&pb.header), "part {i} header");
        }
    }
    // The tiled AOV stayed tiled.
    assert!(b.parts[2].header.has_tile_description());

    // The injected part: new matte, old matte covered, other channel and attribute kept.
    let file = CryptomatteFile::open_for_layer(&out, Some("CryptoObject")).expect("open out");
    assert_eq!((file.part, file.multipart), (1, true));
    let layer = file.select_layer(None).expect("layer");
    assert_eq!(file.extract_names(layer, &["cube"]).expect("cube").sum(), N as f32);
    assert_eq!(file.extract_names(layer, &["bunny"]).expect("bunny").sum(), 0.0);
    let z: Vec<f32> = cryptomatte::channel_as_f32(file.image.channel("Z").expect("Z")).unwrap().into_owned();
    assert_eq!(z, (0..N).map(|i| i as f32).collect::<Vec<_>>());
    assert!(file.header.iter().any(|(k, _)| k.as_bytes() == b"comments"));
    assert_eq!(file.header.view().expect("view survives"), "left");
    assert_eq!(b.parts[1].header.compression().unwrap(), a.parts[1].header.compression().unwrap());
    assert_eq!(b.parts[1].header.type_().unwrap(), SCANLINE_IMAGE);
}

#[test]
fn inject_compression_override_applies_to_the_injected_part_only() {
    let tmp = scratch("mp-inject-comp");
    let src = tmp.path().join("multi.exr");
    let out = src.with_file_name("out.exr");
    four_part_source(&src, 1);
    inject_cube(&src, &out, Some(Compression::Rle)).expect("inject");
    let (a, b) = (raw(&src), raw(&out));
    assert_eq!(b.parts[1].header.compression().unwrap(), Compression::Rle);
    for i in [0, 2, 3] {
        assert_eq!(a.parts[i].chunks, b.parts[i].chunks, "part {i}");
        assert_eq!(b.parts[i].header.compression().unwrap(), Compression::Zip);
    }
}

#[test]
fn inject_drops_stale_metadata_from_other_parts() {
    let tmp = scratch("mp-inject-stale");
    let src = tmp.path().join("multi.exr");
    let out = src.with_file_name("out.exr");
    four_part_source(&src, 0); // keys in the beauty part, channels in part 1
    inject_cube(&src, &out, None).expect("inject");

    let b = raw(&out);
    let keys = |i: usize| b.parts[i].header.iter().filter(|(k, _)| k.as_bytes().starts_with(b"cryptomatte/")).count();
    assert_eq!(keys(0), 0, "the stale keys left the beauty part");
    assert!(keys(1) > 0, "the type's keys sit beside its channels");

    let layers = discover_path(&out).expect("discover");
    assert_eq!(layers.len(), 1);
    let file = CryptomatteFile::open(&out).expect("open");
    let manifest = file.load_layer_manifest(file.select_layer(None).unwrap()).expect("manifest");
    assert!(manifest.by_name.contains_key("bunny") && manifest.by_name.contains_key("cube"), "{manifest:?}");
}

#[test]
fn inject_refuses_to_overwrite_its_multipart_source() {
    let tmp = scratch("mp-inject-same");
    let src = tmp.path().join("multi.exr");
    four_part_source(&src, 1);
    let before = fs::read(&src).expect("read");
    let err = inject_cube(&src, &src, None).unwrap_err();
    assert!(matches!(err, Error::Io(_)), "{err:?}");
    assert_eq!(fs::read(&src).expect("read"), before, "the source is untouched");
}

#[test]
fn inject_into_a_one_part_multipart_file_needs_no_splice() {
    let tmp = scratch("mp-inject-one");
    let src = tmp.path().join("multi.exr");
    let out = src.with_file_name("out.exr");
    let (crypto, attrs) = bunny_payload();
    write_multipart(&src, &[PartSpec { name: "crypto", channels: crypto, attrs, tiled: false, deep: false }]);
    inject_cube(&src, &out, None).expect("inject");
    let file = CryptomatteFile::open(&out).expect("open");
    let layer = file.select_layer(None).expect("layer");
    assert_eq!(file.extract_names(layer, &["cube"]).expect("cube").sum(), N as f32);
}

/// Each part's chunk bytes (leader, `partNumber` and payload), sliced out of a multipart file by
/// the offset tables. `RawImage` does not frame deep chunks, and no chunk holds an absolute
/// offset, so equal regions mean byte-identical chunks wherever the part sits in the file.
fn chunk_regions(bytes: &[u8], parts: usize) -> Vec<Vec<u8>> {
    let mut is = MemIStream::new(bytes.to_vec());
    let _magic = xdr::read_i32(&mut is).expect("magic");
    let version = xdr::read_i32(&mut is).expect("version");
    let counts: Vec<i32> = (0..parts)
        .map(|_| Header::read_attributes_from(&mut is, version).expect("header").chunk_count().expect("chunkCount"))
        .collect();
    let mut term = [0u8; 1];
    is.read(&mut term).expect("terminator");
    let mut firsts = Vec::new();
    for c in counts {
        let offsets: Vec<u64> = (0..c).map(|_| xdr::read_u64(&mut is).expect("offset")).collect();
        firsts.push(*offsets.iter().min().expect("a part has chunks") as usize);
    }
    firsts.push(bytes.len());
    firsts.windows(2).map(|w| bytes[w[0]..w[1]].to_vec()).collect()
}

/// Every header of `path`, as `MultiPartInputFile` reads it.
fn headers_of(path: &Path) -> Vec<Header> {
    let mut is = StdIfStream::open(path.to_str().expect("utf8")).expect("open");
    let file = MultiPartInputFile::new(&mut is).expect("multipart in");
    (0..file.parts()).map(|p| file.header(p).expect("header").clone()).collect()
}

#[test]
fn inject_raw_copies_deep_and_tiled_parts_byte_for_byte() {
    let tmp = scratch("mp-inject-deep");
    let src = tmp.path().join("multi.exr");
    let out = src.with_file_name("out.exr");
    mixed_source(&src, 1, true);
    inject_cube(&src, &out, None).expect("inject");

    let (a, b) = (fs::read(&src).expect("src"), fs::read(&out).expect("out"));
    let (ra, rb) = (chunk_regions(&a, 6), chunk_regions(&b, 6));
    let (ha, hb) = (headers_of(&src), headers_of(&out));
    let types: Vec<String> = hb.iter().map(|h| h.type_().expect("type").to_string()).collect();
    assert_eq!(types, [SCANLINE_IMAGE, SCANLINE_IMAGE, TILED_IMAGE, DEEP_SCANLINE, DEEP_TILE, SCANLINE_IMAGE]);
    for i in [0, 2, 3, 4, 5] {
        assert!(ra[i].len() > 16, "part {i} has chunks");
        assert_eq!(ra[i], rb[i], "part {i} chunks are byte-identical");
        assert_eq!(header_bytes(&ha[i]), header_bytes(&hb[i]), "part {i} header");
    }
    assert_ne!(ra[1], rb[1], "the injected part changed");
    let file = CryptomatteFile::open_for_layer(&out, Some("CryptoObject")).expect("open out");
    let layer = file.select_layer(None).expect("layer");
    assert_eq!(file.extract_names(layer, &["cube"]).expect("cube").sum(), N as f32);
}

// ---- tiled parts -----------------------------------------------------------

#[test]
fn inject_keeps_a_one_level_tiling_of_the_injected_part() {
    let tmp = scratch("mp-inject-tiled");
    let src = tmp.path().join("multi.exr");
    let out = src.with_file_name("out.exr");
    let (crypto, attrs) = bunny_payload();
    write_multipart(
        &src,
        &[
            PartSpec { name: "beauty", channels: beauty(), attrs: Vec::new(), tiled: false, deep: false },
            PartSpec { name: "crypto", channels: crypto, attrs, tiled: true, deep: false },
        ],
    );
    inject_cube(&src, &out, None).expect("inject");
    let b = raw(&out);
    let h = &b.parts[1].header;
    assert_eq!(h.type_().unwrap(), TILED_IMAGE);
    let td = h.tile_description().expect("still tiled");
    assert_eq!((td.x_size, td.y_size, td.mode), (4, 4, LevelMode::OneLevel));
    let file = CryptomatteFile::open(&out).expect("open");
    let layer = file.select_layer(None).expect("layer");
    assert_eq!(file.extract_names(layer, &["cube"]).expect("cube").sum(), N as f32);
}

/// A single-part MIPMAP tiled file with a bunny CryptoObject.
fn write_mip_source(path: &Path) {
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    let mut enc = from_named_mattes(Windows::origin(W, H).unwrap(), &[NamedMatte::new("bunny", &[1.0; N])], &opts)
        .expect("encode");
    let td = TileDescription { x_size: 4, y_size: 4, mode: LevelMode::MipmapLevels, rounding_mode: LevelRoundingMode::RoundDown };
    enc.image.set_tiling(Some(td));
    for l in 1..=2 {
        let win = enc.image.level_window(l, l).expect("level window");
        let n = ((win.max.x - win.min.x + 1) * (win.max.y - win.min.y + 1)) as usize;
        let planes: Vec<(String, exr_core::ChannelData)> = enc
            .image
            .channels()
            .map(|(name, _)| (name.to_string(), exr_core::ChannelData::Float(vec![0.0; n])))
            .collect();
        enc.image.insert_level(l, l, planes).expect("level");
    }
    enc.write(path, Some(Compression::Zip)).expect("write mip");
}

#[test]
fn inject_refuses_a_mip_tiled_part_instead_of_losing_its_levels() {
    let tmp = scratch("mp-inject-mip");
    let src = tmp.path().join("multi.exr");
    write_mip_source(&src);
    // Reading and extracting is fine: level 0 is what a matte is made of.
    let file = CryptomatteFile::open(&src).expect("open");
    let layer = file.select_layer(None).expect("layer");
    assert_eq!(file.extract_names(layer, &["bunny"]).expect("bunny").sum(), N as f32);
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    let err = file.inject(layer, "cube", &[1.0; N], MergeOp::Over, &opts).unwrap_err();
    assert!(matches!(&err, Error::Unsupported(m) if m.contains("MipmapLevels")), "{err:?}");
}

#[test]
fn single_part_inject_keeps_a_one_level_tiling() {
    let tmp = scratch("sp-inject-tiled");
    let src = tmp.path().join("multi.exr");
    let out = src.with_file_name("out.exr");
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    let mut enc = from_named_mattes(Windows::origin(W, H).unwrap(), &[NamedMatte::new("bunny", &[1.0; N])], &opts)
        .expect("encode");
    let td = TileDescription { x_size: 4, y_size: 4, mode: LevelMode::OneLevel, rounding_mode: LevelRoundingMode::RoundDown };
    enc.image.set_tiling(Some(td));
    enc.write(&src, Some(Compression::Zip)).expect("write tiled");
    inject_cube(&src, &out, None).expect("inject");
    let b = raw(&out);
    assert_eq!(b.parts.len(), 1);
    assert_eq!(b.parts[0].header.tile_description().expect("still tiled").x_size, 4);
    let file = CryptomatteFile::open(&out).expect("open");
    let layer = file.select_layer(None).expect("layer");
    assert_eq!(file.extract_names(layer, &["cube"]).expect("cube").sum(), N as f32);
}

/// The spliced multipart file opens and decodes in C++ OpenEXR (`exrcheck`). `#[ignore]`d like the
/// other oracle gates: run with `EXR_TOOLS_DIR` set to the folder `cargo xtask parity-tools` fills
/// (`target/parity-tools`). The flat file goes through the library path in every mode; the file
/// with deep parts only through the core path (`-c`): the library path rejects valid deep files
/// (`exrcheck`'s `readDeepTile` request bug, see `tests/parity_tool_check.rs`, and every deep
/// scanline golden), which `exr-rs check` reproduces on purpose.
#[test]
#[ignore = "needs the C++ exrcheck (EXR_TOOLS_DIR)"]
fn spliced_file_passes_cpp_exrcheck() {
    let dir = std::env::var("EXR_TOOLS_DIR").expect("EXR_TOOLS_DIR is not set");
    let exe = Path::new(&dir).join(if cfg!(windows) { "exrcheck.exe" } else { "exrcheck" });
    let check = |file: &Path, mode: &[&str]| {
        let res = std::process::Command::new(&exe).args(mode).arg(file).output().expect("run exrcheck");
        assert!(
            res.status.success(),
            "exrcheck {mode:?} {}: {}{}",
            file.display(),
            String::from_utf8_lossy(&res.stdout),
            String::from_utf8_lossy(&res.stderr)
        );
    };
    for (deep, modes) in [(false, vec![vec![], vec!["-m"], vec!["-t"], vec!["-c"]]), (true, vec![vec!["-c"]])] {
        let tmp = scratch(if deep { "mp-inject-cpp-deep" } else { "mp-inject-cpp" });
        let src = tmp.path().join("multi.exr");
        let out = src.with_file_name("out.exr");
        mixed_source(&src, 1, deep);
        inject_cube(&src, &out, None).expect("inject");
        for mode in &modes {
            check(&src, mode); // the fixture itself is a valid file
            check(&out, mode);
        }
    }
}

#[test]
fn a_failing_splice_leaves_no_output_and_an_existing_one_untouched() {
    let tmp = scratch("mp-inject-fail");
    let src = tmp.path().join("multi.exr");
    let dir = src.parent().expect("dir").to_path_buf();
    four_part_source(&src, 1);
    let file = CryptomatteFile::open(&src).expect("open");
    let layer = file.select_layer(Some("CryptoObject")).expect("layer");
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    let enc = file.inject(layer, "cube", &[1.0; N], MergeOp::Over, &opts).expect("inject");

    // The source is re-opened at write time: make that fail.
    fs::remove_file(&src).expect("remove source");
    let fresh = dir.join("fresh.exr");
    assert!(enc.write(&fresh, None).is_err());
    assert!(!fresh.exists(), "no output for a failed splice");

    let existing = dir.join("existing.exr");
    fs::write(&existing, b"precious").expect("write");
    assert!(enc.write(&existing, None).is_err());
    assert_eq!(fs::read(&existing).expect("read"), b"precious", "an existing output survives");

    let left: Vec<_> = fs::read_dir(&dir).expect("ls").map(|e| e.expect("entry").file_name()).collect();
    assert_eq!(left.len(), 1, "no partial file left behind: {left:?}");
}

/// The stream write refuses a sidecar with this exact, single-spaced message.
#[test]
fn stream_write_refuses_a_sidecar_with_a_clean_message() {
    let opts = EncodeOptions {
        manifest: cryptomatte::ManifestPlacement::Sidecar { relative_path: "m.json".into() },
        ..EncodeOptions::new("CryptoObject")
    };
    let enc = from_named_mattes(Windows::origin(1, 1).unwrap(), &[NamedMatte::new("a", &[1.0])], &opts).expect("encode");
    let err = enc.write_to(&mut MemOStream::new(), None).unwrap_err();
    assert_eq!(
        err.to_string(),
        "manifest: a stream write cannot place the sidecar manifest `m.json`; use `write(path)` or an inline manifest"
    );
}


// ---- channel spelling -------------------------------------------------------

/// `CryptoObject` with the suffixes of `spelling`, as a single-part file with its metadata.
fn write_spelled_source(path: &Path, spelling: cryptomatte::Spelling) {
    let opts = EncodeOptions { ranked_layers: 1, ..EncodeOptions::new("CryptoObject") };
    let enc = from_named_mattes(Windows::origin(W, H).unwrap(), &[NamedMatte::new("bunny", &[1.0; N])], &opts)
        .expect("encode");
    let mut image = exr_core::Image::new(enc.image.data_window);
    image.insert_channel("R", exr_core::ChannelData::Float(vec![0.5; N])).expect("beauty");
    for (name, data) in enc.image.channels() {
        let suffix = name.rsplit('.').next().expect("suffix");
        let comp = cryptomatte::Component::parse(suffix).expect("component");
        let base = name.strip_suffix(suffix).expect("base");
        image.insert_channel(format!("{base}{}", spelling.suffix(comp)), data.clone()).expect("channel");
    }
    let attrs = enc.attrs.iter().map(|(k, a)| (k.clone(), a.clone_attr())).collect();
    image.write_with_attrs(path, Compression::Zip, attrs).expect("write");
}

#[test]
fn inject_keeps_the_channel_spelling_the_file_uses() {
    use cryptomatte::Spelling;
    for (spelling, rank0) in [
        (Spelling::Long, "CryptoObject00.red"),
        (Spelling::Lower, "CryptoObject00.r"),
        (Spelling::Upper, "CryptoObject00.R"),
    ] {
        let tmp = scratch(&format!("spell-{spelling:?}"));
        let src = tmp.path().join("multi.exr");
        let out = src.with_file_name("out.exr");
        write_spelled_source(&src, spelling);
        inject_cube(&src, &out, None).expect("inject");
        let file = CryptomatteFile::open(&out).expect("open");
        let names: Vec<&str> = file.image.channels().map(|(n, _)| n).collect();
        assert!(names.contains(&rank0), "{spelling:?}: {names:?}");
        assert!(names.contains(&"R"), "the beauty channel is not renamed: {names:?}");
        let ranked = names.iter().filter(|n| n.starts_with("CryptoObject00.")).count();
        assert_eq!(ranked, 4, "{spelling:?}: one spelling only, no stale twins: {names:?}");
        let layer = file.select_layer(None).expect("layer");
        assert_eq!(file.extract_names(layer, &["cube"]).expect("cube").sum(), N as f32);
    }
}
