//! Loader tests: fixtures are written with vfx-io's own writers into a temp dir, and every
//! expectation is a closed-form value (sRGB 128/255 = 0.2158605), not a second decode of the same
//! code path.

use super::*;
use vfx_io::{AttrValue, ImageData};

const SRGB_128: f32 = 0.215_860_5;

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp dir")
}

fn px(img: &Image, ch: &str, i: usize) -> f32 {
    img.channel(ch).and_then(|c| c.sample_f32(i)).unwrap_or(f32::NAN)
}

/// A PNG written with straight alpha kept as given (the writer would otherwise un-premultiply).
fn straight_png(path: &Path, w: u32, h: u32, ch: u32, px: Vec<u8>) {
    let mut img = ImageData::from_u8(w, h, ch, px);
    img.metadata.attrs.set("oiio:UnassociatedAlpha", AttrValue::UInt(1));
    vfx_io::write(path, &img).expect("write png");
}

#[test]
fn srgb_8bit_decodes_to_linear() {
    let dir = tmp();
    let p = dir.path().join("a.png");
    vfx_io::write(&p, &ImageData::from_u8(2, 1, 3, vec![128, 128, 128, 0, 255, 255])).unwrap();
    let l = load(&p).unwrap();
    assert!((px(&l.image, "R", 0) - SRGB_128).abs() < 2e-4, "{}", px(&l.image, "R", 0));
    assert_eq!((px(&l.image, "G", 1), px(&l.image, "R", 1)), (1.0, 0.0));
    assert!(l.image.channel("A").is_none());
    assert!(matches!(l.image.channel("R"), Some(ChannelData::Half(_))), "8-bit stays half");
    assert!((Tf::Srgb.lin(128.0 / 255.0) - SRGB_128).abs() < 1e-6);
}

#[test]
fn straight_alpha_is_premultiplied_in_linear_light() {
    let dir = tmp();
    let p = dir.path().join("a.png");
    straight_png(&p, 1, 1, 4, vec![128, 128, 128, 128]);
    let l = load(&p).unwrap();
    let a = 128.0 / 255.0;
    let want = SRGB_128 * a;
    assert!((px(&l.image, "R", 0) - want).abs() < 2e-4, "{} vs {want}", px(&l.image, "R", 0));
    // Premultiplying the encoded value first (OIIO's default) would give a different number.
    let encoded_first = Tf::Srgb.lin(a * a);
    assert!((encoded_first - want).abs() > 0.02);
    assert!((px(&l.image, "A", 0) - a).abs() < 2e-4);
}

#[test]
fn associated_alpha_is_unpremultiplied_before_the_curve() {
    // 4 channels, no straight tag: colour 64 under alpha 128 is the encoded 128 premultiplied.
    let mut img = ImageData::from_u8(1, 1, 4, vec![64, 64, 64, 128]);
    img.metadata.set_colorspace_attribute(colorspace::SRGB_REC709_SCENE);
    let tf = space_of(&img).unwrap().tf;
    let spec = img.to_spec();
    let p = planes(&img, tf, &spec.channel_names, usize::try_from(spec.alpha_channel).ok());
    let a = 128.0 / 255.0;
    assert!((p[0][0] - Tf::Srgb.lin(64.0 / 255.0 / a) * a).abs() < 1e-6);
    assert_eq!(p[3][0], a);
}

#[test]
fn gray_and_16bit_keep_their_shape() {
    let dir = tmp();
    let p = dir.path().join("g.png");
    vfx_io::write(&p, &ImageData::from_u16(1, 1, 1, vec![65535])).unwrap();
    let l = load(&p).unwrap();
    assert!(matches!(l.image.channel("Y"), Some(ChannelData::Float(_))));
    assert_eq!(px(&l.image, "Y", 0), 1.0);
}

#[test]
fn float_hdr_is_linear() {
    let dir = tmp();
    let p = dir.path().join("a.hdr");
    vfx_io::write(&p, &ImageData::from_f32(1, 1, 3, vec![0.5, 0.25, 4.0])).unwrap();
    let l = load(&p).unwrap();
    // RGBE shares one exponent: a few percent, but no curve (an sRGB decode would give 0.21).
    assert!((px(&l.image, "R", 0) - 0.5).abs() < 0.02);
    assert!((px(&l.image, "B", 0) - 4.0).abs() < 0.1);
}

#[test]
fn cineon_log_maps_white_and_black() {
    assert!((Tf::Cineon.lin(685.0 / 1023.0) - 1.0).abs() < 1e-5);
    assert!(Tf::Cineon.lin(95.0 / 1023.0).abs() < 1e-5);
}

#[test]
fn untagged_rule_and_unknown_tag() {
    let u8img = ImageData::from_u8(1, 1, 3, vec![0; 3]);
    assert_eq!(space_of(&u8img).unwrap().tf, Tf::Srgb);
    let f = ImageData::from_f32(1, 1, 3, vec![0.0; 3]);
    assert_eq!(space_of(&f).unwrap().tf, Tf::Linear);
    let mut odd = ImageData::from_f32(1, 1, 3, vec![0.0; 3]);
    odd.metadata.set_colorspace_attribute("CMYK");
    assert!(matches!(space_of(&odd), Err(ImportError::ColorSpace(n)) if n == "CMYK"));
    let mut g = ImageData::from_f32(1, 1, 3, vec![0.0; 3]);
    g.metadata.set_colorspace_rec709_gamma(2.4);
    assert_eq!(space_of(&g).unwrap().tf, Tf::Gamma(2.4));
}

#[test]
fn registry_lists_exr_first_and_no_heif() {
    let f = families();
    assert_eq!(f[0].key, "EXR");
    for want in ["png", "dpx", "tga", "hdr", "jpg"] {
        assert!(f.iter().any(|x| x.exts.iter().any(|e| e == want)), "{want}");
    }
    assert!(!f.iter().any(|x| x.exts.iter().any(|e| e == "heic" || e == "avif")));
    assert!(is_image(Path::new("a.EXR")) && is_image(Path::new("s.0001.Png")) && !is_image(Path::new("a.txt")));
    assert!(is_foreign(Path::new("a.dpx")) && !is_foreign(Path::new("a.exr")));
}

#[test]
fn exr_is_not_ours_and_garbage_is_typed() {
    let dir = tmp();
    let e = dir.path().join("a.exr");
    std::fs::write(&e, b"v/1\x01\x02\0\0\0").unwrap();
    assert!(matches!(load(&e), Err(ImportError::Exr(_))));
    let bad = dir.path().join("b.png");
    std::fs::write(&bad, b"not a png").unwrap();
    assert!(matches!(load(&bad), Err(ImportError::Decode { .. })));
}

fn tagged(name: &str) -> ImageData {
    let mut img = ImageData::from_f32(1, 1, 3, vec![0.0; 3]);
    img.metadata.set_colorspace_attribute(name);
    img
}

#[test]
fn display_gamma_tag_is_the_inverse_display_eotf() {
    // `g24_rec709_display` is the tag of the 519-frame PNG sequence that used to fail to open.
    let s = space_of(&tagged("g24_rec709_display")).unwrap();
    assert_eq!((s.tf, s.to709), (Tf::Gamma(2.4), None));
    assert!((s.tf.lin(0.5) - 0.5f32.powf(2.4)).abs() < 1e-6);
    // The interop-ID table spells the other transfers out the same way.
    assert_eq!(space_of(&tagged("g22_rec709_display")).unwrap().tf, Tf::Gamma(2.2));
    assert_eq!(space_of(&tagged("srgb_rec709_display")).unwrap().tf, Tf::Srgb);
    let pq = space_of(&tagged("pq_rec2020_display")).unwrap();
    assert_eq!(pq.tf, Tf::Pq);
    assert!((pq.tf.lin(0.508_078) - 1.0).abs() < 0.01, "ST 2084: 100 nits is signal 0.508");
}

#[test]
fn wide_primaries_rotate_into_rec709() {
    let m = space_of(&tagged("lin_p3d65_scene")).unwrap().to709.expect("P3 is not Rec.709");
    // Published linear Display-P3 -> Rec.709 (D65 both): the columns are the P3 primaries.
    let want = [[1.224_940_1, -0.224_940_4, 0.0], [-0.042_056_9, 1.042_057_1, 0.0], [-0.019_637_6, -0.078_636_1, 1.098_273_5]];
    for (r, w) in m.iter().zip(want) {
        for (a, b) in r.iter().zip(w) {
            assert!((a - b).abs() < 2e-4, "{m:?}");
        }
    }
    // Pure P3 red lands out of gamut: the negative stays.
    let mut planes = vec![vec![1.0], vec![0.0], vec![0.0]];
    rotate(&mut planes, &["R".into(), "G".into(), "B".into()], &m);
    assert!((planes[0][0] - 1.224_940_1).abs() < 2e-4 && planes[1][0] < -0.04 && planes[2][0] < -0.01, "{planes:?}");
}

#[test]
fn d60_primaries_keep_neutrals_neutral() {
    // ACEScg white is D60; without the adaptation to D65 it would come out tinted.
    for tag in ["lin_ap1_scene", "lin_ap0_scene"] {
        let m = space_of(&tagged(tag)).unwrap().to709.unwrap_or_else(|| panic!("{tag}"));
        for row in m {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 2e-3, "{tag}: {m:?}");
        }
    }
}

#[test]
fn tags_the_table_cannot_make_rgb_stay_typed_errors() {
    for tag in ["CMYK", "lin_ciexyzd65_scene", "data", "no_such_space"] {
        assert!(matches!(space_of(&tagged(tag)), Err(ImportError::ColorSpace(n)) if n == tag), "{tag}");
    }
}
