//! Loader and tag tests: fixtures are written with vfx-io's own writers into a temp dir, and every
//! expectation is a closed-form value (a code over its maximum, a published matrix), not a second
//! decode of the same code path.

use super::*;
use vfx_io::{AttrValue, ImageData};

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

/// An 8-bit PNG keeps its codes (no linearisation) and says it is sRGB (vfx-io assumes sRGB for a
/// PNG without colour chunks, as OIIO does).
#[test]
fn srgb_8bit_keeps_its_codes() {
    let dir = tmp();
    let p = dir.path().join("a.png");
    vfx_io::write(&p, &ImageData::from_u8(2, 1, 3, vec![128, 128, 128, 0, 255, 255])).unwrap();
    let l = load(&p).unwrap();
    assert!((px(&l.image, "R", 0) - 128.0 / 255.0).abs() < 2e-4, "{}", px(&l.image, "R", 0));
    assert_eq!((px(&l.image, "G", 1), px(&l.image, "R", 1)), (1.0, 0.0));
    assert!(l.image.channel("A").is_none() && !l.straight);
    assert!(matches!(l.image.channel("R"), Some(ChannelData::Half(_))), "8-bit stays half");
    assert_eq!(l.tag, ColorTag { space: "srgb_rec709_scene".into(), origin: TagOrigin::File });
}

/// Every 8-bit code survives the half storage exactly enough to round back to itself.
#[test]
fn half_holds_every_8bit_code() {
    for k in 0..=255u32 {
        let v = f32::from(Half::from(k as f32 / 255.0));
        assert_eq!((v * 255.0).round() as u32, k);
    }
}

/// Straight alpha stays straight: the colour is not multiplied, the flag says so.
#[test]
fn straight_alpha_is_kept() {
    let dir = tmp();
    let p = dir.path().join("a.png");
    straight_png(&p, 1, 1, 4, vec![128, 128, 128, 128]);
    let l = load(&p).unwrap();
    assert!(l.straight);
    assert!((px(&l.image, "R", 0) - 128.0 / 255.0).abs() < 2e-4, "{}", px(&l.image, "R", 0));
    assert!((px(&l.image, "A", 0) - 128.0 / 255.0).abs() < 2e-4);
}

/// The PQ HDR10 PNG WarpBro writes (cICP 9/16/0/1): tagged `pq_rec2020_display` by the file, its
/// 16-bit codes kept exactly.
#[test]
fn pq_png_keeps_codes_and_cicp_tag() {
    let dir = tmp();
    let p = dir.path().join("hdr.png");
    let codes = vec![0u16, 32768, 65535, 20000, 40000, 60000];
    let mut img = ImageData::from_u16(2, 1, 3, codes.clone());
    img.metadata.set_colorspace_attribute("pq_rec2020_display");
    vfx_io::write(&p, &img).unwrap();
    let l = load(&p).unwrap();
    assert_eq!(l.tag, ColorTag { space: "pq_rec2020_display".into(), origin: TagOrigin::File });
    for (i, (ch, k)) in ["R", "G", "B", "R", "G", "B"].iter().zip(&codes).enumerate() {
        let v = px(&l.image, ch, i / 3);
        assert_eq!((v * 65535.0).round() as u16, *k, "{ch}{i}");
    }
    let b = builtin(&l.tag).unwrap();
    assert_eq!((b.tf, b.nits), (Tf::Pq, Some(100.0)));
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
    // RGBE shares one exponent: a few percent, but no curve.
    assert!((px(&l.image, "R", 0) - 0.5).abs() < 0.02);
    assert!((px(&l.image, "B", 0) - 4.0).abs() < 0.1);
    assert_eq!(builtin(&l.tag).unwrap(), Builtin::LINEAR);
}

/// No reader tag: integer data falls back to `srgb_rec709_display`, float to `lin_rec709_scene`,
/// both marked untagged so the config's file rules come first.
#[test]
fn untagged_fallbacks() {
    let int = foreign_tag(&ImageData::from_u8(1, 1, 3, vec![0; 3]));
    assert_eq!(int, ColorTag { space: "srgb_rec709_display".into(), origin: TagOrigin::Untagged });
    let float = foreign_tag(&ImageData::from_f32(1, 1, 3, vec![0.0; 3]));
    assert_eq!(float, ColorTag { space: "lin_rec709_scene".into(), origin: TagOrigin::Untagged });
}

/// OIIO's informal names are stored by their interop ID; names without one stay as named.
#[test]
fn tags_are_stored_by_interop_id() {
    let mut img = ImageData::from_f32(1, 1, 3, vec![0.0; 3]);
    img.metadata.set_colorspace_attribute("sRGB");
    assert_eq!(foreign_tag(&img).space, "srgb_rec709_scene");
    img.metadata.set_colorspace_attribute("KodakLog");
    assert_eq!(foreign_tag(&img).space, "KodakLog");
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

fn tag(name: &str) -> ColorTag {
    ColorTag { space: name.into(), origin: TagOrigin::File }
}

/// The built-in decode spells each tag out as a curve, primaries and absoluteness.
#[test]
fn builtin_curves_of_the_interop_ids() {
    let b = |n: &str| builtin(&tag(n)).unwrap();
    assert_eq!(b("g24_rec709_display").tf, Tf::Gamma(2.4));
    assert_eq!(b("g22_rec709_display").tf, Tf::Gamma(2.2));
    assert_eq!(b("srgb_rec709_display"), Builtin { tf: Tf::Srgb, to709: IDENTITY, nits: None });
    assert_eq!(b("srgb_rec709_scene").tf, Tf::Srgb);
    assert_eq!(b("Rec709").tf, Tf::Rec709);
    assert_eq!(b("KodakLog").tf, Tf::Cineon);
    assert_eq!(b("scene_linear"), Builtin::LINEAR);
    let pq = b("pq_rec2020_display");
    assert_eq!((pq.tf, pq.nits), (Tf::Pq, Some(100.0)));
    assert_ne!(pq.to709, IDENTITY, "Rec.2020 primaries rotate into Rec.709");
    let hlg = b("hlg_rec2020_display");
    assert_eq!((hlg.tf, hlg.nits), (Tf::Hlg, Some(100.0)));
}

#[test]
fn wide_primaries_rotate_into_rec709() {
    let m = builtin(&tag("lin_p3d65_scene")).unwrap().to709;
    // Published linear Display-P3 -> Rec.709 (D65 both): the columns are the P3 primaries.
    let want = [[1.224_940_1, -0.224_940_4, 0.0], [-0.042_056_9, 1.042_057_1, 0.0], [-0.019_637_6, -0.078_636_1, 1.098_273_5]];
    for (r, w) in m.iter().zip(want) {
        for (a, b) in r.iter().zip(w) {
            assert!((a - b).abs() < 2e-4, "{m:?}");
        }
    }
}

#[test]
fn d60_primaries_keep_neutrals_neutral() {
    // ACEScg white is D60; without the adaptation to D65 it would come out tinted.
    for t in ["lin_ap1_scene", "lin_ap0_scene"] {
        let m = builtin(&tag(t)).unwrap().to709;
        for row in m {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 2e-3, "{t}: {m:?}");
        }
    }
}

/// A tag with no curve here stays a typed error: only an OCIO config can show it.
#[test]
fn tags_without_a_builtin_curve_are_typed_errors() {
    for t in ["CMYK", "lin_ciexyzd65_scene", "data", "no_such_space", "ACEScct"] {
        assert!(matches!(builtin(&tag(t)), Err(ImportError::ColorSpace(n)) if n == t), "{t}");
    }
}

fn header() -> Header {
    Header::new()
}

/// EXR rules in order: `colorInteropID` first, then `whiteLuminance`, then `chromaticities`
/// matched to a known gamut, else untagged (`scene_linear` after the file rules).
#[test]
fn exr_tag_rules() {
    let mut h = header();
    assert_eq!(exr_tag(&h).unwrap(), ColorTag { space: "scene_linear".into(), origin: TagOrigin::Untagged });

    h.add_chromaticities(exr_core::Chromaticities::default()).unwrap();
    assert_eq!(exr_tag(&h).unwrap(), ColorTag { space: "lin_rec709_scene".into(), origin: TagOrigin::Chromaticities });
    let ap1 = &vfx_primaries::ACES_AP1;
    *h.chromaticities_mut().unwrap() = exr_core::Chromaticities {
        red: [ap1.r.0, ap1.r.1],
        green: [ap1.g.0, ap1.g.1],
        blue: [ap1.b.0, ap1.b.1],
        white: [ap1.w.0, ap1.w.1],
    };
    assert_eq!(exr_tag(&h).unwrap().space, "lin_ap1_scene");
    // An unknown gamut is not guessed at: the file rules decide.
    h.chromaticities_mut().unwrap().red = [0.7, 0.29];
    assert_eq!(exr_tag(&h).unwrap().origin, TagOrigin::Untagged);

    let mut l = header();
    l.add_white_luminance(203.0).unwrap();
    let t = exr_tag(&l).unwrap();
    assert_eq!(t.space, XYZ_DISPLAY_ROLE);
    let TagOrigin::Luminance { to_xyz, nits } = t.origin else { panic!("{t:?}") };
    // White (1, 1, 1) of Rec.709 is D65 XYZ with Y = 1 (the reference white); its nits ride along.
    let y = to_xyz[1].iter().sum::<f32>();
    assert!((y - 1.0).abs() < 1e-4 && nits == 203.0, "{to_xyz:?}");
    let b = builtin(&t).unwrap();
    assert!(b.nits == Some(203.0) && b.tf == Tf::Linear);
    for (r, row) in b.to709.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            let want = if r == c { 1.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-4, "{:?}", b.to709);
        }
    }
    *l.white_luminance_mut().unwrap() = 0.0;
    assert!(matches!(exr_tag(&l), Err(ImportError::Attr(_))));

    h.add_color_interop_id(exr_core::attr::ExrString::from("lin_ap0_scene")).unwrap();
    assert_eq!(exr_tag(&h).unwrap(), ColorTag { space: "lin_ap0_scene".into(), origin: TagOrigin::File });
}
