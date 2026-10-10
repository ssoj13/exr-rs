//! `api::planar::read_planes`: one part of an in-memory EXR decoded into caller-owned planes, with
//! the data window placed inside a chosen frame (crop, margins, absent channels), for scanline
//! (direct, multi-threaded) and tiled (gather) parts, every codec family, HALF / FLOAT / UINT file
//! channels into FLOAT or HALF planes, at several thread counts. Expected samples are computed
//! from the written channel data by plain coordinate arithmetic (independent of the reader).

use exr_core::api::planar::{read_planes, read_planes_part, Frame, PlaneDst};
use exr_core::api::{ChannelData, Image};
use exr_core::attr::{Compression, LevelMode, LevelRoundingMode, PixelType, TileDescription};
use exr_core::io::{MemIStream, MemOStream};
use exr_core::MultiPartInputFile;
use imath_rs::{Box2i, Half, V2i};

fn bx(x0: i32, y0: i32, x1: i32, y1: i32) -> Box2i {
    Box2i { min: V2i { x: x0, y: y0 }, max: V2i { x: x1, y: y1 } }
}

/// Channels R (HALF), G (FLOAT), B (UINT) with values that differ per channel and pixel.
fn image(dw: Box2i, tiled: bool) -> Image {
    let n = ((dw.max.x - dw.min.x + 1) * (dw.max.y - dw.min.y + 1)) as usize;
    let r: Vec<Half> = (0..n).map(|i| Half::from((i % 1000) as f32 * 0.25 + 0.5)).collect();
    let g: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 100.0).collect();
    let b: Vec<u32> = (0..n).map(|i| (i * 7 % 60000) as u32).collect();
    let mut img = Image::new(dw)
        .with_channel("R", ChannelData::Half(r))
        .with_channel("G", ChannelData::Float(g))
        .with_channel("B", ChannelData::Uint(b));
    if tiled {
        img.set_tiling(Some(TileDescription {
            x_size: 8,
            y_size: 8,
            mode: LevelMode::OneLevel,
            rounding_mode: LevelRoundingMode::RoundDown,
        }));
    }
    img
}

fn sample(img: &Image, ch: &str, dw: Box2i, x: i32, y: i32) -> Option<f32> {
    if x < dw.min.x || x > dw.max.x || y < dw.min.y || y > dw.max.y {
        return None;
    }
    let i = ((y - dw.min.y) * (dw.max.x - dw.min.x + 1) + (x - dw.min.x)) as usize;
    img.channel(ch).and_then(|c| c.sample_f32(i))
}

#[test]
fn planes_equal_the_written_samples_with_crop_margins_and_fill() {
    // (data window, (frame origin x, y, w, h)): equal, smaller + offset, larger, sticking out
    // left/top, out right/bottom, display origin != 0, fully outside.
    let cases = [
        (bx(0, 0, 12, 9), (0, 0, 13, 10)),
        (bx(2, 1, 6, 4), (0, 0, 9, 7)),
        (bx(0, 0, 15, 11), (0, 0, 9, 7)),
        (bx(-3, -2, 5, 4), (0, 0, 9, 7)),
        (bx(5, 4, 14, 12), (0, 0, 9, 7)),
        (bx(3, 2, 10, 8), (3, 2, 9, 7)),
        (bx(30, 30, 35, 35), (0, 0, 9, 7)),
        (bx(-40, -40, -30, -30), (0, 0, 9, 7)),
    ];
    for tiled in [false, true] {
        for comp in [Compression::NoCompression, Compression::Rle, Compression::Zip, Compression::Piz] {
            for (dw, (ox, oy, w, h)) in cases {
                let img = image(dw, tiled);
                let mut out = MemOStream::new();
                img.write_to(&mut out, comp).expect("write");
                let bytes = out.into_data();
                for threads in [0, 1, 4] {
                    for plane_ty in [PixelType::Float, PixelType::Half] {
                        let sz: usize = if plane_ty == PixelType::Half { 2 } else { 4 };
                        // Planes with padded rows (stride > w * sz) to prove strides are honoured.
                        let ys = w as usize * sz + 6;
                        let mut bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![0x5Au8; ys * h as usize]).collect();
                        let names = ["B", "R", "G", "Z"]; // Z is not in the file
                        let mut dst: Vec<PlaneDst> = names
                            .iter()
                            .zip(bufs.iter_mut())
                            .map(|(n, b)| PlaneDst {
                                channel: n,
                                data: b,
                                x_stride: sz,
                                y_stride: ys,
                                type_: plane_ty,
                                fill: 1.0,
                            })
                            .collect();
                        let frame = Frame { origin: V2i { x: ox, y: oy }, width: w as usize, height: h as usize };
                        let hdr = read_planes(&bytes, 0, frame, &mut dst, Some(threads)).unwrap_or_else(|e| {
                            panic!("{comp:?} tiled={tiled} dw={dw:?} frame=({ox},{oy},{w},{h}): {e}")
                        });
                        assert_eq!(hdr.data_window().unwrap(), dw);
                        drop(dst);
                        for (pi, name) in names.iter().enumerate() {
                            for y in 0..h {
                                for x in 0..w {
                                    let at = y as usize * ys + x as usize * sz;
                                    let got = if plane_ty == PixelType::Half {
                                        f32::from(Half::from_bits(u16::from_ne_bytes([bufs[pi][at], bufs[pi][at + 1]])))
                                    } else {
                                        f32::from_ne_bytes(bufs[pi][at..at + 4].try_into().unwrap())
                                    };
                                    let want = if *name == "Z" {
                                        1.0
                                    } else {
                                        sample(&img, name, dw, ox + x, oy + y).map_or(0.0, |v| {
                                            if plane_ty == PixelType::Half {
                                                f32::from(Half::from(v))
                                            } else {
                                                v
                                            }
                                        })
                                    };
                                    assert_eq!(
                                        got, want,
                                        "{name} {comp:?} tiled={tiled} threads={threads} {plane_ty:?} dw={dw:?} frame=({ox},{oy}) at ({x},{y})"
                                    );
                                }
                                // The padding after each row is never touched.
                                for p in (w as usize * sz)..ys {
                                    assert_eq!(bufs[pi][y as usize * ys + p], 0x5A, "row padding written");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bad_requests_are_errors_not_panics() {
    let dw = bx(0, 0, 7, 7);
    let mut out = MemOStream::new();
    image(dw, false).write_to(&mut out, Compression::Zip).unwrap();
    let bytes = out.into_data();
    let frame = Frame { origin: V2i { x: 0, y: 0 }, width: 8, height: 8 };
    let mut small = vec![0u8; 16];
    let mut dst = [PlaneDst { channel: "R", data: &mut small, x_stride: 4, y_stride: 32, type_: PixelType::Float, fill: 0.0 }];
    assert!(read_planes(&bytes, 0, frame, &mut dst, None).is_err(), "plane too small");
    let mut ok = vec![0u8; 8 * 32];
    let mut dst = [PlaneDst { channel: "R", data: &mut ok, x_stride: 4, y_stride: 32, type_: PixelType::Uint, fill: 0.0 }];
    assert!(read_planes(&bytes, 0, frame, &mut dst, None).is_err(), "UINT plane type");
    let mut dst = [PlaneDst { channel: "R", data: &mut ok, x_stride: 4, y_stride: 32, type_: PixelType::Float, fill: 0.0 }];
    assert!(read_planes(&bytes, 3, frame, &mut dst, None).is_err(), "no such part");
    assert!(read_planes(&bytes[..bytes.len() / 2], 0, frame, &mut dst, None).is_err(), "truncated file");
    assert!(read_planes(&bytes, 0, Frame { width: 0, ..frame }, &mut dst, None).is_err(), "empty frame");
}

#[test]
fn two_planes_for_one_channel_are_an_error() {
    let dw = bx(0, 0, 7, 7);
    let mut out = MemOStream::new();
    image(dw, false).write_to(&mut out, Compression::Zip).unwrap();
    let bytes = out.into_data();
    let frame = Frame { origin: V2i { x: 0, y: 0 }, width: 8, height: 8 };
    let (mut a, mut b) = (vec![0xAAu8; 8 * 32], vec![0xBBu8; 8 * 32]);
    let mut dst = [
        PlaneDst { channel: "R", data: &mut a, x_stride: 4, y_stride: 32, type_: PixelType::Float, fill: 0.0 },
        PlaneDst { channel: "R", data: &mut b, x_stride: 4, y_stride: 32, type_: PixelType::Float, fill: 0.0 },
    ];
    let e = match read_planes(&bytes, 0, frame, &mut dst, Some(2)) {
        Err(e) => e,
        Ok(_) => panic!("two destinations for one channel must be refused"),
    };
    assert!(e.to_string().contains("more than one destination"), "{e}");
    assert!(a.iter().all(|&x| x == 0xAA) && b.iter().all(|&x| x == 0xBB), "nothing is written on error");
}

/// One open file, one header parse: the caller opens the file (to describe it), then decodes with
/// the SAME open file - twice, into different frames - and gets what `read_planes` gives from bytes.
#[test]
fn an_open_file_decodes_like_read_planes_without_reopening() {
    let dw = bx(0, 0, 11, 9);
    let img = image(dw, false);
    let mut out = MemOStream::new();
    img.write_to(&mut out, Compression::Zip).unwrap();
    let bytes = out.into_data();
    let mut is = MemIStream::new(&bytes[..]);
    let mut mp = MultiPartInputFile::new(&mut is).unwrap();
    // The description step: the header the caller reads once.
    assert_eq!(mp.header(0).unwrap().data_window().unwrap(), dw);
    for (w, h) in [(12usize, 10usize), (7, 5)] {
        let frame = Frame { origin: V2i { x: 0, y: 0 }, width: w, height: h };
        let (mut a, mut b) = (vec![0u8; w * h * 4], vec![0u8; w * h * 4]);
        let mut open = [
            PlaneDst { channel: "G", data: &mut a, x_stride: 4, y_stride: w * 4, type_: PixelType::Float, fill: 0.0 },
        ];
        read_planes_part(&mut mp, 0, frame, &mut open).expect("open-file decode");
        let mut from_bytes = [
            PlaneDst { channel: "G", data: &mut b, x_stride: 4, y_stride: w * 4, type_: PixelType::Float, fill: 0.0 },
        ];
        read_planes(&bytes, 0, frame, &mut from_bytes, None).expect("bytes decode");
        assert_eq!(a, b, "{w}x{h}");
        assert!(a.iter().any(|&x| x != 0), "pixels were decoded");
    }
}
