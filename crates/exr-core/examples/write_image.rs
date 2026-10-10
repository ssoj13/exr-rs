//! Build an `Image` in memory and write it as a single-part scanline EXR via
//! the `api` facade.
//!
//! Run:
//! ```text
//! cargo run -p exr-core --example write_image -- out.exr
//! ```
//!
//! Writes a 64x64 RGB gradient (HALF channels) compressed with ZIP. Read it
//! back with the `read_image` example or `exr-rs info out.exr`.

use std::process::ExitCode;

use exr_core::attr::Compression;
use exr_core::{ChannelData, Image};
use imath_rs::{Box2i, Half, V2i};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: write_image <out.exr>");
        return ExitCode::FAILURE;
    };

    let (w, h) = (64usize, 64usize);
    let data_window = Box2i {
        min: V2i { x: 0, y: 0 },
        max: V2i { x: w as i32 - 1, y: h as i32 - 1 },
    };

    // Three HALF channels, raster order over the data window: a simple gradient.
    let mut red = Vec::with_capacity(w * h);
    let mut green = Vec::with_capacity(w * h);
    let mut blue = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            red.push(Half::from(x as f32 / w as f32));
            green.push(Half::from(y as f32 / h as f32));
            blue.push(Half::from(0.5));
        }
    }

    let img = Image::new(data_window)
        .with_channel("R", ChannelData::Half(red))
        .with_channel("G", ChannelData::Half(green))
        .with_channel("B", ChannelData::Half(blue));

    match img.write(&path, Compression::Zip) {
        Ok(()) => {
            println!("wrote {path} ({w}x{h}, RGB HALF, ZIP)");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
