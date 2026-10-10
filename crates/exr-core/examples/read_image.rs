//! Read a single-part EXR via the `api` facade and print its dimensions and
//! channels.
//!
//! Run:
//! ```text
//! cargo run -p exr-core --example read_image -- path/to/image.exr
//! ```
//!
//! This is the common "load an image" case. Deep / multipart / subsampled
//! files are out of scope for the facade and produce a clear error pointing at
//! the matching low-level API (see `exr_core::scanline` / `deep` / `multipart`).

use std::process::ExitCode;

use exr_core::Image;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: read_image <file.exr>");
        return ExitCode::FAILURE;
    };

    match Image::read(&path) {
        Ok(img) => {
            println!("{path}");
            println!("  data window:    {:?}", img.data_window);
            println!("  display window: {:?}", img.display_window);
            println!("  size:           {} x {}", img.width(), img.height());
            println!("  compression:    {:?}", img.compression);
            println!("  channels:");
            for (name, data) in img.channels() {
                println!("    {name:<8} {:?} ({} samples)", data.pixel_type(), data.len());
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
