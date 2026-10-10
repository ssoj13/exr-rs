//! `makeMultiView.cpp`: combine single-view images into one multi-view image
//! (`openexr/src/bin/exrmultiview`, v3.4.15).
//!
//! Every input is read into one [`FlatImageLevel`] (the C++ `Image`) spanning the union of the data
//! windows, its channels renamed by `insert_view_name` (`R` -> `right.R`; the first view keeps
//! layer-less names), black where an input does not reach. The image is then written as one scanline
//! file whose header is the last input's, with the channels rebuilt, the codec set and the
//! `multiView` attribute added.
//!
//! Deliberate differences from the C++ (each a defect there):
//! * a last input that is a tiled file crashes `exrmultiview` (its header, `tiles` attribute
//!   included, goes to the scanline `OutputFile`: access violation). Here the output header drops
//!   the tile description, and a `RANDOM_Y` line order becomes `INCREASING_Y`, so a tiled image
//!   combines like any other;
//! * two channels that end up with one name are an error (the C++ replaces the first channel and
//!   leaves the file reader pointing at freed memory).

use exr_core::attr::{ChannelList, Compression, ExrString, LineOrder, Name};
use exr_core::buf::ensure_len;
use exr_core::header::Header;
use exr_core::io::IStream;
use exr_core::multi_view::insert_view_name;
use exr_core::scanline::{InputFile, OutputFile};
use exr_core::{ExrError, Result};
use exr_util::flat_image_level::FlatImageLevel;
use exr_util::image_level::ImageLevel;

use exr_util::flat_image_io::{pack, unpack};
use exr_util::image_io::strip_ranges;

use crate::cli::tool_io::{rows_of, strip_of, strip_of_some, unit_of};

/// A file reader that owns its stream (what `InputFile::open` returns).
type Reader<'a> = InputFile<'a, Box<dyn IStream + 'a>>;

/// Rows of a file in the units its chunks come in: a scanline block or a tile row.
fn chunk_rows(header: &Header) -> Result<i64> {
    Ok(if header.has_tile_description() {
        header.tile_description()?.y_size as i64
    } else {
        header.compression()?.num_scanlines() as i64
    })
}

/// The name the file calls the level channel `level_name`.
fn file_name(names: &[(Name, Name)], level_name: &Name) -> Name {
    names.iter().find(|(_, l)| l == level_name).map_or_else(|| level_name.clone(), |(f, _)| f.clone())
}

/// Read every channel of `file` into `image`, under the names `names` (file name, image name).
fn read_view<'a>(file: &mut Reader<'a>, image: &mut FlatImageLevel, names: &[(Name, Name)]) -> Result<()> {
    let dw = file.header().data_window()?;
    let unit = chunk_rows(file.header())?;
    let rows = rows_of(image, dw.max.x as i64 - dw.min.x as i64 + 1, unit);
    let mut buf = Vec::new();
    for (y0, y1) in strip_ranges(dw.min.y, dw.max.y, rows, unit, false) {
        let strip = strip_of_some(image, names.iter().map(|(_, level)| level), (dw.min.x, dw.max.x), (y0, y1))?;
        ensure_len(&mut buf, strip.bytes);
        file.set_frame_buffer(&strip.frame_buffer_with(|n| file_name(names, n))?)?;
        file.read_pixels(y0, y1, &mut buf)?;
        unpack(image, &strip, &buf)?;
    }
    Ok(())
}

/// Write the whole of `image` through `out` in strips, in the order the file's line order wants.
fn write_image(out: &mut OutputFile<'_>, image: &FlatImageLevel, header: &Header) -> Result<()> {
    let dw = header.data_window()?;
    let unit = unit_of(image, header.compression()?);
    let rows = rows_of(image, dw.max.x as i64 - dw.min.x as i64 + 1, unit);
    let decreasing = header.line_order()? == LineOrder::DecreasingY;
    let mut buf = Vec::new();
    for (y0, y1) in strip_ranges(dw.min.y, dw.max.y, rows, unit, decreasing) {
        let strip = strip_of(image, (dw.min.x, dw.max.x), (y0, y1))?;
        ensure_len(&mut buf, strip.bytes);
        pack(image, &strip, &mut buf)?;
        out.set_frame_buffer(&strip.frame_buffer()?)?;
        out.write_pixels(&buf, y1 - y0 + 1)?;
    }
    Ok(())
}

/// `makeMultiView`: combine `in_files` (one per entry of `views`) into `out_file`.
pub fn make_multi_view(views: &[String], in_files: &[String], out_file: &str, compression: Compression, verbose: bool) -> Result<()> {
    let say = |i: usize| {
        if verbose {
            outln!("reading file {} for {} view", in_files[i], views[i]);
        }
    };
    let view_names: Vec<ExrString> = views.iter().map(|v| ExrString::from(v.as_str())).collect();

    // Find the size of the data window and check the files. The files stay open for the read pass.
    let mut files = Vec::with_capacity(views.len());
    let mut header = None;
    let mut window = None;
    for (i, path) in in_files.iter().enumerate() {
        let file = InputFile::open(path)?;
        say(i);
        if file.header().has_multi_view() {
            return Err(ExrError::Base(format!(
                "The image in file {path} is already a multi-view image.  Cannot combine multiple multi-view images."
            )));
        }
        let h = file.header().clone();
        let dw = h.data_window()?;
        match window.as_mut() {
            None => window = Some(dw),
            Some(d) => d.extend_by_box(&dw),
        }
        header = Some(h);
        files.push(file);
    }
    let (Some(mut header), Some(window)) = (header, window) else {
        return Err(ExrError::Logic("no input views".into()));
    };
    let mut image = FlatImageLevel::new(0, 0, &window)?;
    *header.data_window_mut()? = window;
    // Blow away the channels; they are rebuilt.
    *header.channels_mut()? = ChannelList::new();

    // Read the input image files.
    for (i, file) in files.iter_mut().enumerate() {
        say(i);
        let channels: Vec<_> = file.header().channels()?.iter().map(|(n, c)| (n.clone(), *c)).collect();
        let mut names = Vec::with_capacity(channels.len());
        for (in_name, ch) in &channels {
            let out_name = Name::from(insert_view_name(in_name.as_bytes(), &view_names, i));
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::Arg(format!(
                    "no support for subsampling: channel {in_name} of {} has subsampling {}, {}",
                    in_files[i], ch.x_sampling, ch.y_sampling
                )));
            }
            // The level starts out black (zeroed), as the C++ `black ()`.
            image.insert_channel(&out_name, ch)?;
            header.channels_mut()?.insert(out_name.clone(), *ch)?;
            names.push((in_name.clone(), out_name));
        }
        read_view(file, &mut image, &names)?;
    }

    // The inputs are closed before the output file is created, as in the C++: the output may be one
    // of them (its pixels are all in `image`).
    drop(files);

    // Write the output image file.
    *header.compression_mut()? = compression;
    header.add_multi_view(view_names)?;
    if header.has_tile_description() {
        header.erase("tiles")?;
        if header.line_order()? == LineOrder::RandomY {
            *header.line_order_mut()? = LineOrder::IncreasingY;
        }
    }
    exr_core::io::write_file(out_file, |stream| -> exr_core::Result<_> {
        let mut out = OutputFile::new(stream, &header)?;
        if verbose {
            outln!("writing file {out_file}");
        }
        write_image(&mut out, &image, &header)
    })
}
