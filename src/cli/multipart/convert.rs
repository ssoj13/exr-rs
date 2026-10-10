//! `convert`: a single-part (typically multi-view) image into a multipart file with one part per
//! layer and view (`-convert`).
//!
//! The channels are regrouped by `SplitChannels`, which numbers the parts; the whole image is read
//! into memory, one plane per channel, and each output part is written from its channels' planes.
//! The C++ does the same (a `vector<char>` per channel): a part is written at a time, in order, so
//! the file layout matches the C++ byte for byte, and that needs all the pixels before the first
//! part is complete.
//!
//! Deliberate difference: an output that is the input file is refused (as `combine` refuses it);
//! the C++ creates the output first and so reads from a file it has just emptied.

use exr_core::attr::{Channel, ChannelList, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::io::{write_file, StdIfStream};
use exr_core::multipart::{get_channels_in_multi_part_file, split_channels, MultiPartInputFile, MultiPartOutputFile};

use super::names::{make_unique_names, set_str};
use crate::cli::tool_io::same_file;
use crate::cli::{CliError, CliResult};

/// `sizeof (half)` or `sizeof (float)`: the C++ gives every non-HALF channel, UINT included, 4 bytes.
fn sample_size(t: PixelType) -> usize {
    if t == PixelType::Half { 2 } else { 4 }
}

fn fail(msg: impl Into<String>) -> CliError {
    CliError::Other(msg.into())
}

/// Where each channel's plane sits in the one pixel buffer, and the slice that addresses it.
struct Planes {
    /// Bytes of every plane laid end to end.
    bytes: usize,
    /// Per channel: its plane's first byte, its sample size.
    at: Vec<(usize, usize)>,
}

impl Planes {
    /// Lay out one `width x height` plane for every type in `types`; `vectorSize`'s overflow check.
    fn new(types: impl Iterator<Item = PixelType>, pixels: i64) -> Result<Self, CliError> {
        let (mut bytes, mut at) = (0usize, Vec::new());
        for t in types {
            let size = sample_size(t);
            let plane = usize::try_from(pixels).ok().and_then(|p| p.checked_mul(size)).ok_or_else(|| fail("invalid vector size"))?;
            at.push((bytes, size));
            bytes = bytes.checked_add(plane).ok_or_else(|| fail("invalid vector size"))?;
        }
        Ok(Planes { bytes, at })
    }

    /// The slice of plane `i` for a data window `width` pixels wide whose first pixel is
    /// `first` pixels from the origin of the coordinates (`pixel_base`), at channel type `t`.
    fn slice(&self, i: usize, t: PixelType, width: i64, first: i64) -> Result<Slice, CliError> {
        let (start, size) = self.at[i];
        let bad = || fail("invalid vector size");
        let base = i64::try_from(start).ok().zip(first.checked_mul(i64::try_from(size).map_err(|_| bad())?)).and_then(|(s, f)| s.checked_sub(f)).ok_or_else(bad)?;
        let row = usize::try_from(width).ok().and_then(|w| w.checked_mul(size)).ok_or_else(bad)?;
        Ok(Slice::new(t, base, size, row, 1, 1, 0.0, false, false)?)
    }
}

/// Run `-convert`. `inputs` must hold exactly one file.
pub fn convert(inputs: &[String], out: &str) -> CliResult {
    let [in_name] = inputs else {
        return Err(fail("can only convert one file at once - use 'combine' mode for multiple files"));
    };
    // `get_channels_in_multi_part_file` takes the borrowed-stream reader only.
    let mut in_stream = StdIfStream::open(in_name)?;
    let mut infile = MultiPartInputFile::new(&mut in_stream)?;
    if infile.parts() != 1 {
        return Err(fail("can only convert single part EXRs to multipart EXR-2.0 files: use 'split' mode instead"));
    }
    let in_header = infile.header(0)?.clone();

    // The view that keeps its channel names: the first of the `multiView` attribute.
    let hero: Vec<u8> = if in_header.has_multi_view() {
        in_header.multi_view()?.first().map(|v| v.as_bytes().to_vec()).unwrap_or_default()
    } else {
        Vec::new()
    };

    // Channel names in view-friendly form, then their new parts (`SplitChannels`).
    let input_channels = get_channels_in_multi_part_file(&infile)?;
    let mut output_channels = input_channels.clone();
    let parts = usize::try_from(split_channels(&mut output_channels, true, &hero)).map_err(|_| fail("no output parts"))?;

    // Every output header is the input's without channels (and without `multiView`); a file
    // without a part name gets the empty one, exactly as the C++ does (it names the parts from an
    // empty file name), and the clashes are then numbered.
    let mut headers = Vec::with_capacity(parts);
    for _ in 0..parts {
        let mut h = in_header.clone();
        if h.has_multi_view() {
            h.erase("multiView")?;
        }
        if !h.has_name() {
            set_str(&mut h, "name", b"")?;
        }
        *h.channels_mut()? = ChannelList::new();
        headers.push(h);
    }
    make_unique_names(&mut headers).map_err(fail)?;

    let in_channels = in_header.channels()?;
    let dw = in_header.data_window()?;
    // Sizes in i64: the window's extent does not fit an int.
    let width = i64::from(dw.max.x) - i64::from(dw.min.x) + 1;
    let height = i64::from(dw.max.y) - i64::from(dw.min.y) + 1;
    let pixels = width.checked_mul(height).ok_or_else(|| fail("invalid vector size"))?;
    // Pixels between the start of a plane and the coordinates' origin.
    let first = i64::from(dw.min.y).checked_mul(width).and_then(|r| r.checked_add(i64::from(dw.min.x))).ok_or_else(|| fail("invalid vector size"))?;

    let types = input_channels
        .iter()
        .map(|c| in_channels.get(&c.internal_name).map(|ch| ch.type_))
        .collect::<exr_core::Result<Vec<_>>>()?;
    let planes = Planes::new(types.iter().copied(), pixels)?;

    // Each channel goes into its part's header and frame buffer; the same plane is also the
    // input frame buffer's destination.
    let mut in_fb = FrameBuffer::new();
    let mut out_fbs: Vec<FrameBuffer> = (0..parts).map(|_| FrameBuffer::new()).collect();
    for (i, (inc, outc)) in input_channels.iter().zip(&output_channels).enumerate() {
        let part = usize::try_from(outc.part_number).ok().filter(|p| *p < parts).ok_or_else(|| fail("channel without a part"))?;
        let chan: Channel = *in_channels.get(&inc.internal_name)?;
        headers[part].channels_mut()?.insert(outc.name.clone(), chan)?;
        if !outc.view.is_empty() {
            set_str(&mut headers[part], "view", outc.view.as_bytes())?;
        }
        let slice = planes.slice(i, chan.type_, width, first)?;
        out_fbs[part].insert(outc.name.clone(), slice.clone())?;
        in_fb.insert(inc.internal_name.clone(), slice)?;
    }

    // The pixels' memory before the output file exists, so an image too big to hold leaves no file.
    let mut pixels_buf = Vec::new();
    pixels_buf.try_reserve_exact(planes.bytes).map_err(|_| fail(format!("cannot allocate {} bytes for the pixels", planes.bytes)))?;
    pixels_buf.resize(planes.bytes, 0u8);

    if same_file(in_name, out) {
        return Err(fail("input and output file names cannot be the same"));
    }
    let rows = i32::try_from(height).map_err(|_| fail("the image is too tall"))?;
    write_file(out, |stream| {
        let mut outfile = MultiPartOutputFile::new(stream, &headers)?;

        let mut inpart = infile.input_part_wrapper(0)?;
        inpart.set_frame_buffer(&in_fb)?;
        inpart.read_pixels(dw.min.y, dw.max.y, &mut pixels_buf)?;

        for (i, fb) in out_fbs.iter().enumerate() {
            let mut part = outfile.output_part_wrapper(i32::try_from(i).map_err(|_| fail("too many parts"))?)?;
            part.set_frame_buffer(fb)?;
            part.write_pixels(&pixels_buf, rows)?;
        }
        Ok::<(), CliError>(())
    })?;
    Ok(())
}
