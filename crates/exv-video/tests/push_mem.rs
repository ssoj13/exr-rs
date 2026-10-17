//! Memory gate of the streaming writer: pushing frames one at a time must cost O(1 frame) of heap, NOT
//! O(clip). The sink discards bytes (a real file would not sit in RAM either), frames are generated inside
//! the loop (never materialized), so the only thing that could grow with the frame count is the writer.
//! Own integration binary: it installs the counting allocator as `#[global_allocator]`.

use std::io::{self, Seek, SeekFrom, Write};

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_format::{Channel, Dtype, Header};
use exv_video::{ClipWriter, WriteOpts};

#[global_allocator]
static GLOBAL: exv_format::bufpool::track::TrackingAlloc<std::alloc::System> = exv_format::bufpool::track::TrackingAlloc::new(std::alloc::System);

/// Write+Seek sink that keeps only the position (`Write::write` discards).
struct Null(u64);

impl Write for Null {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.0 += b.len() as u64;
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for Null {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        match p {
            SeekFrom::Start(o) => self.0 = o,
            SeekFrom::End(_) | SeekFrom::Current(_) => {}
        }
        Ok(self.0)
    }
}

const W: u32 = 512;
const H: u32 = 256;

/// One synthetic F32 + U16 frame (`shift` varies the content so P frames carry residuals).
fn frame(shift: u32) -> Image {
    let n = (W * H) as usize;
    let f: Vec<u8> = (0..n).flat_map(|i| ((i as u32 + shift * 3) as f32 * 0.01).to_le_bytes()).collect();
    let u: Vec<u8> = (0..n).flat_map(|i| (((i as u32 + shift) & 0xfff) as u16).to_le_bytes()).collect();
    let header = Header {
        width: W,
        height: H,
        frame_count: 1,
        tile_w: 128,
        tile_h: 128,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![Channel { name: "Z".into(), dtype: Dtype::F32 }, Channel { name: "U".into(), dtype: Dtype::U16 }],
    };
    Image::new(header, vec![f, u]).expect("img")
}

/// Peak extra heap of pushing `n` frames (gop 8, MV search on: P references are kept).
fn peak(n: u32) -> usize {
    assert!(exv_format::bufpool::track::is_installed(), "TrackingAlloc is not the global allocator: the gate would pass vacuously");
    let opts = WriteOpts { gop: 8, search: 2, ..WriteOpts::default() };
    exv_format::bufpool::track::measure(|| {
        let mut w = ClipWriter::new(Null(0), &frame(0).header, &Profile::lossless_fast(), &opts).expect("new");
        for s in 0..n {
            w.push(frame(s)).expect("push");
        }
        w.finish().expect("finish")
    })
    .1
}

#[test]
fn push_peak_does_not_scale_with_clip_length() {
    let frame_bytes = (W * H) as usize * 6;
    let (short, long) = (peak(8), peak(48));
    assert!(long < short * 3 / 2, "peak grew with the clip: 8 frames = {short} B, 48 frames = {long} B");
    assert!(long < 12 * frame_bytes, "peak {long} B is more than ~12 frames of {frame_bytes} B");
}
