//! OOM-gate regression guard (`docs/DESIGN_LOG.md` D-03) — on the unified `.exv` still container.
//!
//! `read_still` (a still = a 1-frame `.exv`) allocates every full-size plane at once, so its peak
//! live heap is the image's whole-image residency (≈ raw bytes). B1's harness (`exv mem`) measured
//! ≈1.0–1.5× raw. This test pins a loose ceiling so a future change that *doubles* residency (e.g. an
//! accidental extra full-image copy on the decode path) fails CI. The out-of-core paths
//! (`FrameReader` per-tile decode, `write_still_windowed`) must stay `O(tile)`/`O(window)`.
//!
//! These tests exercise the REAL container through
//! `exv-video` (a DEV-dependency back-edge) + the frame-body `FrameReader` — the exact production
//! decode path, not a bespoke test layout.
//!
//! This integration-test crate installs the counting allocator as its own `#[global_allocator]`;
//! that is local to this test binary and never affects the library or its other consumers.

use exv_codec::Profile;
use exv_codec::frame::FrameReader;
use exv_codec::image::Image;
use exv_format::{Channel, Dtype, Header, Meta};
use exv_video::{read_still, still_frame_body, write_still, write_still_windowed};

#[global_allocator]
static GLOBAL: exv_format::bufpool::track::TrackingAlloc<std::alloc::System> =
    exv_format::bufpool::track::TrackingAlloc::new(std::alloc::System);

/// `exv_format::bufpool::track` counters are process-global, so two `measure()` regions running on different test
/// threads would pollute each other's peak (documented caveat). cargo runs tests in parallel; this
/// lock serializes the measured tests so each peak reflects only its own allocations.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Byte budget of the streaming encode window (the tile window is `budget / bytes-per-tile`).
const WINDOW_BUDGET: u64 = 64 << 20;

/// `exv_format::bufpool::track::measure` guarded against a vacuous pass: the counters read 0 when no tracker is the
/// global allocator, which would make every "peak stays small" assertion below trivially true.
fn measure<R>(f: impl FnOnce() -> R) -> (R, usize) {
    assert!(exv_format::bufpool::track::is_installed(), "TrackingAlloc is not the #[global_allocator]: the OOM gate would pass vacuously");
    exv_format::bufpool::track::measure(f)
}

/// Synthetic still mirroring the CLI's `mem` harness: R,G,B (f16) + Z (f32) + id (u32).
fn synth(w: u32, h: u32) -> Image {
    let n = w as usize * h as usize;
    let f16: Vec<u8> = (0..n).flat_map(|i| ((i as u16) & 0x3ff).to_le_bytes()).collect();
    let z: Vec<u8> = (0..n).flat_map(|i| (i as f32 * 0.1).to_le_bytes()).collect();
    let id: Vec<u8> = (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect();
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![
            Channel { name: "R".into(), dtype: Dtype::F16 },
            Channel { name: "G".into(), dtype: Dtype::F16 },
            Channel { name: "B".into(), dtype: Dtype::F16 },
            Channel { name: "Z".into(), dtype: Dtype::F32 },
            Channel { name: "id".into(), dtype: Dtype::U32 },
        ],
    };
    Image::new(header, vec![f16.clone(), f16.clone(), f16, z, id]).expect("well-formed synthetic")
}

/// Open a `.exv` still's frame body and wrap it in a [`FrameReader`] (the production tile-read seam).
/// Panics on a malformed buffer — the tests build their own good `.exv`.
fn open_reader(bytes: &[u8]) -> FrameReader<'_> {
    let (header, body) = still_frame_body(bytes).expect("still_frame_body: bad .exv");
    FrameReader::open(body, header).expect("FrameReader::open")
}

/// `read_still` peak heap must stay near the whole-image residency (≈ raw), not a multiple of it.
/// Uses the rANS fast path so the gate is independent of the optional `htj2k-backend` feature.
#[test]
fn read_still_peak_is_near_raw() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let img = synth(512, 512);
    let raw: usize = img.planes.iter().map(Vec::len).sum();
    let bytes = write_still(&img, &Profile::lossless_fast()).expect("encode .exv");

    let (decoded, peak) = measure(|| read_still(&bytes));
    assert!(decoded.is_ok(), "decode must succeed");

    // Loose ceiling: a healthy whole-image decode peaks at ~raw plus small transients. 2× raw flags
    // a gross regression (an extra full-image copy) without being flaky on per-tile scratch.
    let ceiling = raw * 2;
    assert!(
        peak <= ceiling,
        "read_still peak {peak} B exceeded {ceiling} B (raw {raw} B = {:.2}x)",
        peak as f64 / raw as f64
    );
}

/// The per-tile reader must decode tile-by-tile with peak heap `O(tile)`, NOT `O(image)`: proof that
/// the out-of-core read path does not materialize full planes. Decodes every tile/channel keeping
/// only one tile live at a time and asserts the peak stays far below the whole-image raw.
#[test]
fn streaming_reader_peak_is_per_tile() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let img = synth(1024, 1024); // 4x4 tiles of 256²; raw 14 MB, one tile-channel <= 256 KiB
    let raw: usize = img.planes.iter().map(Vec::len).sum();
    let bytes = write_still(&img, &Profile::lossless_fast()).expect("encode .exv");

    let reader = open_reader(&bytes);
    let nch = reader.header().channels.len();
    let (sum, peak) = measure(|| {
        let mut acc = 0u64;
        for t in 0..reader.tile_count() {
            for ci in 0..nch {
                let tile = reader.decode_tile_channel(t, ci).expect("decode tile/channel");
                acc = acc.wrapping_add(tile.iter().map(|&b| b as u64).sum::<u64>());
                // `tile` drops here -> peak stays at one tile (plus the codec's reused scratch).
            }
        }
        acc
    });
    assert!(sum > 0, "decoded data should be non-empty");
    // O(tile), not O(image): peak must be a small fraction of the full-image raw.
    assert!(
        peak < raw / 4,
        "streaming peak {peak} B is not << raw {raw} B ({:.2}x) — reader materializes too much",
        peak as f64 / raw as f64
    );
}

/// A deterministic per-pixel generator that never holds the whole image (the role an EXR-tiled adapter
/// plays in B2.3). U32 channels so it round-trips exactly under a lossless profile.
struct GenSource;
impl GenSource {
    fn pixel(x: u32, y: u32, ci: usize) -> u32 {
        x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503) ^ (ci as u32).wrapping_mul(0x9E37_79B9)
    }
}

fn gen_header(w: u32, h: u32) -> Header {
    Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: 256,
        tile_h: 256,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![Channel { name: "a".into(), dtype: Dtype::U32 }],
    }
}

/// The `produce(t0,t1)` window the streaming writer pulls: tiles `[t0,t1)` IN ORDER, `[i][ci]` =
/// `tw*th*esz` bytes generated from [`GenSource::pixel`] (never materializing the whole image).
fn gen_window(header: &Header, t0: u64, t1: u64) -> Vec<Vec<Vec<u8>>> {
    let tx_n = header.tiles_x() as u64;
    (t0..t1)
        .map(|t| {
            let (tx, ty) = ((t % tx_n) as u32, (t / tx_n) as u32);
            let (x0, y0, tw, th) = header.tile_rect(tx, ty);
            header
                .channels
                .iter()
                .enumerate()
                .map(|(ci, _)| {
                    let mut chan = vec![0u8; tw as usize * th as usize * 4];
                    for r in 0..th {
                        for c in 0..tw {
                            let v = GenSource::pixel(x0 + c, y0 + r, ci);
                            let base = ((r * tw + c) as usize) * 4;
                            chan[base..base + 4].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                    chan
                })
                .collect()
        })
        .collect()
}

/// The streaming ENCODER must be `O(window)`, NOT `O(image)`: 4x the pixels must not mean 4x the peak
/// heap. Both sizes are >= one full encode window (2048² = 64 tiles; the byte window is at most 4 tiles per
/// thread) so the comparison is not skewed by a sub-window image. The generator is never materialized (windowed `produce`).
#[test]
fn streaming_encode_peak_does_not_scale_with_image() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let enc_peak = |w: u32, h: u32| -> usize {
        let header = gen_header(w, h);
        let (written, peak) = measure(|| {
            let mut sink = std::io::sink();
            write_still_windowed(&header, &Profile::lossless_fast(), &Meta::default(), WINDOW_BUDGET, &mut sink, |t0, t1| {
                Ok(gen_window(&header, t0, t1))
            })
            .expect("encode")
        });
        assert!(written > 0, "encoder wrote nothing");
        peak
    };
    let small = enc_peak(2048, 2048); // 8x8 = 64 tiles (one full window)
    let big = enc_peak(4096, 4096); // 16x16 = 256 tiles (4 windows), 4x the pixels
    assert!(
        big < small * 2,
        "encode peak scaled with image (2048={small} B, 4096={big} B) — not O(window)"
    );
}

/// The serial-producer windowed encoder (used by the EXR adapter) must produce a correct `.exv`: the
/// producer yields raw tiles IN ORDER (as a single EXR stream would), they encode in parallel windows
/// and decode back bit-exact (incl. edge tiles) via the container `read_still`.
#[test]
fn streaming_windowed_roundtrips_from_serial_producer() {
    let (w, h) = (300u32, 200u32);
    let header = gen_header(w, h);
    let mut buf = Vec::new();
    write_still_windowed(&header, &Profile::lossless_fast(), &Meta::default(), WINDOW_BUDGET, &mut buf, |t0, t1| {
        Ok(gen_window(&header, t0, t1))
    })
    .expect("windowed encode");
    let img = read_still(&buf).expect("decode");
    let plane = &img.planes[0];
    for &(x, y) in &[(0u32, 0u32), (299, 199), (256, 128), (255, 199)] {
        let idx = (y as usize * w as usize + x as usize) * 4;
        let got = u32::from_le_bytes(plane[idx..idx + 4].try_into().expect("4 bytes"));
        assert_eq!(got, GenSource::pixel(x, y, 0), "pixel ({x},{y})");
    }
}

/// LOD reader gate (B2.4): on a lossless-HTJ2K `.exv`, `decode_tile_channel_level(.., 0)` equals
/// `decode_tile_channel` at the tile's full extent, and a non-HTJ2K profile returns `None` (no LOD).
/// Small image (one 256² tile holds it) so it stays a cheap functional check; still takes the serial
/// lock per convention since it shares the process-global tracking allocator with the measured tests.
#[cfg(feature = "htj2k-backend")]
#[test]
fn lod_tile_level0_equals_full_and_non_htj2k_is_none() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let img = synth(96, 40); // one tile (tile_w/h = 256); channels [R,G,B:f16, Z:f32, id:u32]

    // Lossless HTJ2K: every channel is an HTJ2K codestream -> LOD level 0 reproduces full-res bytes.
    let bytes = write_still(&img, &Profile::htj2k()).expect("encode htj2k .exv");
    let reader = open_reader(&bytes);
    let nch = reader.header().channels.len();
    let (_, _, tw, th) = reader.header().tile_rect(0, 0);
    for ci in 0..nch {
        let full = reader.decode_tile_channel(0, ci).expect("full tile/channel");
        let (lod, rw, rh) = reader.decode_tile_channel_level(0, ci, 0).expect("lod level0");
        assert_eq!((rw, rh), (tw, th), "level0 dims == tile rect (ch {ci})");
        assert_eq!(lod, full, "level0 bytes == decode_tile_channel (ch {ci})");
    }
    // A reduced level shrinks to ceil(dim/2): proves the scalable path actually engages.
    let (_, rw1, rh1) = reader.decode_tile_channel_level(0, 0, 1).expect("lod level1");
    assert_eq!((rw1, rh1), (tw.div_ceil(2), th.div_ceil(2)), "level1 dims = ceil(dim/2)");

    // A non-HTJ2K profile (rANS lossless) has no resolution scalability -> None, caller falls back.
    let bytes_fast = write_still(&img, &Profile::lossless_fast()).expect("encode fast .exv");
    let reader_fast = open_reader(&bytes_fast);
    assert!(
        reader_fast.decode_tile_channel_level(0, 0, 0).is_err(),
        "non-HTJ2K profile must return None from the LOD path"
    );
}

/// The streaming encoder must produce a correct `.exv` from an out-of-core generator: encode the
/// generator, decode with `read_still`, and verify pixels (incl. edge tiles) match.
#[test]
fn streaming_encode_roundtrips_from_generator() {
    let (w, h) = (300u32, 200u32); // non-multiple of the tile size -> edge tiles
    let header = gen_header(w, h);
    let mut buf = Vec::new();
    write_still_windowed(&header, &Profile::lossless_fast(), &Meta::default(), WINDOW_BUDGET, &mut buf, |t0, t1| {
        Ok(gen_window(&header, t0, t1))
    })
    .expect("encode");
    let img = read_still(&buf).expect("decode");
    let plane = &img.planes[0];
    for &(x, y) in &[(0u32, 0u32), (299, 199), (150, 100), (255, 199), (256, 0)] {
        let idx = (y as usize * w as usize + x as usize) * 4;
        let got = u32::from_le_bytes(plane[idx..idx + 4].try_into().expect("4 bytes"));
        assert_eq!(got, GenSource::pixel(x, y, 0), "pixel ({x},{y})");
    }
}

/// A written tile blob is self-describing: `FrameReader::decode_tile_channel` for every (tile, channel)
/// returns exactly the bytes `read_still` scatters into the full planes. Proves the per-channel length
/// table living INSIDE each blob is parsed correctly across all channels/tiles.
#[test]
fn tile_blob_self_describes_channels() {
    let mut img = synth(200, 130);
    img.header.tile_w = 64;
    img.header.tile_h = 64;
    let bytes = write_still(&img, &Profile::lossless_fast()).expect("encode .exv");

    let full = read_still(&bytes).expect("read_still");
    let reader = open_reader(&bytes);
    let h = reader.header().clone();
    let tx_n = h.tiles_x();
    for t in 0..reader.tile_count() {
        let (tx, ty) = (t % tx_n, t / tx_n);
        let (x, y, tw, th) = h.tile_rect(tx, ty);
        for ci in 0..h.channels.len() {
            let esz = h.channels[ci].dtype.size();
            let tile = reader.decode_tile_channel(t, ci).expect("decode tile/channel");
            assert_eq!(tile.len(), tw as usize * th as usize * esz, "tile {t} ch {ci} size");
            // Each tile row must equal the corresponding row of the full plane (round-trip identity).
            let row = tw as usize * esz;
            for r in 0..th as usize {
                let src = r * row;
                let dst = ((y as usize + r) * h.width as usize + x as usize) * esz;
                assert_eq!(
                    &tile[src..src + row],
                    &full.planes[ci][dst..dst + row],
                    "tile {t} ch {ci} row {r}"
                );
            }
        }
    }
}
