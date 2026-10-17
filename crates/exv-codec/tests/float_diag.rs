//! Float-codec diagnostic: WHERE EXV LosslessFast loses bytes against OpenEXR ZIP on a
//! multichannel float frame (robat: 43 AOV channels). Not a unit test, a measurement run.
//!
//! `set EXV_DIAG_EXR=D:\_demo\EXR\Robat\robat.0001.exr`
//! `cargo test -p exv-codec --test float_diag -- --ignored --nocapture`
//!
//! # What we measure (three questions from the plan)
//! 1. **rANS table overhead** — each (tile×channel) = its own rANS stream with a fixed header
//!    ≈778 B (MAGIC4+tl1+lanes1+n4 + freq[256×u16]=512 + lane_len[32×u32]=128 + 32 state flushes=128).
//!    We compare tiled-256 against single-tile: the difference ≈ the cost of duplicating tables over the grid.
//! 2. **Predictor quality** — single-tile EXV (minimum overhead) against EXR-ZIP: a clean
//!    comparison of "reorder+delta+rANS" against "predictor+deflate" on the same data.
//! 3. **Per-channel** — exactly where (which AOVs) EXV loses/wins vs EXR-ZIP.

use exr_core::{ChannelData, Compression, Image as ExrImage, MemOStream, read};
use exv_codec::Profile;
use exv_codec::image::Image as ExvImage;
use exv_video::write_still;
use exv_format::{Channel, Dtype, Header};

/// Fixed overhead of one rANS blob (one tile×channel) at lanes=32 — see the doc above.
const RANS_FIXED_OVERHEAD: u64 = 4 + 1 + 1 + 4 + 512 + 32 * 4 + 32 * 4; // = 778
/// Frequency table only — the one thing a "shared per-channel table" would remove.
const RANS_TABLE_BYTES: u64 = 512;

/// (name, exv-dtype, planar LE bytes) of one channel from the loaded EXR.
fn chan_plane(cd: &ChannelData) -> (Dtype, Vec<u8>) {
    match cd {
        ChannelData::Half(v) => (Dtype::F16, v.iter().flat_map(|x| x.bits().to_le_bytes()).collect()),
        ChannelData::Float(v) => (Dtype::F32, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        ChannelData::Uint(v) => (Dtype::U32, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
    }
}

/// Build a EXV Header for the given tile + planes from an EXR image.
fn build(orig: &ExrImage, tile: u32) -> (ExvImage, u64) {
    let (w, h) = (orig.width() as u32, orig.height() as u32);
    let mut channels = Vec::new();
    let mut planes = Vec::new();
    let mut raw = 0u64;
    for (name, cd) in orig.channels() {
        let (dt, bytes) = chan_plane(cd);
        raw += bytes.len() as u64;
        channels.push(Channel { name: name.to_string(), dtype: dt });
        planes.push(bytes);
    }
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: tile.min(w),
        tile_h: tile.min(h),
        codec_id: 1,
        codec_params: Vec::new(),
        channels,
    };
    (ExvImage::new(header, planes).expect("valid image"), raw)
}

/// Size of one channel as a 1-frame .exv (for per-channel attribution). `tile==0` → single tile.
fn exv_one_channel(name: &str, dt: Dtype, plane: Vec<u8>, w: u32, h: u32, tile: u32, prof: &Profile) -> u64 {
    let (tw, th) = if tile == 0 { (w, h) } else { (tile.min(w), tile.min(h)) };
    let header = Header {
        width: w,
        height: h,
        frame_count: 1,
        tile_w: tw,
        tile_h: th,
        codec_id: 1,
        codec_params: Vec::new(),
        channels: vec![Channel { name: name.to_string(), dtype: dt }],
    };
    let img = ExvImage::new(header, vec![plane]).expect("valid 1-ch");
    write_still(&img, prof).expect("encode .exv").len() as u64
}

/// Size of one channel as a standalone EXR-ZIP (in-memory).
fn exr_one_channel(orig: &ExrImage, name: &str, cd: &ChannelData) -> u64 {
    let mut img = ExrImage::new(orig.data_window);
    img.insert_channel(name, cd.clone()).expect("insert");
    let mut out = MemOStream::new();
    img.write_to(&mut out, Compression::Zip).expect("exr zip write");
    out.data().len() as u64
}

#[test]
#[ignore = "diagnostic: needs a real multichannel float EXR via EXV_DIAG_EXR"]
fn float_codec_diag() {
    let path = std::env::var("EXV_DIAG_EXR")
        .unwrap_or_else(|_| r"D:\_demo\EXR\Robat\robat.0001.exr".to_string());
    let Ok(orig) = read(&path) else {
        eprintln!("skip: could not read {path} (set EXV_DIAG_EXR)");
        return;
    };
    let (w, h) = (orig.width() as u32, orig.height() as u32);
    let nch = orig.channels().count() as u64;
    let prof = Profile::lossless_fast();

    let exr_ondisk = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    let (img_t, raw) = build(&orig, 256);
    let (img_s, _) = build(&orig, w.max(h)); // single tile (tile >= max(w,h))
    let exv_tiled = write_still(&img_t, &prof).expect("encode .exv").len() as u64;
    let exv_single = write_still(&img_s, &prof).expect("encode .exv").len() as u64;

    // EXR-ZIP afresh via exr-core (clean baseline; on-disk may use a different codec).
    let mut exr_zip = MemOStream::new();
    orig.write_to(&mut exr_zip, Compression::Zip).expect("exr zip");
    let exr_zip = exr_zip.data().len() as u64;

    let tx = w.div_ceil(256) as u64;
    let ty = h.div_ceil(256) as u64;
    let n_tiles = tx * ty;
    let blobs = n_tiles * nch;

    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { (a as f64 / b as f64 - 1.0) * 100.0 };

    eprintln!("\n=== float-codec diag — {path} ===");
    eprintln!("{w}x{h}, {nch} channels, tile grid {tx}x{ty} = {n_tiles} tiles, {blobs} rANS blobs");
    eprintln!("raw planar           {:>10} B  {:>7.2} MB  1.00x", raw, mb(raw));
    eprintln!("EXR on-disk          {:>10} B  {:>7.2} MB  {:>5.2}x", exr_ondisk, mb(exr_ondisk), raw as f64 / exr_ondisk.max(1) as f64);
    eprintln!("EXR-ZIP (exr-core)   {:>10} B  {:>7.2} MB  {:>5.2}x   <-- baseline", exr_zip, mb(exr_zip), raw as f64 / exr_zip.max(1) as f64);
    eprintln!("EXV LF tiled-256     {:>10} B  {:>7.2} MB  {:>5.2}x   ({:+.1}% vs EXR-ZIP)", exv_tiled, mb(exv_tiled), raw as f64 / exv_tiled.max(1) as f64, pct(exv_tiled, exr_zip));
    eprintln!("EXV LF single-tile   {:>10} B  {:>7.2} MB  {:>5.2}x   ({:+.1}% vs EXR-ZIP)", exv_single, mb(exv_single), raw as f64 / exv_single.max(1) as f64, pct(exv_single, exr_zip));

    eprintln!("\n--- (1) rANS table overhead ---");
    let measured = exv_tiled.saturating_sub(exv_single);
    let theo_full = blobs.saturating_sub(nch) * RANS_FIXED_OVERHEAD; // extra tiles relative to 1/channel
    let theo_table = blobs.saturating_sub(nch) * RANS_TABLE_BYTES;
    eprintln!("tiled-256 − single-tile (measured)   {:>10} B  {:>6.2} MB", measured, mb(measured));
    eprintln!("theo. fixed overhead of extra blobs   {:>10} B  {:>6.2} MB  (778 B × {} extra blobs)", theo_full, mb(theo_full), blobs - nch);
    eprintln!("  of which freq tables only           {:>10} B  {:>6.2} MB  (512 B × extra)", theo_table, mb(theo_table));
    eprintln!("EXV-single deficit vs EXR-ZIP         {:>+10} B  {:>+6.2} MB", exv_single as i64 - exr_zip as i64, mb(exv_single.saturating_sub(exr_zip)) * if exv_single >= exr_zip { 1.0 } else { -1.0 });

    eprintln!("\n--- (3) per-channel (single-tile, minimum overhead) ---");
    eprintln!("{:<24} {:>4} {:>11} {:>11} {:>11} {:>8}", "channel", "type", "raw", "exv-1tile", "exr-zip", "Δ%");
    let mut sum_exv1 = 0u64;
    let mut sum_exr1 = 0u64;
    for (name, cd) in orig.channels() {
        let (dt, plane) = chan_plane(cd);
        let rawc = plane.len() as u64;
        let g1 = exv_one_channel(name, dt, plane, w, h, 0, &prof);
        let e1 = exr_one_channel(&orig, name, cd);
        sum_exv1 += g1;
        sum_exr1 += e1;
        eprintln!("{:<24} {:>4} {:>11} {:>11} {:>11} {:>+7.1}%", name, format!("{dt:?}"), rawc, g1, e1, pct(g1, e1));
    }
    eprintln!("{:<24} {:>4} {:>11} {:>11} {:>11} {:>+7.1}%", "Σ (1-ch files)", "", raw, sum_exv1, sum_exr1, pct(sum_exv1, sum_exr1));
    eprintln!("(note: per-channel EXR-ZIP carries ~per-file header overhead; look at the relative Δ%)");
}
