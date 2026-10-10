//! Codec-in-isolation comparison: the cleanest answer to "codec or
//! pipeline?" for ZIP/PIZ/DWAA, where the whole-file comparisons show the
//! biggest remaining gap, 2.8x-4.2x.
//!
//! Times ONLY `Compressor::compress`/`uncompress` directly — no stream I/O,
//! no `OutputFile`/`InputFile`, no frame-buffer gather/scatter at all — on
//! synthetic, already-gathered "chunk bytes" (the per-line, per-channel,
//! alphabetically-ordered Xdr layout every scanline codec actually
//! compresses/decompresses). Compares against `oracle.cpp`'s new
//! `bench-codec` mode, which does the SAME thing via C++'s public
//! `Imf::Compressor`/`newCompressor` API (`ImfCompressor.h`) — the same
//! codec-facing API shape our own `Compressor` trait mirrors.
//!
//! Run explicitly (ignored by default — real subprocess + several seconds
//! per codec):
//! ```text
//! cargo test -p exr-core --release --test perf_codec_isolation -- --ignored --nocapture
//! ```

mod support;

use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};

use exr_core::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
use exr_core::compressor::new_compressor;
use exr_core::header::Header;
use imath_rs::{Half, V2f};

const WIDTH: i32 = 16384;
/// Nominal header height (only feeds `sample()`'s cosmetic gradient and the
/// header's own metadata — never how many lines are actually compressed;
/// see `TOTAL_LINES`).
const HEIGHT: i32 = 8192;

/// How many lines' worth of chunks to actually compress/decompress per
/// codec per repeat — the original full-height
/// (`HEIGHT`=8192) run took over an hour with `REPEATS=9` and no visible
/// progress, an unacceptable "silent" long-running benchmark. `TOTAL_LINES`
/// is chosen once per codec as `n_chunks * lines_per_chunk` for the
/// smallest `n_chunks` that reaches at least this many lines (e.g. ZIP's
/// 16-line chunks need 64 of them to reach 1024; HTJ2K256's 256-line chunks
/// need only 4) — enough chunks for a stable ratio without the whole image.
const TOTAL_LINES: i32 = 1024;

fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

fn sample(c: usize, x: usize, y: usize, w: usize, h: usize) -> f32 {
    let gx = x as f32 / w as f32;
    let gy = y as f32 / h as f32;
    let grad = 0.15 + 0.7 * (gx * 0.6 + gy * 0.4) + 0.1 * (c as f32) * 0.25;
    let n = hash01(
        (x as u32).wrapping_mul(73_856_093)
            ^ (y as u32).wrapping_mul(19_349_663)
            ^ (c as u32).wrapping_mul(83_492_791),
    );
    (grad + 0.06 * (n - 0.5)).clamp(0.0, 8.0)
}

fn make_header(comp: Compression) -> Header {
    let mut h = Header::with_dimensions(
        WIDTH, HEIGHT, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, comp,
    )
    .unwrap();
    let mut cl = ChannelList::new();
    cl.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("G", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("B", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
    cl.insert("A", Channel::new(PixelType::Float, 1, 1, false)).unwrap();
    h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
    h
}

/// Build `lines` scanlines' worth of already-gathered, uncompressed chunk
/// bytes: per line, per channel in ALPHABETICAL order (A, B, G, R — how
/// `ChannelList` stores and every scanline codec expects them), `width`
/// samples in Xdr little-endian. Same synthetic pixel pattern as
/// `oracle.cpp`'s `fill_bench_pixels` (not required to be bit-identical
/// across languages for this throughput-only comparison, just similarly
/// non-degenerate so every codec does real compression work).
fn make_chunk_bytes(width: i32, first_y: i32, lines: i32) -> Vec<u8> {
    let (w, h) = (width as usize, HEIGHT as usize);
    let mut buf = Vec::with_capacity(lines as usize * w * 10);
    for y in first_y..first_y + lines {
        let yu = y as usize;
        // A: float
        for x in 0..w {
            buf.extend_from_slice(&sample(3, x, yu, w, h).to_le_bytes());
        }
        // B: half
        for x in 0..w {
            buf.extend_from_slice(&Half::from(sample(2, x, yu, w, h)).bits().to_le_bytes());
        }
        // G: half
        for x in 0..w {
            buf.extend_from_slice(&Half::from(sample(1, x, yu, w, h)).bits().to_le_bytes());
        }
        // R: half
        for x in 0..w {
            buf.extend_from_slice(&Half::from(sample(0, x, yu, w, h)).bits().to_le_bytes());
        }
    }
    buf
}

/// The smallest `n_chunks * lines_per_chunk >= TOTAL_LINES` for a codec —
/// e.g. ZIP (16-line chunks) needs 64 chunks to reach 1024 lines; HTJ2K256
/// (256-line chunks) needs only 4. Returns `(n_chunks, actual_lines)`.
fn bench_size(comp: Compression) -> (usize, i32) {
    let lines_per_chunk = comp.num_scanlines();
    let n_chunks = (((TOTAL_LINES + lines_per_chunk - 1) / lines_per_chunk).max(1)) as usize;
    (n_chunks, n_chunks as i32 * lines_per_chunk)
}

/// Time compress-all-chunks and (separately) uncompress-all-chunks, each
/// with a FRESH compressor per chunk (matching the parallel dispatch
/// pattern this crate uses; a single persistent compressor across all
/// chunks would additionally hide allocation cost, which is not what we
/// want to isolate here — codec throughput, not allocator behavior).
fn time_ours(comp: Compression) -> (f64, f64) {
    let header = make_header(comp);
    let lines_per_chunk = comp.num_scanlines();
    let max_bytes_per_line = 10 * WIDTH as usize; // 2+2+2+4 bytes/pixel
    let (n_chunks, _actual_lines) = bench_size(comp);

    let chunks: Vec<Vec<u8>> = (0..n_chunks)
        .map(|i| make_chunk_bytes(WIDTH, i as i32 * lines_per_chunk, lines_per_chunk))
        .collect();

    let t0 = Instant::now();
    let mut packed: Vec<Vec<u8>> = Vec::with_capacity(n_chunks);
    for (i, chunk) in chunks.iter().enumerate() {
        let min_y = i as i32 * lines_per_chunk;
        match new_compressor(comp, max_bytes_per_line, &header).unwrap() {
            Some(mut c) => packed.push(c.compress(chunk, min_y).unwrap().to_vec()),
            None => packed.push(chunk.clone()),
        }
    }
    let compress_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    for (i, p) in packed.iter().enumerate() {
        let min_y = i as i32 * lines_per_chunk;
        match new_compressor(comp, max_bytes_per_line, &header).unwrap() {
            Some(mut c) => {
                let out = c.uncompress(p, min_y, 0).unwrap();
                assert_eq!(out.len(), chunks[i].len());
            }
            None => assert_eq!(p.len(), chunks[i].len()),
        }
    }
    let uncompress_ms = t1.elapsed().as_secs_f64() * 1e3;

    (compress_ms, uncompress_ms)
}

/// `ABOVE_NORMAL_PRIORITY_CLASS` (`0x00008000`, `winbase.h`) — both sides
/// run with the same process priority: the C++
/// oracle subprocess is bumped to this via `CommandExt::creation_flags`
/// (std, no extra dependency); our own side runs in-process, bumped the
/// same way by whatever launched THIS test binary (see the module doc for
/// the `Start-Process -Priority AboveNormal` wrapper this is meant to run
/// under) — both sides get the same OS scheduling priority relative to
/// whatever background load this shared machine has right now.
#[cfg(windows)]
const ABOVE_NORMAL_PRIORITY_CLASS: u32 = 0x0000_8000;

fn time_cpp(oracle: &std::path::Path, comp_name: &str, actual_lines: i32) -> (f64, f64) {
    #[allow(unused_mut)]
    let mut cmd = Command::new(oracle);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(ABOVE_NORMAL_PRIORITY_CLASS);
    }
    let output = cmd
        .args(["bench-codec", comp_name, &WIDTH.to_string(), &actual_lines.to_string()])
        .output()
        .expect("failed to run oracle bench-codec");
    assert!(
        output.status.success(),
        "oracle bench-codec failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut compress_ms = None;
    let mut uncompress_ms = None;
    for tok in stdout.split_whitespace() {
        if let Some(v) = tok.strip_prefix("compress_ms=") {
            compress_ms = v.parse::<f64>().ok();
        }
        if let Some(v) = tok.strip_prefix("uncompress_ms=") {
            uncompress_ms = v.parse::<f64>().ok();
        }
    }
    (
        compress_ms.unwrap_or_else(|| panic!("no compress_ms in oracle output: {stdout}")),
        uncompress_ms.unwrap_or_else(|| panic!("no uncompress_ms in oracle output: {stdout}")),
    )
}

/// Best-effort current total CPU load (%), via PowerShell's
/// `Get-Counter`, logged once per repeat.
/// One PowerShell process spawn per call (tens of ms) is negligible next to
/// a single compress/uncompress repeat; `None` (never fatal — this is
/// diagnostic logging, not a correctness input) when PowerShell isn't
/// available or the counter can't be read (e.g. non-Windows CI).
fn cpu_load_pct() -> Option<f64> {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse::<f64>().ok()
}

/// Mandatory logging (a run without it had no visible progress for over an
/// hour and had to be killed blind) — appends a
/// timestamped line to BOTH stdout (`--nocapture` shows it live) and
/// `target/bench/codec_isolation.log` (so progress is visible even
/// without an attached terminal), one line per codec-start and per repeat.
struct BenchLog {
    file: std::fs::File,
    start: Instant,
}

impl BenchLog {
    fn open(path: &str) -> Self {
        let dir = std::path::Path::new(path).parent().unwrap();
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("failed to create bench log directory {dir:?}: {e}"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("failed to open bench log file");
        Self { file, start: Instant::now() }
    }

    fn line(&mut self, msg: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let line = format!("[t+{elapsed:>7.1}s] {msg}");
        println!("{line}");
        // Never silently discard a fallible call, even in test-only logging
        // code: a failed write to the mandatory progress log is exactly the
        // kind of thing that would otherwise leave a multi-minute run
        // "silent" again, which is the whole reason this log exists.
        if let Err(e) = writeln!(self.file, "{line}").and_then(|()| self.file.flush()) {
            eprintln!("BenchLog: failed to write progress log line: {e}");
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// `(max - min) / median` as a percentage — the stability check: a
/// ratio computed from a series whose spread exceeds 15% is flagged
/// unreliable rather than reported as a clean number.
fn spread_pct(v: &[f64]) -> f64 {
    let med = median(v.to_vec());
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = min_of(v);
    100.0 * (max - min) / med
}

/// Repeats per codec/side. The first attempt at this
/// harness used `REPEATS=9` at the FULL `HEIGHT` (8192 lines, 1.34 GB per
/// pass) and produced no visible progress for over an hour before being
/// killed blind. `TOTAL_LINES` (above) now bounds the WORK per pass; `5`
/// repeats plus the mandatory per-repeat logging below means a stall is
/// visible within minutes instead of being invisible for an hour.
const REPEATS: usize = 5;

/// Spread threshold: a ratio from a series whose
/// `(max-min)/median` exceeds this is marked unreliable rather than
/// reported as settled.
const UNRELIABLE_SPREAD_PCT: f64 = 15.0;

/// If the up-front time estimate (one real repeat of the first codec,
/// extrapolated to every codec/repeat) exceeds this, the log records the
/// overrun explicitly rather than silently running long.
const ESTIMATE_WARN: Duration = Duration::from_secs(15 * 60);

#[test]
#[ignore] // real subprocess + several minutes total; run explicitly
fn codec_only_ours_vs_cpp() {
    let Some(oracle) = support::oracle_or_skip("codec_only_ours_vs_cpp") else {
        return;
    };

    // `cargo test` runs with CWD set to the CRATE's manifest directory
    // (`crates/exr-core`), not the workspace root -- a relative path here
    // would land the log at `crates/exr-core/target/bench/...` instead of the
    // workspace-root `target/bench/...` where the other bench logs live.
    // Anchor it explicitly off
    // `CARGO_MANIFEST_DIR` instead.
    let log_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/bench/codec_isolation.log");
    let mut log = BenchLog::open(log_path.to_str().unwrap());
    log.line(&format!(
        "codec_only_ours_vs_cpp starting: WIDTH={WIDTH} TOTAL_LINES~={TOTAL_LINES} \
         REPEATS={REPEATS}, single-threaded, ours/cpp interleaved A/B per repeat"
    ));

    // HTJ2K256 -- root cause found and fixed in `oracle.cpp`'s `bench-codec`
    // mode: `Imf::Compressor::uncompress()` (the legacy wrapper this
    // isolation harness calls directly) returns `_decoder.bytes_decompressed`
    // as its result, but OpenEXRCore's internal HT decoder never sets that
    // field (confirmed: there is no `internal_ht.c` counterpart to
    // `internal_zip.c`/`internal_piz.c`/etc. that writes it, and a `memcmp`
    // against the known-correct chunk showed the decoded BYTES were already
    // fully correct even while `bytes_decompressed` stayed 0). Not a bug in
    // our port -- `parity-exr`'s HTJ2K256/32 goldens through the REAL
    // `InputFile`/`OutputFile` pipeline are byte-identical to C++, 108/108 --
    // just a narrow gap in this legacy per-chunk API's HT support that this
    // synthetic harness happens to exercise. `oracle.cpp` now trusts the
    // known expected size for HT when `outSize==0` rather than failing.
    // HTJ2K32 is NOT included below (only HTJ2K256): the codec set is
    // ZIP/PIZ/DWAA/HTJ2K with one representative HTJ2K variant,
    // and HTJ2K32 would add meaningfully more wall time for the same
    // qualitative answer.
    let all_codecs: &[(&str, Compression)] = &[
        ("zip", Compression::Zip),
        ("piz", Compression::Piz),
        ("dwaa", Compression::Dwaa),
        ("htj2k256", Compression::Htj2k256),
    ];
    // `BENCH_CODEC_FILTER=piz,dwaa` (comma-separated, case-insensitive) runs
    // only the named codec(s) instead of the whole list -- re-running
    // everything just to get one codec's numbers wastes machine time.
    let filter: Option<Vec<String>> = std::env::var("BENCH_CODEC_FILTER")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_lowercase()).collect());
    let codecs: Vec<(&str, Compression)> = match &filter {
        Some(names) => all_codecs.iter().copied().filter(|(n, _)| names.iter().any(|f| f == n)).collect(),
        None => all_codecs.to_vec(),
    };
    let codecs: &[(&str, Compression)] = &codecs;

    // Up-front time estimate: time ONE full
    // repeat (ours + cpp, compress + uncompress) of the FIRST codec, then
    // extrapolate to every codec's repeats. Codecs differ enough in cost
    // (ZIP is cheap, PIZ/DWAA/HTJ2K much less so) that this is a rough
    // floor, not a precise prediction -- logged as such.
    let (first_name, first_comp) = codecs[0];
    let (_, first_lines) = bench_size(first_comp);
    let est_t0 = Instant::now();
    let _ = time_ours(first_comp);
    let _ = time_cpp(&oracle, first_name, first_lines);
    let one_repeat_one_codec = est_t0.elapsed();
    let estimate = one_repeat_one_codec * (REPEATS as u32) * (codecs.len() as u32);
    log.line(&format!(
        "ESTIMATE: one repeat of {first_name} (ours+cpp, compress+uncompress) took \
         {:.1}s -> rough total estimate for {} codecs x {REPEATS} repeats = {:.1}s \
         ({:.1} min). This is a FLOOR (other codecs are typically slower than ZIP, \
         never faster), not an upper bound.{}",
        one_repeat_one_codec.as_secs_f64(),
        codecs.len(),
        estimate.as_secs_f64(),
        estimate.as_secs_f64() / 60.0,
        if estimate > ESTIMATE_WARN {
            " EXCEEDS the 15-minute budget -- proceeding anyway since TOTAL_LINES is \
             already cut down from the full image; report actual wall time at the end."
        } else {
            ""
        }
    ));

    println!(
        "\nW3.9 phase B: codec-only (no I/O, no scatter) ours vs C++, {WIDTH}-wide x \
         (per-codec chunk count reaching >= {TOTAL_LINES} lines), single-threaded, \
         median of {REPEATS} (min in parens); ours and C++ run STRICTLY interleaved \
         (A/B/A/B) per repeat so background load hits both sides equally. ratio = \
         ours/cpp median, <1.0 = we are faster. A series whose (max-min)/median \
         exceeds {UNRELIABLE_SPREAD_PCT:.0}% is flagged UNRELIABLE (!)."
    );
    println!(
        "{:>8}  {:>18} {:>18} {:>7}   {:>18} {:>18} {:>7}",
        "codec", "ours_c (min)", "cpp_c (min)", "ratio", "ours_u (min)", "cpp_u (min)", "ratio"
    );

    let mut summary_rows: Vec<String> = Vec::new();

    for &(name, comp) in codecs {
        let (n_chunks, actual_lines) = bench_size(comp);
        log.line(&format!(
            "codec={name} starting (n_chunks={n_chunks}, actual_lines={actual_lines})"
        ));

        let mut ours_cs = Vec::with_capacity(REPEATS);
        let mut ours_us = Vec::with_capacity(REPEATS);
        let mut cpp_cs = Vec::with_capacity(REPEATS);
        let mut cpp_us = Vec::with_capacity(REPEATS);

        for rep in 1..=REPEATS {
            // Strict A/B interleaving: ours THEN cpp, every repeat -- never
            // all-ours-then-all-cpp, so a load spike during this codec's
            // series affects both sides' samples at the same points in time.
            let (ours_c, ours_u) = time_ours(comp);
            ours_cs.push(ours_c);
            ours_us.push(ours_u);
            let (cpp_c, cpp_u) = time_cpp(&oracle, name, actual_lines);
            cpp_cs.push(cpp_c);
            cpp_us.push(cpp_u);

            let load = cpu_load_pct().map(|p| format!("{p:.1}%")).unwrap_or_else(|| "n/a".into());
            log.line(&format!(
                "codec={name} repeat={rep}/{REPEATS} ours_c={ours_c:.1}ms cpp_c={cpp_c:.1}ms \
                 ours_u={ours_u:.1}ms cpp_u={cpp_u:.1}ms cpu_load={load}"
            ));
        }

        let (ours_c_med, ours_c_min) = (median(ours_cs.clone()), min_of(&ours_cs));
        let (cpp_c_med, cpp_c_min) = (median(cpp_cs.clone()), min_of(&cpp_cs));
        let (ours_u_med, ours_u_min) = (median(ours_us.clone()), min_of(&ours_us));
        let (cpp_u_med, cpp_u_min) = (median(cpp_us.clone()), min_of(&cpp_us));

        let ratio_c = ours_c_med / cpp_c_med;
        let ratio_u = ours_u_med / cpp_u_med;
        let unreliable_c = spread_pct(&ours_cs) > UNRELIABLE_SPREAD_PCT
            || spread_pct(&cpp_cs) > UNRELIABLE_SPREAD_PCT;
        let unreliable_u = spread_pct(&ours_us) > UNRELIABLE_SPREAD_PCT
            || spread_pct(&cpp_us) > UNRELIABLE_SPREAD_PCT;

        let row = format!(
            "{:>8}  {:>7.1}({:>5.1}) {:>7.1}({:>5.1}) {:>5.2}x{}   \
             {:>7.1}({:>5.1}) {:>7.1}({:>5.1}) {:>5.2}x{}",
            name.to_uppercase(),
            ours_c_med, ours_c_min, cpp_c_med, cpp_c_min, ratio_c,
            if unreliable_c { "!" } else { " " },
            ours_u_med, ours_u_min, cpp_u_med, cpp_u_min, ratio_u,
            if unreliable_u { "!" } else { " " },
        );
        println!("{row}");
        summary_rows.push(row);
    }

    log.line("FINAL TABLE:");
    log.line(&format!(
        "{:>8}  {:>18} {:>18} {:>7}   {:>18} {:>18} {:>7}",
        "codec", "ours_c (min)", "cpp_c (min)", "ratio", "ours_u (min)", "cpp_u (min)", "ratio"
    ));
    for row in &summary_rows {
        log.line(row);
    }
}

/// Fast, always-run correctness check for this file's own harness plumbing
/// (`make_chunk_bytes`'s alphabetical-channel/Xdr layout, `new_compressor`
/// usage) — a SINGLE chunk per codec at full `WIDTH`, NOT the full
/// `HEIGHT`-tall benchmark, so this stays cheap enough to run in the normal
/// gate rather than needing `--ignored`. Verifies round-trip LENGTH only
/// (byte-exact lossless round-tripping for every codec is already covered
/// by `tests/parity_exr.rs`'s C++ cross-read gate elsewhere) — this test's
/// job is only to catch a broken harness (wrong buffer size, panic, wrong
/// channel order) before it produces silently-meaningless timing numbers
/// once the machine is quiet enough to run those.
#[test]
fn codec_isolation_harness_round_trips_are_correct() {
    for &comp in &[Compression::Zip, Compression::Piz, Compression::Dwaa, Compression::Htj2k256] {
        let header = make_header(comp);
        let lines = comp.num_scanlines();
        let max_bytes_per_line = 10 * WIDTH as usize;
        let chunk = make_chunk_bytes(WIDTH, 0, lines);

        let packed = match new_compressor(comp, max_bytes_per_line, &header).unwrap() {
            Some(mut c) => c.compress(&chunk, 0).unwrap().to_vec(),
            None => chunk.clone(),
        };
        let decoded_len = match new_compressor(comp, max_bytes_per_line, &header).unwrap() {
            Some(mut c) => c.uncompress(&packed, 0, 0).unwrap().len(),
            None => packed.len(),
        };
        assert_eq!(
            decoded_len,
            chunk.len(),
            "{comp:?}: round-trip length mismatch (harness bug, not a codec bug)"
        );
    }
}
