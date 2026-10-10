//! Out-of-core tile source: the
//! file stays on disk; a tile row of a level is produced by decoding the
//! level-0 rows it covers in bands, shading only the channels the view needs
//! to RGBA ([`Shade`]: the active layer's [`Mapping`], or a Cryptomatte
//! object's coverage), and streaming them through a [`BandAccumulator`]. The
//! level-0 image is never resident.
//!
//! Scanline and tiled single-part flat files both go through exr-core's
//! `InputFile` (which reads tiled files as C++ `InputFile` does), so level 0 has
//! one decode path. Subsampled files are not file-backed (their channels do not
//! share one sample grid); the viewer decodes those whole.
//!
//! A file's own levels win when their geometry matches: a tiled MIPMAP (or
//! RIPMAP, its `(k, k)` level) file already stores the coarse images, so level
//! `k > 0` is read from the file, through `TiledInputFile`, instead of being
//! reduced from level 0 - when, and only when, the file's level `k` has exactly
//! the viewer's `ceil(w / 2^k) x ceil(h / 2^k)` size (ROUND_UP files always do;
//! ROUND_DOWN ones only where the division is exact). The pixels then are the
//! writer's own filter, not [`super::reduce`]'s area average, which is the point.
//! The Cryptomatte overlay never uses them: averaged ids are meaningless, its
//! coverage is reduced after it is computed.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use exr_core::api::ChannelData;
use exr_core::attr::{LevelMode, PixelType};
use exr_core::frame_buffer::{FrameBuffer, Slice};
use exr_core::io::StdIfStream;
use exr_core::misc::pixel_type_size;
use exr_core::scanline::InputFile;
use exr_core::tiled::TiledInputFile;
use imath_rs::Half;

use super::grid::{Grid, TileId};
use super::reduce::{BandAccumulator, TileData};
use super::region::{
    RegionAccumulator, RegionBudget, RegionData, RegionPlan, RegionPolicy, pixel_count,
};
use super::source::{Production, ProductionContext, TileSource, Unit};
use crate::image::Mapping;

/// Level-0 RGBA bytes one decode band may hold (the rows per band follow).
const BAND_BYTES: usize = 64 << 20;

/// Levels whose long side is at most this are the OVERVIEW: all of them are
/// built by one pass over the file ([`Unit::Overview`]) instead of one decode
/// of the same rows per level. 2048 keeps the pass's sums (f64 RGBA per level
/// pixel, all overview levels) near 170 MiB. Levels the file stores itself
/// ([`FileSource`]'s `own`) are not part of the pass.
const OVERVIEW_SIDE: u32 = 2048;

/// Decoded channels of a band: `(name, samples)`, in the band's order.
type Band = Vec<(String, ChannelData)>;

/// How a decoded band becomes RGBA.
enum Shade {
    /// The active layer, through its mapping (over the band's channels).
    Layer(Mapping),
    /// Cryptomatte coverage of `ids` over `ranked` (by channel name), as opaque
    /// grey — the viewer's overlay, the same `extract_coverage_with` the
    /// resident frame uses.
    Coverage {
        ranked: Vec<cryptomatte::RankedLayer>,
        ids: Vec<f32>,
    },
}

impl Shade {
    /// Fill `out` with `width * rows` RGBA pixels from `band`.
    fn fill(&self, out: &mut Vec<f32>, band: &Band, width: u32, rows: u32) -> Result<(), String> {
        match self {
            Shade::Layer(m) => m.fill(out, band, width, rows).map_err(|e| e.to_string()),
            Shade::Coverage { ranked, ids } => {
                let matte = cryptomatte::extract_coverage_with(width, rows, ranked, ids, |name| {
                    band.iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, d)| d)
                        .ok_or_else(|| {
                            cryptomatte::Error::Metadata(format!(
                                "missing Cryptomatte channel `{name}`"
                            ))
                        })
                })
                .map_err(|e| e.to_string())?;
                out.clear();
                out.extend(matte.coverage.iter().flat_map(|&c| [c, c, c, 1.0]));
                Ok(())
            }
        }
    }
}

/// One channel the shading reads.
#[derive(Clone, Debug)]
struct Used {
    name: String,
    pixel_type: PixelType,
}

/// A file-backed source for one layer of one flat single-part EXR.
pub struct FileSource {
    path: PathBuf,
    w: u32,
    h: u32,
    /// Data-window origin (file coordinates of level-0 pixel `(0, 0)`).
    min: (i32, i32),
    /// The channels the shading reads, in the band's channel order.
    used: Vec<Used>,
    /// How a band becomes RGBA.
    shade: Shade,
    /// Names of the band's channels, for readout labels.
    labels: Vec<(String, usize)>,
    /// Rows of the used channels for the pixel readout.
    rows: RowCache,
    /// Level-0 RGBA bytes per decode band ([`BAND_BYTES`]; tests shrink it to
    /// exercise several bands per tile row).
    band_bytes: usize,
    /// Long side at or below which a level belongs to the overview
    /// ([`OVERVIEW_SIDE`]; tests shrink it so small images have row levels too).
    overview_side: u32,
    /// `own[k]`: level `k` is read from the file's own MIPMAP/RIPMAP level
    /// instead of reduced from level 0 (`own[0]` is always false).
    own: Vec<bool>,
}

trait RowSink {
    type Sample;
    fn rows(&self) -> std::ops::Range<u32>;
    fn push_row(&mut self, y: u32, row: &[f32]) -> Result<(), String>;
    fn finish(self) -> Result<Vec<Self::Sample>, String>;
}
impl RowSink for BandAccumulator {
    type Sample = (TileId, TileData);
    fn rows(&self) -> std::ops::Range<u32> {
        self.rows()
    }
    fn push_row(&mut self, y: u32, row: &[f32]) -> Result<(), String> {
        self.push_row(y, row)
    }
    fn finish(self) -> Result<Vec<Self::Sample>, String> {
        self.finish()
    }
}
impl RowSink for RegionAccumulator {
    type Sample = f32;
    fn rows(&self) -> std::ops::Range<u32> {
        self.rows()
    }
    fn push_row(&mut self, y: u32, row: &[f32]) -> Result<(), String> {
        self.push_row(y, row)
    }
    fn finish(self) -> Result<Vec<Self::Sample>, String> {
        self.finish()
    }
}

impl FileSource {
    /// Original f32 regions share band decode, Mapping/Shade and reduction with
    /// the display path. Native levels are used only when explicitly requested.
    pub fn region(
        &self,
        plan: RegionPlan,
        context: &ProductionContext,
        budget: &RegionBudget,
    ) -> Result<Arc<RegionData>, super::region::RegionError> {
        if plan.policy() == RegionPolicy::MatchingStoredNative
            && plan.level() > 0
            && !self.is_own(plan.level())
        {
            return Err(super::region::RegionError::Unsupported);
        }
        let result = self.region_pixels(plan, context, budget);
        super::region::RegionError::from_result(result, context.is_cancelled())
    }

    fn region_pixels(
        &self,
        plan: RegionPlan,
        context: &ProductionContext,
        budget: &RegionBudget,
    ) -> Result<Arc<RegionData>, super::region::RegionError> {
        if plan.size() != (self.w, self.h) {
            return Err("region source geometry does not match file".into());
        }
        if context.is_cancelled() {
            return Err("region cancelled".into());
        }
        let own = plan.policy() == RegionPolicy::MatchingStoredNative
            && plan.phase() == super::region::SamplingPhase::TOP_LEFT
            && self.is_own(plan.level());
        let sampling = if own { plan.native()? } else { plan };
        let (w, h) = sampling.size();
        let mut stream = self.open()?;
        if own {
            let mut file = TiledInputFile::new(&mut stream).map_err(|e| e.to_string())?;
            let align = file.tile_y_size().max(1).min(h);
            let (mut permit, mut workspace, per_band) =
                self.region_work(plan, sampling, w, align, budget)?;
            let acc = RegionAccumulator::new(sampling)?;
            permit.resize(
                plan.output_bytes()
                    .checked_add(acc.bytes())
                    .and_then(|n| n.checked_add(self.band_scratch(w, per_band)?))
                    .ok_or("region size overflow")?,
            )?;
            let rows = acc.rows();
            let (raw, rgba) = workspace.buffers();
            let out = self.pump_rows(
                context,
                w,
                align,
                rows,
                vec![acc],
                rgba,
                per_band,
                |r0, r1| {
                    read_own_rows_into(
                        &mut file,
                        &self.used,
                        plan.level(),
                        (w, h),
                        self.min,
                        r0,
                        r1,
                        raw,
                    )
                },
            )?;
            if context.is_cancelled() {
                return Err("region cancelled".into());
            }
            RegionData::finish(plan, out, permit)
        } else {
            let mut file = InputFile::new(&mut stream).map_err(|e| e.to_string())?;
            // InputFile's tiled path decodes whole tile rows even for one requested row.
            let align = file
                .header()
                .tile_description()
                .ok()
                .map_or(1, |td| td.y_size.max(1).min(h));
            let (mut permit, mut workspace, per_band) =
                self.region_work(plan, sampling, w, align, budget)?;
            let acc = RegionAccumulator::new(sampling)?;
            permit.resize(
                plan.output_bytes()
                    .checked_add(acc.bytes())
                    .and_then(|n| n.checked_add(self.band_scratch(w, per_band)?))
                    .ok_or("region size overflow")?,
            )?;
            let rows = acc.rows();
            let (raw, rgba) = workspace.buffers();
            let out = self.pump_rows(
                context,
                w,
                align,
                rows,
                vec![acc],
                rgba,
                per_band,
                |r0, r1| read_rows_into(&mut file, &self.used, w, self.min, r0, r1, raw),
            )?;
            if context.is_cancelled() {
                return Err("region cancelled".into());
            }
            RegionData::finish(plan, out, permit)
        }
    }
    fn plane_row_bytes(&self, w: u32) -> Result<usize, String> {
        self.used.iter().try_fold(0usize, |n, u| {
            let size = pixel_type_size(u.pixel_type).map_err(|e| e.to_string())?;
            n.checked_add((w as usize).checked_mul(size).ok_or("plane row overflow")?)
                .ok_or("plane row overflow".into())
        })
    }
    // Typed band copy + Cryptomatte coverage and its two f32 conversion scratch planes.
    fn band_scratch(&self, w: u32, rows: u32) -> Option<usize> {
        let typed = self.plane_row_bytes(w).ok()?.checked_mul(rows as usize)?;
        let crypto = if matches!(self.shade, Shade::Coverage { .. }) {
            (w as usize)
                .checked_mul(rows as usize)?
                .checked_mul(12)?
                .checked_add(64)?
        } else {
            0
        };
        typed.checked_add(crypto)
    }
    fn region_work(
        &self,
        output: RegionPlan,
        sampling: RegionPlan,
        w: u32,
        align: u32,
        budget: &RegionBudget,
    ) -> Result<(super::region::Permit, super::region::Workspace, u32), super::region::RegionError> {
        let row = self.plane_row_bytes(w)?;
        let rgba_row = (w as usize).checked_mul(16).ok_or("RGBA row overflow")?;
        // Use the minimal aligned band: admission is exact and never silently
        // expands a one-row request to an uncharged native tile row.
        let per_band = align;
        let working_rows = align.min(sampling.size().1);
        let scratch = self
            .band_scratch(w, working_rows)
            .ok_or("band scratch overflow")?;
        let bytes = output
            .output_bytes()
            .checked_add(sampling.sums_bytes())
            .and_then(|n| n.checked_add(scratch))
            .ok_or("region working overflow")?;
        let raw = row
            .checked_mul(working_rows as usize)
            .ok_or("raw band overflow")?;
        let samples = rgba_row
            .checked_mul(working_rows as usize)
            .ok_or("RGBA band overflow")?
            / 4;
        let minimum = bytes.checked_add(raw).and_then(|n| n.checked_add(samples.checked_mul(4)?))
            .ok_or("combined region working size overflows")?;
        if minimum > budget.limit() {
            return Err(super::region::RegionError::Budget { needed: minimum, limit: budget.limit() });
        }
        let permit = budget.reserve(bytes)?;
        let workspace = budget.workspace(raw, samples)?;
        Ok((permit, workspace, per_band))
    }

    /// A source for the layer `mapping` describes over the file's channel
    /// `names`/`types` (header order); `data_window` is `(min, width, height)`.
    pub fn new(
        path: PathBuf,
        data_window: ((i32, i32), u32, u32),
        names: &[&str],
        types: &[PixelType],
        mapping: Mapping,
    ) -> Result<Self, String> {
        let used_idx = mapping.channels();
        let used = used_channels(&used_idx, names, types)?;
        let band = mapping
            .remap(|i| used_idx.iter().position(|&u| u == i))
            .ok_or("mapping reads a channel outside its own list")?;
        let band_names: Vec<&str> = used.iter().map(|u| u.name.as_str()).collect();
        let labels = band.rows(&band_names);
        Ok(Self::with_shade(
            path,
            data_window,
            used,
            Shade::Layer(band),
            labels,
        ))
    }

    /// A source for the Cryptomatte coverage overlay: the coverage of `id`
    /// over the `ranked` layers (whose channels must be among `names`).
    pub fn coverage(
        path: PathBuf,
        data_window: ((i32, i32), u32, u32),
        names: &[&str],
        types: &[PixelType],
        ranked: Vec<cryptomatte::RankedLayer>,
        id: f32,
    ) -> Result<Self, String> {
        let mut used_idx = Vec::new();
        for layer in &ranked {
            for (id_ch, cov_ch) in layer.pairs() {
                for ch in [id_ch, cov_ch] {
                    let i = names
                        .iter()
                        .position(|n| *n == ch)
                        .ok_or_else(|| format!("Cryptomatte channel `{ch}` is not in the file"))?;
                    used_idx.push(i);
                }
            }
        }
        used_idx.sort_unstable();
        used_idx.dedup();
        let used = used_channels(&used_idx, names, types)?;
        let labels = vec![("coverage".to_owned(), 0)];
        Ok(Self::with_shade(
            path,
            data_window,
            used,
            Shade::Coverage {
                ranked,
                ids: vec![id],
            },
            labels,
        ))
    }

    fn with_shade(
        path: PathBuf,
        data_window: ((i32, i32), u32, u32),
        used: Vec<Used>,
        shade: Shade,
        labels: Vec<(String, usize)>,
    ) -> Self {
        let ((x0, y0), w, h) = data_window;
        let rows = RowCache::new(path.clone(), data_window, used.clone());
        // Averaged Cryptomatte ids are meaningless: only a plain layer may use the
        // file's own levels.
        let own = match shade {
            Shade::Layer(_) => own_levels(&path, &Grid::new(w, h)),
            Shade::Coverage { .. } => Vec::new(),
        };
        Self {
            path,
            w,
            h,
            min: (x0, y0),
            used,
            shade,
            labels,
            rows,
            band_bytes: BAND_BYTES,
            overview_side: OVERVIEW_SIDE,
            own,
        }
    }

    /// Whether level `k` comes from the file's own levels.
    fn is_own(&self, k: u32) -> bool {
        self.own.get(k as usize).copied().unwrap_or(false)
    }

    /// The finest overview level of `grid` (at least the top level).
    fn overview_from(&self, grid: &Grid) -> u32 {
        (0..grid.levels())
            .find(|&k| {
                let (w, h) = grid.level_size(k);
                w.max(h) <= self.overview_side
            })
            .unwrap_or(grid.levels() - 1)
    }

    /// Stream level-0 rows `rows` through `accs` in bands; each accumulator
    /// takes the rows of its own range. Returns every accumulator's tiles.
    fn stream(
        &self,
        context: &ProductionContext,
        rows: std::ops::Range<u32>,
        accs: Vec<BandAccumulator>,
    ) -> Result<Vec<(TileId, TileData)>, String> {
        let mut stream = self.open()?;
        let mut file = InputFile::new(&mut stream).map_err(|e| e.to_string())?;
        self.pump(context, self.w, 1, rows, accs, |r0, r1| {
            self.decode_rows(&mut file, r0, r1)
        })
    }

    /// Tile row `ty` of level `k`, read from the file's own level `k`. The row is
    /// the level-0 row of a `level_size(k)` image (same 512 px tiles), so the ONE
    /// [`BandAccumulator`] builds it - at its level 0, i.e. the plain `f16`
    /// conversion - and the ids are moved up to level `k`.
    fn own_row(
        &self,
        context: &ProductionContext,
        grid: &Grid,
        k: u32,
        ty: u32,
    ) -> Result<Vec<(TileId, TileData)>, String> {
        let (lw, lh) = grid.level_size(k);
        let acc = BandAccumulator::new(&Grid::new(lw, lh), 0, ty)
            .ok_or_else(|| format!("tile row L{k} {ty} is outside the {lw}x{lh} level"))?;
        let mut stream = self.open()?;
        let mut file = TiledInputFile::new(&mut stream).map_err(|e| e.to_string())?;
        // Decode whole file tile rows: a band that split one would decode it twice.
        let align = file.tile_y_size().max(1);
        let rows = acc.rows();
        let tiles = self.pump(context, lw, align, rows, vec![acc], |r0, r1| {
            read_own_rows(&mut file, &self.used, k, (lw, lh), self.min, r0, r1)
        })?;
        Ok(tiles
            .into_iter()
            .map(|(id, t)| (TileId { level: k, ..id }, t))
            .collect())
    }

    /// Decode rows `rows` of an image `w` wide in bands (`decode(r0, r1)` yields
    /// one band; bands are a multiple of `align` rows), shade them and feed every
    /// accumulator the rows of its own range.
    #[allow(clippy::too_many_arguments)]
    fn pump(
        &self,
        context: &ProductionContext,
        w: u32,
        align: u32,
        rows: std::ops::Range<u32>,
        accs: Vec<BandAccumulator>,
        decode: impl FnMut(u32, u32) -> Result<Band, String>,
    ) -> Result<Vec<(TileId, TileData)>, String> {
        pixel_count(w, 1)?;
        let row = (w as usize).checked_mul(16).ok_or("band row overflow")?;
        let per_band =
            u32::try_from((self.band_bytes / row).max(1)).map_err(|_| "band rows overflow")?;
        let per_band = (per_band / align.max(1))
            .max(1)
            .checked_mul(align.max(1))
            .ok_or("band alignment overflow")?;
        self.pump_rows(
            context,
            w,
            align,
            rows,
            accs,
            &mut Vec::new(),
            per_band,
            decode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn pump_rows<S: RowSink>(
        &self,
        context: &ProductionContext,
        w: u32,
        align: u32,
        rows: std::ops::Range<u32>,
        mut accs: Vec<S>,
        rgba: &mut Vec<f32>,
        per_band: u32,
        mut decode: impl FnMut(u32, u32) -> Result<Band, String>,
    ) -> Result<Vec<S::Sample>, String> {
        let align = align.max(1);
        let mut r0 = rows.start;
        while r0 < rows.end {
            if context.is_cancelled() {
                return Ok(Vec::new());
            }
            // End at a native-row boundary, including when the requested start
            // lies inside a tile row. This keeps at most per_band native rows.
            let boundary =
                (u64::from(r0) / u64::from(align) + u64::from(per_band / align)) * u64::from(align);
            let r1 = u64::from(rows.end).min(boundary).max(u64::from(r0) + 1) as u32;
            let band = decode(r0, r1)?;
            #[cfg(test)]
            if tests::CANCEL_AFTER_REGION_BAND.with(|c| c.replace(false)) {
                context.cancel();
            }
            if context.is_cancelled() {
                return Ok(Vec::new());
            }
            self.shade.fill(rgba, &band, w, r1 - r0)?;
            for (i, row) in rgba.chunks_exact(w as usize * 4).enumerate() {
                let y = r0 + i as u32;
                for acc in accs.iter_mut().filter(|a| a.rows().contains(&y)) {
                    acc.push_row(y, row)?;
                }
            }
            r0 = r1;
        }
        if accs.len() == 1 {
            return accs.pop().ok_or("missing accumulator")?.finish();
        }
        let mut out = Vec::new();
        for acc in accs {
            out.extend(acc.finish()?);
        }
        Ok(out)
    }

    /// Decode in bands of `bytes` level-0 RGBA bytes (tests).
    #[cfg(test)]
    fn with_band_bytes(mut self, bytes: usize) -> Self {
        self.band_bytes = bytes;
        self
    }

    /// Put levels with a long side up to `side` in the overview (tests).
    #[cfg(test)]
    fn with_overview_side(mut self, side: u32) -> Self {
        self.overview_side = side;
        self
    }

    /// Decode level-0 rows `[r0, r1)` of the used channels.
    fn decode_rows(&self, file: &mut InputFile<'_>, r0: u32, r1: u32) -> Result<Band, String> {
        #[cfg(test)]
        tests::L0_ROWS.with(|c| c.set(c.get() + u64::from(r1 - r0)));
        read_rows(file, &self.used, self.w, self.min, r0, r1)
    }

    fn open(&self) -> Result<StdIfStream, String> {
        open(&self.path)
    }

    /// The pixel readout at level-0 `(x, y)`: each shown channel by name, at
    /// full precision from the file (one decoded row cached).
    pub fn readout(&self, x: u32, y: u32) -> Option<Vec<(String, f32)>> {
        if x >= self.w {
            return None;
        }
        self.rows.with_row(y, |row| {
            let mut px = Vec::new();
            self.shade.fill(&mut px, row, self.w, 1).ok()?;
            let i = x as usize * 4;
            Some(
                self.labels
                    .iter()
                    .map(|(n, slot)| (n.clone(), px[i + slot]))
                    .collect(),
            )
        })?
    }
}

/// Level-0 rows of a file-backed image, decoded on demand for the pixel readouts,
/// the last one kept: the pointer moves along a row far more often than across.
/// The shown-channel readout ([`FileSource::readout`]) and the every-channel probe
/// ([`RowCache::all`]) both read through it.
pub struct RowCache {
    path: PathBuf,
    w: u32,
    h: u32,
    /// Data-window origin (file coordinates of level-0 pixel `(0, 0)`).
    min: (i32, i32),
    /// The channels decoded, in band order.
    used: Vec<Used>,
    /// The last decoded row: `(y, channels)`.
    row: Mutex<Option<(u32, Band)>>,
}

impl RowCache {
    fn new(path: PathBuf, (min, w, h): ((i32, i32), u32, u32), used: Vec<Used>) -> Self {
        Self {
            path,
            w,
            h,
            min,
            used,
            row: Mutex::new(None),
        }
    }

    /// Every channel of a file with channels `names`/`types` (header order) and
    /// `data_window` `(min, width, height)`.
    pub fn all(
        path: PathBuf,
        data_window: ((i32, i32), u32, u32),
        names: &[&str],
        types: &[PixelType],
    ) -> Result<Self, String> {
        let idx: Vec<usize> = (0..names.len()).collect();
        Ok(Self::new(
            path,
            data_window,
            used_channels(&idx, names, types)?,
        ))
    }

    /// `f` over row `y`'s channels (decoded when `y` is not the kept row). `None`
    /// out of bounds or when the row cannot be read (logged).
    pub fn with_row<R>(&self, y: u32, f: impl FnOnce(&Band) -> R) -> Option<R> {
        if y >= self.h {
            return None;
        }
        let mut kept = self.row.lock().ok()?;
        if kept.as_ref().is_none_or(|(ry, _)| *ry != y) {
            let read = || -> Result<Band, String> {
                let mut stream = open(&self.path)?;
                let mut file = InputFile::new(&mut stream).map_err(|e| e.to_string())?;
                read_rows(&mut file, &self.used, self.w, self.min, y, y + 1)
            };
            match read() {
                Ok(row) => *kept = Some((y, row)),
                Err(e) => {
                    log::warn!(
                        "{}: row {y} for the pixel readout: {e}",
                        self.path.display()
                    );
                    return None;
                }
            }
        }
        kept.as_ref().map(|(_, row)| f(row))
    }
}

fn open(path: &std::path::Path) -> Result<StdIfStream, String> {
    let p = path
        .to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))?;
    StdIfStream::open(p).map_err(|e| e.to_string())
}

/// Decode level-0 rows `[r0, r1)` of the channels `used` from `file`, whose data
/// window is `w` wide and starts at `min`.
fn read_rows(
    file: &mut InputFile<'_>,
    used: &[Used],
    w: u32,
    min: (i32, i32),
    r0: u32,
    r1: u32,
) -> Result<Band, String> {
    read_rows_into(file, used, w, min, r0, r1, &mut Vec::new())
}
fn read_rows_into(
    file: &mut InputFile<'_>,
    used: &[Used],
    w: u32,
    min: (i32, i32),
    r0: u32,
    r1: u32,
    buf: &mut Vec<u8>,
) -> Result<Band, String> {
    #[cfg(test)]
    tests::REGION_BANDS.with(|c| c.set(c.get() + 1));
    let rows = r1
        .checked_sub(r0)
        .filter(|&n| n > 0)
        .ok_or("empty or inverted decode band")? as usize;
    let planes = Planes::new(used, w, min, r0, rows)?;
    resize_raw(buf, planes.total)?;
    file.set_frame_buffer(&planes.fb)
        .map_err(|e| e.to_string())?;
    let y0 = i32::try_from(i64::from(min.1) + i64::from(r0)).map_err(|_| "file y overflow")?;
    let y1 = i32::try_from(i64::from(min.1) + i64::from(r1) - 1).map_err(|_| "file y overflow")?;
    file.read_pixels(y0, y1, buf).map_err(|e| e.to_string())?;
    planes.band(used, buf, 0, rows)
}
fn resize_raw(buf: &mut Vec<u8>, n: usize) -> Result<(), String> {
    if n > isize::MAX as usize {
        return Err("band bytes exceed addressable storage".into());
    }
    if buf.capacity() < n {
        buf.try_reserve_exact(n - buf.len())
            .map_err(|e| e.to_string())?;
    }
    buf.resize(n, 0);
    Ok(())
}

/// Decode rows `[r0, r1)` of the file's own level `(k, k)` (`size` = its
/// extent, data window at `min`) of the channels `used`. Whole file tile rows
/// are read; the requested rows are cut out.
fn read_own_rows(
    file: &mut TiledInputFile<'_>,
    used: &[Used],
    k: u32,
    size: (u32, u32),
    min: (i32, i32),
    r0: u32,
    r1: u32,
) -> Result<Band, String> {
    read_own_rows_into(file, used, k, size, min, r0, r1, &mut Vec::new())
}
#[allow(clippy::too_many_arguments)]
fn read_own_rows_into(
    file: &mut TiledInputFile<'_>,
    used: &[Used],
    k: u32,
    size: (u32, u32),
    min: (i32, i32),
    r0: u32,
    r1: u32,
    buf: &mut Vec<u8>,
) -> Result<Band, String> {
    #[cfg(test)]
    tests::REGION_BANDS.with(|c| c.set(c.get() + 1));
    if r1 <= r0 || r1 > size.1 {
        return Err("native band outside level".into());
    }
    let (lw, lh) = size;
    let th = file.tile_y_size().max(1);
    let (f0, f1) = (r0 / th, (r1 - 1) / th);
    let top = f0 * th;
    let bottom = ((f1 + 1) * th).min(lh);
    let planes = Planes::new(used, lw, min, top, (bottom - top) as usize)?;
    resize_raw(buf, planes.total)?;
    let (lvl, tx_last) = (
        k as i32,
        file.num_x_tiles(k as i32).map_err(|e| e.to_string())? - 1,
    );
    file.set_frame_buffer(&planes.fb)
        .map_err(|e| e.to_string())?;
    file.read_tiles(0, tx_last, f0 as i32, f1 as i32, lvl, lvl, buf)
        .map_err(|e| e.to_string())?;
    planes.band(used, buf, (r0 - top) as usize, (r1 - r0) as usize)
}

/// The frame-buffer layout of `rows` full-width rows of `used`, planar, the
/// first at level row `r0` of an image `w` wide whose window starts at `min`.
struct Planes {
    fb: FrameBuffer,
    /// Per channel: byte offset of its plane and bytes per row.
    planes: Vec<(usize, usize)>,
    total: usize,
}

impl Planes {
    fn new(used: &[Used], w: u32, min: (i32, i32), r0: u32, rows: usize) -> Result<Self, String> {
        let height = u32::try_from(rows).map_err(|_| "band height overflow")?;
        pixel_count(w, height)?;
        let y0 =
            i32::try_from(i64::from(min.1) + i64::from(r0)).map_err(|_| "window y overflow")?;
        let xmax =
            i32::try_from(i64::from(min.0) + i64::from(w) - 1).map_err(|_| "window x overflow")?;
        let ymax = i32::try_from(i64::from(y0) + i64::from(height) - 1)
            .map_err(|_| "window y overflow")?;
        let window = imath_rs::Box2i {
            min: imath_rs::V2i { x: min.0, y: y0 },
            max: imath_rs::V2i { x: xmax, y: ymax },
        };
        let mut fb = FrameBuffer::new();
        let mut planes = Vec::with_capacity(used.len());
        let mut total = 0usize;
        for u in used {
            let size = pixel_type_size(u.pixel_type).map_err(|e| e.to_string())?;
            let row = (w as usize).checked_mul(size).ok_or("plane row overflow")?;
            let len = row.checked_mul(rows).ok_or("plane bytes overflow")?;
            let end = total
                .checked_add(len)
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or("plane total overflow")?;
            let slice = Slice::make_from_window(
                u.pixel_type,
                total,
                window,
                size,
                row,
                1,
                1,
                0.0,
                false,
                false,
            )
            .map_err(|e| e.to_string())?;
            slice.validate_layout().map_err(|e| e.to_string())?;
            fb.insert(u.name.as_str(), slice)
                .map_err(|e| e.to_string())?;
            planes.push((total, row));
            total = end;
        }
        Ok(Self { fb, planes, total })
    }

    /// Rows `[skip, skip + take)` of every plane of `buf`, as typed channels.
    fn band(&self, used: &[Used], buf: &[u8], skip: usize, take: usize) -> Result<Band, String> {
        used.iter()
            .zip(&self.planes)
            .map(|(u, &(off, row))| {
                let bytes = buf
                    .get(
                        off.checked_add(skip.checked_mul(row).ok_or("band offset overflow")?)
                            .ok_or("band offset overflow")?
                            ..off
                                .checked_add(
                                    skip.checked_add(take)
                                        .and_then(|n| n.checked_mul(row))
                                        .ok_or("band extent overflow")?,
                                )
                                .ok_or("band extent overflow")?,
                    )
                    .ok_or_else(|| {
                        format!(
                            "channel `{}`: rows {skip}+{take} are outside the buffer",
                            u.name
                        )
                    })?;
                Ok((u.name.clone(), to_channel(u.pixel_type, bytes)))
            })
            .collect()
    }
}

/// Which levels of the pyramid `grid` the file at `path` stores itself with
/// exactly the viewer's geometry (see the module docs): `own[k]`. A file that is
/// not a tiled MIPMAP/RIPMAP one (every scanline file) has none.
fn own_levels(path: &Path, grid: &Grid) -> Vec<bool> {
    let mut own = vec![false; grid.levels() as usize];
    let mut stream = match open(path) {
        Ok(s) => s,
        Err(e) => {
            log::debug!("{}: no own levels: {e}", path.display());
            return own;
        }
    };
    let file = match TiledInputFile::new(&mut stream) {
        Ok(f) => f,
        Err(e) => {
            log::debug!("{}: no own levels: {e}", path.display());
            return own;
        }
    };
    if !matches!(
        file.level_mode(),
        LevelMode::MipmapLevels | LevelMode::RipmapLevels
    ) {
        return own;
    }
    for k in 1..grid.levels() {
        let l = k as i32;
        let (w, h) = grid.level_size(k);
        let same = file.is_valid_level(l, l)
            && file.data_window_for_level(l, l).is_ok_and(|b| {
                i64::from(b.max.x) - i64::from(b.min.x) + 1 == i64::from(w)
                    && i64::from(b.max.y) - i64::from(b.min.y) + 1 == i64::from(h)
            });
        if let Some(slot) = own.get_mut(k as usize) {
            *slot = same;
        }
    }
    own
}

/// The channels at `idx` (ascending) of `names`/`types`, as band channels.
fn used_channels(idx: &[usize], names: &[&str], types: &[PixelType]) -> Result<Vec<Used>, String> {
    idx.iter()
        .map(|&i| match (names.get(i), types.get(i)) {
            (Some(name), Some(&pixel_type)) => Ok(Used {
                name: (*name).to_owned(),
                pixel_type,
            }),
            _ => Err(format!("channel {i} of {} requested", names.len())),
        })
        .collect()
}

/// Native-endian bytes of one plane as typed samples.
fn to_channel(t: PixelType, bytes: &[u8]) -> ChannelData {
    match t {
        PixelType::Half => ChannelData::Half(
            bytes
                .as_chunks::<2>().0.iter()
                .map(|b| Half::from_bits(u16::from_ne_bytes(*b)))
                .collect(),
        ),
        PixelType::Float => ChannelData::Float(
            bytes
                .as_chunks::<4>().0.iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect(),
        ),
        // `pixel_type_size` already rejected anything else.
        _ => ChannelData::Uint(
            bytes
                .as_chunks::<4>().0.iter()
                .map(|b| u32::from_ne_bytes(*b))
                .collect(),
        ),
    }
}

impl TileSource for FileSource {
    fn read_region(
        &self,
        plan: RegionPlan,
        context: &ProductionContext,
        budget: &RegionBudget,
    ) -> Result<Arc<RegionData>, super::region::RegionError> {
        self.region(plan, context, budget)
    }
    fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    fn unit(&self, id: TileId) -> Unit {
        // The grid is a function of the size, so its geometry is rebuilt here.
        if self.is_own(id.level) {
            return Unit::Row {
                level: id.level,
                ty: id.ty,
            };
        }
        let from = self.overview_from(&Grid::new(self.w, self.h));
        if id.level >= from {
            Unit::Overview { from }
        } else {
            Unit::Row {
                level: id.level,
                ty: id.ty,
            }
        }
    }

    fn produce(&self, grid: &Grid, unit: Unit, context: &ProductionContext) -> Production {
        if context.is_cancelled() {
            return Production::Cancelled;
        }
        context.finish((|| match unit {
            #[cfg(feature = "ofx")]
            Unit::Level => Err("a file source does not render a whole effect level".into()),
            Unit::Row { level, ty } if self.is_own(level) => self.own_row(context, grid, level, ty),
            Unit::Row { level, ty } => {
                let acc = BandAccumulator::new(grid, level, ty).ok_or_else(|| {
                    format!(
                        "tile row L{level} {ty} is outside the {}x{} image",
                        self.w, self.h
                    )
                })?;
                let rows = acc.rows();
                self.stream(context, rows, vec![acc])
            }
            Unit::Overview { from } => {
                let mut accs = Vec::new();
                for level in (from..grid.levels()).filter(|&k| !self.is_own(k)) {
                    let (_, rows) = grid.tiles(level);
                    for ty in 0..rows {
                        accs.push(
                            BandAccumulator::new(grid, level, ty)
                                .ok_or("overview row outside the grid")?,
                        );
                    }
                }
                if accs.is_empty() {
                    return Ok(Vec::new());
                }
                log::debug!(
                    "overview pass: levels {from}..{} of {}x{}",
                    grid.levels(),
                    self.w,
                    self.h
                );
                self.stream(context, 0..self.h, accs)
            }
            Unit::Tile(id) => Err(format!(
                "the file source produces rows and the overview, not tile {id:?}"
            )),
        })())
    }

    fn scope_level(
        &self,
        _grid: &Grid,
        _max_pixels: u64,
        _max_dim: u32,
    ) -> Option<(u32, Arc<Vec<f32>>)> {
        // Assembled from the tile cache by the scheduler; the file is never
        // decoded whole just for a scope.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::DecodedFrame;
    use crate::tiles::reduce::tile_from_rgba;
    use exr_core::Compression;

    #[test]
    fn shifted_file_region_uses_three_enclosing_bins_through_dyn_provider() {
        use crate::tiles::grid::PixRect;
        use crate::tiles::region::{RegionPolicy, SamplingPhase};
        let path = scratch("roi-shifted-lattice.exr");
        Image::new(Box2i {
            min: V2i { x: -3, y: 7 },
            max: V2i { x: 0, y: 7 },
        })
        .with_channel("Y", ChannelData::Float(vec![10., 20., 30., 40.]))
        .write(&path, Compression::NoCompression)
        .unwrap();
        let groups = exr_core::attr::group_layers([("Y", PixelType::Float)]);
        let source = FileSource::new(
            path,
            ((-3, 7), 4, 1),
            &["Y"],
            &[PixelType::Float],
            Mapping::of(&["Y"], groups.first()),
        )
        .unwrap();
        let provider: &dyn TileSource = &source;
        let plan = RegionPlan::on_lattice(
            (4, 1),
            1,
            PixRect {
                x: 0,
                y: 0,
                w: 3,
                h: 1,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
            (1, 0),
        )
        .unwrap();
        REGION_BANDS.with(|c| c.set(0));
        let tiny = RegionBudget::new(1);
        assert!(
            provider
                .read_region(plan, &ProductionContext::default(), &tiny)
                .is_err()
        );
        assert_eq!(REGION_BANDS.with(|c| c.get()), 0);
        assert_eq!(tiny.used(), 0);
        let budget = RegionBudget::new(4096);
        let output = provider
            .read_region(plan, &ProductionContext::default(), &budget)
            .unwrap();
        assert_eq!(output.plan().level_bounds(), (0, 0, 3, 1));
        for (pixel, expected) in output.rgba().as_chunks::<4>().0.iter().zip([10., 25., 40.]) {
            assert_eq!(*pixel, [expected, expected, expected, 1.]);
        }
        assert!(REGION_BANDS.with(|c| c.get()) > 0);
        drop(output);
        budget.clear_workspace();
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn file_level_zero_region_preserves_float_bits() {
        use crate::tiles::grid::PixRect;
        use crate::tiles::region::{RegionPolicy, SamplingPhase};
        let bits = [0x80000000u32, 0x7fc12345, 0x3f800347, 0x7f800000];
        let path = scratch("roi-float-bits.exr");
        Image::new(Box2i {
            min: V2i { x: -3, y: -7 },
            max: V2i { x: -3, y: -4 },
        })
        .with_channel("Y", ChannelData::Float(bits.map(f32::from_bits).to_vec()))
        .write(&path, Compression::NoCompression)
        .unwrap();
        let groups = exr_core::attr::group_layers([("Y", PixelType::Float)]);
        let source = FileSource::new(
            path,
            ((-3, -7), 1, 4),
            &["Y"],
            &[PixelType::Float],
            Mapping::of(&["Y"], groups.first()),
        )
        .unwrap();
        let plan = RegionPlan::new(
            (1, 4),
            0,
            PixRect {
                x: 0,
                y: 0,
                w: 1,
                h: 4,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
        )
        .unwrap();
        let budget = RegionBudget::new(1024);
        let output = source
            .read_region(plan, &ProductionContext::default(), &budget)
            .unwrap();
        for (px, bits) in output.rgba().as_chunks::<4>().0.iter().zip(bits) {
            assert_eq!(
                px[..3].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                [bits; 3]
            );
            assert_eq!(px[3], 1.0);
        }
    }

    #[test]
    fn region_cancel_after_decode_releases_output_and_active_admission() {
        use crate::tiles::grid::PixRect;
        use crate::tiles::region::{RegionError, RegionPolicy, SamplingPhase};
        let path = scratch("roi-cancel.exr");
        Image::new(Box2i {
            min: V2i { x: 5, y: -9 },
            max: V2i { x: 8, y: -4 },
        })
        .with_channel(
            "Y",
            ChannelData::Float((0..24).map(|i| i as f32 / 7.0).collect()),
        )
        .write(&path, Compression::Zip)
        .unwrap();
        let groups = exr_core::attr::group_layers([("Y", PixelType::Float)]);
        let source = FileSource::new(
            path,
            ((5, -9), 4, 6),
            &["Y"],
            &[PixelType::Float],
            Mapping::of(&["Y"], groups.first()),
        )
        .unwrap();
        let plan = RegionPlan::new(
            (4, 6),
            1,
            PixRect {
                x: 0,
                y: 0,
                w: 2,
                h: 3,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
        )
        .unwrap();
        let budget = RegionBudget::new(4096);
        let context = ProductionContext::default();
        CANCEL_AFTER_REGION_BAND.with(|c| c.set(true));
        REGION_BANDS.with(|c| c.set(0));
        assert!(matches!(
            source.read_region(plan, &context, &budget),
            Err(RegionError::Cancelled)
        ));
        assert_eq!(
            REGION_BANDS.with(|c| c.get()),
            1,
            "cancelled ROI must stop after the first actual band"
        );
        assert_eq!(
            budget.used(),
            4 * 4 + 4 * 16,
            "only retained workspace remains charged"
        );
        budget.clear_workspace();
        assert_eq!(budget.used(), 0, "no active or output permit remains");
    }

    #[test]
    fn f32_region_coverage_is_extracted_before_averaging_ids() {
        use crate::tiles::grid::PixRect;
        use crate::tiles::region::{RegionPolicy, SamplingPhase};
        use cryptomatte::{EncodeOptions, NamedMatte, Windows};
        let coverage = (0..35)
            .map(|i| if i % 3 == 0 { 0.125 } else { 0.875 })
            .collect::<Vec<_>>();
        let inverse = coverage.iter().map(|&v| 1.0 - v).collect::<Vec<_>>();
        let encoded = cryptomatte::from_named_mattes(
            Windows::origin(7, 5).unwrap(),
            &[
                NamedMatte::new("bunny", &coverage),
                NamedMatte::new("cube", &inverse),
            ],
            &EncodeOptions::new("CryptoObject"),
        )
        .unwrap();
        let path = scratch("roi-crypto.exr");
        encoded.write(&path, Some(Compression::Zip)).unwrap();
        let frame = DecodedFrame::load(&path).unwrap();
        let names = frame
            .channels
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>();
        let types = frame
            .channels
            .iter()
            .map(|(_, d)| d.pixel_type())
            .collect::<Vec<_>>();
        let ranked = cryptomatte::ranked_layers_for("CryptoObject", names.iter().copied());
        let source = FileSource::coverage(
            path,
            ((0, 0), 7, 5),
            &names,
            &types,
            ranked,
            cryptomatte::name_to_float_id("bunny"),
        )
        .unwrap();
        let plan = RegionPlan::new(
            (7, 5),
            1,
            PixRect {
                x: 0,
                y: 0,
                w: 4,
                h: 3,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
        )
        .unwrap();
        let budget = RegionBudget::new(4096);
        let source: &dyn TileSource = &source;
        let got = source
            .read_region(plan, &ProductionContext::default(), &budget)
            .unwrap();
        let full = coverage
            .iter()
            .flat_map(|&v| [v, v, v, 1.0])
            .collect::<Vec<_>>();
        assert_eq!(
            got.rgba(),
            super::super::region::resident(&full, plan, || false).unwrap()
        );
        assert_eq!(
            got.rgba()[0],
            0.6875,
            "first 2x2 covers one low-coverage and three high-coverage source pixels"
        );
        let context = ProductionContext::default();
        context.cancel();
        assert!(matches!(
            source.read_region(plan, &context, &budget),
            Err(super::super::region::RegionError::Cancelled)
        ));
        drop(got);
        budget.clear_workspace();
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn f32_regions_match_resident_mixed_planes_at_negative_origin() {
        use crate::tiles::grid::PixRect;
        use crate::tiles::region::{RegionBudget, RegionPlan, RegionPolicy, SamplingPhase};
        let path = scratch("roi-mixed.exr");
        let dw = Box2i {
            min: V2i { x: -9, y: 13 },
            max: V2i { x: -3, y: 17 },
        };
        let image = Image::new(dw)
            .with_channel(
                "R",
                ChannelData::Float((0..35).map(|i| i as f32 / 7.0 + 0.0001).collect()),
            )
            .with_channel(
                "G",
                ChannelData::Half((0..35).map(|i| Half::from(i as f32 / 11.0)).collect()),
            )
            .with_channel("B", ChannelData::Uint((0..35).map(|i| i * 13).collect()));
        image.write(&path, Compression::Zip).unwrap();
        let frame = DecodedFrame::load(&path).unwrap();
        let names = frame
            .channels
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>();
        let types = frame
            .channels
            .iter()
            .map(|(_, d)| d.pixel_type())
            .collect::<Vec<_>>();
        let mapping = Mapping::of(&names, frame.layers.first());
        let mut rgba = Vec::new();
        mapping.fill(&mut rgba, &frame.channels, 7, 5).unwrap();
        let source = FileSource::new(path, ((-9, 13), 7, 5), &names, &types, mapping).unwrap();
        let resident = crate::tiles::source::FrameSource {
            rgba: Arc::new(rgba),
            w: 7,
            h: 5,
        };
        let budget = RegionBudget::new(4096);
        for phase in [SamplingPhase::TOP_LEFT, SamplingPhase::BOTTOM_LEFT] {
            for (level, rect) in [
                (
                    0,
                    PixRect {
                        x: 2,
                        y: 1,
                        w: 5,
                        h: 4,
                    },
                ),
                (
                    1,
                    PixRect {
                        x: 1,
                        y: 0,
                        w: 3,
                        h: 3,
                    },
                ),
                (
                    2,
                    PixRect {
                        x: 0,
                        y: 0,
                        w: 2,
                        h: 2,
                    },
                ),
            ] {
                let plan = RegionPlan::new((7, 5), level, rect, RegionPolicy::OriginalArea, phase)
                    .unwrap();
                let got = source
                    .region(plan, &ProductionContext::default(), &budget)
                    .unwrap();
                let want = resident
                    .region(plan, &ProductionContext::default(), &budget)
                    .unwrap();
                assert_eq!(got.rgba(), want.rgba());
                assert!(
                    got.rgba()[0] > 0.0,
                    "selected actual float plane must not silently become black"
                );
                if level == 0 && phase == SamplingPhase::TOP_LEFT {
                    assert_eq!(got.rgba()[0], 9.0f32 / 7.0 + 0.0001);
                    assert_eq!(got.rgba()[1], f32::from(Half::from(9.0f32 / 11.0)));
                    assert_eq!(got.rgba()[2], 117.0f32 / u32::MAX as f32);
                    assert_ne!(got.rgba()[0], f32::from(Half::from(got.rgba()[0])));
                }
                drop(got);
                drop(want);
                budget.clear_workspace();
                assert_eq!(budget.used(), 0);
            }
        }
        assert!(budget.reuse_hits() == 0);
        let plan = RegionPlan::new(
            (7, 5),
            1,
            PixRect {
                x: 0,
                y: 0,
                w: 4,
                h: 3,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
        )
        .unwrap();
        let first = source
            .region(plan, &ProductionContext::default(), &budget)
            .unwrap();
        drop(first);
        let second = source
            .region(plan, &ProductionContext::default(), &budget)
            .unwrap();
        assert!(
            budget.reuse_hits() > 0,
            "file provider must actually reuse its workspace"
        );
        drop(second);
        budget.clear_workspace();
    }

    #[test]
    fn f32_region_native_policy_preserves_stored_filter_and_budget_refuses_before_decode() {
        use crate::tiles::grid::PixRect;
        use crate::tiles::region::{RegionBudget, RegionPlan, RegionPolicy, SamplingPhase};
        let path = scratch("roi-native.exr");
        write_mip(&path, 1301, 5, LevelRoundingMode::RoundUp);
        let (source, rgba, grid) = mip_fixture(&path, 300);
        let rect = PixRect {
            x: 11,
            y: 0,
            w: 3,
            h: 3,
        };
        let native = RegionPlan::new(
            grid.size(),
            1,
            rect,
            RegionPolicy::MatchingStoredNative,
            SamplingPhase::TOP_LEFT,
        )
        .unwrap();
        let area = RegionPlan::new(
            grid.size(),
            1,
            rect,
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
        )
        .unwrap();
        let budget = RegionBudget::new(1 << 20);
        REGION_BANDS.with(|c| c.set(0));
        let got = source
            .read_region(native, &ProductionContext::default(), &budget)
            .unwrap();
        assert!(
            REGION_BANDS.with(|c| c.get()) > 0,
            "native ROI decode must engage"
        );
        let stored = read_level(&source.path, 1);
        for y in 0..3 {
            for x in 0..3 {
                let i = (y * 3 + x) as usize * 4;
                let want = stored[(y * 651 + x + 11) as usize];
                assert_eq!(&got.rgba()[i..i + 3], &want);
            }
        }
        let reduced = source
            .region(area, &ProductionContext::default(), &budget)
            .unwrap();
        assert_eq!(
            reduced.rgba(),
            super::super::region::resident(&rgba, area, || false).unwrap()
        );
        assert_ne!(got.rgba(), reduced.rgba());
        L0_ROWS.with(|c| c.set(0));
        REGION_BANDS.with(|c| c.set(0));
        let tight = RegionBudget::new(native.output_bytes() + native.sums_bytes() + 1);
        assert!(
            source
                .read_region(native, &ProductionContext::default(), &tight)
                .is_err()
        );
        assert_eq!(tight.used(), 0);
        assert_eq!(
            REGION_BANDS.with(|c| c.get()),
            0,
            "budget rejection precedes any pixel decode"
        );
        assert_eq!(L0_ROWS.with(|c| c.get()), 0);
        budget.clear_workspace();
    }

    #[test]
    fn file_native_preflight_rejects_absent_and_incompatible_levels_without_buffers_or_decode() {
        use crate::tiles::grid::PixRect;
        use crate::tiles::region::{RegionError, SamplingPhase};
        let path = scratch("roi-native-absent.exr");
        Image::new(Box2i {
            min: V2i { x: 0, y: 0 },
            max: V2i { x: 2, y: 2 },
        })
        .with_channel("Y", ChannelData::Float(vec![1.0001; 9]))
        .write(&path, Compression::Zip)
        .unwrap();
        let groups = exr_core::attr::group_layers([("Y", PixelType::Float)]);
        let scanline = FileSource::new(
            path,
            ((0, 0), 3, 3),
            &["Y"],
            &[PixelType::Float],
            Mapping::of(&["Y"], groups.first()),
        )
        .unwrap();
        let native = |size, level, w, h| {
            RegionPlan::new(
                size,
                level,
                PixRect { x: 0, y: 0, w, h },
                RegionPolicy::MatchingStoredNative,
                SamplingPhase::TOP_LEFT,
            )
            .unwrap()
        };
        let context = ProductionContext::default();
        let budget = RegionBudget::new(4096);
        let provider: &dyn TileSource = &scanline;
        let original = provider
            .read_region(native((3, 3), 0, 3, 3), &context, &budget)
            .unwrap();
        assert_eq!(&original.rgba()[..4], &[1.0001, 1.0001, 1.0001, 1.0]);
        drop(original);
        budget.clear_workspace();
        let path = scratch("roi-native-incompatible.exr");
        write_mip(&path, 7, 5, LevelRoundingMode::RoundDown);
        let (mip, _, _) = mip_fixture(&path, 300);
        // A missing backing file proves rejection precedes even reader creation.
        let absent = FileSource {
            path: scratch("never-written-native.exr"),
            ..scanline
        };
        let denied = RegionBudget::new(0);
        for (source, size, extent) in [(&absent, (3, 3), (2, 2)), (&mip, (7, 5), (4, 3))] {
            let p = native(size, 1, extent.0, extent.1);
            let provider: &dyn TileSource = source;
            REGION_BANDS.with(|c| c.set(0));
            L0_ROWS.with(|c| c.set(0));
            assert!(matches!(
                provider.read_region(p, &context, &denied),
                Err(RegionError::Unsupported)
            ));
            assert!(matches!(
                source.region(p, &context, &denied),
                Err(RegionError::Unsupported)
            ));
            assert_eq!(denied.used(), 0);
            assert_eq!(REGION_BANDS.with(|c| c.get()), 0);
            assert_eq!(L0_ROWS.with(|c| c.get()), 0);
        }
    }

    #[test]
    fn cancelling_after_an_actual_file_band_skips_the_remaining_decodes() {
        let path = scratch("cancel-after-band.exr");
        let window = Box2i {
            min: V2i { x: -9, y: 13 },
            max: V2i { x: -6, y: 18 },
        };
        Image::new(window)
            .with_channel("R", ChannelData::Float((0..24).map(|i| i as f32).collect()))
            .write(&path, Compression::Zip)
            .unwrap();
        let source = FileSource::new(
            path.clone(),
            ((-9, 13), 4, 6),
            &["R"],
            &[PixelType::Float],
            Mapping::of(&["R"], None),
        )
        .unwrap()
        .with_band_bytes(64);
        let grid = Grid::new(4, 6);
        let context = ProductionContext::default();
        let mut stream = source.open().unwrap();
        let mut input = InputFile::new(&mut stream).unwrap();
        let mut decoded_bands = 0;
        let result = source.pump(
            &context,
            4,
            1,
            0..6,
            vec![BandAccumulator::new(&grid, 0, 0).unwrap()],
            |r0, r1| {
                let band = source.decode_rows(&mut input, r0, r1)?;
                decoded_bands += 1;
                context.cancel();
                Ok(band)
            },
        );
        assert_eq!(decoded_bands, 1, "must not decode the remaining five bands");
        assert!(matches!(context.finish(result), Production::Cancelled));
        assert!(matches!(
            source.produce(&grid, Unit::Row { level: 0, ty: 0 }, &context),
            Production::Cancelled
        ));
        assert_eq!(decoded_bands, 1);
    }

    #[test]
    fn cancellation_stops_file_bands_and_wins_over_a_decode_failure() {
        let mapping = Mapping::of(&["R"], None);
        let source = FileSource::with_shade(
            std::path::PathBuf::from("cancelled-file-must-not-be-opened.exr"),
            ((-9, 13), 4, 4),
            vec![Used {
                name: "R".into(),
                pixel_type: PixelType::Float,
            }],
            Shade::Layer(mapping),
            vec![("R".into(), 0)],
        )
        .with_band_bytes(64);
        let grid = Grid::new(4, 4);
        let context = ProductionContext::default();
        let mut bands = 0;
        let result = source.pump(
            &context,
            4,
            1,
            0..4,
            vec![BandAccumulator::new(&grid, 0, 0).unwrap()],
            |r0, r1| {
                bands += 1;
                assert_eq!((r0, r1), (0, 1));
                context.cancel();
                Ok(vec![("R".into(), ChannelData::Float(vec![1.0; 4]))])
            },
        );
        assert_eq!(bands, 1);
        assert!(matches!(context.finish(result), Production::Cancelled));
        assert!(matches!(
            context.finish(Err("decode failed after cancellation".into())),
            Production::Cancelled
        ));
        assert!(matches!(
            source.produce(&grid, Unit::Row { level: 0, ty: 0 }, &context),
            Production::Cancelled
        ));
        bands = 0;
        let result = source.pump(
            &context,
            4,
            1,
            0..4,
            vec![BandAccumulator::new(&grid, 0, 0).unwrap()],
            |_, _| {
                bands += 1;
                Err("must not decode".into())
            },
        );
        assert_eq!(bands, 0, "cancelled generation must not start another band");
        assert!(matches!(context.finish(result), Production::Cancelled));
    }

    fn ready(source: &FileSource, grid: &Grid, unit: Unit) -> Vec<(TileId, TileData)> {
        match source.produce(grid, unit, &ProductionContext::default()) {
            Production::Ready(tiles) => tiles,
            _ => panic!("fixture source must be ready"),
        }
    }
    use exr_core::api::Image;
    use exr_core::attr::LevelRoundingMode;
    use imath_rs::{Box2i, V2i};

    thread_local! {
        /// Level-0 rows `FileSource::decode_rows` has decoded on this thread.
        pub(super) static CANCEL_AFTER_REGION_BAND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        pub(super) static REGION_BANDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(super) static L0_ROWS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("exr-view-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// 1100 x 600 at (3, -2): R G B A (half), depth.Z, N.x N.y N.z (float).
    fn write_scanline(path: &std::path::Path, comp: Compression) {
        let (w, h) = (1100usize, 600usize);
        let dw = Box2i {
            min: V2i { x: 3, y: -2 },
            max: V2i {
                x: 3 + w as i32 - 1,
                y: -2 + h as i32 - 1,
            },
        };
        let plane = |f: &dyn Fn(usize, usize) -> f32| {
            (0..w * h).map(|i| f(i % w, i / w)).collect::<Vec<f32>>()
        };
        let half = |v: Vec<f32>| ChannelData::Half(v.into_iter().map(Half::from).collect());
        Image::new(dw)
            .with_channel(
                "R",
                half(plane(&|x, y| (x % 97) as f32 / 50.0 + y as f32 * 1e-3)),
            )
            .with_channel("G", half(plane(&|x, y| ((x + y) % 13) as f32 / 7.0)))
            .with_channel("B", half(plane(&|x, _| (x as f32 * 0.01).sin() + 1.0)))
            .with_channel("A", half(plane(&|x, y| ((x ^ y) % 17) as f32 / 16.0)))
            .with_channel(
                "depth.Z",
                ChannelData::Float(plane(&|x, y| (x * 3 + y * 7) as f32)),
            )
            .with_channel("N.x", ChannelData::Float(plane(&|x, _| x as f32 / 1100.0)))
            .with_channel("N.y", ChannelData::Float(plane(&|_, y| y as f32 / 600.0)))
            .with_channel(
                "N.z",
                ChannelData::Float(plane(&|x, y| ((x * y) % 1000) as f32 / 1000.0)),
            )
            .write(path, comp)
            .unwrap();
    }

    /// 700 x 530 tiled (64 x 64) RGBA, ZIP.
    fn write_tiled(path: &std::path::Path) {
        use exr_core::attr::LineOrder;
        use exr_core::header::Header;
        use exr_core::rgba::{Rgba, RgbaChannels, TiledRgbaOutputFile};
        let (w, h) = (700usize, 530usize);
        let px: Vec<Rgba> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                Rgba {
                    r: Half::from(x / 700.0),
                    g: Half::from(y / 530.0),
                    b: Half::from(((x + y) % 50.0) / 50.0),
                    a: Half::from(1.0),
                }
            })
            .collect();
        let header = Header::with_dimensions(
            w as i32,
            h as i32,
            1.0,
            imath_rs::V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap();
        let mut os = exr_core::io::StdOfStream::create(path.to_str().unwrap()).unwrap();
        TiledRgbaOutputFile::write_stream(&mut os, &header, RgbaChannels::WRITE_RGBA, 64, 64, &px)
            .unwrap();
    }

    /// The resident path (DecodedFrame + Mapping + tile_from_rgba) and the
    /// file path give the same tiles on every level and tile row, for `layer`,
    /// and the same readout.
    fn check(path: &std::path::Path, layer: &str) {
        let frame = DecodedFrame::load(path).unwrap();
        let names: Vec<&str> = frame.channels.iter().map(|(n, _)| n.as_str()).collect();
        let types: Vec<PixelType> = frame.channels.iter().map(|(_, c)| c.pixel_type()).collect();
        let group = frame.layers.iter().find(|g| g.name == layer).unwrap();
        let mapping = Mapping::of(&names, Some(group));
        let (w, h) = (frame.width, frame.height);
        let mut rgba = Vec::new();
        mapping.fill(&mut rgba, &frame.channels, w, h).unwrap();

        let min = (frame.data_window.min.x, frame.data_window.min.y);
        // ~40 rows per band: several bands per level-0 tile row.
        // ~40 rows per band, and an overview from long side 300 down, so both
        // the row units and the overview unit run.
        let src = FileSource::new(path.to_path_buf(), (min, w, h), &names, &types, mapping)
            .unwrap()
            .with_band_bytes(w as usize * 16 * 40)
            .with_overview_side(300);
        let grid = Grid::new(w, h);
        assert!(
            src.own.iter().all(|&o| !o),
            "{}: no own levels expected",
            path.display()
        );
        let from = src.overview_from(&grid);
        assert!(from > 0 && from < grid.levels(), "both paths are exercised");
        let mut seen = std::collections::HashSet::new();
        for level in 0..grid.levels() {
            let (cols, rows) = grid.tiles(level);
            for ty in 0..rows {
                for tx in 0..cols {
                    let unit = src.unit(TileId { level, tx, ty });
                    assert_eq!(matches!(unit, Unit::Overview { .. }), level >= from);
                    if !seen.insert(unit) {
                        continue;
                    }
                    for (id, tile) in ready(&src, &grid, unit) {
                        assert_eq!(
                            Some(tile),
                            tile_from_rgba(&rgba, &grid, id),
                            "{} layer {layer:?} {id:?}",
                            path.display()
                        );
                    }
                }
            }
        }
        let rows = mapping.rows(&names);
        for (x, y) in [(0, 0), (w - 1, h - 1), (w / 2, h / 3), (17, h - 1)] {
            let want: Vec<(String, f32)> = rows
                .iter()
                .map(|(n, slot)| (n.clone(), rgba[((y * w + x) * 4) as usize + slot]))
                .collect();
            assert_eq!(
                src.readout(x, y),
                Some(want),
                "{} layer {layer:?} readout ({x},{y})",
                path.display()
            );
        }
        assert_eq!(src.readout(w, 0), None);
    }

    #[test]
    fn file_tiles_equal_resident_tiles_scanline() {
        for comp in [Compression::Zip, Compression::NoCompression] {
            let path = scratch(&format!("scan-{comp:?}.exr"));
            write_scanline(&path, comp);
            for layer in ["", "depth", "N"] {
                check(&path, layer);
            }
        }
    }

    #[test]
    fn file_tiles_equal_resident_tiles_tiled() {
        let path = scratch("tiled.exr");
        write_tiled(&path);
        check(&path, "");
    }

    const MIN: (i32, i32) = (5, -3);

    /// Level-local sample of channel `c` of level `l`: distinct per level (so a
    /// level read from the file cannot pass for a reduction of level 0) and an
    /// exact `f16`.
    fn val(l: u32, c: u32, x: u32, y: u32) -> f32 {
        ((x * 3 + y * 5 + l * 7 + c * 11) % 200) as f32 / 16.0
    }

    /// A tiled MIPMAP file (64 x 96 tiles, ZIP, window at `MIN`), R G B half,
    /// every level filled by [`val`].
    fn write_mip(path: &std::path::Path, w: i32, h: i32, round: LevelRoundingMode) {
        use exr_core::attr::{
            Channel, ChannelList, ChannelListAttribute, LineOrder, TileDescription,
        };
        use exr_core::header::Header;
        use exr_core::tiled::TiledOutputFile;
        let mut hd = Header::with_dimensions(
            w,
            h,
            1.0,
            imath_rs::V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::Zip,
        )
        .unwrap();
        let (x0, y0) = MIN;
        *hd.data_window_mut().unwrap() = Box2i {
            min: V2i { x: x0, y: y0 },
            max: V2i {
                x: x0 + w - 1,
                y: y0 + h - 1,
            },
        };
        let td = TileDescription {
            x_size: 64,
            y_size: 96,
            mode: LevelMode::MipmapLevels,
            rounding_mode: round,
        };
        hd.set_tile_description(td).unwrap();
        let mut cl = ChannelList::new();
        for n in ["B", "G", "R"] {
            cl.insert(n, Channel::new(PixelType::Half, 1, 1, false))
                .unwrap();
        }
        hd.insert("channels", ChannelListAttribute::new(cl))
            .unwrap();
        let mut os = exr_core::io::StdOfStream::create(path.to_str().unwrap()).unwrap();
        let mut f = TiledOutputFile::new(&mut os, &hd).unwrap();
        for l in 0..f.num_levels().unwrap() {
            let (lw, lh) = (
                f.level_width(l).unwrap() as u32,
                f.level_height(l).unwrap() as u32,
            );
            let plane = lw as usize * lh as usize * 2;
            let mut buf = vec![0u8; plane * 3];
            let mut fb = FrameBuffer::new();
            for (c, name) in ["R", "G", "B"].into_iter().enumerate() {
                for y in 0..lh {
                    for x in 0..lw {
                        let i = c * plane + (y * lw + x) as usize * 2;
                        buf[i..i + 2].copy_from_slice(
                            &Half::from(val(l as u32, c as u32, x, y))
                                .bits()
                                .to_ne_bytes(),
                        );
                    }
                }
                let row = lw as usize * 2;
                let base = (c * plane) as i64 - i64::from(x0) * 2 - i64::from(y0) * row as i64;
                let slice = Slice {
                    type_: PixelType::Half,
                    base_offset: base,
                    x_stride: 2,
                    y_stride: row,
                    x_sampling: 1,
                    y_sampling: 1,
                    fill_value: 0.0,
                    x_tile_coords: false,
                    y_tile_coords: false,
                };
                fb.insert(name, slice).unwrap();
            }
            f.set_frame_buffer(&fb).unwrap();
            f.write_tiles_l(
                &buf,
                0,
                f.num_x_tiles(l).unwrap() - 1,
                0,
                f.num_y_tiles(l).unwrap() - 1,
                l,
            )
            .unwrap();
        }
    }

    /// The file's own level `k` RGB, read independently through exr-core's
    /// tiled reader as `f32`.
    fn read_level(path: &std::path::Path, k: u32) -> Vec<[f32; 3]> {
        let mut stream = open(path).unwrap();
        let mut f = TiledInputFile::new(&mut stream).unwrap();
        let l = k as i32;
        let (lw, lh) = (
            f.level_width(l).unwrap() as usize,
            f.level_height(l).unwrap() as usize,
        );
        let mut buf = vec![0u8; lw * lh * 12];
        let mut fb = FrameBuffer::new();
        for (c, name) in ["R", "G", "B"].into_iter().enumerate() {
            let slice = Slice {
                type_: PixelType::Float,
                base_offset: (c * lw * lh * 4) as i64
                    - i64::from(MIN.0) * 4
                    - i64::from(MIN.1) * (lw * 4) as i64,
                x_stride: 4,
                y_stride: lw * 4,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            };
            fb.insert(name, slice).unwrap();
        }
        f.set_frame_buffer(&fb).unwrap();
        f.read_tiles(
            0,
            f.num_x_tiles(l).unwrap() - 1,
            0,
            f.num_y_tiles(l).unwrap() - 1,
            l,
            l,
            &mut buf,
        )
        .unwrap();
        let px = |c: usize, i: usize| {
            let o = (c * lw * lh + i) * 4;
            f32::from_ne_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
        };
        (0..lw * lh)
            .map(|i| [px(0, i), px(1, i), px(2, i)])
            .collect()
    }

    /// A [`FileSource`] over `path` (band and overview limits shrunk) with the
    /// resident level-0 RGBA the reduction rule is checked against.
    fn mip_fixture(path: &std::path::Path, overview_side: u32) -> (FileSource, Vec<f32>, Grid) {
        let frame = DecodedFrame::load(path).unwrap();
        let names: Vec<&str> = frame.channels.iter().map(|(n, _)| n.as_str()).collect();
        let types: Vec<PixelType> = frame.channels.iter().map(|(_, c)| c.pixel_type()).collect();
        let group = frame.layers.iter().find(|g| g.name.is_empty()).unwrap();
        let mapping = Mapping::of(&names, Some(group));
        let (w, h) = (frame.width, frame.height);
        let mut rgba = Vec::new();
        mapping.fill(&mut rgba, &frame.channels, w, h).unwrap();
        let src = FileSource::new(path.to_path_buf(), (MIN, w, h), &names, &types, mapping)
            .unwrap()
            .with_band_bytes(w as usize * 16 * 40)
            .with_overview_side(overview_side);
        (src, rgba, Grid::new(w, h))
    }

    /// Every unit of level `k`, produced once: `(unit, tiles)`.
    fn produce_level(
        src: &FileSource,
        grid: &Grid,
        k: u32,
    ) -> Vec<(Unit, Vec<(TileId, TileData)>)> {
        let (cols, rows) = grid.tiles(k);
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for ty in 0..rows {
            for tx in 0..cols {
                let unit = src.unit(TileId { level: k, tx, ty });
                if seen.insert(unit) {
                    out.push((unit, ready(src, grid, unit)));
                }
            }
        }
        out
    }

    /// Level `k`'s tiles equal the file's own level `k`, texel for texel.
    fn assert_own_level(src: &FileSource, grid: &Grid, path: &std::path::Path, k: u32) {
        let (lw, _) = grid.level_size(k);
        let want = read_level(path, k);
        let half = |v: f32| f32::from(Half::from(v));
        let mut tiles = 0;
        for (unit, made) in produce_level(src, grid, k) {
            assert!(
                matches!(unit, Unit::Row { level, .. } if level == k),
                "own level {k} is a row unit: {unit:?}"
            );
            for (id, tile) in made {
                let r = grid.rect(id).unwrap();
                assert_eq!(id.level, k);
                for y in 0..tile.h {
                    for x in 0..tile.w {
                        let p = want[((r.y + y) * lw + r.x + x) as usize];
                        let px = [half(p[0]), half(p[1]), half(p[2]), 1.0];
                        assert_eq!(tile.texel(x, y), Some(px), "L{k} {id:?} texel {x},{y}");
                    }
                }
                tiles += 1;
            }
        }
        let (cols, rows) = grid.tiles(k);
        assert_eq!(tiles, cols * rows, "every tile of level {k}");
    }

    /// Level `k`'s tiles are the reduction of level 0 (`tile_from_rgba`).
    fn assert_reduced_level(src: &FileSource, grid: &Grid, rgba: &[f32], k: u32) {
        for (_, made) in produce_level(src, grid, k) {
            for (id, tile) in made {
                assert_eq!(Some(tile), tile_from_rgba(rgba, grid, id), "reduced {id:?}");
            }
        }
    }

    /// ROUND_UP: every file level has the viewer's size, so every coarse level
    /// is the file's own, never the reduction; level 0 is unchanged; and no
    /// level-0 row is decoded for them (no overview pass at all).
    #[test]
    fn matching_file_levels_are_served_from_the_file() {
        let path = scratch("mip-up.exr");
        write_mip(&path, 1301, 1101, LevelRoundingMode::RoundUp);
        let (src, rgba, grid) = mip_fixture(&path, 300);
        assert_eq!(grid.levels(), 3);
        assert_eq!(src.own, [false, true, true]);
        L0_ROWS.with(|c| c.set(0));
        for k in 1..grid.levels() {
            assert_own_level(&src, &grid, &path, k);
        }
        assert_eq!(
            L0_ROWS.with(|c| c.get()),
            0,
            "no level-0 row decoded for own levels"
        );
        let from = src.overview_from(&grid);
        for k in from..grid.levels() {
            assert!(
                !matches!(
                    src.unit(TileId {
                        level: k,
                        tx: 0,
                        ty: 0
                    }),
                    Unit::Overview { .. }
                ),
                "L{k}"
            );
        }
        // Level 0 keeps the band path.
        for (_, made) in produce_level(&src, &grid, 0).into_iter().take(2) {
            for (id, tile) in made {
                assert_eq!(Some(tile), tile_from_rgba(&rgba, &grid, id), "{id:?}");
            }
        }
        assert!(L0_ROWS.with(|c| c.get()) > 0, "level 0 was decoded");
    }

    /// ROUND_DOWN with odd sizes: no level matches, so all keep the reduction.
    #[test]
    fn mismatched_file_levels_keep_the_reduction() {
        let path = scratch("mip-down.exr");
        write_mip(&path, 1301, 1101, LevelRoundingMode::RoundDown);
        let (src, rgba, grid) = mip_fixture(&path, 300);
        assert_eq!(src.own, [false, false, false]);
        for k in 0..grid.levels() {
            assert_reduced_level(&src, &grid, &rgba, k);
        }
    }

    /// ROUND_DOWN where only level 1 divides exactly: level 1 is the file's,
    /// level 2 is reduced, and the overview pass covers level 2 alone.
    #[test]
    fn mixed_levels_overview_only_covers_the_reduced_ones() {
        let path = scratch("mip-mixed.exr");
        write_mip(&path, 1302, 1100, LevelRoundingMode::RoundDown);
        let (src, rgba, grid) = mip_fixture(&path, 300);
        assert_eq!(grid.levels(), 3);
        assert_eq!(src.own, [false, true, false]);
        assert_own_level(&src, &grid, &path, 1);
        let top = produce_level(&src, &grid, 2);
        assert_eq!(top.len(), 1);
        assert!(matches!(top[0].0, Unit::Overview { from: 2 }));
        assert!(
            top[0].1.iter().all(|(id, _)| id.level >= 2),
            "the pass built no level-1 tile"
        );
        assert_reduced_level(&src, &grid, &rgba, 2);
    }

    /// Own levels that lie INSIDE the overview range (levels 1 and 2 divide
    /// exactly, level 3 does not, the overview starts at level 2): the pass
    /// builds level 3 alone, level 2 stays the file's.
    #[test]
    fn overview_pass_skips_own_levels_inside_its_range() {
        let path = scratch("mip-wide.exr");
        write_mip(&path, 2604, 104, LevelRoundingMode::RoundDown);
        let (src, rgba, grid) = mip_fixture(&path, 700);
        assert_eq!(grid.levels(), 4);
        assert_eq!(src.overview_from(&grid), 2);
        assert_eq!(src.own, [false, true, true, false]);
        assert_own_level(&src, &grid, &path, 2);
        let top = produce_level(&src, &grid, 3);
        assert_eq!(top.len(), 1);
        assert!(matches!(top[0].0, Unit::Overview { from: 2 }));
        assert!(
            top[0].1.iter().all(|(id, _)| id.level == 3),
            "the pass built only level 3"
        );
        assert_reduced_level(&src, &grid, &rgba, 3);
    }

    /// The Cryptomatte overlay on a file-backed frame shows the same tiles
    /// and readout as the resident overlay (coverage computed over the whole
    /// frame, then tiled).
    #[test]
    fn coverage_tiles_equal_resident_coverage() {
        use cryptomatte::{EncodeOptions, NamedMatte, Windows};
        let (w, h) = (600u32, 300u32);
        let n = (w * h) as usize;
        let bunny: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) / 6.0).collect();
        let cube: Vec<f32> = bunny.iter().map(|c| 1.0 - c).collect();
        let enc = cryptomatte::from_named_mattes(
            Windows::origin(w as usize, h as usize).unwrap(),
            &[
                NamedMatte::new("bunny", &bunny),
                NamedMatte::new("cube", &cube),
            ],
            &EncodeOptions::new("CryptoObject"),
        )
        .unwrap();
        let path = scratch("crypto.exr");
        enc.write(&path, Some(Compression::Zip)).unwrap();

        let frame = DecodedFrame::load(&path).unwrap();
        let names: Vec<&str> = frame.channels.iter().map(|(n, _)| n.as_str()).collect();
        let types: Vec<PixelType> = frame.channels.iter().map(|(_, c)| c.pixel_type()).collect();
        let ranked = cryptomatte::ranked_layers_for("CryptoObject", names.iter().copied());
        let id = cryptomatte::name_to_float_id("bunny");
        let matte = cryptomatte::extract_coverage_with(w, h, &ranked, &[id], |name| {
            frame
                .channels
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, d)| d)
                .ok_or_else(|| cryptomatte::Error::Metadata(name.to_string()))
        })
        .unwrap();
        let rgba: Vec<f32> = matte
            .coverage
            .iter()
            .flat_map(|&c| [c, c, c, 1.0])
            .collect();

        let src = FileSource::coverage(path.clone(), ((0, 0), w, h), &names, &types, ranked, id)
            .unwrap()
            .with_band_bytes(w as usize * 16 * 30)
            .with_overview_side(200);
        let grid = Grid::new(w, h);
        let mut seen = std::collections::HashSet::new();
        for level in 0..grid.levels() {
            let (cols, rows) = grid.tiles(level);
            for ty in 0..rows {
                for tx in 0..cols {
                    let unit = src.unit(TileId { level, tx, ty });
                    if !seen.insert(unit) {
                        continue;
                    }
                    for (id, tile) in ready(&src, &grid, unit) {
                        assert_eq!(
                            Some(tile),
                            tile_from_rgba(&rgba, &grid, id),
                            "coverage {id:?}"
                        );
                    }
                }
            }
        }
        for (x, y) in [(0u32, 0u32), (599, 299), (123, 45)] {
            let c = matte.coverage[(y * w + x) as usize];
            assert_eq!(src.readout(x, y), Some(vec![("coverage".to_string(), c)]));
        }
    }
}
