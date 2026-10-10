//! Checked local image regions, streaming area sampling and byte admission.
use super::grid::PixRect;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

/// A structured provider outcome; callers never classify display error strings.
#[derive(Debug, PartialEq, Eq)]
pub enum RegionError {
    Unsupported,
    /// Intrinsically larger than the immutable cap.
    Budget { needed: usize, limit: usize },
    /// Another live owner temporarily holds admission.
    Pending,
    Cancelled,
    Failed(String),
}
impl std::error::Error for RegionError {}

impl RegionError {
    pub(crate) fn from_result<T>(result: Result<T, Self>, cancelled: bool) -> Result<T, Self> {
        if cancelled {
            Err(Self::Cancelled)
        } else {
            result
        }
    }
}

impl From<String> for RegionError {
    fn from(value: String) -> Self { Self::Failed(value) }
}
impl From<&str> for RegionError {
    fn from(value: &str) -> Self { Self::Failed(value.into()) }
}
impl std::fmt::Display for RegionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("region format or sampling policy is unsupported"),
            Self::Budget { needed, limit } => write!(f,"region needs {needed} bytes; immutable cap is {limit}"),
            Self::Pending => f.write_str("region admission waits for a live owner"),
            Self::Cancelled => f.write_str("region cancelled"),
            Self::Failed(error) => f.write_str(error),
        }
    }
}

/// Stored mip samples and original-image area means are deliberately distinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionPolicy {
    OriginalArea,
    MatchingStoredNative,
}

/// The edge at which integer area cells begin. Bottom/right cells retain their
/// own phase on odd extents; reversing rows cannot substitute for this choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    Start,
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        allow(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingPhase {
    pub x: Anchor,
    pub y: Anchor,
}
impl SamplingPhase {
    pub const TOP_LEFT: Self = Self {
        x: Anchor::Start,
        y: Anchor::Start,
    };
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        expect(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    pub const BOTTOM_LEFT: Self = Self {
        x: Anchor::Start,
        y: Anchor::End,
    };
}

/// One checked integer axis, shared by ROI plans and processed display geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LatticeAxis {
    start: i64,
    end: i64,
}
impl LatticeAxis {
    pub(crate) fn new(origin: i64, extent: u32) -> Result<Self, String> {
        if extent == 0 { return Err("lattice extent must be positive".into()); }
        Ok(Self { start: origin, end: origin.checked_add(i64::from(extent)).ok_or("lattice extent overflows")? })
    }
    /// Positive integer sampling step; all arithmetic is wider than the checked endpoints.
    pub(crate) fn bounds(self, step: i128) -> (i64, i64) {
        (i128::from(self.start).div_euclid(step) as i64, (-(-i128::from(self.end)).div_euclid(step)) as i64)
    }
    pub(crate) fn span(self, first: u32, count: u32, step: i128) -> std::ops::Range<u32> {
        let origin_bin = i128::from(self.bounds(step).0);
        let n = i128::from(self.end) - i128::from(self.start);
        let a = ((origin_bin + i128::from(first)) * step - i128::from(self.start)).clamp(0,n);
        let b = ((origin_bin + i128::from(first) + i128::from(count)) * step - i128::from(self.start)).clamp(0,n);
        a as u32..b as u32
    }
    pub(crate) fn texel(self, local: u32, step: i128) -> f64 {
        (self.start as f64 + f64::from(local)) / step as f64 - self.bounds(step).0 as f64
    }
    pub(crate) fn start(self) -> i64 { self.start }
}

/// Immutable half-open rectangle in the level's local sampling lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionPlan {
    size: (u32, u32),
    level: u32,
    rect: PixRect,
    policy: RegionPolicy,
    phase: SamplingPhase,
    origin: (i64, i64),
    bounds: (i64, i64, i64, i64),
}

pub fn pixel_count(w: u32, h: u32) -> Result<usize, String> {
    if w == 0 || h == 0 {
        return Err("region extent must be positive".into());
    }
    (w as usize)
        .checked_mul(h as usize)
        .filter(|&n| n <= isize::MAX as usize / 48)
        .ok_or_else(|| "region pixel bytes overflow addressable storage".into())
}

impl RegionPlan {
    pub fn new(
        size: (u32, u32),
        level: u32,
        rect: PixRect,
        policy: RegionPolicy,
        phase: SamplingPhase,
    ) -> Result<Self, String> {
        Self::on_lattice(size, level, rect, policy, phase, (0, 0))
    }
    /// The origin is the coordinate of logical source sample zero relative to
    /// the immutable integer lattice. For End it is measured in the reflected
    /// axis, not copied blindly from the file's unreflected minimum.
    /// The rectangle indexes the enclosing signed bin bounds locally.
    pub fn on_lattice(
        size: (u32, u32),
        level: u32,
        rect: PixRect,
        policy: RegionPolicy,
        phase: SamplingPhase,
        origin: (i64, i64),
    ) -> Result<Self, String> {
        pixel_count(size.0, size.1)?;
        let step = i128::from(1u64.checked_shl(level).ok_or("sampling level overflows")?);
        if policy == RegionPolicy::MatchingStoredNative
            && (origin != (0, 0) || phase != SamplingPhase::TOP_LEFT)
        {
            return Err("native levels require their stored zero-origin top-left lattice".into());
        }
        let (x0, x1) = LatticeAxis::new(origin.0, size.0)?.bounds(step);
        let (y0, y1) = LatticeAxis::new(origin.1, size.1)?.bounds(step);
        let lw = u64::try_from(i128::from(x1) - i128::from(x0))
            .map_err(|_| "lattice width overflows")?;
        let lh = u64::try_from(i128::from(y1) - i128::from(y0))
            .map_err(|_| "lattice height overflows")?;
        if rect.w == 0
            || rect.h == 0
            || u64::from(rect.x) + u64::from(rect.w) > lw
            || u64::from(rect.y) + u64::from(rect.h) > lh
        {
            return Err("region is outside the level lattice".into());
        }
        pixel_count(rect.w, rect.h)?;
        Ok(Self {
            size,
            level,
            rect,
            policy,
            phase,
            origin,
            bounds: (x0, y0, x1, y1),
        })
    }
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        expect(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    pub fn lattice_origin(self) -> (i64, i64) {
        self.origin
    }
    /// Signed half-open enclosing bin bounds: (x0, y0, x1, y1).
    pub fn level_bounds(self) -> (i64, i64, i64, i64) {
        self.bounds
    }
    pub fn size(self) -> (u32, u32) {
        self.size
    }
    pub fn level(self) -> u32 {
        self.level
    }
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        allow(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    pub fn rect(self) -> PixRect {
        self.rect
    }
    pub fn policy(self) -> RegionPolicy {
        self.policy
    }
    pub fn phase(self) -> SamplingPhase {
        self.phase
    }
    pub fn output_bytes(self) -> usize {
        self.rect.w as usize * self.rect.h as usize * 16
    }
    pub fn sums_bytes(self) -> usize {
        self.rect.w as usize * self.rect.h as usize * 32
    }
    fn span(self, i: u32, n: u32, anchor: Anchor, origin: i64) -> std::ops::Range<u32> {
        let step = i128::from(1u64 << self.level);
        let first = i128::from(origin).div_euclid(step);
        let k = first + i128::from(i);
        let a = (k * step - i128::from(origin)).clamp(0, i128::from(n)) as u32;
        let b = ((k + 1) * step - i128::from(origin)).clamp(0, i128::from(n)) as u32;
        match anchor {
            Anchor::Start => a..b,
            Anchor::End => n - b..n - a,
        }
    }
    pub fn rows(self) -> std::ops::Range<u32> {
        let a = self.span(self.rect.y, self.size.1, self.phase.y, self.origin.1);
        let b = self.span(
            self.rect.y + self.rect.h - 1,
            self.size.1,
            self.phase.y,
            self.origin.1,
        );
        a.start.min(b.start)..a.end.max(b.end)
    }
    pub(crate) fn native(self) -> Result<Self, String> {
        if self.phase != SamplingPhase::TOP_LEFT || self.origin != (0, 0) {
            return Err("native levels require their stored top-left lattice".into());
        }
        let step = 1u64 << self.level;
        Self::new(
            (
                u64::from(self.size.0).div_ceil(step) as u32,
                u64::from(self.size.1).div_ceil(step) as u32,
            ),
            0,
            self.rect,
            RegionPolicy::OriginalArea,
            self.phase,
        )
    }
}

/// Shared row-ordered f64 kernel for resident regions and streamed file bands.
pub(crate) struct RegionAccumulator {
    plan: RegionPlan,
    next: u32,
    sums: Vec<[f64; 4]>,
    exact: Vec<f32>,
    // Present only for disjoint mixed-LOD cells; uniform row readers allocate none.
    weights: Vec<u64>,
    first_bits: Vec<[u32; 4]>,
    uniform: Vec<u8>,
}
impl RegionAccumulator {
    pub(crate) fn new(plan: RegionPlan) -> Result<Self, String> {
        #[cfg(test)]
        tests::ACCUMULATOR_BUILDS.with(|c| c.set(c.get() + 1));
        let n = pixel_count(plan.rect.w, plan.rect.h)?;
        let mut sums = Vec::new();
        let mut exact = Vec::new();
        if plan.level == 0 {
            exact.try_reserve_exact(n * 4).map_err(|e| e.to_string())?;
            exact.resize(n * 4, 0.0);
        } else {
            sums.try_reserve_exact(n).map_err(|e| e.to_string())?;
            sums.resize(n, [0.; 4]);
        }
        Ok(Self {
            plan,
            next: plan.rows().start,
            sums,
            exact,
            weights: Vec::new(),
            first_bits: Vec::new(),
            uniform: Vec::new(),
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        self.sums.capacity() * 32 + self.exact.capacity() * 4 + self.weights.capacity() * 8
            + self.first_bits.capacity() * 16 + self.uniform.capacity()
    }

    /// Scratch for one checked destination strip, including coverage proof.
    #[cfg(feature = "ofx")]
    pub(crate) fn cells_required_bytes(plan: RegionPlan) -> Result<usize, String> {
        if plan.phase != SamplingPhase::TOP_LEFT || plan.policy != RegionPolicy::OriginalArea
            || plan.level > 40 || plan.rect.h != 1 {
            return Err("mosaic reduction requires one original-area top-first row".into());
        }
        let pixels = pixel_count(plan.rect.w, plan.rect.h)?;
        pixels.checked_mul(if plan.level == 0 { 24 } else { 57 })
            .ok_or_else(|| "mosaic reducer scratch overflows".into())
    }

    #[cfg(feature = "ofx")]
    pub(crate) fn for_cells(plan: RegionPlan) -> Result<Self, String> {
        Self::cells_required_bytes(plan)?;
        let mut accumulator = Self::new(plan)?;
        let count = pixel_count(plan.rect.w, plan.rect.h)?;
        accumulator.weights.try_reserve_exact(count).map_err(|error| error.to_string())?;
        accumulator.weights.resize(count, 0);
        if plan.level > 0 {
            accumulator.first_bits.try_reserve_exact(count).map_err(|error| error.to_string())?;
            accumulator.first_bits.resize(count, [0; 4]);
            accumulator.uniform.try_reserve_exact(count).map_err(|error| error.to_string())?;
            accumulator.uniform.resize(count, 0);
        }
        Ok(accumulator)
    }

    /// Reuse storage for another row of the same immutable sampling lattice.
    #[cfg(feature = "ofx")]
    pub(crate) fn restart_cells(&mut self, y: u32) -> Result<(), String> {
        if self.weights.len() != self.plan.rect.w as usize || self.plan.rect.h != 1 {
            return Err("accumulator has no mosaic row storage".into());
        }
        let rect = PixRect { y, ..self.plan.rect };
        self.plan = RegionPlan::on_lattice(self.plan.size, self.plan.level, rect,
            self.plan.policy, self.plan.phase, self.plan.origin)?;
        self.next = self.plan.rows().start;
        self.sums.fill([0.0; 4]);
        self.exact.fill(0.0);
        self.weights.fill(0);
        self.first_bits.fill([0; 4]);
        self.uniform.fill(0);
        Ok(())
    }

    /// Disjoint accepted native cells contribute their exact clipped original area.
    /// No common full-resolution raster or intermediate f16 values are constructed.
    #[cfg(feature = "ofx")]
    pub(crate) fn add_cell(&mut self, cell: PixRect, pixel: [f32; 4]) -> Result<(), String> {
        if self.weights.len() != self.plan.rect.w as usize || self.plan.rect.h != 1 {
            return Err("accumulator has no mosaic row storage".into());
        }
        let p = self.plan;
        let right = cell.x.checked_add(cell.w).ok_or("mosaic cell edge overflows")?;
        let bottom = cell.y.checked_add(cell.h).ok_or("mosaic cell edge overflows")?;
        let rows = p.rows();
        if cell.w == 0 || cell.h == 0 || right > p.size.0 || bottom > p.size.1
            || cell.y < rows.start || bottom > rows.end {
            return Err("mosaic cell is outside the requested original row".into());
        }
        let step = 1i128 << p.level;
        let bin = |local: u32| (i128::from(p.origin.0) + i128::from(local))
            .div_euclid(step) - i128::from(p.bounds.0);
        let first = bin(cell.x).max(i128::from(p.rect.x));
        let end = (bin(right - 1) + 1).min(i128::from(p.rect.x) + i128::from(p.rect.w));
        if first >= end { return Err("mosaic cell misses the destination strip".into()); }
        let ys = p.span(p.rect.y, p.size.1, p.phase.y, p.origin.1);
        for bin_x in first..end {
            let x = u32::try_from(bin_x).map_err(|_| "mosaic column exceeds checked lattice")?;
            let xs = p.span(x, p.size.0, p.phase.x, p.origin.0);
            let width = xs.end.min(right) - xs.start.max(cell.x);
            let height = ys.end.min(bottom) - ys.start.max(cell.y);
            let weight = u64::from(width) * u64::from(height);
            let index = (x - p.rect.x) as usize;
            let area = u64::from(xs.end - xs.start) * u64::from(ys.end - ys.start);
            let total = self.weights[index].checked_add(weight).ok_or("mosaic coverage overflows")?;
            if total > area { return Err("mosaic cells overlap a destination sample".into()); }
            if p.level > 0 {
                let bits = pixel.map(f32::to_bits);
                if self.weights[index] == 0 {
                    self.first_bits[index] = bits;
                    self.uniform[index] = 0b1111;
                } else {
                    for (channel, bits) in bits.into_iter().enumerate() {
                        if self.first_bits[index][channel] != bits {
                            self.uniform[index] &= !(1u8 << channel);
                        }
                    }
                }
            }
            self.weights[index] = total;
            if p.level == 0 {
                // A level-zero destination has area one: preserve NaN payloads and signed zero.
                self.exact[index * 4..index * 4 + 4].copy_from_slice(&pixel);
            } else {
                for (sum, value) in self.sums[index].iter_mut().zip(pixel) {
                    *sum += f64::from(value) * weight as f64;
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "ofx")]
    pub(crate) fn finish_cells_into(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        let p = self.plan;
        if self.weights.len() != p.rect.w as usize || p.rect.h != 1 {
            return Err("accumulator has no mosaic row storage".into());
        }
        let ys = p.span(p.rect.y, p.size.1, p.phase.y, p.origin.1);
        for x in 0..p.rect.w {
            let xs = p.span(p.rect.x + x, p.size.0, p.phase.x, p.origin.0);
            let area = u64::from(xs.end - xs.start) * u64::from(ys.end - ys.start);
            if self.weights[x as usize] != area {
                return Err("accepted mosaic does not cover every destination sample".into());
            }
        }
        if out.capacity() < p.output_bytes() / 4 {
            return Err("mosaic output row was not admitted before reduction".into());
        }
        self.next = self.rows().end;
        self.write_into(out)?;
        // Matching native proxy samples need no arithmetic: preserve payload bits
        // per component even when ragged cells split one sample across sweep bands.
        if p.level > 0 {
            for (index, flags) in self.uniform.iter().copied().enumerate() {
                for channel in 0..4 {
                    if flags & (1u8 << channel) != 0 {
                        out[index * 4 + channel] = f32::from_bits(self.first_bits[index][channel]);
                    }
                }
            }
        }
        Ok(())
    }
    pub(crate) fn rows(&self) -> std::ops::Range<u32> {
        self.plan.rows()
    }
    pub(crate) fn push_row(&mut self, y: u32, row: &[f32]) -> Result<(), String> {
        if y != self.next || y >= self.rows().end || row.len() != self.plan.size.0 as usize * 4 {
            return Err("region rows must arrive in order at full source width".into());
        }
        let p = self.plan;
        let local = match p.phase.y {
            Anchor::Start => y,
            Anchor::End => p.size.1 - 1 - y,
        };
        let bin =
            (i128::from(p.origin.1) + i128::from(local)).div_euclid(i128::from(1u64 << p.level));
        let ly = u32::try_from(bin - i128::from(p.level_bounds().1) - i128::from(p.rect.y))
            .map_err(|_| "row is outside the requested lattice")?;
        for lx in 0..p.rect.w {
            let span = p.span(p.rect.x + lx, p.size.0, p.phase.x, p.origin.0);
            if p.level == 0 {
                let target = (ly as usize * p.rect.w as usize + lx as usize) * 4;
                self.exact[target..target + 4]
                    .copy_from_slice(&row[span.start as usize * 4..span.start as usize * 4 + 4]);
                continue;
            }
            let acc = &mut self.sums[ly as usize * p.rect.w as usize + lx as usize];
            for px in row[span.start as usize * 4..span.end as usize * 4].as_chunks::<4>().0 {
                for c in 0..4 {
                    acc[c] += f64::from(px[c]);
                }
            }
        }
        self.next += 1;
        Ok(())
    }
    /// Weighted proxy samples use their clipped original lattice area.
    #[cfg(feature = "ofx")]
    pub(crate) fn push_level_row(&mut self, fine: u32, y: u32, row: &[f32]) -> Result<(), String> {
        let width = LatticeAxis::new(self.plan.origin.0, self.plan.size.0)?
            .bounds(1i128 << fine);
        if row.len() != usize::try_from(width.1 - width.0).map_err(|_|"processed width overflows")? * 4 {
            return Err("processed row has wrong width".into());
        }
        if fine == 0 { return self.push_row(y, row); }
        self.push_level_pixels(fine, y, |x| Ok(row[x as usize * 4..x as usize * 4 + 4].try_into().unwrap()))
    }
    /// Stream only contributing samples, including valid unaligned host rows;
    /// the caller never needs a full proxy row allocation.
    #[cfg(feature = "ofx")]
    pub(crate) fn push_level_pixels(&mut self, fine: u32, y: u32,
        mut pixel: impl FnMut(u32) -> Result<[f32; 4], String>) -> Result<(), String> {
        let p = self.plan;
        if fine > p.level || p.phase != SamplingPhase::TOP_LEFT {
            return Err("processed reduction requires a finer top-first lattice".into());
        }
        let step = 1i128 << fine;
        let x = LatticeAxis::new(p.origin.0, p.size.0)?;
        let axis_y = LatticeAxis::new(p.origin.1, p.size.1)?;
        let fine_width = u32::try_from(x.bounds(step).1 - x.bounds(step).0)
            .map_err(|_| "processed fine width overflows")?;
        let ys = axis_y.span(y, 1, step);
        if ys.start != self.next || ys.start >= self.rows().end || ys.end > self.rows().end {
            return Err("processed rows must cover the requested original rows in order".into());
        }
        let ly = u32::try_from((i128::from(p.origin.1) + i128::from(ys.start))
            .div_euclid(1i128 << p.level) - i128::from(p.bounds.1) - i128::from(p.rect.y))
            .map_err(|_| "processed row is outside its destination")?;
        for fx in 0..fine_width {
            let xs = x.span(fx, 1, step);
            let bin = (i128::from(p.origin.0) + i128::from(xs.start)).div_euclid(1i128 << p.level)
                - i128::from(p.bounds.0) - i128::from(p.rect.x);
            let Ok(lx) = u32::try_from(bin) else { continue; };
            if lx >= p.rect.w { continue; }
            let value = pixel(fx)?;
            let target = ly as usize * p.rect.w as usize + lx as usize;
            if p.level == 0 {
                self.exact[target * 4..target * 4 + 4].copy_from_slice(&value);
            } else {
                let weight = f64::from(xs.end - xs.start) * f64::from(ys.end - ys.start);
                let acc = &mut self.sums[target];
                for c in 0..4 { acc[c] += f64::from(value[c]) * weight; }
            }
        }
        self.next = ys.end;
        Ok(())
    }
    #[cfg(feature = "ofx")]
    pub(crate) fn write_row(&self, out: &mut Vec<f32>) -> Result<(), String> {
        self.write_into(out)
    }

    fn write_into(&self, out: &mut Vec<f32>) -> Result<(), String> {
        if self.next != self.rows().end {
            return Err("region stopped before all source rows arrived".into());
        }
        out.clear();
        let p = self.plan;
        if p.level == 0 {
            out.extend_from_slice(&self.exact);
            return Ok(());
        }
        for ly in 0..p.rect.h {
            let ys = p.span(p.rect.y + ly, p.size.1, p.phase.y, p.origin.1);
            for lx in 0..p.rect.w {
                let xs = p.span(p.rect.x + lx, p.size.0, p.phase.x, p.origin.0);
                let n = f64::from(ys.end - ys.start) * f64::from(xs.end - xs.start);
                out.extend(
                    self.sums[ly as usize * p.rect.w as usize + lx as usize]
                        .map(|v| (v / n) as f32),
                );
            }
        }
        Ok(())
    }
    fn restart_row(&mut self, y: u32) {
        self.plan.rect.y = y;
        self.next = self.rows().start;
        self.sums.fill([0.0; 4]);
    }
    pub(crate) fn finish(self) -> Result<Vec<f32>, String> {
        if self.next != self.rows().end {
            return Err("region stopped before all source rows arrived".into());
        }
        if self.plan.level == 0 {
            return Ok(self.exact);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(self.plan.output_bytes() / 4)
            .map_err(|e| e.to_string())?;
        self.write_into(&mut out)?;
        Ok(out)
    }
}

/// One admission ledger shared across concurrent jobs and retained region owners.
/// This bounds provider-owned pixel buffers, not codec/reader internal allocations.
#[derive(Clone)]
pub struct RegionBudget {
    inner: Arc<Budget>,
    attempt: Option<Arc<Attempt>>,
}
/// A request factory keeps attribution, but does not pin a workspace claim.
/// Only a successful owner-attached permit retains the upgraded attempt.
#[cfg(feature = "ofx")]
#[derive(Clone)]
pub(crate) struct AttemptAdmission {
    inner: Arc<Budget>,
    attempt: std::sync::Weak<Attempt>,
}
#[cfg(feature = "ofx")]
impl AttemptAdmission {
    pub(crate) fn used(&self) -> usize { self.inner.used.load(Ordering::Acquire) }
    pub(crate) fn attempt_capacity(&self) -> usize {
        self.attempt.upgrade().map_or(0, |attempt| {
            if attempt.retained.load(Ordering::Acquire) { self.inner.limit / 2 }
            else { self.inner.limit - self.inner.limit / 2 }
        })
    }
    pub(crate) fn reserve(&self, bytes: usize) -> Result<Permit, RegionError> {
        let attempt = self.attempt.upgrade().ok_or(RegionError::Cancelled)?;
        let budget = RegionBudget { inner: self.inner.clone(), attempt: Some(attempt) };
        budget.reserve(bytes)
    }
}

struct Budget {
    limit: usize,
    used: AtomicUsize,
    ledger: Mutex<Ledger>,
    pool: Mutex<Option<Buffers>>,
    hits: AtomicUsize,
    released: AtomicU64,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}
#[derive(Default)]
struct Ledger {
    retained: usize,
    candidate: Option<u64>,
    #[cfg(any(feature = "ofx", test))]
    next: u64,
}
/// Membership belongs to physical permits, including host frames surviving a call.
struct Attempt {
    id: u64,
    live: AtomicUsize,
    retained: std::sync::atomic::AtomicBool,
    budget: std::sync::Weak<Budget>,
}
impl Drop for Attempt {
    fn drop(&mut self) {
        if let Some(budget) = self.budget.upgrade() {
            let mut ledger = budget.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let changed = ledger.candidate == Some(self.id);
            if changed { ledger.candidate = None; }
            drop(ledger);
            if changed { budget.notify(); }
        }
    }
}
struct Buffers {
    raw: Vec<u8>,
    rgba: Vec<f32>,
}
impl Buffers {
    fn bytes(&self) -> usize {
        self.raw.capacity() + self.rgba.capacity() * 4
    }
}
impl Budget {
    fn notify(&self) {
        self.released.fetch_add(1, Ordering::AcqRel);
        if let Some(wake) = &self.wake { wake(); }
    }
    fn discard_buffers(&self, buffers: Buffers) {
        let bytes = buffers.bytes();
        drop(buffers);
        self.release(bytes);
    }
    fn release(&self, bytes: usize) {
        if bytes == 0 { return; }
        let ledger = self.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.used.fetch_sub(bytes, Ordering::AcqRel);
        drop(ledger);
        self.notify();
    }
}
impl RegionBudget {
    /// Request metadata can outlive a call without becoming a physical owner.
    #[cfg(feature = "ofx")]
    pub(crate) fn admission_view(&self) -> Result<AttemptAdmission, RegionError> {
        let attempt = self.attempt.as_ref().ok_or(RegionError::Unsupported)?;
        Ok(AttemptAdmission { inner: self.inner.clone(), attempt: Arc::downgrade(attempt) })
    }
    #[cfg(any(feature = "ofx", test))]
    pub fn new(limit: usize) -> Self {
        Self::with_wake(limit, None)
    }
    #[cfg(any(feature = "ofx", test))]
    pub(crate) fn with_wake(limit: usize, wake: Option<Arc<dyn Fn() + Send + Sync>>) -> Self {
        Self {
            inner: Arc::new(Budget {
                limit,
                used: AtomicUsize::new(0),
                ledger: Mutex::new(Ledger::default()),
                pool: Mutex::new(None),
                hits: AtomicUsize::new(0),
                released: AtomicU64::new(0),
                wake,
            }),
            attempt: None,
        }
    }
    #[cfg(any(feature = "ofx", test))]
    pub fn used(&self) -> usize {
        self.inner.used.load(Ordering::Acquire)
    }
    #[cfg(test)]
    pub fn reuse_hits(&self) -> usize {
        self.inner.hits.load(Ordering::Acquire)
    }
    /// Retained outputs and one replacement share this physical cap equally.
    pub(crate) fn retained_limit(&self) -> usize { self.inner.limit / 2 }
    pub(crate) fn candidate_limit(&self) -> usize { self.inner.limit - self.retained_limit() }
    /// Claim the sole workspace before plan/render; clones share this membership.
    #[cfg(any(feature = "ofx", test))]
    pub(crate) fn begin_attempt(&self) -> Result<Self, RegionError> {
        let mut ledger = self.inner.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if ledger.candidate.is_some() { return Err(RegionError::Pending); }
        let id = ledger.next.checked_add(1).ok_or("effect attempt identity exhausted")?;
        ledger.next = id;
        ledger.candidate = Some(id);
        let attempt = Arc::new(Attempt {
            id, live: AtomicUsize::new(0), retained: std::sync::atomic::AtomicBool::new(false),
            budget: Arc::downgrade(&self.inner),
        });
        drop(ledger);
        Ok(Self { inner: self.inner.clone(), attempt: Some(attempt) })
    }
    /// Transfer all surviving physical owners, without global debit/recharge.
    /// Pending leaves the completed candidate claimed; it must not be rerendered.
    #[cfg(any(feature = "ofx", test))]
    pub(crate) fn promote(&self) -> Result<(), RegionError> {
        let Some(attempt) = &self.attempt else { return Ok(()); };
        let mut ledger = self.inner.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if attempt.retained.load(Ordering::Acquire) { return Ok(()); }
        let live = attempt.live.load(Ordering::Acquire);
        if live > self.retained_limit() {
            return Err(RegionError::Budget { needed: live, limit: self.retained_limit() });
        }
        if ledger.retained.checked_add(live).is_none_or(|n| n > self.retained_limit()) {
            return Err(RegionError::Pending);
        }
        ledger.retained += live;
        ledger.candidate = None;
        attempt.retained.store(true, Ordering::Release);
        drop(ledger);
        self.inner.notify();
        Ok(())
    }
    #[cfg(feature = "ofx")]
    pub(crate) fn physical(&self) -> Self { Self { inner: self.inner.clone(), attempt: None } }
    #[cfg(feature = "ofx")]
    pub(crate) fn attempt_live(&self) -> Option<usize> {
        self.attempt.as_ref().map(|a| a.live.load(Ordering::Acquire))
    }
    #[cfg(feature = "ofx")]
    pub(crate) fn attempt_members(&self) -> Option<usize> {
        self.attempt.as_ref().map(Arc::strong_count)
    }
    #[cfg(feature = "ofx")]
    pub(crate) fn can_promote_after(&self, retiring: usize) -> Result<(), RegionError> {
        let Some(attempt) = &self.attempt else { return Ok(()); };
        if attempt.retained.load(Ordering::Acquire) { return Ok(()); }
        let ledger = self.inner.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let live = attempt.live.load(Ordering::Acquire);
        if live > self.retained_limit() {
            return Err(RegionError::Budget { needed: live, limit: self.retained_limit() });
        }
        if ledger.retained.saturating_sub(retiring).checked_add(live).is_none_or(|n| n > self.retained_limit()) {
            Err(RegionError::Pending)
        } else { Ok(()) }
    }
    /// Temporary display storage shares the physical cap but is never promoted
    /// into retained effect ownership. A live candidate must fit its own peak.
    #[cfg(feature = "ofx")]
    pub(crate) fn can_temporary(&self, bytes: usize) -> Result<(), RegionError> {
        let Some(attempt) = &self.attempt else { return Ok(()); };
        if attempt.retained.load(Ordering::Acquire) { return Ok(()); }
        let live = attempt.live.load(Ordering::Acquire);
        let needed = live.checked_add(bytes).ok_or("effect display peak overflows")?;
        if needed > self.candidate_limit() {
            Err(RegionError::Budget { needed, limit: self.candidate_limit() })
        } else { Ok(()) }
    }
    /// Explicitly discard retained scratch after a generation or memory-pressure event.
    #[cfg(any(feature = "ofx", test))]
    pub fn clear_workspace(&self) {
        let buffers = self.inner.pool.lock().ok().and_then(|mut pool| pool.take());
        if let Some(buffers) = buffers { self.inner.discard_buffers(buffers); }
    }
    #[cfg(any(feature = "ofx", test))]
    pub(crate) fn readiness_epoch(&self) -> u64 {
        self.inner.released.load(Ordering::Acquire)
    }
    pub(crate) fn limit(&self) -> usize { self.inner.limit }
    fn charge_membership(&self, bytes: usize, ledger: &mut Ledger) -> Result<(), RegionError> {
        let Some(attempt) = &self.attempt else { return Ok(()); };
        let live = attempt.live.load(Ordering::Acquire);
        let needed = live.checked_add(bytes).ok_or("effect compound working bytes overflow")?;
        let limit = if attempt.retained.load(Ordering::Acquire) {
            self.retained_limit()
        } else { self.candidate_limit() };
        if needed > limit { return Err(RegionError::Budget { needed, limit }); }
        if attempt.retained.load(Ordering::Acquire) {
            if ledger.retained.checked_add(bytes).is_none_or(|n| n > limit) {
                return Err(RegionError::Pending);
            }
            ledger.retained += bytes;
        }
        attempt.live.store(needed, Ordering::Release);
        Ok(())
    }
    fn uncharge_membership(&self, bytes: usize, ledger: &mut Ledger) {
        if let Some(attempt) = &self.attempt {
            attempt.live.fetch_sub(bytes, Ordering::AcqRel);
            if attempt.retained.load(Ordering::Acquire) { ledger.retained -= bytes; }
        }
    }
    /// Pool transfer changes membership only: the physical allocation is already charged.
    fn adopt_workspace(&self, bytes: usize) -> Result<(), RegionError> {
        let mut ledger = self.inner.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.charge_membership(bytes, &mut ledger)
    }
    fn release(&self, bytes: usize, physical: bool) {
        if bytes == 0 { return; }
        let mut ledger = self.inner.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.uncharge_membership(bytes, &mut ledger);
        if physical { self.inner.used.fetch_sub(bytes, Ordering::AcqRel); }
        drop(ledger);
        if physical || self.attempt.is_some() { self.inner.notify(); }
    }
    pub(crate) fn workspace(
        &self,
        raw_bytes: usize,
        rgba_samples: usize,
    ) -> Result<Workspace, RegionError> {
        let pooled = self.inner.pool.lock().map_err(|_| "workspace lock poisoned")?.take();
        if let Some(b) = pooled {
            if b.raw.capacity() >= raw_bytes && b.rgba.capacity() >= rgba_samples
                && self.adopt_workspace(b.bytes()).is_ok() {
                    self.inner.hits.fetch_add(1, Ordering::AcqRel);
                    return Ok(Workspace { budget: self.clone(), buffers: Some(b) });
            }
            // Optional retained capacity cannot turn a smaller intrinsic-fit
            // request into Budget. Free it and admit the exact fresh request.
            self.inner.discard_buffers(b);
        }
        let bytes = raw_bytes.checked_add(rgba_samples.checked_mul(4).ok_or("workspace overflow")?)
            .ok_or("workspace overflow")?;
        let mut permit = self.reserve(bytes)?;
        let mut b = Buffers { raw: Vec::new(), rgba: Vec::new() };
        b.raw.try_reserve_exact(raw_bytes).map_err(|e| e.to_string())?;
        b.rgba.try_reserve_exact(rgba_samples).map_err(|e| e.to_string())?;
        permit.resize(b.bytes())?;
        permit.bytes = 0;
        Ok(Workspace { budget: self.clone(), buffers: Some(b) })
    }
    pub(crate) fn reserve(&self, bytes: usize) -> Result<Permit, RegionError> {
        if bytes > self.inner.limit {
            return Err(RegionError::Budget { needed: bytes, limit: self.inner.limit });
        }
        let mut ledger = self.inner.ledger.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.charge_membership(bytes, &mut ledger)?;
        let used = self.inner.used.load(Ordering::Acquire);
        let Some(next) = used.checked_add(bytes).filter(|&n| n <= self.inner.limit) else {
            self.uncharge_membership(bytes, &mut ledger);
            return Err(RegionError::Pending);
        };
        self.inner.used.store(next, Ordering::Release);
        drop(ledger);
        Ok(Permit { budget: self.clone(), bytes })
    }
}
pub(crate) struct Workspace {
    budget: RegionBudget,
    buffers: Option<Buffers>,
}
impl Workspace {
    pub(crate) fn buffers(&mut self) -> (&mut Vec<u8>, &mut Vec<f32>) {
        let b = self.buffers.as_mut().expect("workspace is present until drop");
        (&mut b.raw, &mut b.rgba)
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        if let Some(b) = self.buffers.take() {
            let bytes = b.bytes();
            let mut pool = self.budget.inner.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if pool.is_none() {
                *pool = Some(b);
                drop(pool);
                self.budget.release(bytes, false);
                return;
            }
            drop(pool);
            drop(b);
            self.budget.release(bytes, true);
        }
    }
}
pub(crate) struct Permit {
    budget: RegionBudget,
    bytes: usize,
}
impl Permit {
    pub(crate) fn resize(&mut self, bytes: usize) -> Result<(), RegionError> {
        if bytes > self.budget.inner.limit {
            return Err(RegionError::Budget { needed: bytes, limit: self.budget.inner.limit });
        }
        if bytes > self.bytes {
            let mut extra = self.budget.reserve(bytes - self.bytes)?;
            extra.bytes = 0;
        } else {
            self.budget.release(self.bytes - bytes, true);
        }
        self.bytes = bytes;
        Ok(())
    }
    #[cfg(all(test, feature = "ofx"))]
    pub(crate) fn budget(&self) -> &RegionBudget { &self.budget }
}
impl Drop for Permit {
    fn drop(&mut self) { self.budget.release(self.bytes, true); }
}

/// Immutable original-precision output. Its allocation stays charged until the
/// final Arc owner disappears; copying a handle never releases admission.
pub struct RegionData {
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        allow(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    plan: RegionPlan,
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        allow(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    rgba: Vec<f32>,
    _permit: Permit,
}
impl RegionData {
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        expect(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    pub fn plan(&self) -> RegionPlan {
        self.plan
    }
    #[cfg_attr(
        not(any(feature = "ofx", test)),
        expect(
            dead_code,
            reason = "region consumers are absent when the OpenFX feature is disabled"
        )
    )]
    pub fn rgba(&self) -> &[f32] {
        &self.rgba
    }
    pub(crate) fn finish(
        plan: RegionPlan,
        rgba: Vec<f32>,
        permit: Permit,
    ) -> Result<Arc<Self>, RegionError> {
        // Bind storage to its permit before any fallible capacity adjustment.
        // Parameter drop order alone would release admission before the Vec.
        let mut data = Self { plan, rgba, _permit: permit };
        let bytes = data.rgba.capacity().checked_mul(4).ok_or("output capacity overflow")?;
        data._permit.resize(bytes)?;
        Ok(Arc::new(data))
    }
}

/// Resident sources stream one output row at a time through the same kernel,
/// so tile production never allocates an entire f64 tile beside its Half output.
pub(crate) fn resident_rows(
    src: &[f32],
    plan: RegionPlan,
    mut cancelled: impl FnMut() -> bool,
    mut emit: impl FnMut(&[f32]),
) -> Result<(), String> {
    if src.len()
        != pixel_count(plan.size.0, plan.size.1)?
            .checked_mul(4)
            .ok_or("source size overflow")?
    {
        return Err("source has wrong length".into());
    }
    let stride = plan.size.0 as usize * 4;
    let mut row_plan = plan;
    row_plan.rect.h = 1;
    let mut acc = RegionAccumulator::new(row_plan)?;
    let mut row = Vec::new();
    row.try_reserve_exact(plan.rect.w as usize * 4)
        .map_err(|e| e.to_string())?;
    for y in 0..plan.rect.h {
        if cancelled() {
            return Err("region cancelled".into());
        }
        acc.restart_row(plan.rect.y + y);
        for sy in acc.rows() {
            if cancelled() {
                return Err("region cancelled".into());
            }
            acc.push_row(sy, &src[sy as usize * stride..(sy as usize + 1) * stride])?;
        }
        if cancelled() {
            return Err("region cancelled".into());
        }
        acc.write_into(&mut row)?;
        emit(&row);
    }
    Ok(())
}
pub(crate) fn resident(
    src: &[f32],
    plan: RegionPlan,
    cancelled: impl FnMut() -> bool,
) -> Result<Vec<f32>, String> {
    if src.len()
        != pixel_count(plan.size.0, plan.size.1)?
            .checked_mul(4)
            .ok_or("source size overflow")?
    {
        return Err("source has wrong length".into());
    }
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(plan.output_bytes() / 4)
        .map_err(|e| e.to_string())?;
    resident_rows(src, plan, cancelled, |row| rgba.extend_from_slice(row))?;
    Ok(rgba)
}

#[cfg(test)]
impl Drop for Buffers {
    fn drop(&mut self) {
        tests::BUFFERS_DROP_CHECK.with(|check| {
            if let Some(check) = check.borrow_mut().take() {
                check();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "ofx")]
    #[test]
    fn mosaic_cells_weight_mixed_native_areas_and_reuse_signed_ragged_rows() {
        let plan = RegionPlan::on_lattice((5, 7), 2,
            PixRect { x: 0, y: 0, w: 2, h: 1 }, RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT, (-3, -7)).unwrap();
        let mut accumulator = RegionAccumulator::for_cells(plan).unwrap();
        let mut out = Vec::with_capacity(8);
        assert!(accumulator.finish_cells_into(&mut out).is_err(), "missing coverage is refused");
        accumulator.add_cell(PixRect { x: 0, y: 0, w: 1, h: 1 }, [100.0; 4]).unwrap();
        accumulator.add_cell(PixRect { x: 1, y: 0, w: 2, h: 1 }, [10.0; 4]).unwrap();
        accumulator.add_cell(PixRect { x: 0, y: 1, w: 3, h: 2 }, [10.0; 4]).unwrap();
        accumulator.add_cell(PixRect { x: 3, y: 0, w: 2, h: 3 }, [20.0; 4]).unwrap();
        accumulator.finish_cells_into(&mut out).unwrap();
        assert_eq!(out, vec![20.0; 8], "one fine pixel plus eight coarse-area pixels");
        assert!(accumulator.add_cell(PixRect { x: 0, y: 0, w: 1, h: 1 }, [0.0; 4]).is_err());
        let bytes = accumulator.bytes();
        accumulator.restart_cells(1).unwrap();
        accumulator.add_cell(PixRect { x: 0, y: 3, w: 3, h: 4 }, [3.0; 4]).unwrap();
        accumulator.add_cell(PixRect { x: 3, y: 3, w: 2, h: 4 }, [9.0; 4]).unwrap();
        accumulator.finish_cells_into(&mut out).unwrap();
        assert_eq!(out, [vec![3.0; 4], vec![9.0; 4]].concat());
        assert_eq!(accumulator.bytes(), bytes, "row reuse does not grow scratch");
        assert!(accumulator.restart_cells(2).is_err());
    }

    #[cfg(feature = "ofx")]
    #[test]
    fn mosaic_matching_proxy_samples_preserve_bits_across_fragmented_cells() {
        let plan = RegionPlan::on_lattice((2, 1), 1,
            PixRect { x: 0, y: 0, w: 1, h: 1 }, RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT, (0, -1)).unwrap();
        let mut accumulator = RegionAccumulator::for_cells(plan).unwrap();
        let value = [-0.0, f32::from_bits(0x7fc1_2345), f32::from_bits(1), f32::INFINITY];
        accumulator.add_cell(PixRect { x: 0, y: 0, w: 1, h: 1 }, value).unwrap();
        accumulator.add_cell(PixRect { x: 1, y: 0, w: 1, h: 1 }, value).unwrap();
        let mut out = Vec::with_capacity(4);
        accumulator.finish_cells_into(&mut out).unwrap();
        assert_eq!(out.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), value.map(f32::to_bits));
    }

    #[cfg(feature = "ofx")]
    #[test]
    fn mosaic_level_zero_preserves_native_sample_bits_and_requires_admitted_output() {
        let plan = RegionPlan::on_lattice((2, 1), 0,
            PixRect { x: 0, y: 0, w: 2, h: 1 }, RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT, (-4, -1)).unwrap();
        let mut accumulator = RegionAccumulator::for_cells(plan).unwrap();
        let first = [f32::from_bits(0x7fc1_2345), -0.0, f32::INFINITY, -f32::INFINITY];
        let second = [f32::from_bits(0x0000_0001), -3.25, 1.0, 0.0];
        accumulator.add_cell(PixRect { x: 0, y: 0, w: 1, h: 1 }, first).unwrap();
        accumulator.add_cell(PixRect { x: 1, y: 0, w: 1, h: 1 }, second).unwrap();
        assert!(accumulator.finish_cells_into(&mut Vec::new()).is_err());
        let mut out = Vec::with_capacity(8);
        accumulator.finish_cells_into(&mut out).unwrap();
        let actual: Vec<_> = out.iter().map(|value| value.to_bits()).collect();
        let expected: Vec<_> = first.into_iter().chain(second).map(f32::to_bits).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn typed_pressure_retries_only_after_actual_release() {
        use super::*;
        let budget = RegionBudget::new(16);
        let before = budget.readiness_epoch();
        let held = budget.reserve(16).unwrap();
        assert_eq!(budget.readiness_epoch(), before, "admission is not readiness");
        assert!(matches!(budget.reserve(8), Err(RegionError::Pending)));
        assert!(matches!(budget.reserve(17), Err(RegionError::Budget { needed: 17, limit: 16 })));
        drop(held);
        assert!(budget.readiness_epoch() > before);
        let _retry = budget.reserve(8).unwrap();
    }

    #[test]
    fn every_release_wakes_after_pool_guard_and_accounting() {
        use super::*;
        let link = Arc::new(Mutex::new(None::<std::sync::Weak<Budget>>));
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_link = link.clone();
        let callback_calls = calls.clone();
        let budget = RegionBudget::with_wake(1024, Some(Arc::new(move || {
            let inner = callback_link.lock().unwrap().as_ref().unwrap().upgrade().unwrap();
            let _pool = inner.pool.try_lock().expect("release callback ran under pool guard");
            assert!(inner.used.load(Ordering::Acquire) <= inner.limit);
            callback_calls.fetch_add(1, Ordering::Relaxed);
        })));
        *link.lock().unwrap() = Some(Arc::downgrade(&budget.inner));
        drop(budget.workspace(8, 2).unwrap());
        budget.clear_workspace();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        drop(budget.workspace(8, 2).unwrap());
        let larger = budget.workspace(16, 4).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2, "undersized pooled buffers discarded");
        let second = budget.workspace(8, 2).unwrap();
        drop(larger);
        drop(second);
        assert_eq!(calls.load(Ordering::Relaxed), 3, "occupied-pool discard");
        budget.clear_workspace();
        let mut permit = budget.reserve(64).unwrap();
        permit.resize(32).unwrap();
        drop(permit);
        assert_eq!(budget.used(), 0);
        assert_eq!(calls.load(Ordering::Relaxed), 6, "pool plus resize plus permit drop");
    }

    thread_local! {
        pub(super) static BUFFERS_DROP_CHECK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
            const { std::cell::RefCell::new(None) };
    }

    #[test]
    fn discarded_workspace_buffers_remain_charged_at_destruction_entry() {
        for path in 0..3 {
            let budget = RegionBudget::new(256);
            let first = budget.workspace(128, 0).unwrap();
            let active = if path == 2 {
                Some(budget.workspace(128, 0).unwrap())
            } else {
                None
            };
            drop(first);
            let expected = if path == 2 { 256 } else { 128 };
            let observed = budget.clone();
            let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let entered = Arc::clone(&called);
            BUFFERS_DROP_CHECK.with(|check| {
                *check.borrow_mut() = Some(Box::new(move || {
                    // Drop entry precedes both Vec field destructors: their allocations are live.
                    assert_eq!(observed.used(), expected);
                    assert!(observed.reserve(256).is_err());
                    entered.store(true, Ordering::Release);
                }));
            });
            match path {
                0 => budget.clear_workspace(),
                1 => drop(budget.workspace(256, 0).unwrap()),
                2 => drop(active),
                _ => unreachable!(),
            }
            assert!(called.load(Ordering::Acquire));
            budget.clear_workspace();
            assert_eq!(budget.used(), 0);
        }
    }

    thread_local! { pub(super) static ACCUMULATOR_BUILDS: std::cell::Cell<usize> = const {std::cell::Cell::new(0)}; }
    use super::*;
    use crate::tiles::source::{FrameSource, ProductionContext};
    fn plan(phase: SamplingPhase) -> RegionPlan {
        RegionPlan::new(
            (3, 5),
            1,
            PixRect {
                x: 0,
                y: 0,
                w: 2,
                h: 3,
            },
            RegionPolicy::OriginalArea,
            phase,
        )
        .unwrap()
    }

    #[test]
    fn frame_native_preflight_is_shared_by_direct_and_dyn_providers() {
        use crate::tiles::source::TileSource;
        let source = FrameSource {
            rgba: Arc::new(vec![1.0001; 16]),
            w: 2,
            h: 2,
        };
        let context = ProductionContext::default();
        let native = |level, side| {
            RegionPlan::new(
                (2, 2),
                level,
                PixRect {
                    x: 0,
                    y: 0,
                    w: side,
                    h: side,
                },
                RegionPolicy::MatchingStoredNative,
                SamplingPhase::TOP_LEFT,
            )
            .unwrap()
        };
        let denied = RegionBudget::new(0);
        ACCUMULATOR_BUILDS.with(|c| c.set(0));
        let provider: &dyn TileSource = &source;
        for result in [
            provider.read_region(native(1, 1), &context, &denied),
            source.region(native(1, 1), &context, &denied),
        ] {
            assert!(matches!(result, Err(RegionError::Unsupported)));
        }
        assert_eq!(denied.used(), 0);
        assert_eq!(ACCUMULATOR_BUILDS.with(|c| c.get()), 0);
        let budget = RegionBudget::new(1024);
        let got = provider
            .read_region(native(0, 2), &context, &budget)
            .unwrap();
        assert_eq!(got.rgba(), &[1.0001; 16]);
        assert_eq!(got.plan().policy(), RegionPolicy::MatchingStoredNative);
        drop(got);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn shifted_integer_lattice_keeps_the_immutable_anchor_and_clipped_bins() {
        let source: Vec<f32> = [10., 20., 30., 40.]
            .into_iter()
            .flat_map(|v| [v; 4])
            .collect();
        let make = |origin, x, w| {
            RegionPlan::on_lattice(
                (4, 1),
                1,
                PixRect { x, y: 0, w, h: 1 },
                RegionPolicy::OriginalArea,
                SamplingPhase::TOP_LEFT,
                (origin, 0),
            )
            .unwrap()
        };
        let shifted = make(1, 0, 3);
        assert_eq!(shifted.lattice_origin(), (1, 0));
        assert_eq!(shifted.level_bounds(), (0, 0, 3, 1));
        // Immutable bins [0,2),[2,4),[4,6) intersect samples at [1,5).
        assert_eq!(
            resident(&source, shifted, || false)
                .unwrap()
                .as_chunks::<4>().0.iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            [10., 25., 40.]
        );
        assert_eq!(
            resident(&source, make(1, 1, 1), || false).unwrap(),
            [25.; 4]
        );
        let negative = make(-1, 0, 3);
        assert_eq!(negative.level_bounds(), (-1, 0, 2, 1));
        assert_eq!(
            resident(&source, negative, || false)
                .unwrap()
                .as_chunks::<4>().0.iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            [10., 25., 40.]
        );
        // A temporal DW shift by one pixel changes enclosing bounds and means.
        let next_frame = make(2, 0, 2);
        assert_eq!(next_frame.level_bounds(), (1, 0, 3, 1));
        assert_eq!(
            resident(&source, next_frame, || false)
                .unwrap()
                .as_chunks::<4>().0.iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            [15., 35.]
        );
        assert!(
            RegionPlan::new(
                (4, 1),
                1,
                PixRect {
                    x: 0,
                    y: 0,
                    w: 3,
                    h: 1
                },
                RegionPolicy::OriginalArea,
                SamplingPhase::TOP_LEFT
            )
            .is_err()
        );
    }
    #[test]
    fn reflected_shifted_odd_height_uses_independent_bottom_cells() {
        let source: Vec<f32> = [0., 1., 2., 3., 4.]
            .into_iter()
            .flat_map(|v| [v; 4])
            .collect();
        let p = RegionPlan::on_lattice(
            (1, 5),
            1,
            PixRect {
                x: 0,
                y: 0,
                w: 1,
                h: 3,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::BOTTOM_LEFT,
            (0, 1),
        )
        .unwrap();
        assert_eq!(p.level_bounds(), (0, 0, 1, 3));
        // Bottom logical values [4,3,2,1,0] at [1,6): [4], [3,2], [1,0].
        assert_eq!(
            resident(&source, p, || false)
                .unwrap()
                .as_chunks::<4>().0.iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            [4., 2.5, 0.5]
        );
        let mut clipped = p;
        clipped.rect.y = 1;
        clipped.rect.h = 1;
        assert_eq!(resident(&source, clipped, || false).unwrap(), [2.5; 4]);
        let old = RegionPlan::new(
            (1, 5),
            1,
            PixRect {
                x: 0,
                y: 0,
                w: 1,
                h: 3,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::BOTTOM_LEFT,
        )
        .unwrap();
        assert_eq!(
            resident(&source, old, || false)
                .unwrap()
                .as_chunks::<4>().0.iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            [3.5, 1.5, 0.]
        );
    }
    #[test]
    fn lattice_limits_native_rejection_budget_and_cancellation_preflight() {
        let rect = PixRect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        };
        assert!(
            RegionPlan::on_lattice(
                (1, 1),
                0,
                rect,
                RegionPolicy::OriginalArea,
                SamplingPhase::TOP_LEFT,
                (i64::MAX, 0)
            )
            .is_err()
        );
        let min = RegionPlan::on_lattice(
            (1, 1),
            63,
            rect,
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
            (i64::MIN, i64::MIN),
        )
        .unwrap();
        assert_eq!(min.level_bounds(), (-1, -1, 0, 0));
        assert_eq!(resident(&[7.; 4], min, || false).unwrap(), [7.; 4]);
        let max = RegionPlan::on_lattice(
            (1, 1),
            63,
            rect,
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
            (i64::MAX - 1, i64::MAX - 1),
        )
        .unwrap();
        assert_eq!(max.level_bounds(), (0, 0, 1, 1));
        assert_eq!(resident(&[7.; 4], max, || false).unwrap(), [7.; 4]);
        assert!(
            RegionPlan::on_lattice(
                (1, 1),
                64,
                rect,
                RegionPolicy::OriginalArea,
                SamplingPhase::TOP_LEFT,
                (0, 0)
            )
            .is_err()
        );
        for (phase, origin) in [
            (SamplingPhase::TOP_LEFT, (1, 0)),
            (SamplingPhase::BOTTOM_LEFT, (0, 0)),
        ] {
            assert!(
                RegionPlan::on_lattice(
                    (1, 1),
                    1,
                    rect,
                    RegionPolicy::MatchingStoredNative,
                    phase,
                    origin
                )
                .is_err()
            );
        }
        let source = FrameSource {
            rgba: Arc::new(vec![1.; 16]),
            w: 4,
            h: 1,
        };
        let p = RegionPlan::on_lattice(
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
        ACCUMULATOR_BUILDS.with(|c| c.set(0));
        let budget = RegionBudget::new(1);
        assert!(
            source
                .region(p, &ProductionContext::default(), &budget)
                .is_err()
        );
        assert_eq!(budget.used(), 0);
        assert_eq!(ACCUMULATOR_BUILDS.with(|c| c.get()), 0);
        let mut calls = 0;
        assert!(
            resident(&source.rgba, p, || {
                calls += 1;
                calls == 2
            })
            .is_err()
        );
    }

    #[test]
    fn resident_stream_reuses_its_bounded_row_output() {
        let source = vec![1.0001; 3 * 5 * 4];
        let mut pointers = Vec::new();
        ACCUMULATOR_BUILDS.with(|c| c.set(0));
        resident_rows(
            &source,
            plan(SamplingPhase::TOP_LEFT),
            || false,
            |row| {
                pointers.push(row.as_ptr());
                assert_eq!(row.len(), 2 * 4);
            },
        )
        .unwrap();
        assert_eq!(pointers.len(), 3);
        assert_eq!(
            ACCUMULATOR_BUILDS.with(|c| c.get()),
            1,
            "resident stream must construct one accumulator, not one per row"
        );
        assert!(
            pointers.iter().all(|&p| p == pointers[0]),
            "row workspace must be reused inside actual resident streaming"
        );
    }

    #[test]
    fn full_precision_level_zero_preserves_bits_before_display_conversion() {
        use crate::tiles::source::TileSource;
        let bits = [0x80000000u32, 0x7fc12345, 0x3f800347, 0x7f800000];
        let source = FrameSource {
            rgba: Arc::new(bits.map(f32::from_bits).to_vec()),
            w: 1,
            h: 1,
        };
        let plan = RegionPlan::new(
            (1, 1),
            0,
            PixRect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            RegionPolicy::OriginalArea,
            SamplingPhase::TOP_LEFT,
        )
        .unwrap();
        let budget = RegionBudget::new(128);
        let provider: &dyn TileSource = &source;
        let output = provider
            .read_region(plan, &ProductionContext::default(), &budget)
            .unwrap();
        assert_eq!(
            output
                .rgba()
                .iter()
                .map(|p| p.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
    }

    #[test]
    fn odd_height_anchors_have_distinct_sampling_cells() {
        let rgba: Vec<f32> = (0..15)
            .flat_map(|i| [i as f32 / 7.0, 0.0, 0.0, 1.0])
            .collect();
        let top = resident(&rgba, plan(SamplingPhase::TOP_LEFT), || false).unwrap();
        let bottom = resident(&rgba, plan(SamplingPhase::BOTTOM_LEFT), || false).unwrap();
        // Independent explicit source indices, including each ragged column.
        let mean = |indices: &[usize]| {
            (indices.iter().map(|&i| f64::from(rgba[i * 4])).sum::<f64>() / indices.len() as f64)
                as f32
        };
        let top_expected = [
            mean(&[0, 1, 3, 4]),
            mean(&[2, 5]),
            mean(&[6, 7, 9, 10]),
            mean(&[8, 11]),
            mean(&[12, 13]),
            mean(&[14]),
        ];
        let bottom_expected = [
            mean(&[9, 10, 12, 13]),
            mean(&[11, 14]),
            mean(&[3, 4, 6, 7]),
            mean(&[5, 8]),
            mean(&[0, 1]),
            mean(&[2]),
        ];
        assert_eq!(
            top.as_chunks::<4>().0.iter().map(|p| p[0]).collect::<Vec<_>>(),
            top_expected
        );
        assert_eq!(
            bottom.as_chunks::<4>().0.iter().map(|p| p[0]).collect::<Vec<_>>(),
            bottom_expected
        );
        assert_ne!(
            bottom[0], top[16],
            "reversing top rows cannot repair odd-height phase"
        );
    }

    #[test]
    fn compound_attempt_peak_is_intrinsic_and_old_owners_are_pressure() {
        let budget = RegionBudget::new(128);
        let attempt = budget.begin_attempt().unwrap();
        let output = attempt.reserve(40).unwrap();
        assert!(matches!(attempt.reserve(25), Err(RegionError::Budget { needed: 65, limit: 64 })));
        assert_eq!(budget.used(), 40);
        assert!(matches!(budget.begin_attempt(), Err(RegionError::Pending)));
        drop(output);
        drop(attempt);
        let old = budget.reserve(80).unwrap();
        let candidate = budget.begin_attempt().unwrap();
        assert!(matches!(candidate.reserve(64), Err(RegionError::Pending)));
        drop(old);
        let live = candidate.reserve(64).unwrap();
        assert_eq!(budget.used(), 64);
        drop(live);
        drop(candidate);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn promotion_waits_for_external_owner_without_recharging_or_rerendering() {
        let budget = RegionBudget::new(128);
        let first = budget.begin_attempt().unwrap();
        let old = first.reserve(48).unwrap();
        first.promote().unwrap();
        let old_pin = first.clone();
        let candidate = budget.begin_attempt().unwrap();
        let completed = candidate.reserve(40).unwrap();
        assert!(matches!(candidate.promote(), Err(RegionError::Pending)));
        assert_eq!(budget.used(), 88);
        assert!(matches!(budget.begin_attempt(), Err(RegionError::Pending)));
        drop(first);
        assert!(matches!(candidate.promote(), Err(RegionError::Pending)));
        let epoch = budget.readiness_epoch();
        drop(old);
        drop(old_pin);
        assert!(budget.readiness_epoch() > epoch);
        candidate.promote().unwrap();
        assert_eq!(budget.used(), 40, "promotion is a tag transfer");
        let next = budget.begin_attempt().unwrap();
        let replacement = next.reserve(64).unwrap();
        assert_eq!(budget.used(), 104, "retained output has bounded replacement headroom");
        drop(replacement);
        drop(next);
        drop(completed);
        drop(candidate);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn optional_oversized_pool_does_not_coarsen_a_small_intrinsic_fit() {
        let budget = RegionBudget::new(128);
        drop(budget.workspace(96, 0).unwrap());
        let attempt = budget.begin_attempt().unwrap();
        let output = attempt.reserve(32).unwrap();
        let mut small = attempt.workspace(16, 0).unwrap();
        assert_eq!(small.buffers().0.capacity(), 16);
        assert_eq!(budget.used(), 48);
        assert_eq!(budget.reuse_hits(), 0, "oversized retained capacity was discarded");
        drop(small);
        drop(output);
        drop(attempt);
        budget.clear_workspace();
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn pooled_physical_charge_transfers_into_the_compound_attempt_once() {
        let budget = RegionBudget::new(128);
        drop(budget.workspace(8, 4).unwrap());
        assert_eq!(budget.used(), 24);
        let attempt = budget.begin_attempt().unwrap();
        let workspace = attempt.workspace(8, 4).unwrap();
        assert_eq!(budget.used(), 24);
        let output = attempt.reserve(40).unwrap();
        assert!(matches!(attempt.reserve(1), Err(RegionError::Budget { needed:65, limit:64 })));
        drop(workspace);
        let remainder = attempt.reserve(24).unwrap();
        assert_eq!(budget.used(), 88);
        drop(remainder);
        drop(output);
        drop(attempt);
        budget.clear_workspace();
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn retained_outputs_and_parallel_admission_count_until_last_owner() {
        let source = FrameSource {
            rgba: Arc::new(vec![1.0001; 3 * 5 * 4]),
            w: 3,
            h: 5,
        };
        let p = plan(SamplingPhase::TOP_LEFT);
        assert_eq!(
            p.rect(),
            PixRect {
                x: 0,
                y: 0,
                w: 2,
                h: 3
            }
        );
        assert_eq!(p.phase(), SamplingPhase::TOP_LEFT);
        let peak = p.output_bytes() + p.sums_bytes() + p.rect().w as usize * 16;
        let budget = RegionBudget::new(peak);
        let output = source
            .region(p, &ProductionContext::default(), &budget)
            .unwrap();
        assert_eq!(output.plan(), p);
        assert_eq!(output.rgba()[0], 1.0001);
        assert_ne!(
            output.rgba()[0],
            f32::from(imath_rs::Half::from(output.rgba()[0]))
        );
        let held = Arc::clone(&output);
        drop(output);
        assert_eq!(budget.used(), held.rgba.capacity() * 4);
        let source2 = source.clone();
        let b2 = budget.clone();
        assert!(
            std::thread::spawn(move || source2.region(p, &ProductionContext::default(), &b2))
                .join()
                .unwrap()
                .is_err()
        );
        drop(held);
        assert_eq!(budget.used(), 0);
        assert!(
            source
                .region(p, &ProductionContext::default(), &budget)
                .is_ok()
        );
    }
    #[test]
    fn cancellation_drops_admission_and_rejects_before_sampling() {
        let source = FrameSource {
            rgba: Arc::new(vec![1.0; 60]),
            w: 3,
            h: 5,
        };
        let budget = RegionBudget::new(1024);
        let context = ProductionContext::default();
        context.cancel();
        assert!(
            source
                .region(plan(SamplingPhase::TOP_LEFT), &context, &budget)
                .is_err()
        );
        assert_eq!(budget.used(), 0);
        let mut calls = 0;
        assert!(
            resident(&source.rgba, plan(SamplingPhase::TOP_LEFT), || {
                calls += 1;
                calls == 2
            })
            .is_err()
        );
        assert_eq!(calls, 2);
    }
    #[test]
    fn pooled_workspace_reuses_exact_allocations_and_remains_charged() {
        let budget = RegionBudget::new(512);
        let pointers;
        {
            let mut work = budget.workspace(128, 32).unwrap();
            let (raw, rgba) = work.buffers();
            pointers = (raw.as_ptr(), rgba.as_ptr());
            assert_eq!(budget.used(), raw.capacity() + rgba.capacity() * 4);
        }
        assert!(budget.used() > 0);
        {
            let mut work = budget.workspace(128, 32).unwrap();
            let (raw, rgba) = work.buffers();
            assert_eq!((raw.as_ptr(), rgba.as_ptr()), pointers);
            assert_eq!(budget.reuse_hits(), 1);
            assert!(
                budget.workspace(256, 64).is_err(),
                "concurrent scratch must be admitted too"
            );
        }
        budget.clear_workspace();
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn geometry_preflight_rejects_invalid_or_unaddressable_work() {
        assert!(pixel_count(0, 3).is_err());
        assert!(pixel_count(u32::MAX, u32::MAX).is_err());
        assert!(
            RegionPlan::new(
                (3, 5),
                1,
                PixRect {
                    x: u32::MAX,
                    y: 0,
                    w: 1,
                    h: 1
                },
                RegionPolicy::OriginalArea,
                SamplingPhase::TOP_LEFT
            )
            .is_err()
        );
        assert!(
            RegionPlan::new(
                (3, 5),
                64,
                PixRect {
                    x: 0,
                    y: 0,
                    w: 1,
                    h: 1
                },
                RegionPolicy::OriginalArea,
                SamplingPhase::TOP_LEFT
            )
            .is_err()
        );
    }
}
