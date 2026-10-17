//! The tile SOURCE seam: one fallible, thread-safe trait every tile producer implements (native `.exv`
//! container, a foreign linear image, a generator), plus the pull driver [`stream_tiles`].
//!
//! # Why
//! Out-of-core work never holds an image, it holds a *source* and pulls tiles. Before this module the same
//! "read tile (tx,ty), all channels" loop existed in five places with five error conventions (panic on a bad
//! index, `Option`, `String`); here a tile read is a [`TileResult`] and an out-of-range `(tx,ty)` is
//! [`TileError::Oob`], never a panic or a silently empty tile.

use std::fmt;
use std::sync::Arc;

/// A payload the pool can charge (`exv_format::bufpool::Weigh`; `Vec<u8>` and `PlaneBuf` are weighed there).
pub use exv_format::bufpool::Weigh;
use exv_format::{Dtype, TileGrid};

/// One error type for source, pool and budget (the pool's `exv_format::bufpool::PoolError` maps into it). `Clone` so a single-flight pool can hand the SAME failure to
/// every thread that waited on the load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileError {
    /// `(tx, ty)` is outside the `tiles_x x tiles_y` grid.
    Oob { tx: u32, ty: u32, tiles_x: u32, tiles_y: u32 },
    /// Channel index `ci` is outside the `n` channels of the source.
    Chan { ci: usize, n: usize },
    /// The codec refused the tile (corrupt blob, unknown codec, over a decode limit).
    Decode { tx: u32, ty: u32, ci: Option<usize>, why: Arc<str> },
    /// The backing store failed (read error, bad plane size).
    Io(Arc<str>),
    /// A P/B frame: its tiles reference another frame and are not independently decodable (FORMAT.md section 5).
    NotIntra { frame: u32 },
    /// A load needs `need` bytes but `used` of `cap` are already committed (and pinned): a hard error, never a
    /// silent over-commit.
    OverBudget { need: usize, used: usize, cap: usize },
    /// A size or device limit is exceeded (allocation guard, tile does not fit the GPU).
    Limit(Arc<str>),
    /// The caller cancelled the load.
    Cancelled,
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oob { tx, ty, tiles_x, tiles_y } => write!(f, "tile ({tx},{ty}) outside the {tiles_x}x{tiles_y} grid"),
            Self::Chan { ci, n } => write!(f, "channel {ci} out of range (source has {n})"),
            Self::Decode { tx, ty, ci: Some(ci), why } => write!(f, "tile ({tx},{ty}) channel {ci}: {why}"),
            Self::Decode { tx, ty, ci: None, why } => write!(f, "tile ({tx},{ty}): {why}"),
            Self::Io(why) => write!(f, "tile source I/O: {why}"),
            Self::NotIntra { frame } => write!(f, "frame {frame} is not INTRA: its tiles are not independently decodable"),
            Self::OverBudget { need, used, cap } => {
                write!(f, "tile needs {need} B but {used} of {cap} B are committed and pinned (raise the memory budget)")
            }
            Self::Limit(why) => write!(f, "limit exceeded: {why}"),
            Self::Cancelled => f.write_str("tile load cancelled"),
        }
    }
}

impl std::error::Error for TileError {}

/// The pool's own failures as tile errors (`?` on `Cancel::check` and every pool result).
impl From<exv_format::bufpool::PoolError> for TileError {
    fn from(e: exv_format::bufpool::PoolError) -> Self {
        match e {
            exv_format::bufpool::PoolError::OverBudget { need, used, cap } => Self::OverBudget { need, used, cap },
            exv_format::bufpool::PoolError::Cancelled => Self::Cancelled,
            exv_format::bufpool::PoolError::Panicked => Self::Io("tile loader panicked".into()),
        }
    }
}

/// The reverse mapping single-flight waiters need ("the leader was cancelled / could not evict": retry). A panicked
/// leader maps to `Io` and is shared with the waiters, never retried.
impl exv_format::bufpool::PoolErr for TileError {
    fn as_pool(&self) -> Option<exv_format::bufpool::PoolError> {
        match *self {
            Self::OverBudget { need, used, cap } => Some(exv_format::bufpool::PoolError::OverBudget { need, used, cap }),
            Self::Cancelled => Some(exv_format::bufpool::PoolError::Cancelled),
            _ => None,
        }
    }
}

impl From<TileError> for std::io::Error {
    fn from(e: TileError) -> Self {
        let kind = match e {
            TileError::Io(_) => std::io::ErrorKind::Other,
            TileError::Cancelled => std::io::ErrorKind::Interrupted,
            _ => std::io::ErrorKind::InvalidData,
        };
        std::io::Error::new(kind, e)
    }
}

/// Result of every tile operation.
pub type TileResult<T> = Result<T, TileError>;

/// All channels of one tile, in header channel order (`planes[ci]` = `tw*th*esz` bytes). Reusable: [`Self::clear`]
/// keeps the allocations so a streaming loop does not churn the allocator.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct TileBuf {
    pub planes: Vec<Vec<u8>>,
}

impl TileBuf {
    /// Payload bytes (what the pool charges).
    pub fn bytes(&self) -> usize {
        self.planes.iter().map(Vec::len).sum()
    }

    /// Empty every plane, keeping capacity.
    pub fn clear(&mut self) {
        self.planes.iter_mut().for_each(Vec::clear);
    }

    /// Capacity actually held (the honest peak of a reused buffer).
    pub fn cap(&self) -> usize {
        self.planes.iter().map(Vec::capacity).sum()
    }
}

impl Weigh for TileBuf {
    fn bytes(&self) -> usize {
        TileBuf::bytes(self)
    }
}

/// The tile rect `(x, y, w, h)` of `(tx, ty)`, or [`TileError::Oob`]. THE bounds guard of every source
/// ([`TileGrid::tile_rect`] saturates, so on its own it turns an out-of-range index into an empty/garbage rect).
pub fn check_tile(g: &TileGrid, tx: u32, ty: u32) -> TileResult<(u32, u32, u32, u32)> {
    let (tiles_x, tiles_y) = (g.tiles_x(), g.tiles_y());
    if tx >= tiles_x || ty >= tiles_y {
        return Err(TileError::Oob { tx, ty, tiles_x, tiles_y });
    }
    Ok(g.tile_rect(tx, ty))
}

/// Exact decoded bytes of tile `(tx, ty)` over `dtypes` (the pool's reservation cost).
pub fn tile_bytes_of(g: &TileGrid, dtypes: &[Dtype], tx: u32, ty: u32) -> TileResult<usize> {
    let (_, _, w, h) = check_tile(g, tx, ty)?;
    let px = w as u64 * h as u64;
    let esz: u64 = dtypes.iter().map(|d| d.size() as u64).sum();
    usize::try_from(px * esz).map_err(|_| TileError::Limit("tile bytes overflow usize".into()))
}

/// Yields decoded tiles on demand without materializing the image. `Send + Sync`: pool workers and prefetch
/// threads share one source through an `Arc<dyn TileSource>`.
pub trait TileSource: Send + Sync {
    /// Tile geometry.
    fn grid(&self) -> TileGrid;

    /// Per-channel element types (header order).
    fn dtypes(&self) -> &[Dtype];

    /// Whole tile, every channel, into `out` (planes resized to the exact tile size).
    ///
    /// # Errors
    /// [`TileError::Oob`] for a bad index, [`TileError::Decode`] / [`TileError::Io`] from the backend. Never panics.
    fn read_tile(&self, tx: u32, ty: u32, out: &mut TileBuf) -> TileResult<()>;

    /// ONE channel of a tile. The default decodes the whole tile and picks `ci` (right for HTJ2K, whose tile is
    /// one multi-component codestream); per-channel codecs override it to decode only `ci`.
    fn read_chan(&self, tx: u32, ty: u32, ci: usize, out: &mut Vec<u8>) -> TileResult<()> {
        let mut t = TileBuf::default();
        self.read_tile(tx, ty, &mut t)?;
        let n = t.planes.len();
        *out = t.planes.into_iter().nth(ci).ok_or(TileError::Chan { ci, n })?;
        Ok(())
    }

    /// Exact decoded bytes of tile `(tx, ty)`: the reservation cost a pool charges BEFORE decoding.
    fn tile_bytes(&self, tx: u32, ty: u32) -> TileResult<usize> {
        tile_bytes_of(&self.grid(), self.dtypes(), tx, ty)
    }
}

/// A foreign linear image (generator output, an EXR scanline buffer, test data): one row-major full-size plane
/// per channel, tiles are strided gathers of it.
pub struct PlanesSource<'a> {
    grid: TileGrid,
    dtypes: Vec<Dtype>,
    planes: Vec<&'a [u8]>,
}

impl<'a> PlanesSource<'a> {
    /// # Errors
    /// [`TileError::Io`] when the plane count differs from `dtypes` or any plane is not `width*height*esz` bytes
    /// (checked here once so [`TileSource::read_tile`] can never index out of bounds).
    pub fn new(grid: TileGrid, dtypes: Vec<Dtype>, planes: Vec<&'a [u8]>) -> TileResult<Self> {
        if planes.len() != dtypes.len() {
            return Err(TileError::Io(format!("{} planes for {} channels", planes.len(), dtypes.len()).into()));
        }
        for (ci, (p, d)) in planes.iter().zip(&dtypes).enumerate() {
            let want = grid.width as u64 * grid.height as u64 * d.size() as u64;
            if p.len() as u64 != want {
                return Err(TileError::Io(format!("plane {ci} is {} B, expected {want}", p.len()).into()));
            }
        }
        Ok(Self { grid, dtypes, planes })
    }
}

impl TileSource for PlanesSource<'_> {
    fn grid(&self) -> TileGrid {
        self.grid
    }

    fn dtypes(&self) -> &[Dtype] {
        &self.dtypes
    }

    fn read_tile(&self, tx: u32, ty: u32, out: &mut TileBuf) -> TileResult<()> {
        let (x, y, tw, th) = check_tile(&self.grid, tx, ty)?;
        out.planes.resize_with(self.planes.len(), Vec::new);
        for ((dst, src), d) in out.planes.iter_mut().zip(&self.planes).zip(&self.dtypes) {
            let esz = d.size();
            let row = tw as usize * esz;
            dst.clear();
            dst.reserve(row * th as usize);
            for r in 0..th {
                let s = ((y + r) as usize * self.grid.width as usize + x as usize) * esz;
                // `new` proved the plane is width*height*esz and `check_tile` the rect is inside the grid.
                dst.extend_from_slice(src.get(s..s + row).ok_or_else(|| TileError::Io("plane row out of range".into()))?);
            }
        }
        Ok(())
    }
}

/// What a [`stream_tiles`] pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamStats {
    pub tiles: u64,
    /// Capacity of the single reused buffer at the end: the proof the pass is `O(tile)`, not `O(image)`.
    pub peak_buf: usize,
}

/// Pull every tile in row-major order through ONE reused [`TileBuf`]; the first error (from the source or `f`)
/// aborts. Peak memory is one tile.
pub fn stream_tiles<S: TileSource + ?Sized>(
    src: &S,
    mut f: impl FnMut(u32, u32, &TileBuf) -> TileResult<()>,
) -> TileResult<StreamStats> {
    let g = src.grid();
    let mut buf = TileBuf::default();
    let mut st = StreamStats::default();
    for ty in 0..g.tiles_y() {
        for tx in 0..g.tiles_x() {
            src.read_tile(tx, ty, &mut buf)?;
            f(tx, ty, &buf)?;
            st.tiles += 1;
            st.peak_buf = st.peak_buf.max(buf.cap());
        }
    }
    Ok(st)
}

/// Tiles per parallel window for tiles that each hold `tile_bytes` in flight, from a byte budget: the ONE window
/// rule (replaces the fixed `ENCODE_WINDOW = 64` tile count). Capped at a few tiles per rayon thread: more only
/// adds resident bytes, not speed.
pub fn par_window(budget: u64, tile_bytes: u64) -> u64 {
    crate::budget::window_tiles(budget, tile_bytes, 4 * rayon::current_num_threads() as u64)
}

/// Ordered windowed pipeline: `work(0..n)` runs in parallel `win` items at a time, `sink` receives the results
/// strictly in index order (so a file writer or plane scatter stays deterministic for any thread count). Peak
/// memory = one window of results. The shared loop behind whole-image decode, region reads and tiled encode.
///
/// # Errors
/// The first `work` or `sink` error in index order aborts the pipeline.
pub fn pipe<T: Send, E: Send>(
    n: u64,
    win: u64,
    work: impl Fn(u64) -> Result<T, E> + Sync,
    mut sink: impl FnMut(u64, T) -> Result<(), E>,
) -> Result<(), E> {
    use rayon::prelude::*;
    let win = win.max(1);
    let mut t0 = 0u64;
    while t0 < n {
        let len = usize::try_from((n - t0).min(win)).unwrap_or(usize::MAX);
        // Collect every result, THEN walk them in index order: the reported error is the lowest-index one, not
        // whichever worker failed first.
        let batch = (0..len).into_par_iter().map(|i| work(t0 + i as u64)).collect::<Vec<Result<T, E>>>();
        for (i, v) in batch.into_iter().enumerate() {
            sink(t0 + i as u64, v?)?;
        }
        t0 += len as u64;
    }
    Ok(())
}
