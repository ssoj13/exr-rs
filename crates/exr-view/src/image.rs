//! Image backend: load an EXR into per-channel CPU buffers (source bit depth) +
//! drive the GPU display path (materialize the active layer to linear RGBA f32,
//! feed it to the tiled display pyramid, [`crate::tiles`], each frame).
//!
//! ## Layers
//!
//! An EXR can carry many channels beyond `R/G/B/A` (AOVs: `diffuse.R/G/B`,
//! `depth.Z`, `N.x/y/z`, `velocity.x/y`, …). They are grouped into Nuke-style
//! **layers** by [`exr_core::attr::group_layers`] (the shared dot-split rule).
//! The viewer keeps EVERY channel resident (in its source bit depth) and
//! *materializes* the user-selected layer into an interleaved linear RGBA f32
//! buffer on demand:
//!
//! - the **main** (unlayered) layer uses the classic `R/G/B/A`-by-name mapping
//!   with a `Y`/single-channel luminance fallback (so ordinary images are
//!   unchanged, bit-for-bit);
//! - a **named** layer packs its up-to-4 canonical components into RGBA
//!   (1→grey, 2→R,G, 3→R,G,B, 4→R,G,B,A; extra components beyond 4 are dropped).
//!
//! Switching the active layer re-materializes from the resident channels (a CPU
//! repack — never a disk re-read) and marks the GPU texture for re-upload.
//!
//! ## Split of responsibility
//!
//! [`DecodedFrame`] is the immutable, cacheable unit (`Arc`, `Send + Sync`): all
//! channels + the layer list + metadata. [`ImageBackend`] is the per-view display
//! object: it owns the GPU textures, the navigation state, the active-layer
//! selection, and the materialized RGBA buffer for the current layer. The colour
//! pass and the GPU scopes both read the SAME uploaded `linear_tex` (single
//! upload). `device`/`queue`/`renderer` are passed in (this viewer owns its wgpu
//! device directly, no eframe).

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use exr_core::ExrError;
use exr_core::api::ChannelData;
use exr_core::attr::{LayerGroup, PixelType, group_layers};
use exr_core::header::Header;
use exr_gpu::UploadLayout;
use imath_rs::Box2i;

use crate::tiles::file::{FileSource, RowCache};
use crate::tiles::render::{Look, TileCallback, TileGpu};
use crate::tiles::source::{FrameSource, TileSource};
use crate::tiles::view::{Placement, SourceDesc, TileState};
use crate::transfer;

/// Bytes per texel of the linear source texture (`Rgba32Float`) — the sizing
/// input for the shared [`UploadLayout`] check.
const LINEAR_TEXEL_SIZE: u32 = 16;

// ── Channel display ───────────────────────────────────────────────────────────

/// Channel display mode. Variants map 1-to-1 to the `CH_*` constants in
/// `crate::transfer::mode` (Rgba=0, R=1, G=2, B=3, A=4, Luminance=5).
///
/// This selects which component(s) of the **materialized active layer's** RGBA
/// the colour pass shows — it composes with the layer selection (e.g. pick layer
/// `N`, then isolate its X via [`ChannelDisplay::R`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelDisplay {
    #[default]
    Rgba,
    R,
    G,
    B,
    A,
    /// Rec.709 luminance as greyscale.
    Luminance,
}

impl ChannelDisplay {
    /// Short label for the toolbar buttons.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba => "RGBA",
            Self::R => "R",
            Self::G => "G",
            Self::B => "B",
            Self::A => "A",
            Self::Luminance => "Lum",
        }
    }

    /// Convert to the `channel_mode` u32 the GPU shader expects.
    pub fn to_channel_mode(self) -> u32 {
        match self {
            Self::Rgba => transfer::mode::RGBA,
            Self::R => transfer::mode::R,
            Self::G => transfer::mode::G,
            Self::B => transfer::mode::B,
            Self::A => transfer::mode::A,
            Self::Luminance => transfer::mode::LUM,
        }
    }
}

// ── Metadata ──────────────────────────────────────────────────────────────────

/// One header attribute for the Metadata panel: its name and its value as C++
/// `exrheader` prints it (`exr_core::api::describe`, the shared formatter).
#[derive(Debug, PartialEq)]
pub struct MetaEntry {
    pub key: String,
    pub value: String,
}

/// One stored sample, as the file holds it: `Half`/`Float` as a float, `Uint` as
/// the integer (an object id, a count) rather than the display's `v / u32::MAX`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sample {
    /// A `Half` or `Float` sample.
    F(f32),
    /// A `Uint` sample.
    U(u32),
}

impl std::fmt::Display for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F(v) => write!(f, "{v:.6}"),
            Self::U(v) => write!(f, "{v}"),
        }
    }
}

/// Sample `i` of `ch` as stored.
fn stored(ch: &ChannelData, i: usize) -> Option<Sample> {
    match ch {
        ChannelData::Uint(v) => v.get(i).map(|&u| Sample::U(u)),
        _ => ch.sample_f32(i).map(Sample::F),
    }
}

// ── DecodedFrame ──────────────────────────────────────────────────────────────

/// Immutable decoded frame: every channel in its source bit depth, the layer
/// grouping, and header metadata. This is the unit the frame cache stores
/// (`Arc<DecodedFrame>`), shared across threads (`Send + Sync`) — it holds NO GPU
/// handles, no per-view state, and no materialized RGBA (those live on
/// [`ImageBackend`]). Built by [`DecodedFrame::load`].
pub struct DecodedFrame {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    /// The EXR data window — the pixels this frame actually carries. `width` /
    /// `height` are its extent; `min` is where it sits in image space.
    pub data_window: Box2i,
    /// The EXR display window — the canvas the data window sits on. The viewport
    /// anchors on THIS, so a sequence whose data window changes per frame does
    /// not make the image jump; the pixel readout reports both.
    pub display_window: Box2i,
    /// Validated typed header pixel aspect ratio, retained for canonical processing.
    pub pixel_aspect: f32,
    /// EXR channel names, in header (alphabetical) order — same order as
    /// [`Self::channels`] and the indices in [`Self::layers`].
    pub channel_names: Vec<String>,
    /// Every channel's pixels, in source bit depth (`Half`/`Float`/`Uint`), in
    /// header order. Raster order over the data window (`width * height` samples).
    pub channels: Vec<(String, ChannelData)>,
    /// Channel grouping into layers (main-first); `LayerGroup::channels` are
    /// indices into [`Self::channels`].
    pub layers: Vec<LayerGroup>,
    pub metadata: Vec<MetaEntry>,
    /// Process-unique identity assigned at load. Part of the tile source identity
    /// (`SourceDesc`): a new frame starts a new tile generation;
    /// a monotonic id never aliases, unlike an `Arc` pointer reused after eviction.
    pub id: u64,
    /// `Some` for a FILE-BACKED frame (out-of-core tiling): too
    /// large to hold, so `channels` is empty and tiles/readout decode from the
    /// file on demand. The types of `channel_names`, in the same order.
    pub file_types: Option<Vec<PixelType>>,
}

/// What the viewer shows of a frame: one layer, or one channel alone in grey.
///
/// Shared by the toolbar, the channel tree and every frame of a sequence; a name
/// absent in the frame on screen falls back (channel -> its layer -> the main layer),
/// so scrubbing across frames with different channels never fails.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Selection {
    /// Layer name (`""` = the main, unlayered group).
    pub layer: String,
    /// A channel shown alone (its full name), `None` = the whole layer.
    pub channel: Option<String>,
}

impl Selection {
    /// The whole layer `name`.
    pub fn layer(name: &str) -> Self {
        Self { layer: name.to_owned(), channel: None }
    }

    /// Channel `name` of layer `layer`, alone.
    pub fn channel(layer: &str, name: &str) -> Self {
        Self { layer: layer.to_owned(), channel: Some(name.to_owned()) }
    }
}

/// Process-global monotonic source of [`DecodedFrame::id`].
static FRAME_ID: AtomicU64 = AtomicU64::new(0);

impl DecodedFrame {
    /// Open an EXR as a resident or file-backed CPU frame (no GPU).
    ///
    /// Resident single-grid files use `exr_core::api::read`, preserve channel
    /// source types, and group channels into layers. Deep and multipart inputs
    /// surface as [`LoadError::Unsupported`]. Subsampled channels are decoded at
    /// their stored sampling and scaled to the data window; luminance/chroma
    /// reconstruction also supplies RGB. Subsampled frames are always resident.
    /// Layer materialization and GPU upload happen later, UI-side.
    ///
    /// Open `path`: decoded whole when it fits `budget` bytes (channels plus the
    /// 16 B/px materialized RGBA), else FILE-BACKED — header only, pixels
    /// decoded per tile band (`tiles::file::FileSource`). An unreadable header
    /// uses the same decode path as [`Self::load`], whose typed error the UI shows.
    pub fn open(path: &Path, budget: u64) -> Result<DecodedFrame, LoadError> {
        if exr_import::is_foreign(path) {
            return Self::import(path);
        }
        match peek_header(path) {
            Some(header) if needs_file_backing(&header, budget) => Self::open_backed(path, &header),
            header => Self::decode(path, header.as_ref()),
        }
    }

    /// A file-backed frame from an already-read header.
    fn open_backed(path: &Path, header: &Header) -> Result<DecodedFrame, LoadError> {
        let bad = |e: ExrError| LoadError::from_exr(e);
        let data_window = header.data_window().map_err(bad)?;
        let display_window = header.display_window().map_err(bad)?;
        let width = (i64::from(data_window.max.x) - i64::from(data_window.min.x) + 1).max(0) as u32;
        let height = (i64::from(data_window.max.y) - i64::from(data_window.min.y) + 1).max(0) as u32;
        let list = header.channels().map_err(bad)?;
        let channel_names: Vec<String> = list.iter().map(|(n, _)| n.to_string()).collect();
        let types: Vec<PixelType> = list.iter().map(|(_, c)| c.type_).collect();
        let layers = group_layers(channel_names.iter().map(String::as_str).zip(types.iter().copied()));
        let metadata = metadata_of(header);
        log::info!(
            "open {}: {width}x{height}, file-backed ({} layers; tiles decode on demand)",
            path.display(),
            layers.len()
        );
        Ok(DecodedFrame {
            path: path.to_path_buf(),
            width,
            height,
            data_window,
            display_window,
            channel_names,
            channels: Vec::new(),
            layers,
            metadata,
            id: FRAME_ID.fetch_add(1, Ordering::Relaxed),
            file_types: Some(types),
            pixel_aspect: validated_aspect(header.pixel_aspect_ratio().map_err(LoadError::from_exr)?)?,
        })
    }

    /// Decode the whole file (every channel resident).
    pub fn load(path: &Path) -> Result<DecodedFrame, LoadError> {
        if exr_import::is_foreign(path) {
            return Self::import(path);
        }
        Self::decode(path, peek_header(path).as_ref())
    }

    /// A non-EXR still (`exr-import`: PNG, DPX, HDR, ...), always resident. It arrives as
    /// scene-linear associated-alpha channels, so everything downstream (exposure, OCIO, scopes,
    /// readout, effects, Save view, sequences) sees a frame like any EXR's. The source colour
    /// space, depth and alpha handling head the Metadata panel.
    fn import(path: &Path) -> Result<DecodedFrame, LoadError> {
        log::info!("load start {} (exr-import)", path.display());
        let loaded = exr_import::load(path).map_err(|e| LoadError::Read(e.to_string()))?;
        let image = loaded.image;
        let (data_window, display_window) = (image.data_window, image.display_window);
        let (width, height) = (image.width().max(0) as u32, image.height().max(0) as u32);
        let channels: Vec<(String, ChannelData)> = image.into_channels().map_err(LoadError::from_exr)?.collect();
        let layers = group_layers(channels.iter().map(|(n, c)| (n.as_str(), c.pixel_type())));
        log::info!("load ok {}: {width}x{height} ({} layers, imported)", path.display(), layers.len());
        Ok(DecodedFrame {
            path: path.to_path_buf(),
            width,
            height,
            data_window,
            display_window,
            channel_names: channels.iter().map(|(n, _)| n.clone()).collect(),
            channels,
            layers,
            metadata: loaded.meta.into_iter().map(|(key, value)| MetaEntry { key, value }).collect(),
            id: FRAME_ID.fetch_add(1, Ordering::Relaxed),
            file_types: None,
            pixel_aspect: 1.0,
        })
    }

    /// A subsampled file: every channel read at its own sampling
    /// (`exr_core::api::planes`) and scaled up to the data window by the sample
    /// covering each pixel; a luminance-chroma file (`Y`, `RY`, `BY`) also gets the
    /// `R`, `G`, `B` (and `A` when it has none) of C++ `RgbaInputFile`'s
    /// reconstruction (27-tap chroma filter, bit-exact with C++), so the main layer
    /// shows colour while `Y`/`RY`/`BY` stay in the channel tree. Always resident
    /// (never file-backed: the channels share no sample grid).
    fn decode_subsampled(path: &Path, header: Option<&Header>) -> Result<DecodedFrame, LoadError> {
        let bad = |e: ExrError| LoadError::from_exr(e);
        let header = header.ok_or_else(|| LoadError::Read(format!("{}: header unreadable", path.display())))?;
        let dw = header.data_window().map_err(bad)?;
        let min = (dw.min.x, dw.min.y);
        let width = (i64::from(dw.max.x) - i64::from(dw.min.x) + 1).max(0) as u32;
        let height = (i64::from(dw.max.y) - i64::from(dw.min.y) + 1).max(0) as u32;

        let p = path.to_str().ok_or_else(|| LoadError::Read(format!("path is not valid UTF-8: {}", path.display())))?;
        let mut stream = exr_core::io::StdIfStream::open(p).map_err(bad)?;
        let mut file = exr_core::scanline::InputFile::new(&mut stream).map_err(bad)?;
        let planes = exr_core::api::planes::planes(&mut file).map_err(bad)?;
        let mut channels = Vec::with_capacity(planes.len() + 4);
        for plane in &planes {
            channels.push((plane.name.clone(), upsample(plane, min, width, height)?));
        }
        if planes.iter().any(|p| p.name == "Y") {
            let rgba = exr_core::rgba::RgbaInputFile::read(path).map_err(bad)?;
            let n = width as usize * height as usize;
            if rgba.pixels.len() != n {
                return Err(LoadError::Read(format!("RGBA reconstruction has {} pixels, expected {n}", rgba.pixels.len())));
            }
            let has = |name: &str| planes.iter().any(|p| p.name == name);
            for (c, name) in ["R", "G", "B", "A"].into_iter().enumerate() {
                if !has(name) {
                    let plane = rgba.pixels.iter().map(|p| [p.r, p.g, p.b, p.a][c]).collect();
                    channels.push((name.to_owned(), ChannelData::Half(plane)));
                }
            }
        }
        // Header order (byte-wise by name), like every other frame.
        channels.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

        let channel_names: Vec<String> = channels.iter().map(|(n, _)| n.clone()).collect();
        let layers = group_layers(channels.iter().map(|(n, c)| (n.as_str(), c.pixel_type())));
        log::info!("load ok {}: {width}x{height} subsampled ({} layers)", path.display(), layers.len());
        Ok(DecodedFrame {
            path: path.to_path_buf(),
            width,
            height,
            data_window: dw,
            display_window: header.display_window().map_err(bad)?,
            channel_names,
            channels,
            layers,
            metadata: metadata_of(header),
            id: FRAME_ID.fetch_add(1, Ordering::Relaxed),
            file_types: None,
            pixel_aspect: validated_aspect(header.pixel_aspect_ratio().map_err(LoadError::from_exr)?)?,
        })
    }

    /// [`Self::load`] with the header already read (`None`: unreadable as a
    /// single flat part; the decode below then reports why).
    fn decode(path: &Path, header: Option<&Header>) -> Result<DecodedFrame, LoadError> {
        // Lifecycle milestone (info): a load is starting (the #1 thing to see).
        log::info!("load start {}", path.display());
        let img = match exr_core::api::read(path) {
            Ok(img) => img,
            // The facade takes one sample grid; a subsampled file is read as stored.
            Err(ExrError::Unsupported(exr_core::api::Unsupported::Subsampled)) => {
                return Self::decode_subsampled(path, header);
            }
            Err(e) => {
                let err = LoadError::from_exr(e);
                // Surface the exact reason (deep/multipart escape hatch, I/O, …).
                log::warn!("load failed {}: {err}", path.display());
                return Err(err);
            }
        };

        let width = img.width().max(0) as u32;
        let height = img.height().max(0) as u32;
        let channel_names: Vec<String> = img.channels().map(|(n, _)| n.to_owned()).collect();
        // The facade image does not keep every attribute (it drops the derived and
        // structural ones), so the panel reads the file's own header. A file the
        // facade decodes is one `peek_header` reads, so `None` here is not expected.
        let metadata = match header {
            Some(h) => metadata_of(h),
            None => {
                log::warn!("{}: header unreadable for the Metadata panel", path.display());
                Vec::new()
            }
        };
        let pixel_aspect = validated_aspect(
            img.attributes().get("pixelAspectRatio")
                .and_then(|a| a.as_any().downcast_ref::<exr_core::attr::FloatAttribute>())
                .ok_or_else(|| LoadError::Read("pixelAspectRatio is missing or not a float attribute".into()))?.value
        )?;
        let compression = img.compression;
        let data_window = img.data_window;
        let display_window = img.display_window;

        // Take ownership of the channels instead of cloning every pixel
        // buffer (`img` is a freshly-read, about-to-be-dropped Image, so
        // `into_channels` moving its BTreeMap entries out directly is free
        // where the old `img.channels().map(|(n, c)| (n.to_owned(),
        // c.clone()))` cloned the whole image just to move it here). Keeps
        // every channel in its source bit depth (header order). `api::read`
        // rejects subsampled files, so `into_channels` (which refuses a
        // subsampled channel rather than drop its sampling) cannot fail here;
        // the error still propagates instead of being assumed away.
        let channels: Vec<(String, ChannelData)> = img.into_channels().map_err(LoadError::from_exr)?.collect();
        // Group into layers via the shared exr-core rule (indices into `channels`).
        let layers = group_layers(channels.iter().map(|(n, c)| (n.as_str(), c.pixel_type())));

        // Structural detail (debug): channels + layer count.
        log::debug!(
            "api::read ok {}: {width}x{height} channels=[{}] layers={} compression={:?}",
            path.display(),
            channel_names.join(" "),
            layers.len(),
            compression
        );
        // User-meaningful milestone (info): the frame finished loading.
        log::info!(
            "load ok {}: {width}x{height} ({} layers)",
            path.display(),
            layers.len()
        );

        let id = FRAME_ID.fetch_add(1, Ordering::Relaxed);
        Ok(DecodedFrame {
            path: path.to_path_buf(),
            width,
            height,
            data_window,
            display_window,
            channel_names,
            channels,
            layers,
            metadata,
            id,
            file_types: None,
            pixel_aspect,
        })
    }

    /// Approximate resident CPU byte size — the sum of all channel buffers in
    /// their source bit depth. Used by the frame cache's memory budget; it MUST
    /// count every channel (not just the materialized RGBA), or a many-AOV
    /// sequence would under-report and blow the RAM cap.
    pub fn byte_size(&self) -> usize {
        self.channels
            .iter()
            .map(|(_, c)| c.len() * exr_core::misc::pixel_type_size(c.pixel_type()).unwrap_or(0))
            .sum()
    }

    /// File name for titles/status.
    pub fn file_name(&self) -> &str {
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(image)")
    }

    /// Resolve a requested layer name to one of this frame's layers. An empty
    /// name (or a name absent in this frame — e.g. while scrubbing a sequence
    /// whose frames differ) falls back to the main/first layer, deterministically.
    fn resolve_layer(&self, name: &str) -> Option<&LayerGroup> {
        self.layers
            .iter()
            .find(|g| g.name == name)
            .or_else(|| self.layers.first())
    }

    /// The group `sel` shows: a solo channel is a one-channel group named after
    /// the channel, which [`Mapping::of`] broadcasts to grey - the same rule a
    /// one-channel layer (`depth.Z`) follows, so a solo needs no mapping of its
    /// own. A channel absent in this frame falls back to its layer.
    pub(crate) fn resolve(&self, sel: &Selection) -> Option<Cow<'_, LayerGroup>> {
        if let Some(name) = &sel.channel
            && let Some(i) = self.channel_names.iter().position(|n| n == name)
            && let Some(pixel_type) = self.channel_type(i)
        {
            return Some(Cow::Owned(LayerGroup { name: name.clone(), channels: vec![i], pixel_type }));
        }
        self.resolve_layer(&sel.layer).map(Cow::Borrowed)
    }

    /// The pixel type of channel `i` ([`Self::channel_names`] order).
    fn channel_type(&self, i: usize) -> Option<PixelType> {
        match &self.file_types {
            Some(types) => types.get(i).copied(),
            None => self.channels.get(i).map(|(_, c)| c.pixel_type()),
        }
    }

    /// The pixel type of every channel, in [`Self::channel_names`] order.
    pub fn channel_types(&self) -> Vec<PixelType> {
        match &self.file_types {
            Some(types) => types.clone(),
            None => self.channels.iter().map(|(_, c)| c.pixel_type()).collect(),
        }
    }

    /// The two EXR windows in the form the viewport and the pixel readout need.
    ///
    /// A degenerate display window (`min > max`, which a malformed header can
    /// carry) falls back to the data window, so the viewport always has a finite
    /// canvas to anchor on instead of dividing by a zero extent.
    pub fn windows(&self) -> Windows {
        let data_min = (self.data_window.min.x, self.data_window.min.y);
        let d = self.display_window;
        let dw = i64::from(d.max.x) - i64::from(d.min.x) + 1;
        let dh = i64::from(d.max.y) - i64::from(d.min.y) + 1;
        if dw > 0 && dh > 0 && dw <= i64::from(u32::MAX) && dh <= i64::from(u32::MAX) {
            Windows {
                display_min: (d.min.x, d.min.y),
                display_size: (dw as u32, dh as u32),
                data_min,
            }
        } else {
            Windows {
                display_min: data_min,
                display_size: (self.width, self.height),
                data_min,
            }
        }
    }
}

/// The data/display window geometry the viewport anchors on and the pixel
/// readout reports. Pixel coordinates are EXR image space (the data window's
/// `min` is NOT necessarily the origin).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Windows {
    /// Display-window origin — the canvas the image is composited onto.
    pub display_min: (i32, i32),
    /// Display-window extent in pixels (always non-zero).
    pub display_size: (u32, u32),
    /// Data-window origin — where the decoded pixels sit inside the canvas.
    pub data_min: (i32, i32),
}

// ── Display state ─────────────────────────────────────────────────────────────

/// Most pixels a scope scatters from: the
/// scopes read the finest pyramid level within this many pixels. 16 M keeps a
/// 4K x 4K image exact at level 0.
const SCOPE_MAX_PIXELS: u64 = 16 << 20;

/// Which display path drew the last frame. Shown in the status bar: a float
/// sequence silently falling back to CPU tiles is what made playback stutter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayPath {
    /// One resident GPU texture, reduced on the GPU (`tiles::direct`): `exact` =
    /// the global fixed-point kernel, else the per-cell window (`main_wide`).
    Direct { exact: bool },
    /// The tile pyramid produced by CPU workers from `feed`.
    Tiles(TileFeed),
}

/// Where the tile path's pixels come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFeed {
    /// The decoded frame in RAM.
    Ram,
    /// An OpenFX effect's output.
    Effect,
    /// The file itself, decoded per tile band (a frame too large to hold).
    File,
}

impl DisplayPath {
    /// Short status-bar label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Direct { exact: true } => "direct · exact",
            Self::Direct { exact: false } => "direct · wide",
            Self::Tiles(TileFeed::Ram) => "tiles",
            Self::Tiles(TileFeed::Effect) => "tiles · effect",
            Self::Tiles(TileFeed::File) => "tiles · file",
        }
    }

    /// Status-bar hover text: what the path means.
    pub fn hint(self) -> &'static str {
        match self {
            Self::Direct { exact: true } => "One GPU texture, reduced on the GPU (exact fixed-point mean).",
            Self::Direct { exact: false } => "One GPU texture, reduced on the GPU (per-cell window: wide-range float data).",
            Self::Tiles(TileFeed::Ram) => "512 px tile pyramid built by CPU workers from the frame in RAM.",
            Self::Tiles(TileFeed::Effect) => "512 px tile pyramid of the OpenFX effect's output.",
            Self::Tiles(TileFeed::File) => "512 px tile pyramid decoded from the file band by band (frame too large for RAM).",
        }
    }
}

/// The app-owned display state, reused across every displayed frame: the tile
/// path's CPU side (source, worker results, RAM cache) and the scopes' linear
/// source texture. The tile atlas and pipeline live in egui's callback
/// resources ([`TileGpu`]), where the paint callback can bind them.
pub struct DisplayTarget {
    tiles: TileState,
    scope: Option<ScopeTex>,
    direct_frame: Option<u64>,
    blocked_frame: Option<u64>,
    retired_scopes: crate::tiles::retire::RetiredTextures,
    /// Tile bytes uploaded per frame at most (`persist::Budgets::upload_mib`).
    pub upload: usize,
    /// The path that drew the last frame; `None` before the first one.
    path: Option<DisplayPath>,
}

/// The scopes' linear `Rgba32Float` source: one pyramid level of the shown
/// content, grow-only like the old display target.
struct ScopeTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    cap: (u32, u32),
    /// `(processed, source generation or accepted revision, level)` uploaded.
    key: Option<(bool, u64, u32)>,
}

impl DisplayTarget {
    #[cfg(feature = "ofx")]
    pub(crate) fn accepted_effect(&self) -> Option<crate::tiles::view::AcceptedEffect> {
        self.tiles.accepted_effect()
    }

    #[cfg(feature = "ofx")]
    pub(crate) fn accepted_pixels(&self) -> Option<crate::consumers::AcceptedPixels> {
        self.tiles.accepted_pixels()
    }

    #[cfg(feature = "ofx")]
    fn ensure_processed_linear(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixels: crate::consumers::AcceptedPixels,
    ) -> Option<(&wgpu::TextureView, u32, u32)> {
        let max_dim = device.limits().max_texture_dimension_2d;
        let level = pixels.level(SCOPE_MAX_PIXELS, max_dim);
        let (w, h) = pixels.grid().level_size(level);
        let key = (true, pixels.revision(), level);
        if self.scope.as_ref().is_some_and(|scope| scope.key == Some(key)) {
            return self.scope.as_ref().map(|scope| (&scope.view, w, h));
        }
        // A failed admission consumes the temporary snapshot, so a scope never
        // waits for scratch while pinning a previous presentation.
        let mut rows = match crate::consumers::PixelRows::new(pixels, level) {
            Ok(rows) => rows,
            Err(crate::tiles::region::RegionError::Pending) => return None,
            Err(error) => {
                log::error!("processed scope source: {error}");
                return None;
            }
        };
        let (w, h) = rows.size();
        self.retired_scopes.reap();
        if !self.scope.as_ref().is_some_and(|scope| fits(scope.cap, (w, h))) {
            let next = ScopeTex::new(device, w, h, max_dim)?;
            if let Some(old) = self.scope.replace(next) {
                let bytes = UploadLayout::compute(old.cap.0, old.cap.1, LINEAR_TEXEL_SIZE, max_dim).ok()?.byte_size;
                self.retired_scopes.retire(queue, old.tex, bytes);
            }
        }
        let scope = self.scope.as_mut()?;
        scope.key = None;
        for y in 0..h {
            let row = match rows.row(y) {
                Ok(row) => row,
                Err(error) => {
                    log::error!("processed scope row: {error}");
                    return None;
                }
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &scope.tex, mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(row),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: None, rows_per_image: None },
                wgpu::Extent3d { width: w, height: 1, depth_or_array_layers: 1 },
            );
        }
        scope.key = Some(key);
        Some((&scope.view, w, h))
    }

    /// Whether playback may replace this frame's backend without cancelling an
    /// unfinished viewport presentation.
    pub(crate) fn can_advance(&self, frame: u64) -> bool {
        self.blocked_frame == Some(frame)
            || self.direct_frame == Some(frame)
            || self.tiles.can_advance(frame)
    }

    /// Drop the held viewport content when deliberately opening another file.
    pub(crate) fn clear_presentation(&mut self) {
        self.direct_frame = None;
        self.blocked_frame = None;
        self.tiles.clear_presentation();
    }
    /// A display state whose RAM tile cache holds at most `ram_budget` bytes.
    pub fn new(ram_budget: usize, upload: usize) -> Self {
        Self {
            tiles: TileState::new(ram_budget),
            scope: None,
            direct_frame: None,
            blocked_frame: None,
            retired_scopes: Default::default(),
            upload,
            path: None,
        }
    }

    /// The path that drew the last frame (the status bar); `None` before the first.
    pub fn path(&self) -> Option<DisplayPath> {
        self.path
    }

    /// Reserve the existing scope allocation and any replacement needed for
    /// this source. The old texture may still be referenced by submitted work.
    fn scope_reserve(&self, max_dim: u32) -> u64 {
        let current = self.scope.as_ref().map_or(0, |s| {
            UploadLayout::compute(s.cap.0, s.cap.1, LINEAR_TEXEL_SIZE, max_dim)
                .map_or(u64::MAX, |l| l.byte_size)
        });
        let current = current.saturating_add(self.retired_scopes.bytes());
        let Some(grid) = self.tiles.grid() else {
            return current;
        };
        let level = crate::tiles::source::finest_level(grid, SCOPE_MAX_PIXELS, max_dim);
        let size = grid.level_size(level);
        if self.scope.as_ref().is_some_and(|s| fits(s.cap, size)) {
            return current;
        }
        current.saturating_add(
            UploadLayout::compute(size.0, size.1, LINEAR_TEXEL_SIZE, max_dim)
                .map_or(u64::MAX, |l| l.byte_size),
        )
    }

    /// Wake the UI loop when a tile worker finishes (see `TileState::set_wake`).
    pub fn set_wake(&mut self, wake: crate::tiles::view::RegionWake) {
        self.tiles.set_wake(wake);
    }

    /// Tiles still being produced (the viewport keeps repainting until 0).
    pub fn tiles_pending(&self) -> usize {
        self.tiles.pending()
    }

    #[cfg(test)]
    pub(crate) fn install_test_source(
        &mut self,
        image: &ImageBackend,
        source: Arc<dyn TileSource>,
    ) {
        self.tiles.set_source(
            SourceDesc {
                frame: image.frame.id,
                view: image.view.clone(),
                pixels_gen: image.pixels_gen,
                stage: 0,
            },
            source,
        );
    }

    #[cfg(test)]
    pub(crate) fn scheduled_units(&self) -> usize {
        self.tiles.scheduled_units
    }

    /// RAM held by the tile cache, and its budget.
    pub fn tiles_ram(&self) -> (usize, usize) {
        self.tiles.ram()
    }
}

/// Can textures allocated at `cap` display a `need`-sized image without churn?
///
/// Grow-only with a shrink escape hatch: the extent must cover the image, and the
/// allocation must not be wildly oversized (>4x the pixels), so switching from a
/// large level to a small proxy does give the VRAM back instead of holding the
/// high-water mark for the session.
fn fits(cap: (u32, u32), need: (u32, u32)) -> bool {
    if cap.0 < need.0 || cap.1 < need.1 {
        return false;
    }
    let have = u64::from(cap.0) * u64::from(cap.1);
    let want = u64::from(need.0) * u64::from(need.1);
    have <= want.saturating_mul(4)
}

// ── ImageBackend ──────────────────────────────────────────────────────────────

/// UI display object for a decoded EXR frame: a shared [`DecodedFrame`] plus
/// navigation, the active-layer selection, and the materialized RGBA buffer for
/// that layer. The `frame` field is the cacheable unit; nav and the materialized
/// buffer are per-display-instance. GPU state lives in the shared, persistent
/// `DisplayTarget` and the tile atlas (owned by the app, reused across
/// frames) — NOT here — so a sequence frame swap rebuilds only cheap CPU state.
pub struct ImageBackend {
    /// The immutable decoded frame (all channels + layers + metadata). Shared via
    /// `Arc` so the cache can hold `Arc<DecodedFrame>` without duplicating data.
    pub frame: Arc<DecodedFrame>,
    /// Cached copy of `frame.width` — avoids indirection in hot viewport code.
    pub width: u32,
    /// Cached copy of `frame.height` — avoids indirection in hot viewport code.
    pub height: u32,

    /// Which component(s) of the active layer's RGBA the colour pass shows.
    pub channel_display: ChannelDisplay,
    /// What is shown (layer or solo channel). Applied each frame from the shared
    /// `ViewControls`; names absent in this frame fall back ([`Selection`]).
    view: Selection,
    /// Optional Cryptomatte coverage overlay (`width * height` samples). When set,
    /// `prepare_cpu` replaces the materialized RGB with greyscale coverage (A=1).
    coverage_overlay: Option<Vec<f32>>,
    /// Identity of the overlay currently requested: `(crypto_layer, object_name)`.
    /// Used to skip re-extract when the selection has not changed after a rematerialize.
    coverage_sel: Option<(String, String)>,
    /// A selection whose coverage extraction FAILED for this frame. Without it,
    /// the failure path cleared `coverage_sel`, so the next UI frame saw
    /// `want != sel`, retried, re-materialized and re-uploaded the whole image
    /// (~132 MB at 4K) — every repaint, forever.
    coverage_failed: Option<(String, String)>,
    /// Bumped whenever baked `pixels` content may change without a layer rename
    /// (coverage overlay set/clear). Part of the tile source identity ([`SourceDesc`]).
    pixels_gen: u64,
    /// The layer name actually materialized into `pixels` (the dirty marker:
    /// `None` = nothing materialized yet, re-materialize on next present).
    materialized: Option<Selection>,
    /// Interleaved linear RGBA f32 of the materialized active layer (`w*h*4`).
    /// The tile source and the pixel-readout source. Built lazily. Shared with
    /// the tile workers (`Arc`); re-materializing reuses the allocation unless a
    /// worker still holds it.
    pixels: Arc<Vec<f32>>,
    /// What each RGBA slot of `pixels` holds, as `(channel name, slot)` rows for
    /// the pixel readout: the real channel names (`Z`, `N.x`), not fixed
    /// `R/G/B/A`. Rebuilt with `pixels` (see [`readout_rows`]).
    readout: Vec<(String, usize)>,
    /// A file-backed frame's tile source for the active layer (the readout
    /// reads through it); `None` for a resident frame.
    file_source: Option<Arc<FileSource>>,
    /// Every channel's rows of a FILE-BACKED frame, for [`Self::probe_at`] (a
    /// resident frame reads its channels directly).
    probe: Option<RowCache>,
    /// A file-backed frame's Cryptomatte overlay: the ranked layers and the
    /// object id whose coverage the tiles show (the resident frame keeps the
    /// computed plane in `coverage_overlay` instead).
    coverage_file: Option<(Vec<cryptomatte::RankedLayer>, f32)>,

    // Viewport navigation
    pub pan: egui::Vec2,
    pub zoom: f32,
    pub fit_zoom: f32,
    pub picked_pixel: Option<(u32, u32)>,
}

/// A stage between the materialized linear pixels and the display: what the viewer shows instead
/// of the decoded image (its OpenFX effects, feature `ofx`), and what it draws over the image.
/// The viewer knows nothing of what the stage does; it uploads what [`Self::process`] returns and
/// re-uploads whenever [`Self::key`] changes.
pub trait ImageStage {
    /// What [`Self::process`] would make now of the same input (the effect, its parameters, the
    /// time): part of the upload key, so a change re-processes and re-uploads.
    fn key(&self) -> u64;

    /// The pixels to show for `pixels` (`w * h * 4` linear RGBA, top row first): the stage's own
    /// result, or `pixels` themselves when it has none (its failure is its own to report).
    fn process<'a>(&'a mut self, pixels: &'a [f32], _w: u32, _h: u32) -> &'a [f32] { pixels }

    /// An immutable stage runs on the existing tile workers.
    #[cfg(feature = "ofx")]
    fn uses_worker(&self) -> bool { false }

    #[cfg(feature = "ofx")]
    fn worker(&self, _frame: Arc<DecodedFrame>, _view: Selection,
        _budget: crate::tiles::region::RegionBudget) -> Result<Option<Arc<dyn TileSource>>, String> { Ok(None) }

    #[cfg(feature = "ofx")]
    fn worker_wake(&self) -> Option<Arc<dyn Fn() + Send + Sync>> { None }

    #[cfg(feature = "ofx")]
    fn accepted(&mut self, _shown: Option<crate::tiles::view::AcceptedEffect>) {}

    /// Why the stage cannot run on the shown image (`None`: it can). The viewer
    /// does not call [`Self::process`] while blocked; the stage reports the
    /// reason in its own UI.
    fn set_blocked(&mut self, _reason: Option<String>) {}

    /// Whether the stage takes the viewport's primary pointer (an effect overlay): the viewer's own
    /// left-button actions (pixel pick, scrub) are then off.
    fn takes_pointer(&self) -> bool {
        false
    }

    /// Draw over the image shown in `image_rect` (`w` x `h` pixels) and take the viewport's pointer
    /// input (`response`); `true` when the stage changed and the image must be processed again.
    fn overlay(
        &mut self,
        _painter: &egui::Painter,
        _response: &egui::Response,
        _image_rect: egui::Rect,
        _size: (u32, u32),
    ) -> bool {
        false
    }
}

impl ImageBackend {
    /// Approximate resident CPU byte size (delegates to the cached frame, which
    /// counts all channels). The per-view materialized `pixels` are NOT counted —
    /// only the displayed backend holds one, not every cached frame.
    pub fn byte_size(&self) -> usize {
        self.frame.byte_size()
    }

    /// Load an EXR file into a CPU backend (no GPU yet).
    ///
    /// Deep / multipart files surface as a clear [`LoadError::Unsupported`] so the
    /// UI can show a status message — never a panic.
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        Ok(Self::from_frame(Arc::new(DecodedFrame::load(path)?)))
    }

    /// Build a UI display object over a decoded frame (fresh nav, no GPU yet, no
    /// materialized buffer). Used for a single image and for each sequence frame.
    /// Materialization is deferred to the first `present`/`ensure_linear` so the
    /// active layer (carried by `ViewControls`, applied per frame) is honoured
    /// without an extra repack.
    pub fn from_frame(frame: Arc<DecodedFrame>) -> Self {
        let (width, height) = (frame.width, frame.height);
        let probe = probe_rows(&frame);
        Self {
            frame,
            width,
            height,
            channel_display: ChannelDisplay::default(),
            view: Selection::default(),
            coverage_overlay: None,
            coverage_sel: None,
            coverage_failed: None,
            pixels_gen: 0,
            materialized: None,
            pixels: Arc::new(Vec::new()),
            readout: Vec::new(),
            file_source: None,
            probe,
            coverage_file: None,
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            fit_zoom: 0.0,
            picked_pixel: None,
        }
    }

    #[cfg(feature = "ofx")]
    pub(crate) fn decoded(&self) -> Arc<DecodedFrame> { self.frame.clone() }

    pub(crate) fn display_aspect(&self) -> f32 {
        let aspect = self.frame.pixel_aspect;
        if aspect.is_finite() && aspect > 0.0
            && (self.windows().display_size.0 as f32 * aspect).is_finite() { aspect }
        else { 1.0 }
    }

    /// File name for window titles / status.
    pub fn file_name(&self) -> &str {
        self.frame.file_name()
    }

    /// The frame's data/display window geometry (see `Windows`).
    pub fn windows(&self) -> Windows {
        self.frame.windows()
    }

    /// Swap in the next sequence frame, KEEPING this backend's materialized
    /// buffer allocation and navigation.
    ///
    /// This is why a sequence step is cheap: `pixels` is a `w*h*4` f32 buffer
    /// (~132 MB at 4K) that used to be freed and re-allocated for every displayed
    /// frame — at 24 fps, ~3 GB/s of allocator traffic on the UI thread. Re-using
    /// the backend keeps the capacity and only re-fills it. The Cryptomatte
    /// overlay is cleared because coverage is per-frame (the caller re-extracts
    /// it for the new frame).
    pub fn set_frame(&mut self, frame: Arc<DecodedFrame>) {
        let resized = (frame.width, frame.height) != (self.width, self.height);
        self.width = frame.width;
        self.height = frame.height;
        self.probe = probe_rows(&frame);
        self.frame = frame;
        self.coverage_overlay = None;
        self.coverage_sel = None;
        // A new frame is a new chance: its channels may differ from the one that
        // failed.
        self.coverage_failed = None;
        self.coverage_file = None;
        self.materialized = None; // re-materialize into the retained buffer
        self.pixels_gen = self.pixels_gen.wrapping_add(1);
        if resized {
            // A different resolution invalidates the fit factor; the viewport
            // recomputes it from the live rect on the next draw.
            self.fit_zoom = 0.0;
        }
    }

    /// Set what is shown (the toolbar combo and the channel tree both write the
    /// shared `ViewControls.view`). Cheap: the re-materialize is deferred to the
    /// next `present`/`ensure_linear`.
    pub fn set_view(&mut self, view: &Selection) {
        if self.view != *view {
            self.view = view.clone();
        }
    }

    /// The available layers for this frame, for building the picker UI.
    pub fn layers(&self) -> &[LayerGroup] {
        &self.frame.layers
    }

    /// What is requested to be shown.
    pub fn view(&self) -> &Selection {
        &self.view
    }

    /// Set or clear a Cryptomatte coverage overlay (row-major `width*height`).
    /// Forces rematerialize + GPU re-upload on the next present.
    /// Setting the SAME empty state twice is a no-op: it must not bump
    /// `pixels_gen`, or clearing an overlay that was already clear would force a
    /// re-materialize and a full GPU re-upload every frame.
    pub fn set_coverage_overlay(
        &mut self,
        coverage: Option<Vec<f32>>,
        sel: Option<(String, String)>,
    ) {
        let unchanged = coverage.is_none()
            && self.coverage_overlay.is_none()
            && self.coverage_file.is_none()
            && self.coverage_sel == sel;
        self.coverage_overlay = coverage;
        self.coverage_file = None;
        self.coverage_sel = sel;
        self.coverage_failed = None;
        if !unchanged {
            self.materialized = None;
            self.pixels_gen = self.pixels_gen.wrapping_add(1);
        }
    }

    /// Record that `sel`'s coverage could not be extracted for THIS frame, and
    /// clear any overlay. The caller checks [`Self::coverage_failed`] before
    /// trying again, so a missing manifest / channel costs one attempt, not one
    /// per repaint.
    pub fn mark_coverage_failed(&mut self, sel: (String, String)) {
        self.set_coverage_overlay(None, None);
        self.coverage_failed = Some(sel);
    }

    /// Whether this frame is file-backed (its pixels are decoded per tile band).
    pub fn is_file_backed(&self) -> bool {
        self.frame.file_types.is_some()
    }

    /// The Cryptomatte overlay of a FILE-BACKED frame: the tiles show the
    /// coverage of `id` over `ranked`, computed per band from the file (a
    /// resident frame gets its plane through [`Self::set_coverage_overlay`]).
    pub fn set_coverage_file(&mut self, sel: (String, String), ranked: Vec<cryptomatte::RankedLayer>, id: f32) {
        self.coverage_overlay = None;
        self.coverage_file = Some((ranked, id));
        self.coverage_sel = Some(sel);
        self.coverage_failed = None;
        self.materialized = None;
        self.pixels_gen = self.pixels_gen.wrapping_add(1);
    }

    /// Current Cryptomatte overlay selection identity, if any.
    pub fn coverage_sel(&self) -> Option<&(String, String)> {
        self.coverage_sel.as_ref()
    }

    /// The selection whose extraction failed for this frame, if any.
    pub fn coverage_failed(&self) -> Option<&(String, String)> {
        self.coverage_failed.as_ref()
    }

    /// Materialize the active layer into `pixels` on the CPU if it is stale
    /// (different layer than last time, or never built). No GPU work; safe to call
    /// from tests. Marks the GPU linear texture dirty when the buffer changed.
    /// When a coverage overlay is set, RGB becomes greyscale coverage (A=1).
    ///
    /// # Errors
    /// [`ExrError::Logic`] if a channel's sample count does not cover the
    /// data window (this used to silently paint 0.0/1.0
    /// for the mismatched pixels instead of surfacing the corrupt frame).
    /// `self.pixels` is left at whatever `materialize_into` wrote before the
    /// failure (a valid-length, partially-filled buffer; still safe to read).
    pub fn prepare_cpu(&mut self) -> Result<(), ExrError> {
        if self.frame.file_types.is_some() {
            // File-backed: nothing to materialize; the tile source decodes bands.
            self.materialized = Some(self.view.clone());
            return Ok(());
        }
        let want_fresh = self.materialized.as_ref() == Some(&self.view)
            && self.pixels.len() == (self.width as usize) * (self.height as usize) * 4;
        if want_fresh {
            return Ok(());
        }
        let mut lap = crate::timing::Lap::start("materialize");
        lap.bytes(u64::from(self.width) * u64::from(self.height) * 16);
        let group = self.frame.resolve(&self.view);
        let group = group.as_deref();
        let mut rows = readout_rows(&self.frame.channels, group);
        // A worker/export may still hold the previous snapshot. Allocate an
        // empty replacement rather than cloning pixels which will be overwritten.
        if Arc::get_mut(&mut self.pixels).is_none() {
            self.pixels = Arc::new(Vec::with_capacity(self.pixels.capacity()));
        }
        // Fill IN PLACE: the buffer's capacity is reused across frames and layer
        // switches (see `set_frame`) instead of a fresh ~132 MB allocation each time.
        materialize_into(
            Arc::make_mut(&mut self.pixels),
            &self.frame,
            self.width,
            self.height,
            group,
        )?;
        if let Some(ref cov) = self.coverage_overlay {
            let n = (self.width as usize) * (self.height as usize);
            if cov.len() == n && self.pixels.len() == n * 4 {
                for (px, &c) in Arc::make_mut(&mut self.pixels).as_chunks_mut::<4>().0.iter_mut().zip(cov) {
                    px.copy_from_slice(&[c, c, c, 1.0]);
                }
                rows = vec![("coverage".to_owned(), 0)];
            }
        }
        self.readout = rows;
        self.materialized = Some(self.view.clone());
        Ok(())
    }

    /// Raw materialized pixel at `(x, y)` of the active layer (linear RGBA) —
    /// used by the pixel-readout HUD. Reads the per-view materialized buffer, so
    /// it reflects the layer currently on screen. `None` out of bounds or before
    /// the first materialize.
    pub fn pixel_at(&self, x: u32, y: u32) -> Option<[f32; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        // usize, not u32: `(y * width + x) * 4` overflows u32 past 2^30 pixels,
        // which 32768x32768 (allowed by Vulkan adapters reporting a 32768 limit)
        // already exceeds — a debug panic, or a silently wrong readout in release.
        let i = (y as usize * self.width as usize + x as usize) * 4;
        Some([
            *self.pixels.get(i)?,
            *self.pixels.get(i + 1)?,
            *self.pixels.get(i + 2)?,
            *self.pixels.get(i + 3)?,
        ])
    }

    /// The pixel readout at `(x, y)`: each channel the active view shows, by its
    /// real name, with its linear value — from the file at full precision for a
    /// file-backed frame. `None` out of bounds or before the first present.
    pub fn readout_at(&self, x: u32, y: u32) -> Option<Vec<(String, f32)>> {
        if let Some(src) = &self.file_source {
            return src.readout(x, y);
        }
        let px = self.pixel_at(x, y)?;
        Some(
            self.readout
                .iter()
                .filter_map(|(name, slot)| px.get(*slot).map(|&v| (name.clone(), v)))
                .collect(),
        )
    }

    /// Every channel of the frame at `(x, y)` (not only the shown ones), with its
    /// stored value, in header order - from the file for a file-backed frame.
    /// `None` out of bounds.
    pub fn probe_at(&self, x: u32, y: u32) -> Option<Vec<(String, Sample)>> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let at = |channels: &[(String, ChannelData)], i: usize| {
            channels.iter().filter_map(|(n, c)| stored(c, i).map(|s| (n.clone(), s))).collect()
        };
        match &self.probe {
            Some(rows) => rows.with_row(y, |row| at(row, x as usize)),
            None => Some(at(&self.frame.channels, y as usize * self.width as usize + x as usize)),
        }
    }

    /// Reset pan to centre and zoom to the fit factor (or 1:1 if no fit yet).
    pub fn reset_view(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.zoom = if self.fit_zoom > 0.0 {
            self.fit_zoom
        } else {
            1.0
        };
    }

    /// Set zoom to 1:1 (100%) centred — the `Home`/`1` action.
    pub fn home_view(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.zoom = 1.0;
    }

    /// Keep the tile source in step with what is shown: a new source generation
    /// when the frame, layer, overlay or effect changes. The effect stage runs
    /// here (once per change) unless the image is larger than one texture, where
    /// it is blocked with a reason (design decision 2026-09-29: OpenFX has no
    /// tile/RoI contract here yet).
    fn sync_source(
        &mut self,
        max_dim: u32,
        tiles: &mut TileState,
        stage: Option<&mut (dyn ImageStage + '_)>,
    ) {
        let (w, h) = (self.width, self.height);
        #[cfg(feature = "ofx")]
        if let Some(stage) = stage.as_ref()
            && stage.uses_worker() {
                let desc = SourceDesc { frame: self.frame.id, view: self.view.clone(),
                    pixels_gen: self.pixels_gen, stage: stage.key() };
                tiles.set_region_wake(stage.worker_wake());
                if tiles.is_stale(&desc) {
                    match stage.worker(self.frame.clone(), self.view.clone(), tiles.region_budget()) {
                        Ok(Some(source)) => {
                            self.file_source = None;
                            tiles.set_source(desc, source);
                        }
                        Ok(None) => {}
                        Err(error) => log::error!("immutable effect source refused: {error}"),
                    }
                }
                return;
        }
        let backed = self.frame.file_types.is_some();
        let stage = match stage {
            Some(s) if backed => {
                s.set_blocked(Some(format!(
                    "a {w}x{h} image is read from disk tile by tile; \
                     OpenFX effects need it whole and are off for it"
                )));
                None
            }
            Some(s) if w > max_dim || h > max_dim => {
                s.set_blocked(Some(format!(
                    "a {w}x{h} image is larger than one GPU texture ({max_dim}); \
                     OpenFX effects are off for it"
                )));
                None
            }
            Some(s) => {
                s.set_blocked(None);
                Some(s)
            }
            None => None,
        };
        let desc = SourceDesc {
            frame: self.frame.id,
            view: self.view.clone(),
            pixels_gen: self.pixels_gen,
            stage: stage.as_ref().map_or(0, |s| s.key()),
        };
        if !tiles.is_stale(&desc) {
            return;
        }
        if let Some(types) = &self.frame.file_types {
            let names: Vec<&str> = self.frame.channel_names.iter().map(String::as_str).collect();
            let dw = self.frame.data_window;
            let window = ((dw.min.x, dw.min.y), w, h);
            let path = self.frame.path.clone();
            let source = match &self.coverage_file {
                Some((ranked, id)) => FileSource::coverage(path, window, &names, types, ranked.clone(), *id),
                None => {
                    let mapping = Mapping::of(&names, self.frame.resolve(&self.view).as_deref());
                    FileSource::new(path, window, &names, types, mapping)
                }
            };
            match source {
                Ok(src) => {
                    let src = Arc::new(src);
                    self.file_source = Some(Arc::clone(&src));
                    tiles.set_source(desc, src as Arc<dyn TileSource>);
                }
                Err(e) => log::error!("file-backed source for {}: {e}", self.file_name()),
            }
            return;
        }
        self.file_source = None;
        let rgba = match stage {
            Some(s) => Arc::new(s.process(&self.pixels, w, h).to_vec()),
            None => Arc::clone(&self.pixels),
        };
        tiles.set_source(desc, Arc::new(FrameSource { rgba, w, h }));
    }

    fn prepare_source_cpu(&mut self, target: &mut DisplayTarget) -> Result<(), ExrError> {
        if self.materialized.as_ref() != Some(&self.view) {
            target.tiles.retire_cpu_source();
        }
        self.prepare_cpu()
    }

    /// Materialize the active layer (if stale), keep the tile source current,
    /// and plan this frame's tiles: produce missing ones off the UI thread,
    /// upload what is ready (within `DisplayTarget::upload`) and return the paint
    /// callback that draws them with `look`, the colour image through `colour` (an
    /// OCIO transform; `None` = the built-in one). Call each frame from the viewport.
    #[allow(clippy::too_many_arguments)]
    pub fn present(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut egui_wgpu::Renderer,
        target: &mut DisplayTarget,
        stage: Option<&mut (dyn ImageStage + '_)>,
        place: Placement,
        look: Look,
        colour: Option<&crate::ocio::Transform>,
    ) -> Result<TileCallback, String> {
        log::trace!(
            "present {}: {}x{} zoom={}",
            self.file_name(),
            self.width,
            self.height,
            place.zoom
        );
        // Times the whole present of a new frame / layer (the stale materialize marks it).
        let _lap = (self.materialized.as_ref() != Some(&self.view))
            .then(|| crate::timing::Lap::start("present"));
        #[cfg(feature = "ofx")]
        let worker = stage.as_ref().is_some_and(|stage| stage.uses_worker());
        #[cfg(not(feature = "ofx"))]
        let worker = false;
        if !worker { self.prepare_source_cpu(target).map_err(|e| e.to_string())?; }
        let sync = _lap.as_ref().map(|_| crate::timing::Lap::start("present sync_source"));
        self.sync_source(
            device.limits().max_texture_dimension_2d,
            &mut target.tiles,
            stage,
        );
        drop(sync);
        let gpu = renderer
            .callback_resources
            .get_mut::<TileGpu>()
            .ok_or("the tile renderer is not initialised")?;
        let colour_lap = _lap.as_ref().map(|_| crate::timing::Lap::start("present set_colour"));
        gpu.pipe.set_colour(device, queue, colour)?;
        drop(colour_lap);
        let reap = _lap.as_ref().map(|_| crate::timing::Lap::start("present poll+reap"));
        let _ = device.poll(wgpu::PollType::Poll);
        gpu.atlas.reap();
        target.retired_scopes.reap();
        if let Some(direct) = &mut gpu.direct {
            direct.reap();
        }
        drop(reap);
        let max_dim = device.limits().max_texture_dimension_2d;
        let reserve = target
            .scope_reserve(max_dim)
            .saturating_add(gpu.canvas_bytes);
        let eligible = !worker && !self.is_file_backed()
            && self.width <= max_dim.min(crate::tiles::direct::MAX_DIM)
            && self.height <= max_dim.min(crate::tiles::direct::MAX_DIM);
        if eligible {
            let desc = target
                .tiles
                .description()
                .cloned()
                .ok_or("display source is absent")?;
            let (_, rgba) = target
                .tiles
                .scope_level(u64::MAX, u32::MAX)
                .ok_or("resident display pixels are absent")?;
            let allowance = gpu
                .vram_budget
                .saturating_sub(reserve)
                .saturating_sub(gpu.atlas.bytes().0);
            let direct = gpu
                .direct
                .get_or_insert_with(|| crate::tiles::direct::DirectGpu::new(device));
            match direct.prepare(
                device,
                queue,
                &gpu.pipe.page_bgl,
                &desc,
                &rgba,
                (self.width, self.height),
                place,
                allowance,
            ) {
                Ok(true) => {
                    gpu.atlas.clear(queue);
                    target.tiles.clear_presentation();
                    target.blocked_frame = None;
                    target.direct_frame = Some(self.frame.id);
                    target.path = Some(DisplayPath::Direct { exact: direct.exact() });
                    return Ok(TileCallback {
                        list: direct.list(place, false),
                        look,
                        direct: true,
                    });
                }
                Ok(false) => {}
                Err(e) => log::warn!("direct display fallback: {e}"),
            }
        }
        let held =
            target.direct_frame.is_some() && gpu.direct.as_ref().is_some_and(|d| d.drawable());
        if held {
            // Retain the only drawable texture until a new atlas overview can be
            // allocated. The original is no longer needed while holding it.
            let direct = gpu.direct.as_mut().expect("held direct image");
            direct.retire_original(queue);
            target.tiles.prepare_frame(place);
            let available = gpu
                .vram_budget
                .saturating_sub(reserve)
                .saturating_sub(direct.bytes());
            let minimum = crate::tiles::atlas::PageLayout::new(max_dim, 0).page_bytes();
            if available < minimum {
                // Preserve the last drawable image, but permit playback to
                // move past an image whose transition cannot fit the allowance.
                target.blocked_frame = Some(self.frame.id);
            }
            if !target.tiles.overview_ready() || available < minimum {
                return Ok(TileCallback {
                    list: direct.list(place, true),
                    look,
                    direct: true,
                });
            }
        }
        if gpu.atlas.bytes().0 == 0 {
            let direct_bytes = gpu.direct.as_ref().map_or(0, |d| d.bytes());
            let allowance = gpu
                .vram_budget
                .saturating_sub(reserve)
                .saturating_sub(direct_bytes);
            gpu.atlas = crate::tiles::atlas::TileAtlas::new(crate::tiles::atlas::PageLayout::new(
                max_dim, allowance,
            ));
        }
        let direct_bytes = gpu.direct.as_ref().map_or(0, |d| d.bytes());
        let atlas_allowance = gpu
            .vram_budget
            .saturating_sub(reserve)
            .saturating_sub(direct_bytes);
        gpu.atlas.limit_growth(atlas_allowance);
        let list = crate::tiles::view::frame(
            &mut target.tiles,
            &mut gpu.atlas,
            device,
            queue,
            &gpu.pipe.page_bgl,
            place,
            target.upload,
        );
        if held && list.instances.is_empty() {
            return Ok(TileCallback {
                list: gpu
                    .direct
                    .as_ref()
                    .expect("held direct image")
                    .list(place, true),
                look,
                direct: true,
            });
        }
        if let Some(direct) = &mut gpu.direct {
            direct.retire_display(queue);
        }
        target.direct_frame = None;
        let feed = if self.is_file_backed() {
            TileFeed::File
        } else if target.tiles.description().is_some_and(|d| d.stage != 0) {
            TileFeed::Effect
        } else {
            TileFeed::Ram
        };
        target.path = Some(DisplayPath::Tiles(feed));
        target.blocked_frame = if list.instances.is_empty()
            && atlas_allowance < crate::tiles::atlas::PageLayout::new(max_dim, 0).page_bytes()
        {
            Some(self.frame.id)
        } else {
            None
        };
        Ok(TileCallback {
            list,
            look,
            direct: false,
        })
    }

    /// The scopes' linear source: the finest pyramid level of the shown content
    /// within `SCOPE_MAX_PIXELS` (level 0, exact, for images up to 4K x 4K),
    /// uploaded to a small `Rgba32Float` texture when the content or level
    /// changes. Returns its view and extent; `None` for an empty image or a
    /// texture the device cannot hold.
    pub fn ensure_linear<'t>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &'t mut DisplayTarget,
        stage: Option<&mut (dyn ImageStage + '_)>,
    ) -> Option<(&'t wgpu::TextureView, u32, u32)> {
        #[cfg(feature = "ofx")]
        if let Some(pixels) = target.accepted_pixels() {
            return target.ensure_processed_linear(device, queue, pixels);
        }
        if self.width == 0 || self.height == 0 {
            return None;
        }
        #[cfg(feature = "ofx")]
        if stage.as_ref().is_some_and(|stage| stage.uses_worker()) { return None; }
        if let Err(e) = self.prepare_source_cpu(target) {
            log::error!(
                "ensure_linear: materialize failed for {}: {e}",
                self.file_name()
            );
            return None;
        }
        target.retired_scopes.reap();
        let max_dim = device.limits().max_texture_dimension_2d;
        self.sync_source(max_dim, &mut target.tiles, stage);
        let grid = *target.tiles.grid()?;
        let level = crate::tiles::source::finest_level(&grid, SCOPE_MAX_PIXELS, max_dim);
        let (lw, lh) = grid.level_size(level);
        let key = (false, target.tiles.generation(), level);

        if !target.scope.as_ref().is_some_and(|t| fits(t.cap, (lw, lh))) {
            let next = ScopeTex::new(device, lw, lh, max_dim)?;
            if let Some(old) = target.scope.replace(next) {
                let bytes = UploadLayout::compute(old.cap.0, old.cap.1, LINEAR_TEXEL_SIZE, max_dim)
                    .ok()?
                    .byte_size;
                target.retired_scopes.retire(queue, old.tex, bytes);
            }
        }
        if target.scope.as_ref()?.key != Some(key) {
            // `None` while a file-backed level is still streaming in; the scope
            // shows its placeholder and asks again next frame.
            let (_, pixels) = target.tiles.scope_level(SCOPE_MAX_PIXELS, max_dim)?;
            log::debug!("scope source: level {level} ({lw}x{lh})");
            let scope = target.scope.as_mut()?;
            let mut lap = crate::timing::Lap::start("scope upload");
            lap.bytes(u64::from(lw) * u64::from(lh) * u64::from(LINEAR_TEXEL_SIZE));
            upload_linear(queue, &scope.tex, &pixels, lw, lh);
            drop(lap);
            scope.key = Some(key);
        }
        let scope = target.scope.as_ref()?;
        Some((&scope.view, lw, lh))
    }
}

impl ImageBackend {
    /// The shown content for saving (`export.rs`) as `(level, w, h, linear RGBA)`:
    /// level 0 of an in-RAM frame (the effect's output when one runs), or the
    /// scopes' overview level of a file-backed frame - level 0 of a 65k plate is
    /// never assembled. `None` while that level is still streaming in (it has been
    /// requested) or when the frame cannot be materialized (logged).
    pub fn export_level(
        &mut self,
        target: &mut DisplayTarget,
        max_dim: u32,
        stage: Option<&mut (dyn ImageStage + '_)>,
    ) -> Option<(u32, u32, u32, Arc<Vec<f32>>)> {
        #[cfg(feature = "ofx")]
        if stage.as_ref().is_some_and(|stage| stage.uses_worker()) { return None; }
        if let Err(e) = self.prepare_source_cpu(target) {
            log::error!("export: materialize failed for {}: {e}", self.file_name());
            return None;
        }
        self.sync_source(max_dim, &mut target.tiles, stage);
        target.tiles.drain();
        let (max_pixels, max_dim) =
            if self.is_file_backed() { (SCOPE_MAX_PIXELS, max_dim) } else { (u64::MAX, u32::MAX) };
        let (level, pixels) = target.tiles.scope_level(max_pixels, max_dim)?;
        let (w, h) = target.tiles.grid()?.level_size(level);
        Some((level, w, h, pixels))
    }
}

// ── GPU helpers ───────────────────────────────────────────────────────────────

/// Upload an interleaved linear RGBA f32 buffer (`w*h*4`) into a `Rgba32Float`
/// texture (the scopes' source).
fn upload_linear(queue: &wgpu::Queue, tex: &wgpu::Texture, pixels: &[f32], w: u32, h: u32) {
    // The texture is grow-only, so the copy extent is the IMAGE, not the texture.
    // A short buffer would be a wgpu validation error (i.e. a reported uncaptured
    // error and a wrong frame); refuse it here with a reason instead.
    let want = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    let Some(pixels) = pixels.get(..want) else {
        log::error!(
            "gpu: linear upload skipped — {} samples for a {w}x{h} image (need {want})",
            pixels.len()
        );
        return;
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice::<f32, u8>(pixels),
        // `pixels` was truncated to exactly `w * h * 4` above.
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4 * 4), // w * RGBA * f32; pixels is always w*h*4
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

impl ScopeTex {
    /// Allocate a `w x h` linear texture, sized through the shared
    /// `exr_gpu::UploadLayout` check against the device limit.
    fn new(device: &wgpu::Device, w: u32, h: u32, max_dim: u32) -> Option<Self> {
        let layout = match UploadLayout::compute(w.max(1), h.max(1), LINEAR_TEXEL_SIZE, max_dim) {
            Ok(l) => l,
            Err(e) => {
                log::error!("scope source {w}x{h} not allocated: {e}");
                return None;
            }
        };
        log::debug!("scope source texture {w}x{h} ({} MiB)", layout.byte_size >> 20);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("exr_view.scope_source"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
                | if cfg!(test) { wgpu::TextureUsages::COPY_SRC } else { wgpu::TextureUsages::empty() },
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Some(Self { tex, view, cap: (w.max(1), h.max(1)), key: None })
    }
}

// ── EXR seam: channels → linear RGBA f32 ───────────────────────────────────────

/// Read one channel's sample at flat index `i` as an f32 (the display value).
///
/// Half promotes losslessly; Float passes through; Uint is normalized to
/// `[0,1]` by `u32::MAX` so integer (e.g. id) channels are at least viewable
/// (a display policy — the canonical `ChannelData::sample_f32` itself gives
/// the raw UINT cast, matching C++'s `uintToFloat`).
///
/// # Errors
/// [`ExrError::Logic`] if `i` is out of range for `ch` — a length mismatch
/// between channels of the same materialized frame is a genuine decode bug
/// (all channels of a layer share the data window), so this is a hard error
/// instead of the previous silent `unwrap_or(0.0)`.
fn sample(ch: &ChannelData, i: usize) -> Result<f32, ExrError> {
    let v = ch.sample_f32(i).ok_or_else(|| {
        ExrError::Logic(format!(
            "sample index {i} out of range for a {}-sample channel",
            ch.len()
        ))
    })?;
    Ok(match ch {
        ChannelData::Uint(_) => v / u32::MAX as f32,
        ChannelData::Half(_) | ChannelData::Float(_) => v,
    })
}

/// Where one RGBA slot of a materialized buffer comes from: a channel (an index
/// into the channel list the [`Mapping`] was built over) or a constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SlotSrc {
    /// The channel at this index.
    Chan(usize),
    /// A constant (missing colour 0, missing alpha 1).
    Const(f32),
}

/// A layer's channel -> RGBA mapping: the single definition of what the viewer
/// shows in each slot. Materializing a resident frame, materializing a band of
/// a file (the out-of-core tile source) and labelling the pixel readout all
/// read it, so they cannot drift apart.
///
/// - no layer (no representable channels): black, opaque;
/// - the main (unlayered) layer: `R/G/B/A` by name (case-insensitive); with no
///   R/G/B, a luminance channel (`Y`, else the image's only channel) broadcast
///   to grey; missing colour 0, missing alpha 1 (the original `to_rgba_f32`
///   rule, so ordinary images are bit-for-bit unchanged);
/// - a named layer: a single component broadcast to grey with alpha 1; else up
///   to four components in canonical order into R, G, B, A (missing 0, 0, 0, 1;
///   extra components dropped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Mapping {
    /// R, G, B, A.
    pub slots: [SlotSrc; 4],
}

impl Mapping {
    /// The mapping of `group` over `names`, the channel list (in the order
    /// `group`'s indices refer to) of the whole image.
    pub(crate) fn of(names: &[&str], group: Option<&LayerGroup>) -> Self {
        use SlotSrc::{Chan, Const};
        let Some(g) = group else {
            return Self { slots: [Const(0.0), Const(0.0), Const(0.0), Const(1.0)] };
        };
        if g.name.is_empty() {
            let find = |want: &str| names.iter().position(|n| n.eq_ignore_ascii_case(want));
            let (r, gr, b, a) = (find("R"), find("G"), find("B"), find("A"));
            let alpha = a.map_or(Const(1.0), Chan);
            if r.is_none() && gr.is_none() && b.is_none() {
                // Luminance fallback: explicit `Y`, else the sole channel of a
                // 1-channel image.
                let lum = find("Y").or(if names.len() == 1 { Some(0) } else { None });
                if let Some(l) = lum {
                    return Self { slots: [Chan(l), Chan(l), Chan(l), alpha] };
                }
            }
            let c = |i: Option<usize>| i.map_or(Const(0.0), Chan);
            return Self { slots: [c(r), c(gr), c(b), alpha] };
        }
        // `get` + range check rather than `[idx]`: a `LayerGroup` is a plain index
        // list, and a stale one (a layer from another frame of a sequence whose
        // channels differ) must not panic here.
        let comp = |slot: usize| g.channels.get(slot).copied().filter(|&i| i < names.len());
        if g.channels.len() == 1 {
            // Single-channel layer (depth.Z, a matte): greyscale.
            let v = comp(0).map_or(Const(0.0), Chan);
            return Self { slots: [v, v, v, Const(1.0)] };
        }
        let c = |slot: usize, missing: f32| comp(slot).map_or(Const(missing), Chan);
        Self { slots: [c(0, 0.0), c(1, 0.0), c(2, 0.0), c(3, 1.0)] }
    }

    /// The channel indices this mapping reads, ascending and unique.
    pub(crate) fn channels(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .slots
            .iter()
            .filter_map(|s| match s {
                SlotSrc::Chan(i) => Some(*i),
                SlotSrc::Const(_) => None,
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The same mapping over another channel list: `to(i)` is channel `i`'s
    /// index there (a band decoded with only [`Self::channels`]).
    pub(crate) fn remap(&self, to: impl Fn(usize) -> Option<usize>) -> Option<Self> {
        let mut slots = self.slots;
        for s in &mut slots {
            if let SlotSrc::Chan(i) = s {
                *i = to(*i)?;
            }
        }
        Some(Self { slots })
    }

    /// Bind original selected channel planes once without copying pixels.
    pub(crate) fn bind<'a>(&self, channels: &'a [(String, ChannelData)]) -> Result<BoundMapping<'a>, ExrError> {
        let mut src: [Option<&ChannelData>; 4] = [None; 4];
        let mut konst = [0f32; 4];
        for (slot, s) in self.slots.iter().enumerate() {
            match *s {
                SlotSrc::Chan(i) => {
                    src[slot] = Some(&channels.get(i).ok_or_else(|| {
                        ExrError::Logic(format!("mapping reads channel {i} of {}", channels.len()))
                    })?.1);
                }
                SlotSrc::Const(v) => konst[slot] = v,
            }
        }
        Ok(BoundMapping { src, konst })
    }

    /// Fill `out` with `width * height` RGBA pixels from `channels` (each
    /// `width * height` samples). `out` is resized in place, keeping its
    /// capacity (see [`ImageBackend::set_frame`]).
    ///
    /// # Errors
    /// [`ExrError::Logic`] for a channel index outside `channels` or a channel
    /// shorter than the pixel count (a corrupt frame is
    /// reported, not painted as 0.0/1.0).
    pub(crate) fn fill(
        &self,
        out: &mut Vec<f32>,
        channels: &[(String, ChannelData)],
        width: u32,
        height: u32,
    ) -> Result<(), ExrError> {
        let n = reset(out, width, height)?;
        let bound = self.bind(channels)?;
        for i in 0..n {
            out[i * 4..i * 4 + 4].copy_from_slice(&bound.sample(i)?);
        }
        Ok(())
    }

    /// The pixel readout's rows: each channel shown, by name, at the first slot
    /// that holds it (grey broadcasts and a lone `A` are listed once).
    pub(crate) fn rows(&self, names: &[&str]) -> Vec<(String, usize)> {
        let mut seen = Vec::new();
        let mut rows = Vec::new();
        for (slot, s) in self.slots.iter().enumerate() {
            if let SlotSrc::Chan(i) = *s
                && !seen.contains(&i)
            {
                seen.push(i);
                if let Some(n) = names.get(i) {
                    rows.push(((*n).to_owned(), slot));
                }
            }
        }
        rows
    }
}

/// Checked original-channel sampling shared by full frames and bounded rows.
pub(crate) struct BoundMapping<'a> {
    src: [Option<&'a ChannelData>; 4],
    konst: [f32; 4],
}
impl BoundMapping<'_> {
    pub(crate) fn sample(&self, i: usize) -> Result<[f32; 4], ExrError> {
        let mut pixel = self.konst;
        for (slot, channel) in self.src.iter().enumerate() {
            if let Some(channel) = channel { pixel[slot] = sample(channel, i)?; }
        }
        Ok(pixel)
    }
}

/// Channel names of a channel list, in order (the `names` a [`Mapping`] takes).
fn names_of(channels: &[(String, ChannelData)]) -> Vec<&str> {
    channels.iter().map(|(n, _)| n.as_str()).collect()
}

/// Materialize the given layer of `frame` into an interleaved linear RGBA f32
/// buffer (`w*h*4`) through its [`Mapping`].
fn materialize_into(
    out: &mut Vec<f32>,
    frame: &DecodedFrame,
    width: u32,
    height: u32,
    group: Option<&LayerGroup>,
) -> Result<(), ExrError> {
    Mapping::of(&names_of(&frame.channels), group).fill(out, &frame.channels, width, height)
}

/// Resize `out` to `width * height * 4` zeroed samples, reusing its allocation
/// whenever the capacity already suffices. Returns the pixel count.
fn reset(out: &mut Vec<f32>, width: u32, height: u32) -> Result<usize, ExrError> {
    let n = crate::tiles::region::pixel_count(width,height).map_err(ExrError::Arg)?;
    let samples=n.checked_mul(4).ok_or_else(||ExrError::Arg("RGBA sample count overflow".into()))?;
    out.clear();
    out.try_reserve_exact(samples).map_err(|e|ExrError::Arg(e.to_string()))?;
    out.resize(samples,0.0);
    Ok(n)
}

/// Which real channel each RGBA slot of a materialized buffer holds, as
/// `(channel name, slot)` rows — the pixel readout's labels, read off the same
/// [`Mapping`] the buffer was filled with.
fn readout_rows(channels: &[(String, ChannelData)], group: Option<&LayerGroup>) -> Vec<(String, usize)> {
    let names = names_of(channels);
    Mapping::of(&names, group).rows(&names)
}

/// `plane` scaled up to the `w x h` data window at `min`: each pixel takes the
/// sample covering it (`Plane::at`).
fn upsample(plane: &exr_core::api::planes::Plane, min: (i32, i32), w: u32, h: u32) -> Result<ChannelData, LoadError> {
    let idx: Option<Vec<usize>> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| plane.at(min, x, y)).collect();
    let idx = idx.ok_or_else(|| LoadError::Read(format!("channel {}: a pixel has no sample", plane.name)))?;
    let gather = |len: usize| idx.iter().all(|&i| i < len);
    let out = match &plane.data {
        ChannelData::Half(v) if gather(v.len()) => ChannelData::Half(idx.iter().map(|&i| v[i]).collect()),
        ChannelData::Float(v) if gather(v.len()) => ChannelData::Float(idx.iter().map(|&i| v[i]).collect()),
        ChannelData::Uint(v) if gather(v.len()) => ChannelData::Uint(idx.iter().map(|&i| v[i]).collect()),
        _ => return Err(LoadError::Read(format!("channel {}: fewer samples than its sampling", plane.name))),
    };
    Ok(out)
}

/// The row reader of [`ImageBackend::probe_at`] for a file-backed `frame`.
fn probe_rows(frame: &DecodedFrame) -> Option<RowCache> {
    let types = frame.file_types.as_ref()?;
    let names: Vec<&str> = frame.channel_names.iter().map(String::as_str).collect();
    let dw = frame.data_window;
    match RowCache::all(frame.path.clone(), ((dw.min.x, dw.min.y), frame.width, frame.height), &names, types) {
        Ok(rows) => Some(rows),
        Err(e) => {
            log::warn!("{}: no every-channel probe: {e}", frame.path.display());
            None
        }
    }
}

/// Every attribute of `header`, in header order, as `exrheader` prints it. A type
/// `exrheader` prints no value for (an unknown, opaque one) shows its type name,
/// so its row is not blank.
fn metadata_of(header: &Header) -> Vec<MetaEntry> {
    exr_core::api::describe::rows(header)
        .into_iter()
        .map(|r| {
            let value = if r.value.is_empty() { format!("({})", r.type_name) } else { r.value };
            MetaEntry { key: r.name, value }
        })
        .collect()
}

/// The header of a single-part flat file, without decoding or allocating any
/// pixel storage; `None` for anything else (multipart, deep, unreadable).
fn peek_header(path: &Path) -> Option<Header> {
    let mut stream = exr_core::io::StdIfStream::open(path.to_str()?).ok()?;
    let file = exr_core::scanline::InputFile::new(&mut stream).ok()?;
    Some(file.header().clone())
}

/// Whether a file with `header` is too large to hold within `budget` bytes
/// (every channel plus the 16 B/px materialized RGBA). Subsampled files never
/// are: they are always decoded whole.
fn needs_file_backing(header: &Header, budget: u64) -> bool {
    let (Ok(dw), Ok(list)) = (header.data_window(), header.channels()) else { return false };
    let w = (i64::from(dw.max.x) - i64::from(dw.min.x) + 1).max(0) as u64;
    let h = (i64::from(dw.max.y) - i64::from(dw.min.y) + 1).max(0) as u64;
    let mut bytes = w * h * 16;
    for (_, c) in list.iter() {
        if (c.x_sampling, c.y_sampling) != (1, 1) {
            return false;
        }
        bytes += w * h * exr_core::misc::pixel_type_size(c.type_).unwrap_or(4) as u64;
    }
    bytes > budget
}

// ── Load error ─────────────────────────────────────────────────────────────────

/// Why an EXR failed to load — distinguishes "unsupported by the facade"
/// (deep / multipart — point the user at a specific part) from a genuine read error.
#[derive(Debug)]
pub enum LoadError {
    /// Deep / multipart / subsampled: the `api::read` escape hatch.
    Unsupported(String),
    /// A real I/O or decode failure.
    Read(String),
}

impl LoadError {
    /// Classify an [`ExrError`]: the facade returns [`ExrError::Unsupported`]
    /// for the deep/multipart/subsampled escape hatch; every other error —
    /// including an unrelated [`ExrError::Arg`] — is a genuine read failure.
    ///
    /// Matching the structured variant (not `ExrError::Arg`) fixes the old bug
    /// where every argument error was mislabeled "unsupported"; only the real
    /// escape hatch now shows the "use a specific part" hint.
    fn from_exr(e: ExrError) -> Self {
        match e {
            ExrError::Unsupported(u) => LoadError::Unsupported(u.to_string()),
            other => LoadError::Read(format!("{other}")),
        }
    }

    /// One-line status message for the UI.
    pub fn status(&self) -> String {
        match self {
            LoadError::Unsupported(_) => {
                "deep/multipart: use a specific part".to_owned()
            }
            LoadError::Read(msg) => format!("load failed: {msg}"),
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            LoadError::Read(msg) => write!(f, "read error: {msg}"),
        }
    }
}

impl std::error::Error for LoadError {}

fn validated_aspect(value: f32) -> Result<f32, LoadError> {
    if value.is_finite() && value > 0.0 { Ok(value) }
    else { Err(LoadError::Read("pixelAspectRatio must be finite and positive".into())) }
}

// ── Tests (headless — the load + materialize seam, no GPU) ─────────────────────

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// A resident 1x1 frame whose `Float` channels `names` hold their own index
    /// (`names[i]` = `i`), grouped into layers the way a loaded file is.
    pub(crate) fn frame_with(names: &[&str]) -> DecodedFrame {
        let channels: Vec<(String, ChannelData)> = names
            .iter()
            .enumerate()
            .map(|(i, n)| ((*n).to_owned(), ChannelData::Float(vec![i as f32])))
            .collect();
        let layers = group_layers(channels.iter().map(|(n, c)| (n.as_str(), c.pixel_type())));
        let win = Box2i::from_min_max(imath_rs::V2i::new(0, 0), imath_rs::V2i::new(0, 0));
        DecodedFrame {
            path: PathBuf::new(),
            width: 1,
            height: 1,
            data_window: win,
            display_window: win,
            channel_names: names.iter().map(|n| (*n).to_owned()).collect(),
            channels,
            layers,
            metadata: vec![],
            id: 0,
            file_types: None,
            pixel_aspect: 1.0,
        }
    }
}

#[cfg(all(test, feature = "ofx"))]
#[path = "image_consumer_tests.rs"]
pub(crate) mod consumer_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A 1x1 8-bit RGBA PNG written by the `png` crate, independent of the importer.
    pub(crate) fn write_png(path: &Path, px: [u8; 4]) {
        let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        let mut enc = png::Encoder::new(file, 1, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&px).unwrap();
    }

    /// A PNG becomes a resident linear frame through `exr-import` (the only door for non-EXR
    /// files): sRGB 128 decodes to 0.2158605, straight alpha 128 premultiplies in linear light,
    /// and the Metadata panel says where it came from.
    #[test]
    fn a_png_enters_through_exr_import() {
        let dir = std::env::temp_dir().join(format!("exr-view-png-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.png");
        write_png(&path, [128, 128, 128, 128]);
        let frame = DecodedFrame::open(&path, 0).unwrap();
        let sample = |n: &str| frame.channels.iter().find(|(c, _)| c == n).and_then(|(_, d)| d.sample_f32(0)).unwrap();
        let alpha = 128.0 / 255.0;
        assert!((sample("R") - 0.215_860_5 * alpha).abs() < 2e-4, "{}", sample("R"));
        assert!((sample("A") - alpha).abs() < 2e-4);
        assert_eq!((frame.width, frame.height, frame.file_types.is_none()), (1, 1, true));
        let meta = |k: &str| frame.metadata.iter().find(|m| m.key == k).map(|m| m.value.as_str());
        assert_eq!(meta("Source format"), Some("Png"), "came through exr-import");
        assert_eq!(meta("Source alpha"), Some("straight, premultiplied in linear light"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn populated_source_reuses_cpu_allocation_and_keeps_export_snapshot() {
        let frame = |id| {
            let mut f = tests_support::frame_with(&["R", "G", "B"]);
            f.id = id;
            Arc::new(f)
        };
        let mut image = ImageBackend::from_frame(frame(1));
        let mut target = DisplayTarget::new(8 << 20, usize::MAX);
        image.prepare_source_cpu(&mut target).unwrap();
        image.sync_source(8192, &mut target.tiles, None);
        assert_eq!(
            Arc::strong_count(&image.pixels),
            2,
            "backend plus live source"
        );
        let first = image.pixels.as_ptr();
        image.set_frame(frame(2));
        image.prepare_source_cpu(&mut target).unwrap();
        assert_eq!(
            first,
            image.pixels.as_ptr(),
            "source retirement allows in-place reuse"
        );
        image.sync_source(8192, &mut target.tiles, None);
        let (_, snapshot) = target.tiles.scope_level(u64::MAX, u32::MAX).unwrap();
        let saved = snapshot.clone();
        image.set_frame(frame(3));
        image.prepare_source_cpu(&mut target).unwrap();
        assert_ne!(
            snapshot.as_ptr(),
            image.pixels.as_ptr(),
            "an outstanding export owns its old allocation"
        );
        assert_eq!(&snapshot[..], &saved[..]);
        assert_eq!(image.pixel_at(0, 0), Some([0.0, 1.0, 2.0, 1.0]));
    }

    use std::path::PathBuf;

    /// A `0,0 .. w-1,h-1` window, for the synthetic frames below.
    fn win(w: u32, h: u32) -> Box2i {
        Box2i::from_min_max(
            imath_rs::V2i::new(0, 0),
            imath_rs::V2i::new(w as i32 - 1, h as i32 - 1),
        )
    }

    /// Resolve a corpus EXR path relative to the workspace root.
    #[test]
    #[cfg_attr(not(feature = "test-data"), ignore = "needs the exr-rs test image set (--features test-data)")]
    fn load_seam_desk_scanline() {
        let path = exr_test_data::openexr_images().join("ScanLines/Desk.exr");
        let mut img = ImageBackend::load(&path).expect("Desk.exr should load via exr-core");

        assert!(img.width > 0 && img.height > 0, "non-empty dimensions");
        assert!(!img.frame.channel_names.is_empty(), "at least one channel");
        assert!(!img.frame.layers.is_empty(), "at least one layer");
        // A normal RGB image groups into the main (unlayered) layer first.
        assert_eq!(img.frame.layers[0].name, "", "main layer first");

        // Materialize the active (main) layer on the CPU and check the buffer.
        img.prepare_cpu().unwrap();
        let expected = (img.width as usize) * (img.height as usize) * 4;
        assert_eq!(
            img.pixels.len(),
            expected,
            "materialized RGBA buffer is w*h*4"
        );
        assert!(
            img.pixels.iter().all(|v| v.is_finite()),
            "all RGBA samples finite"
        );

        // Pixel readout works and is finite at the corners.
        let p0 = img.pixel_at(0, 0).expect("top-left pixel");
        assert!(p0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn named_layer_materializes_to_rgba() {
        // Synthesize a 2x1 frame with a main RGB + a single-channel `depth.Z` layer
        // and a 2-component `velocity` layer; verify materialization shapes.
        use exr_core::api::ChannelData;
        let w = 2u32;
        let h = 1u32;
        let n = (w * h) as usize;
        let channels = vec![
            ("R".to_string(), ChannelData::Float(vec![0.1, 0.2])),
            ("G".to_string(), ChannelData::Float(vec![0.3, 0.4])),
            ("B".to_string(), ChannelData::Float(vec![0.5, 0.6])),
            ("depth.Z".to_string(), ChannelData::Float(vec![7.0, 8.0])),
            ("velocity.x".to_string(), ChannelData::Float(vec![1.0, 2.0])),
            (
                "velocity.y".to_string(),
                ChannelData::Float(vec![-1.0, -2.0]),
            ),
        ];
        let layers = group_layers(channels.iter().map(|(nm, c)| (nm.as_str(), c.pixel_type())));

        // depth → single-channel grey.
        let z = layers
            .iter()
            .find(|g| g.name == "depth")
            .expect("depth layer");
        let mut buf = Vec::new();
        let names = names_of(&channels);
        Mapping::of(&names, Some(z)).fill(&mut buf, &channels, w, h).unwrap();
        assert_eq!(buf.len(), n * 4);
        assert_eq!(&buf[0..4], &[7.0, 7.0, 7.0, 1.0]); // grey broadcast, alpha 1

        // velocity → 2 components into R,G; B=0, A=1.
        let vel = layers
            .iter()
            .find(|g| g.name == "velocity")
            .expect("velocity layer");
        Mapping::of(&names, Some(vel)).fill(&mut buf, &channels, w, h).unwrap();
        assert_eq!(&buf[0..4], &[1.0, -1.0, 0.0, 1.0]);
        assert_eq!(&buf[4..8], &[2.0, -2.0, 0.0, 1.0]);

        // main → R,G,B + alpha 1 (unchanged classic mapping).
        let main = layers
            .iter()
            .find(|g| g.name.is_empty())
            .expect("main layer");
        materialize_into(
            &mut buf,
            &DecodedFrame {
                path: PathBuf::new(),
                width: w,
                height: h,
                data_window: win(w, h),
                display_window: win(w, h),
                channel_names: vec![],
                channels: channels.clone(),
                layers: layers.clone(),
                metadata: vec![],
                id: 0,
                file_types: None,
            pixel_aspect: 1.0,
            },
            w,
            h,
            Some(main),
        )
        .unwrap();
        assert_eq!(&buf[0..4], &[0.1, 0.3, 0.5, 1.0]);
    }

    #[test]
    fn switch_layer_rematerializes_and_falls_back() {
        // No GPU: prepare_cpu repacks `pixels` from the resident channels whenever
        // the active layer changes, and an unknown name falls back to main.
        let channels = vec![
            ("R".to_string(), ChannelData::Float(vec![0.2])),
            ("G".to_string(), ChannelData::Float(vec![0.4])),
            ("B".to_string(), ChannelData::Float(vec![0.6])),
            ("depth.Z".to_string(), ChannelData::Float(vec![9.0])),
        ];
        let layers = group_layers(channels.iter().map(|(n, c)| (n.as_str(), c.pixel_type())));
        let frame = Arc::new(DecodedFrame {
            path: PathBuf::new(),
            width: 1,
            height: 1,
            data_window: win(1, 1),
            display_window: win(1, 1),
            channel_names: vec![],
            channels,
            layers,
            metadata: vec![],
            id: 0,
            file_types: None,
            pixel_aspect: 1.0,
        });
        let mut img = ImageBackend::from_frame(frame);

        // Default = main layer → R,G,B + alpha 1.
        img.prepare_cpu().unwrap();
        assert_eq!(img.materialized, Some(Selection::layer("")));
        assert_eq!(img.pixel_at(0, 0), Some([0.2, 0.4, 0.6, 1.0]));

        // Switch to the depth layer → single-channel grey, re-materialized.
        img.set_view(&Selection::layer("depth"));
        img.prepare_cpu().unwrap();
        assert_eq!(img.materialized, Some(Selection::layer("depth")));
        assert_eq!(img.pixel_at(0, 0), Some([9.0, 9.0, 9.0, 1.0]));

        // Unknown layer name → resolve_layer falls back to main (no panic).
        img.set_view(&Selection::layer("nonexistent"));
        img.prepare_cpu().unwrap();
        assert_eq!(img.pixel_at(0, 0), Some([0.2, 0.4, 0.6, 1.0]));
    }

    /// The readout names each slot by the channel really in it: every row's
    /// channel sample equals that slot of the materialized buffer, and the rows
    /// are exactly the channels shown (no fixed R/G/B/A, no defaulted slots).
    #[test]
    fn readout_rows_name_the_materialized_slots() {
        let f = |v: f32| ChannelData::Float(vec![v]);
        // (active layer, channels with their pixel value, expected readout names)
        type Case<'a> = (&'a str, Vec<(&'a str, f32)>, Vec<&'a str>);
        let cases: Vec<Case> = vec![
            ("", vec![("R", 0.1), ("G", 0.2), ("B", 0.3), ("A", 0.4)], vec!["R", "G", "B", "A"]),
            ("", vec![("R", 0.1), ("G", 0.2), ("B", 0.3)], vec!["R", "G", "B"]),
            ("", vec![("Y", 0.7)], vec!["Y"]),
            ("", vec![("Z", 5.0)], vec!["Z"]),
            ("", vec![("A", 0.5)], vec!["A"]),
            ("", vec![("Y", 0.7), ("A", 0.5)], vec!["Y", "A"]),
            ("depth", vec![("depth.Z", 9.0)], vec!["depth.Z"]),
            ("velocity", vec![("velocity.x", 1.0), ("velocity.y", -1.0)], vec!["velocity.x", "velocity.y"]),
            ("N", vec![("N.x", 0.1), ("N.y", 0.2), ("N.z", 0.3)], vec!["N.x", "N.y", "N.z"]),
            (
                "diffuse",
                vec![("diffuse.A", 0.4), ("diffuse.B", 0.3), ("diffuse.G", 0.2), ("diffuse.R", 0.1)],
                vec!["diffuse.R", "diffuse.G", "diffuse.B", "diffuse.A"],
            ),
            // Five components outside the known R/G/B/A, X/Y/Z, U/V set sort
            // alphabetically (`component_priority`); only the first four are shown.
            (
                "c",
                vec![("c.w", 5.0), ("c.t", 4.0), ("c.s", 3.0), ("c.q", 2.0), ("c.p", 1.0)],
                vec!["c.p", "c.q", "c.s", "c.t"],
            ),
        ];
        for (layer, chans, want) in cases {
            let channels: Vec<(String, ChannelData)> =
                chans.iter().map(|(n, v)| (n.to_string(), f(*v))).collect();
            let layers = group_layers(channels.iter().map(|(n, c)| (n.as_str(), c.pixel_type())));
            let frame = DecodedFrame {
                path: PathBuf::new(),
                width: 1,
                height: 1,
                data_window: win(1, 1),
                display_window: win(1, 1),
                channel_names: vec![],
                channels: channels.clone(),
                layers: layers.clone(),
                metadata: vec![],
                id: 0,
                file_types: None,
            pixel_aspect: 1.0,
            };
            let group = frame.resolve_layer(layer);
            let mut buf = Vec::new();
            materialize_into(&mut buf, &frame, 1, 1, group).unwrap();
            let rows = readout_rows(&channels, group);
            let names: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, want, "layer {layer:?} {chans:?}");
            for (name, slot) in &rows {
                let v = chans.iter().find(|(n, _)| n == name).unwrap().1;
                assert_eq!(buf[*slot], v, "layer {layer:?}: `{name}` is not in slot {slot}");
            }
        }
    }

    #[test]
    fn readout_at_uses_channel_names_and_the_overlay_label() {
        let channels = vec![
            ("R".to_string(), ChannelData::Float(vec![0.2])),
            ("G".to_string(), ChannelData::Float(vec![0.4])),
            ("B".to_string(), ChannelData::Float(vec![0.6])),
            ("depth.Z".to_string(), ChannelData::Float(vec![9.0])),
        ];
        let layers = group_layers(channels.iter().map(|(n, c)| (n.as_str(), c.pixel_type())));
        let frame = Arc::new(DecodedFrame {
            path: PathBuf::new(),
            width: 1,
            height: 1,
            data_window: win(1, 1),
            display_window: win(1, 1),
            channel_names: vec![],
            channels,
            layers,
            metadata: vec![],
            id: 0,
            file_types: None,
            pixel_aspect: 1.0,
        });
        let mut img = ImageBackend::from_frame(frame);
        img.set_view(&Selection::layer("depth"));
        img.prepare_cpu().unwrap();
        assert_eq!(img.readout_at(0, 0), Some(vec![("depth.Z".to_string(), 9.0)]));
        assert_eq!(img.readout_at(1, 0), None);

        img.coverage_overlay = Some(vec![0.25]);
        img.materialized = None;
        img.prepare_cpu().unwrap();
        assert_eq!(img.readout_at(0, 0), Some(vec![("coverage".to_string(), 0.25)]));
    }

    /// Grow-only reuse: the scope source texture must survive a data
    /// window that wobbles frame to frame, but not hold a 16K allocation for a
    /// 1K image for the rest of the session.
    #[test]
    fn display_target_grows_and_shrinks_deliberately() {
        assert!(fits((1920, 1080), (1920, 1080)), "exact size reuses");
        assert!(fits((1920, 1080), (1900, 1060)), "a smaller image reuses");
        assert!(
            !fits((1920, 1080), (1921, 1080)),
            "a wider image must reallocate"
        );
        assert!(
            !fits((1920, 1080), (1920, 1081)),
            "a taller image must reallocate"
        );
        // 4x the pixels is still reuse; beyond that we give the VRAM back.
        assert!(fits((2000, 2000), (1000, 1000)), "exactly 4x stays");
        assert!(
            !fits((16384, 16384), (1024, 1024)),
            "wildly oversized shrinks"
        );
        // Degenerate need is clamped by the caller to 1x1, never divides by zero.
        assert!(fits((1, 1), (1, 1)));
    }

    /// `pixel_at` indexes in `usize`. `(y * width + x) * 4` computed in u32 has
    /// only one bit of headroom left at the largest texture any adapter allows
    /// (32768x32768 = 2^30 px lands the last index on 2^32-4), and none at all
    /// for the 65536-wide plates this project is aiming at — so the index is
    /// computed in `usize`, where the margin does not depend on the device limit.
    #[test]
    fn pixel_at_index_does_not_overflow_u32() {
        let (w, h) = (32768u32, 32768u32);
        let (x, y) = (w - 1, h - 1);
        let i = (y as usize * w as usize + x as usize) * 4;
        assert_eq!(i, ((w as usize * h as usize) - 1) * 4);
        assert_eq!(i, u32::MAX as usize - 3, "one bit of headroom in u32");
        // Past that limit the u32 form wraps; the usize form does not.
        let (w, h) = (65536usize, 65536usize);
        assert_eq!(((h - 1) * w + (w - 1)) * 4, (w * h - 1) * 4);
        // And the real accessor rejects out-of-range coordinates without panicking.
        let img = ImageBackend::from_frame(Arc::new(DecodedFrame {
            path: PathBuf::new(),
            width: 2,
            height: 2,
            data_window: win(2, 2),
            display_window: win(2, 2),
            channel_names: vec![],
            channels: vec![],
            layers: vec![],
            metadata: vec![],
            id: 0,
            file_types: None,
            pixel_aspect: 1.0,
        }));
        assert_eq!(img.pixel_at(2, 0), None);
        assert_eq!(img.pixel_at(0, 2), None);
        assert_eq!(img.pixel_at(u32::MAX, u32::MAX), None);
    }

    /// The data window can sit anywhere inside the display window; a degenerate
    /// display window falls back to the data window rather than a zero canvas.
    #[test]
    fn windows_report_both_and_survive_a_degenerate_display_window() {
        let mk = |data: Box2i, display: Box2i, w: u32, h: u32| DecodedFrame {
            path: PathBuf::new(),
            width: w,
            height: h,
            data_window: data,
            display_window: display,
            channel_names: vec![],
            channels: vec![],
            layers: vec![],
            metadata: vec![],
            id: 0,
            file_types: None,
            pixel_aspect: 1.0,
        };
        let b = |x0, y0, x1, y1| {
            Box2i::from_min_max(imath_rs::V2i::new(x0, y0), imath_rs::V2i::new(x1, y1))
        };

        // Data window offset inside a larger display window.
        let f = mk(b(10, 20, 29, 39), b(0, 0, 99, 49), 20, 20);
        let w = f.windows();
        assert_eq!(w.display_min, (0, 0));
        assert_eq!(w.display_size, (100, 50));
        assert_eq!(w.data_min, (10, 20));

        // Degenerate display window (min > max) -> fall back to the data window.
        let f = mk(b(0, 0, 7, 3), b(5, 5, 1, 1), 8, 4);
        let w = f.windows();
        assert_eq!(w.display_min, (0, 0));
        assert_eq!(w.display_size, (8, 4));
    }

    /// A sequence step must REUSE the materialized buffer, not re-allocate
    /// ~132 MB per displayed frame.
    #[test]
    fn set_frame_reuses_the_materialized_buffer() {
        let frame = |v: f32, id: u64| {
            Arc::new(DecodedFrame {
                path: PathBuf::new(),
                width: 4,
                height: 4,
                data_window: win(4, 4),
                display_window: win(4, 4),
                channel_names: vec![],
                channels: vec![
                    ("R".to_string(), ChannelData::Float(vec![v; 16])),
                    ("G".to_string(), ChannelData::Float(vec![v; 16])),
                    ("B".to_string(), ChannelData::Float(vec![v; 16])),
                ],
                layers: group_layers([("R", exr_core::attr::PixelType::Float)]),
                metadata: vec![],
                id,
                file_types: None,
            pixel_aspect: 1.0,
            })
        };
        let mut img = ImageBackend::from_frame(frame(0.25, 1));
        img.prepare_cpu().unwrap();
        let cap = img.pixels.capacity();
        let ptr = img.pixels.as_ptr();
        assert_eq!(img.pixel_at(0, 0), Some([0.25, 0.25, 0.25, 1.0]));

        img.set_frame(frame(0.75, 2));
        img.prepare_cpu().unwrap();
        assert_eq!(
            img.pixel_at(0, 0),
            Some([0.75, 0.75, 0.75, 1.0]),
            "new frame materialized"
        );
        assert_eq!(img.pixels.capacity(), cap, "capacity kept");
        assert_eq!(
            img.pixels.as_ptr(),
            ptr,
            "same allocation - no realloc per frame"
        );
        // A resolution change invalidates the fit factor so the viewport recomputes it.
        img.fit_zoom = 2.0;
        img.set_frame(Arc::new(DecodedFrame {
            width: 2,
            height: 2,
            ..unwrap_frame(frame(0.5, 3))
        }));
        assert_eq!(img.fit_zoom, 0.0);
    }

    /// Clone a frame out of an `Arc` for the struct-update syntax above.
    fn unwrap_frame(f: Arc<DecodedFrame>) -> DecodedFrame {
        DecodedFrame {
            path: f.path.clone(),
            width: f.width,
            height: f.height,
            data_window: f.data_window,
            display_window: f.display_window,
            channel_names: f.channel_names.clone(),
            channels: f.channels.clone(),
            layers: f.layers.clone(),
            metadata: vec![],
            id: f.id,
            file_types: f.file_types.clone(),
            pixel_aspect: f.pixel_aspect,
        }
    }

    /// Regression: only the structured escape-hatch error is "unsupported".
    /// The old `from_exr` mapped EVERY `ExrError::Arg` to `Unsupported`,
    /// mislabeling unrelated argument errors in the status bar; an `Arg` must
    /// now be a plain read error, and only `ExrError::Unsupported` the hint.
    #[test]
    fn load_error_classifies_only_structured_unsupported() {
        use exr_core::Unsupported;
        assert!(matches!(
            LoadError::from_exr(ExrError::Unsupported(Unsupported::Multipart)),
            LoadError::Unsupported(_)
        ));
        // An unrelated argument error must NOT be labeled unsupported.
        assert!(matches!(
            LoadError::from_exr(ExrError::Arg("some unrelated argument error".into())),
            LoadError::Read(_)
        ));
        assert!(matches!(
            LoadError::from_exr(ExrError::Io("disk gone".into())),
            LoadError::Read(_)
        ));
    }

    /// A test image written to disk: 700 x 530 at (2, -3), R G B A half plus a
    /// float `depth.Z`.
    fn write_test_exr(name: &str) -> PathBuf {
        use exr_core::Compression;
        let (w, h) = (700usize, 530usize);
        let dw = Box2i { min: imath_rs::V2i { x: 2, y: -3 }, max: imath_rs::V2i { x: 2 + w as i32 - 1, y: -3 + h as i32 - 1 } };
        let plane = |f: &dyn Fn(usize, usize) -> f32| (0..w * h).map(|i| f(i % w, i / w)).collect::<Vec<f32>>();
        let half = |v: Vec<f32>| ChannelData::Half(v.into_iter().map(imath_rs::Half::from).collect());
        let dir = std::env::temp_dir().join(format!("exr-view-backed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        exr_core::api::Image::new(dw)
            .with_channel("R", half(plane(&|x, _| x as f32 / 700.0)))
            .with_channel("G", half(plane(&|_, y| y as f32 / 530.0)))
            .with_channel("B", half(plane(&|x, y| ((x * y) % 97) as f32 / 97.0)))
            .with_channel("A", half(plane(&|x, y| ((x + y) % 5) as f32 / 4.0)))
            .with_channel("depth.Z", ChannelData::Float(plane(&|x, y| (x * 2 + y) as f32)))
            .write(&path, Compression::Zip)
            .unwrap();
        path
    }

    /// The every-channel probe gives each channel's stored value (a `Uint` as its
    /// integer), and a file-backed frame probes the same values from the file.
    #[test]
    fn probe_reads_every_channel_as_stored() {
        let mut frame = super::tests_support::frame_with(&["A", "B", "G", "R", "depth.Z"]);
        frame.channels.push(("id".to_owned(), ChannelData::Uint(vec![u32::MAX - 1])));
        frame.channel_names.push("id".to_owned());
        let img = ImageBackend::from_frame(Arc::new(frame));
        let got = img.probe_at(0, 0).unwrap();
        let text: Vec<String> = got.iter().map(|(n, s)| format!("{n}={s}")).collect();
        assert_eq!(
            text,
            ["A=0.000000", "B=1.000000", "G=2.000000", "R=3.000000", "depth.Z=4.000000", "id=4294967294"]
        );
        assert_eq!(img.probe_at(1, 0), None);

        let path = write_test_exr("probe.exr");
        let whole = ImageBackend::from_frame(Arc::new(DecodedFrame::open(&path, u64::MAX).unwrap()));
        let backed = ImageBackend::from_frame(Arc::new(DecodedFrame::open(&path, 1).unwrap()));
        for (x, y) in [(0, 0), (123, 45), (699, 529)] {
            let want = whole.probe_at(x, y).unwrap();
            assert_eq!(want.len(), 5);
            assert_eq!(backed.probe_at(x, y), Some(want), "({x}, {y})");
        }
    }

    /// A parity golden of exr-core (`cargo run -p xtask -- parity-exr`).
    fn golden(name: &str) -> PathBuf {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../exr-core/tests/parity").join(name);
        assert!(p.exists(), "missing golden {}; run `cargo run -p xtask -- parity-exr`", p.display());
        p
    }

    /// A luminance-chroma file opens: `R`,`G`,`B` are C++ `RgbaInputFile`'s
    /// reconstruction bit for bit (its dump), and `Y`/`RY`/`BY`/`A` are the stored
    /// samples scaled up (a 2x2 chroma pixel repeats over its 2x2 block).
    #[test]
    fn luminance_chroma_file_opens_with_cpp_rgb() {
        let frame = DecodedFrame::load(&golden("golden_yc.exr")).unwrap();
        let names: Vec<&str> = frame.channel_names.iter().map(String::as_str).collect();
        assert_eq!(names, ["A", "B", "BY", "G", "R", "RY", "Y"]);
        let bytes = std::fs::read(golden("golden_yc.exr.dump")).unwrap();
        let (w, h) = (u32::from_le_bytes(bytes[0..4].try_into().unwrap()), u32::from_le_bytes(bytes[4..8].try_into().unwrap()));
        assert_eq!((frame.width, frame.height), (w, h));
        let cpp: Vec<u16> = bytes[8..].as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        let chan = |n: &str| &frame.channels.iter().find(|(c, _)| c == n).unwrap().1;
        for (c, name) in ["R", "G", "B"].into_iter().enumerate() {
            let ChannelData::Half(v) = chan(name) else { panic!("{name} is half") };
            let got: Vec<u16> = v.iter().map(|h| h.bits()).collect();
            let want: Vec<u16> = cpp.iter().skip(c).step_by(4).copied().collect();
            assert_eq!(got, want, "{name} differs from C++ RgbaInputFile");
        }
        // RY stored at 2x2 (origin data window): pixel (x, y) is sample (x/2, y/2).
        let mut stream = exr_core::io::StdIfStream::open(golden("golden_yc.exr").to_str().unwrap()).unwrap();
        let mut file = exr_core::scanline::InputFile::new(&mut stream).unwrap();
        let planes = exr_core::api::planes::planes(&mut file).unwrap();
        let ry = planes.iter().find(|p| p.name == "RY").unwrap();
        assert_eq!(ry.sampling, (2, 2));
        let (ChannelData::Half(full), ChannelData::Half(native)) = (chan("RY"), &ry.data) else { panic!("half") };
        for (x, y) in [(0u32, 0u32), (1, 1), (5, 2), (w - 1, h - 1)] {
            let want = native[(y / 2) as usize * ry.width as usize + (x / 2) as usize];
            assert_eq!(full[(y * w + x) as usize], want, "RY at ({x}, {y})");
        }
    }

    /// A plain subsampled channel (not luminance-chroma) opens scaled up, and no
    /// RGB is invented for it.
    #[test]
    fn subsampled_channel_opens_scaled_up() {
        let frame = DecodedFrame::load(&golden("golden_sub_zip_x2y2.exr")).unwrap();
        let names: Vec<&str> = frame.channel_names.iter().map(String::as_str).collect();
        assert_eq!(names, ["A", "S"]);
        let n = frame.width as usize * frame.height as usize;
        assert!(frame.channels.iter().all(|(_, c)| c.len() == n), "every channel full resolution");
    }

    /// The Metadata panel lists EVERY header attribute, in header order, with the
    /// shared `exrheader` text - not a hand-picked few (it once showed 5).
    #[test]
    fn metadata_lists_every_header_attribute() {
        let path = write_test_exr("meta.exr");
        let header = peek_header(&path).unwrap();
        let names: Vec<String> = header.iter().map(|(n, _)| n.to_string()).collect();
        for frame in [DecodedFrame::open(&path, u64::MAX).unwrap(), DecodedFrame::open(&path, 1).unwrap()] {
            let keys: Vec<&str> = frame.metadata.iter().map(|e| e.key.as_str()).collect();
            assert_eq!(keys, names);
            let value = |k: &str| frame.metadata.iter().find(|e| e.key == k).map(|e| e.value.as_str());
            assert_eq!(value("pixelAspectRatio"), Some("1"));
            assert_eq!(value("dataWindow"), Some("(2 -3) - (701 526)"));
            assert!(value("channels").is_some_and(|v| v.lines().count() == 5), "{:?}", value("channels"));
        }
    }

    /// Wait until `tiles` holds every tile of `level`.
    fn produced(tiles: &mut TileState, level: u32) -> Vec<(crate::tiles::grid::TileId, crate::tiles::view::CachedTile)> {
        let grid = *tiles.grid().unwrap();
        let (c, r) = grid.tiles(level);
        let ids: Vec<_> = (0..r)
            .flat_map(|ty| (0..c).map(move |tx| crate::tiles::grid::TileId { level, tx, ty }))
            .collect();
        tiles.request(ids.iter().copied());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            tiles.drain();
            let got: Vec<_> = ids.iter().filter_map(|&id| tiles.cached(id).map(|t| (id, t))).collect();
            if got.len() == ids.len() {
                return got;
            }
            assert!(std::time::Instant::now() < deadline, "tiles never landed");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// Out-of-core tiling through the viewer: a file-backed frame (budget too small to
    /// hold it) shows the same tiles and the same readout as the same file
    /// decoded whole, for the main layer and a named one.
    #[test]
    fn file_backed_frame_matches_the_resident_one() {
        let path = write_test_exr("backed.exr");
        let whole = DecodedFrame::open(&path, u64::MAX).unwrap();
        assert!(whole.file_types.is_none() && !whole.channels.is_empty(), "fits: decoded whole");
        let backed = DecodedFrame::open(&path, 1).unwrap();
        assert!(backed.channels.is_empty(), "too large: nothing resident");
        assert_eq!(backed.file_types.as_deref().map(<[_]>::len), Some(5));
        assert_eq!(backed.channel_names, whole.channel_names);
        let layers = |f: &DecodedFrame| f.layers.iter().map(|g| (g.name.clone(), g.channels.clone())).collect::<Vec<_>>();
        assert_eq!(layers(&backed), layers(&whole));
        assert_eq!(backed.metadata, whole.metadata);
        assert_eq!((backed.width, backed.height, backed.data_window), (whole.width, whole.height, whole.data_window));

        for layer in ["", "depth"] {
            let mut want_img = ImageBackend::from_frame(Arc::new(DecodedFrame::open(&path, u64::MAX).unwrap()));
            let mut got_img = ImageBackend::from_frame(Arc::new(DecodedFrame::open(&path, 1).unwrap()));
            let (mut want, mut got) = (TileState::new(usize::MAX), TileState::new(usize::MAX));
            for (img, tiles) in [(&mut want_img, &mut want), (&mut got_img, &mut got)] {
                img.set_view(&Selection::layer(layer));
                img.prepare_cpu().unwrap();
                img.sync_source(16384, tiles, None);
            }
            assert!(got_img.pixels.is_empty(), "file-backed frames never materialize");
            let grid = *want.grid().unwrap();
            for level in 0..grid.levels() {
                let a = produced(&mut want, level);
                let b = produced(&mut got, level);
                assert_eq!(a.len(), b.len());
                for ((ia, ta), (ib, tb)) in a.iter().zip(&b) {
                    assert_eq!((ia, &**ta), (ib, &**tb), "layer {layer:?} tile {ia:?}");
                }
            }
            for (x, y) in [(0, 0), (699, 529), (350, 200)] {
                assert_eq!(got_img.readout_at(x, y), want_img.readout_at(x, y), "layer {layer:?} ({x},{y})");
                assert!(got_img.readout_at(x, y).is_some());
            }
        }
    }

    #[test]
    fn file_backing_is_decided_by_size_and_never_for_subsampled() {
        use exr_core::attr::{Channel, ChannelList, ChannelListAttribute};
        let mut h = Header::with_dimensions(
            100, 100, 1.0, imath_rs::V2f { x: 0.0, y: 0.0 }, 1.0,
            exr_core::attr::LineOrder::IncreasingY, exr_core::Compression::Zip,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        cl.insert("Y", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl.clone())).unwrap();
        // 100 x 100 x (16 + 2) bytes = 180 000.
        assert!(!needs_file_backing(&h, 180_000));
        assert!(needs_file_backing(&h, 179_999));
        cl.insert("RY", Channel::new(PixelType::Half, 2, 2, false)).unwrap();
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        assert!(!needs_file_backing(&h, 1), "subsampled files are decoded whole");
    }

    /// The Cryptomatte overlay on a file-backed frame: `set_coverage_file`
    /// switches the tile source to coverage (same tiles as the resident
    /// overlay), clearing the selection returns to the layer, and a new frame
    /// drops it.
    #[test]
    fn file_backed_coverage_overlay_wiring() {
        use cryptomatte::{EncodeOptions, NamedMatte, Windows};
        let (w, h) = (300usize, 200usize);
        let bunny: Vec<f32> = (0..w * h).map(|i| ((i % 5) as f32) / 4.0).collect();
        let enc = cryptomatte::from_named_mattes(
            Windows::origin(w, h).unwrap(),
            &[NamedMatte::new("bunny", &bunny)],
            &EncodeOptions::new("CryptoObject"),
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("exr-view-backed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("crypto_overlay.exr");
        enc.write(&path, Some(exr_core::Compression::Zip)).unwrap();

        let resident = DecodedFrame::open(&path, u64::MAX).unwrap();
        let names: Vec<&str> = resident.channel_names.iter().map(String::as_str).collect();
        let ranked = cryptomatte::ranked_layers_for("CryptoObject", names.iter().copied());
        let id = cryptomatte::name_to_float_id("bunny");
        let matte = cryptomatte::extract_coverage(
            &exr_core::api::Image::read(&path).unwrap(),
            &ranked,
            &[id],
        )
        .unwrap();
        let sel = ("CryptoObject".to_string(), "bunny".to_string());

        let mut want_img = ImageBackend::from_frame(Arc::new(resident));
        want_img.set_coverage_overlay(Some(matte.coverage.clone()), Some(sel.clone()));
        let mut got_img = ImageBackend::from_frame(Arc::new(DecodedFrame::open(&path, 1).unwrap()));
        assert!(got_img.is_file_backed());
        got_img.set_coverage_file(sel.clone(), ranked.clone(), id);
        assert_eq!(got_img.coverage_sel(), Some(&sel));

        let (mut want, mut got) = (TileState::new(usize::MAX), TileState::new(usize::MAX));
        for (img, tiles) in [(&mut want_img, &mut want), (&mut got_img, &mut got)] {
            img.prepare_cpu().unwrap();
            img.sync_source(16384, tiles, None);
        }
        let top = want.grid().unwrap().levels() - 1;
        for level in [0, top] {
            let (a, b) = (produced(&mut want, level), produced(&mut got, level));
            for ((ia, ta), (ib, tb)) in a.iter().zip(&b) {
                assert_eq!((ia, &**ta), (ib, &**tb), "coverage tile {ia:?}");
            }
        }
        assert_eq!(got_img.readout_at(7, 3), want_img.readout_at(7, 3));

        // Clearing the selection returns to the layer; a new frame drops it.
        got_img.set_coverage_overlay(None, None);
        got_img.sync_source(16384, &mut got, None);
        // A crypto-only file's first layer is `CryptoObject00` (R, G, B, A).
        let first = got_img.readout_at(0, 0).map(|r| r[0].0.clone());
        assert_eq!(first.as_deref(), Some("CryptoObject00.R"), "back to the layer, not coverage");
        got_img.set_coverage_file(sel, ranked, id);
        let again = Arc::new(DecodedFrame::open(&path, 1).unwrap());
        got_img.set_frame(again);
        assert_eq!(got_img.coverage_sel(), None);
        assert!(got_img.coverage_file.is_none());
    }
    /// A solo channel is shown alone in grey (opaque) with its own readout row; a
    /// channel absent in the frame falls back to its layer, an absent layer to main.
    #[test]
    fn solo_channel_shows_that_channel_in_grey() {
        let frame = super::tests_support::frame_with(&["A", "B", "G", "N.x", "N.y", "N.z", "R"]);
        let mut img = ImageBackend::from_frame(Arc::new(frame));
        let shown = |img: &mut ImageBackend, sel: Selection| {
            img.set_view(&sel);
            img.prepare_cpu().unwrap();
            (img.pixel_at(0, 0).unwrap(), img.readout.clone())
        };

        let (px, rows) = shown(&mut img, Selection::channel("N", "N.y"));
        assert_eq!(px, [4.0, 4.0, 4.0, 1.0]);
        assert_eq!(rows, [("N.y".to_owned(), 0)]);

        let (px, _) = shown(&mut img, Selection::channel("", "G"));
        assert_eq!(px, [2.0, 2.0, 2.0, 1.0]);

        let (px, _) = shown(&mut img, Selection::channel("N", "N.w"));
        assert_eq!(px, [3.0, 4.0, 5.0, 1.0], "absent channel -> its layer");

        let (px, _) = shown(&mut img, Selection::channel("P", "P.x"));
        assert_eq!(px, [6.0, 2.0, 1.0, 0.0], "absent layer -> main");
    }
}
