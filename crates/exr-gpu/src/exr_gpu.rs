//! # exr-gpu — decoded OpenEXR layers in GPU VRAM
//!
//! Decode an EXR on the CPU with [`exr-core`](exr_core), then upload chosen
//! **Nuke-style channel layers** into `wgpu::Texture`s (typed, linear,
//! scene-referred) for readback or direct wgpu compute/render.
//!
//! A file is exposed as one or more [`Part`]s (a single-part file = one part; a
//! multipart AOV render = one part per AOV). Each part groups its channels into
//! [`Layer`]s; you pick a part + layer to upload.
//!
//! ```no_run
//! # fn demo(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), exr_gpu::GpuError> {
//! let loader = exr_gpu::GpuLoader::open("comp.exr")?;
//! for part in loader.parts() {
//!     for layer in part.layers() {
//!         println!("part {:?} layer {:?}: {} ch", part.name(), layer.label(), layer.len());
//!     }
//! }
//! let part = loader.default_part();
//! let layer = part.default_layer().expect("non-empty");
//! let img = part.load_layer(layer, device, queue, wgpu::TextureUsages::TEXTURE_BINDING)?;
//! # let _ = img; Ok(())
//! # }
//! ```
//!
//! ## Scope
//! Phase A/B: single-part + **multipart** scanline/tiled, full resolution, 1–4-channel
//! uniform-type layers. Deep parts are skipped (no flat texture mapping). Subsampled
//! channels surface as [`GpuError::Unsupported`] (Phase C will add CPU upsample).
//! Mixed-type / >4-channel layers → hard [`GpuError::FormatMismatch`]. Decode-on-GPU
//! and colour management are non-goals (pixels are linear scene-referred).

mod buffer;
mod error;
mod format;
mod layer;
mod multipart;
mod readback;
mod source;
mod upload;

use std::path::Path;

pub use buffer::{BufferChannel, GpuBuffer};
pub use error::{GpuError, Result};
pub use format::{LayerFormat, UploadLayout};
pub use layer::Layer;

use source::SourceImage;

/// One decoded EXR layer uploaded to GPU VRAM. The caller owns `texture` + `view`
/// and drops them to free the memory.
pub struct GpuImage {
    /// The GPU texture holding the layer's pixels.
    pub texture: wgpu::Texture,
    /// A default view over the whole texture, ready to bind.
    pub view: wgpu::TextureView,
    /// The format the texture was created with (e.g. `Rgba16Float`).
    pub format: wgpu::TextureFormat,
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels.
    pub height: u32,
    /// The layer name (empty string = the main / unlayered group).
    pub layer_name: String,
    /// Full EXR channel name per GPU component slot; an empty string marks a pad
    /// slot (e.g. the 4th slot of a 3→RGBA-padded layer).
    pub channel_names: Vec<String>,
    /// Bytes per packed texel (`components * bytes_per_component`). Needed to size
    /// the readback buffer and un-pad its rows.
    pub texel_size: u32,
}

impl GpuImage {
    /// Copy this texture back to CPU as tightly-packed native-endian bytes
    /// (`width * height * texel_size`, row-major, no padding).
    ///
    /// The "process on GPU, get the pixels back" path. The texture MUST have been
    /// uploaded with [`wgpu::TextureUsages::COPY_SRC`] (see [`Part::load_layer`]).
    /// Blocks until the GPU copy + buffer mapping complete.
    pub fn read_back(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u8>> {
        readback::read_back(self, device, queue)
    }

    /// Same as [`GpuImage::read_back`], but un-pads into a caller-owned `out`
    /// buffer instead of allocating a fresh one every call.
    ///
    /// V3: `out` is grown-or-truncated (never reallocated once big enough)
    /// -- pass the SAME `Vec` across repeated readbacks of the same-sized
    /// texture to pay for at most one realloc across the whole run.
    pub fn read_back_into(&self, device: &wgpu::Device, queue: &wgpu::Queue, out: &mut Vec<u8>) -> Result<()> {
        readback::read_back_into(self, device, queue, out)
    }
}

/// One part of an EXR: its name, its decoded channels, and the layers they group
/// into. A single-part file has exactly one (unnamed) part.
pub struct Part {
    /// Part name — empty for a single-part file's only part, else the part's
    /// `name` header attribute (e.g. `"diffuse"`, `"specular"`).
    name: String,
    source: SourceImage,
    layers: Vec<Layer>,
}

impl Part {
    /// Build a part from a decoded source + a name (groups channels into layers).
    pub(crate) fn new(name: String, source: SourceImage) -> Self {
        let layers = layer::group_layers(&source);
        Self { name, source, layers }
    }

    /// The part name (empty string = the single unnamed part).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The part's layers — main (unlayered) first, then named (byte order); channels
    /// within a layer are in canonical component order (R,G,B,A / X,Y,Z / U,V).
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The default layer (main/unlayered if present, else first). `None` only for a
    /// zero-channel part.
    pub fn default_layer(&self) -> Option<&Layer> {
        self.layers.first()
    }

    /// Data-window width / height in pixels.
    pub fn width(&self) -> u32 {
        self.source.width
    }
    pub fn height(&self) -> u32 {
        self.source.height
    }

    /// A decoded channel on the CPU, before any upload: its native-endian
    /// samples and its own sample grid (`width x height`; smaller than the part
    /// for a subsampled channel). `None` for an unknown name.
    pub fn channel_bytes(&self, name: &str) -> Option<(&[u8], u32, u32)> {
        self.source
            .channels
            .iter()
            .find(|c| c.name == name)
            .map(|c| (c.bytes.as_slice(), c.width, c.height))
    }

    /// Upload one of this part's layers to a fresh texture with the given `usage`
    /// (`COPY_DST` is added for you). `> 4`-channel or mixed-type layers →
    /// [`GpuError::FormatMismatch`].
    pub fn load_layer(
        &self,
        layer: &Layer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        usage: wgpu::TextureUsages,
    ) -> Result<GpuImage> {
        upload::upload_layer(&self.source, layer, device, queue, usage)
    }

    /// Same as [`Part::load_layer`], but packs into a caller-owned `scratch`
    /// buffer instead of allocating a fresh one every call.
    ///
    /// V3: `scratch` is grown-or-truncated (never reallocated once big
    /// enough) -- pass the SAME `Vec` across repeated uploads (e.g. a
    /// viewer stepping through a frame sequence) to pay for at most one
    /// realloc across the whole run.
    pub fn load_layer_into(
        &self,
        layer: &Layer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        usage: wgpu::TextureUsages,
        scratch: &mut Vec<u8>,
    ) -> Result<GpuImage> {
        upload::upload_layer_into(&self.source, layer, device, queue, usage, scratch)
    }

    /// VRAM bytes uploading `layer` would allocate (the layer's own
    /// `width * height * texel_size` of its chosen format — a subsampled layer
    /// is smaller than the part). Use against `wgpu_info::vram_budget_bytes(adapter)`
    /// before a bulk `load_all` to avoid OOM on multi-AOV files. Returns
    /// [`GpuError::FormatMismatch`] for a > 4-channel / mixed-type layer.
    pub fn vram_estimate(&self, layer: &Layer) -> Result<u64> {
        let fmt = format::select_format(layer.len(), layer.pixel_type)?;
        // Same checked layout the upload path uses (device-free variant — no limit
        // to enforce at estimate time), so the estimate can never disagree with the
        // bytes the upload/readback actually allocate.
        let layout = UploadLayout::for_texture(layer.width, layer.height, fmt.texel_size())?;
        Ok(layout.byte_size)
    }

    /// Upload ALL of this part's channels planar into one [`GpuBuffer`] with a
    /// channel-offset table — the compute escape hatch (every AOV addressable by
    /// index, no 1–4-channel / texture-format constraints). `usage` should include
    /// `STORAGE` (compute) and/or `COPY_SRC` (readback); `COPY_DST` is added.
    pub fn load_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        usage: wgpu::BufferUsages,
    ) -> Result<GpuBuffer> {
        buffer::load_buffer(&self.source, device, queue, usage)
    }
}

/// A decoded EXR ready for part/layer GPU upload.
///
/// Two-phase: [`GpuLoader::open`] decodes on the CPU (no GPU), so you can inspect
/// [`GpuLoader::parts`] before committing any VRAM, then [`Part::load_layer`]
/// uploads exactly what you want.
pub struct GpuLoader {
    parts: Vec<Part>,
}

impl GpuLoader {
    /// Decode an EXR file (CPU only — no GPU).
    ///
    /// Single-part files become one part via `exr_core::read`; multipart files
    /// become one [`Part`] per image part (deep parts skipped); subsampled
    /// (luminance-chroma) files are reconstructed to full-res RGBA via
    /// `RgbaInputFile`. Deep-only files return [`GpuError::Unsupported`]; a corrupt
    /// file returns [`GpuError::Exr`]. Never panics.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // Try the single-part facade first (the common case). On its escape-hatch
        // errors we branch: multipart → the multipart reader; subsampled → every
        // channel at its native sampling (the low-level FrameBuffer path; no
        // reconstruction — [`GpuLoader::open_rgba`] reconstructs luminance-chroma).
        match exr_core::read(path) {
            Ok(image) => {
                let source = SourceImage::from_image(&image)?;
                let parts = vec![Part::new(String::new(), source)];
                log_open(&parts);
                Ok(Self { parts })
            }
            Err(e) => {
                use exr_core::Unsupported;
                // Branch on the structured escape-hatch kind, not on error text.
                let parts = match exr_core::classify_unsupported(&e) {
                    Some(Unsupported::Multipart) => multipart::load_parts(path)?,
                    Some(Unsupported::Subsampled) => {
                        let path_str = path.to_str().ok_or_else(|| {
                            GpuError::Unsupported(format!("path is not valid UTF-8: {}", path.display()))
                        })?;
                        let mut stream =
                            exr_core::io::StdIfStream::open(path_str).map_err(GpuError::from)?;
                        let mut input =
                            exr_core::scanline::InputFile::new(&mut stream).map_err(GpuError::from)?;
                        vec![Part::new(String::new(), SourceImage::from_reader(&mut input)?)]
                    }
                    // Deep files (and any genuine decode error) have no GPU
                    // fallback — surface them. Deep still maps to
                    // `GpuError::Unsupported` via `From`, preserving the
                    // documented "deep-only files return Unsupported" behavior.
                    Some(Unsupported::Deep) | None => return Err(GpuError::from(e)),
                };
                log_open(&parts);
                Ok(Self { parts })
            }
        }
    }

    /// Open a luminance-chroma (`Y`/`RY`/`BY`) file reconstructed to full-res
    /// RGBA, as C++ `Imf::RgbaInputFile` does (27-tap chroma filter, YCA→RGB,
    /// saturation fix); bit-exact with the C++ decode (exr-core's
    /// `rgba_input_file_matches_cpp_rgba_input_file_bit_exact`). One part with
    /// the main `R,G,B,A` layer. [`GpuLoader::open`] returns the stored channels
    /// instead.
    pub fn open_rgba(path: impl AsRef<Path>) -> Result<Self> {
        let rgba = exr_core::rgba::RgbaInputFile::read(path.as_ref()).map_err(GpuError::from)?;
        let parts = vec![Part::new(String::new(), SourceImage::from_rgba(&rgba)?)];
        log_open(&parts);
        Ok(Self { parts })
    }

    /// All parts (one for a single-part file; one per image part for multipart).
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// The default part (the first). Always present after a successful `open`.
    pub fn default_part(&self) -> &Part {
        &self.parts[0]
    }

    // -- single-part convenience: delegate to part 0 -------------------------

    /// Layers of the default part (single-part convenience).
    pub fn layers(&self) -> &[Layer] {
        self.parts[0].layers()
    }

    /// Default layer of the default part (single-part convenience).
    pub fn default_layer(&self) -> Option<&Layer> {
        self.parts[0].default_layer()
    }

    /// Width / height of the default part.
    pub fn width(&self) -> u32 {
        self.parts[0].width()
    }
    pub fn height(&self) -> u32 {
        self.parts[0].height()
    }

    /// Upload a layer of the DEFAULT part (single-part convenience). For multipart,
    /// use [`Part::load_layer`] on the specific [`Part`].
    pub fn load_layer(
        &self,
        layer: &Layer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        usage: wgpu::TextureUsages,
    ) -> Result<GpuImage> {
        self.parts[0].load_layer(layer, device, queue, usage)
    }
}

/// Log a one-line `info` summary of the opened parts/layers.
fn log_open(parts: &[Part]) {
    log::info!(
        "exr-gpu: opened {} part(s): [{}]",
        parts.len(),
        parts
            .iter()
            .map(|p| {
                let pn = if p.name.is_empty() { "<main>" } else { &p.name };
                format!(
                    "{pn} {}x{} {{{}}}",
                    p.width(),
                    p.height(),
                    p.layers().iter().map(Layer::label).collect::<Vec<_>>().join(",")
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    );
}
