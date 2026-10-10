//! Port of OpenEXR `ImfCompositeDeepScanLine.{h,cpp}` -- `class
//! CompositeDeepScanLine`, which reads one or more deep scanline sources and
//! composites their samples into a flat (non-deep) [`FrameBuffer`], using a
//! [`DeepCompositing`] engine (see `deep::compositing`).
//!
//! Port target (pinned tag v3.3.11 -- unchanged in v3.4.15, confirmed by
//! diffing the file against the oracle checkout):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfCompositeDeepScanLine.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfCompositeDeepScanLine.h>
//!
//! # What this ports
//!
//! * [`CompositeDeepScanLine::add_source`] -- port of the 2 C++ `addSource`
//!   overloads (`DeepScanLineInputFile*` / `DeepScanLineInputPart*`), unified
//!   behind the [`DeepScanLineCompositeSource`] trait (implemented for both
//!   concrete types, mirroring the crate's existing `*CopySource` trait
//!   pattern -- see `deep::scanline_output::DeepScanLineCopySource`).
//! * `Data::check_valid` -- Z/alpha-channel presence checks, display-window
//!   agreement across sources, data-window union.
//! * [`CompositeDeepScanLine::set_frame_buffer`] -- the `channels` /
//!   `buffer_map` bookkeeping (C++ `setFrameBuffer`).
//! * [`CompositeDeepScanLine::read_pixels`] -- the two-pass read (counts,
//!   then samples) + per-pixel composite + flat write-back
//!   (`Data::handleDeepFrameBuffer` + `composite_line`).
//! * [`set_maximum_sample_count`] / [`get_maximum_sample_count`] -- the C++
//!   process-wide static limit (`CompositeDeepScanLine::setMaximumSampleCount`).
//!
//! # Threading
//! C++ spawns one `LineCompositeTask` per scanline on the global
//! `ThreadPool`. This compositor is single-threaded; `read_pixels` here is a
//! plain per-scanline loop, which is
//! observationally identical for the pixels it produces.
//!
//! # Safe-Rust adaptation of the per-pixel pointer arrays
//!
//! C++ allocates ONE contiguous `vector<float>` PER CHANNEL shared across
//! ALL sources, laying out each pixel's samples as [source0's samples for
//! this pixel][source1's samples for this pixel]... contiguously, so that a
//! raw pointer to the FIRST source's block for a pixel is also valid as the
//! start of the COMBINED multi-source block for that pixel (pure pointer
//! arithmetic, `ImfCompositeDeepScanLine.cpp:555-566`). This crate has no
//! raw pointers (exr-core contains no `unsafe` code)
//! and, more fundamentally, every OTHER deep reader in this crate already
//! addresses samples via ONE reader's OWN flat per-channel heap with
//! per-reader cumulative-count addressing (`sample_table::RowPrefixCache`) --
//! there is no existing primitive for INTERLEAVING several readers' outputs
//! into one shared heap the way C++'s pointer trick does (the same flat-heap
//! addressing model is why `DeepSlice::x_stride`/`y_stride` are not used as
//! per-pixel pointer-slot strides).
//!
//! The adaptation here keeps each source's samples in its OWN separate flat
//! heap (exactly like every other deep reader `read_pixels` call in this
//! crate), and the composite step COPIES each contributing source's
//! per-pixel sample slice, in source order, into one small per-pixel scratch
//! array before calling [`DeepCompositing::composite_pixel`]. This is
//! semantically identical to C++'s pointer trick (same sample VALUES, same
//! per-pixel ORDER: source 0's samples first, then source 1's, ...) at the
//! cost of one copy per contributing sample per pixel instead of zero -- a
//! deliberate, documented trade for staying entirely in safe Rust.

use crate::attr::PixelType;
use crate::deep::compositing::{DeepCompositing, DefaultDeepCompositing};
use crate::deep::frame_buffer::{DeepFrameBuffer, DeepSlice};
use crate::error::{ExrError, Result};
use crate::frame_buffer::{FrameBuffer, Slice};
use crate::header::Header;

use imath_rs::Box2i;

use std::sync::atomic::{AtomicI64, Ordering as AtomicOrdering};

/// A deep scanline reader usable as a [`CompositeDeepScanLine`] source --
/// unifies the 2 C++ `addSource` overloads (`DeepScanLineInputFile*` /
/// `DeepScanLineInputPart*`) behind one trait, mirroring the crate's
/// existing `*CopySource` pattern (`deep::scanline_output::
/// DeepScanLineCopySource`). Implemented directly for
/// [`crate::deep::DeepScanLineInputFile`] (identity forwarders); implemented
/// for `multipart::part::DeepScanLineInputPart` in `multipart/part.rs`
/// (same reasoning as `TiledInputPart`'s own forwarding impl: trait bounds
/// on a generic/trait-object parameter are not satisfied via `Deref`
/// coercion the way a concrete-type call site is).
pub trait DeepScanLineCompositeSource {
    /// The part/file's header -- C++ `header()`.
    fn header(&self) -> &Header;
    /// Set the pixel destination -- C++ `setFrameBuffer`.
    fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()>;
    /// Read per-pixel sample counts for `[scan_line1, scan_line2]` --
    /// C++ `readPixelSampleCounts`.
    fn read_pixel_sample_counts(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &mut [u8],
    ) -> Result<()>;
    /// Read sample data for `[scan_line1, scan_line2]` -- C++ `readPixels`.
    fn read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()>;
}

// Generic over the stream holder `S` (not just the default `&'a mut
// dyn IStream`) so `scanline::InputFile<'a, S>` can add its OWNED
// `DeepScanLineInputFile<'a, S>` as a compositor source too, not only the
// borrowed-stream one — mirrors the crate-wide owned-stream genericization
// pattern (`ScanLineInputFile`/`TiledInputFile`/`DeepScanLineInputFile`
// themselves).
impl<'a, S> DeepScanLineCompositeSource for crate::deep::DeepScanLineInputFile<'a, S>
where
    S: std::ops::DerefMut<Target = dyn crate::io::IStream + 'a>,
{
    fn header(&self) -> &Header {
        crate::deep::DeepScanLineInputFile::header(self)
    }
    fn set_frame_buffer(&mut self, frame_buffer: &DeepFrameBuffer) -> Result<()> {
        crate::deep::DeepScanLineInputFile::set_frame_buffer(self, frame_buffer)
    }
    fn read_pixel_sample_counts(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &mut [u8],
    ) -> Result<()> {
        crate::deep::DeepScanLineInputFile::read_pixel_sample_counts(
            self,
            scan_line1,
            scan_line2,
            sample_counts,
        )
    }
    fn read_pixels(
        &mut self,
        scan_line1: i32,
        scan_line2: i32,
        sample_counts: &[u8],
        sample_data: &mut [u8],
    ) -> Result<()> {
        crate::deep::DeepScanLineInputFile::read_pixels(
            self,
            scan_line1,
            scan_line2,
            sample_counts,
            sample_data,
        )
    }
}

/// Process-wide maximum composited sample count -- C++'s file-static
/// `int64_t maximumSampleCount = 0` (`ImfCompositeDeepScanLine.cpp:421`).
/// `AtomicI64` for safe global mutability without `unsafe` (C++'s plain
/// global is not thread-safe either; this is at least as safe).
static MAXIMUM_SAMPLE_COUNT: AtomicI64 = AtomicI64::new(0);

/// Set the maximum number of samples [`CompositeDeepScanLine::read_pixels`]
/// will composite on a single call -- port of
/// `CompositeDeepScanLine::setMaximumSampleCount`. Matches the C++ check
/// verbatim: the limit is enforced only when `count > 0` (a `count` of `0`
/// or negative disables enforcement entirely -- see the doc comment on
/// [`CompositeDeepScanLine::read_pixels`] for the divergence between this
/// literal C++ behaviour and the header's own doc comment).
pub fn set_maximum_sample_count(count: i64) {
    MAXIMUM_SAMPLE_COUNT.store(count, AtomicOrdering::SeqCst);
}

/// The current maximum sample count -- port of
/// `CompositeDeepScanLine::getMaximumSampleCount`.
pub fn get_maximum_sample_count() -> i64 {
    MAXIMUM_SAMPLE_COUNT.load(AtomicOrdering::SeqCst)
}

/// Composites one or more deep scanline sources into a flat [`FrameBuffer`]
/// -- port of `class CompositeDeepScanLine`.
pub struct CompositeDeepScanLine<'a> {
    sources: Vec<&'a mut dyn DeepScanLineCompositeSource>,
    output_frame_buffer: FrameBuffer,
    /// `true` once any added source's header has a `ZBack` channel --
    /// C++ `Data::_zback` (an `if`, never reset once set).
    zback: bool,
    /// Union of every added source's data window -- C++ `Data::_dataWindow`.
    /// `None` until the first source is added.
    data_window: Option<Box2i>,
    /// The FIRST added source's display window, checked against every
    /// subsequent source -- C++ compares against `_part[0]`/`_file[0]`'s
    /// header directly; kept explicitly here since sources are trait
    /// objects, not a concrete indexable type.
    display_window: Option<Box2i>,
    /// `["Z", "ZBack" or "Z", "A", ...extra channels]` in first-seen order --
    /// C++ `Data::_channels`.
    channels: Vec<String>,
    /// `output_frame_buffer`'s Nth slice maps to `channels[buffer_map[N]]` --
    /// C++ `Data::_bufferMap`.
    buffer_map: Vec<usize>,
    /// User-supplied compositor, or `None` to use [`DefaultDeepCompositing`]
    /// (C++'s local fallback `DeepCompositing d;`).
    compositor: Option<&'a mut dyn DeepCompositing>,
}

impl<'a> Default for CompositeDeepScanLine<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CompositeDeepScanLine<'a> {
    /// Create an empty compositor with no sources -- C++
    /// `CompositeDeepScanLine::CompositeDeepScanLine`.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            output_frame_buffer: FrameBuffer::new(),
            zback: false,
            data_window: None,
            display_window: None,
            channels: Vec::new(),
            buffer_map: Vec::new(),
            compositor: None,
        }
    }

    /// Add a deep scanline source (a file or a multipart part; anything
    /// implementing [`DeepScanLineCompositeSource`]) -- port of the 2 C++
    /// `addSource` overloads, unified.
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if the source's header is missing a `Z` or `A`
    ///   channel, or (for the 2nd+ source) its display window disagrees with
    ///   the first source's (matching the C++ `THROW`s in `check_valid`).
    pub fn add_source(&mut self, source: &'a mut dyn DeepScanLineCompositeSource) -> Result<()> {
        self.check_valid(source.header())?;
        self.sources.push(source);
        Ok(())
    }

    /// C++ `Data::check_valid`.
    fn check_valid(&mut self, header: &Header) -> Result<()> {
        let channels = header.channels()?;
        let mut has_z = false;
        let mut has_alpha = false;
        for (name, _ch) in channels.iter() {
            match name.to_str() {
                Some("ZBack") => self.zback = true,
                Some("Z") => has_z = true,
                Some("A") => has_alpha = true,
                _ => {}
            }
        }
        if !has_z {
            return Err(ExrError::Arg(
                "Deep data provided to CompositeDeepScanLine is missing a Z channel".into(),
            ));
        }
        if !has_alpha {
            return Err(ExrError::Arg(
                "Deep data provided to CompositeDeepScanLine is missing an alpha channel".into(),
            ));
        }

        if self.sources.is_empty() {
            // first in - update and return.
            self.data_window = Some(header.data_window()?);
            self.display_window = Some(header.display_window()?);
            return Ok(());
        }

        let dw = header.display_window()?;
        if self.display_window != Some(dw) {
            return Err(ExrError::Arg(
                "Deep data provided to CompositeDeepScanLine has a different displayWindow \
                 to previously provided data"
                    .into(),
            ));
        }

        let mut data_window = self.data_window.ok_or_else(|| {
            ExrError::Logic("CompositeDeepScanLine: data window unset with sources present".into())
        })?;
        data_window.extend_by_box(&header.data_window()?);
        self.data_window = Some(data_window);
        Ok(())
    }

    /// Number of sources added so far -- C++ `sources()`.
    pub fn sources(&self) -> usize {
        self.sources.len()
    }

    /// The union data window of every added source -- C++ `dataWindow()`.
    /// Returns the default (empty/inverted) `Box2i` when no source has been
    /// added yet, matching a default-constructed C++ `Box2i`.
    pub fn data_window(&self) -> Box2i {
        self.data_window.unwrap_or_default()
    }

    /// Install a custom compositing engine -- C++ `setCompositing`.
    pub fn set_compositing(&mut self, compositor: &'a mut dyn DeepCompositing) {
        self.compositor = Some(compositor);
    }

    /// The currently set output frame buffer -- C++ `frameBuffer()`.
    pub fn frame_buffer(&self) -> &FrameBuffer {
        &self.output_frame_buffer
    }

    /// Set the flat output frame buffer -- C++ `setFrameBuffer`.
    ///
    /// # Errors
    /// [`ExrError::Arg`] if any requested slice has x/y sampling other than
    /// `1` (matching the C++ `THROW` -- deep files never subsample, so a
    /// subsampled destination slice can never be filled meaningfully).
    pub fn set_frame_buffer(&mut self, frame_buffer: &FrameBuffer) -> Result<()> {
        let mut channels = vec![
            "Z".to_string(),
            if self.zback { "ZBack".to_string() } else { "Z".to_string() },
            "A".to_string(),
        ];
        let mut buffer_map = Vec::new();

        for (name, slice) in frame_buffer.iter() {
            if slice.x_sampling != 1 || slice.y_sampling != 1 {
                return Err(ExrError::Arg(format!(
                    "X and/or y subsampling factors of \"{name}\" channel in framebuffer \
                     are not 1"
                )));
            }
            match name.to_str() {
                Some("ZBack") => buffer_map.push(1),
                Some("Z") => buffer_map.push(0),
                Some("A") => buffer_map.push(2),
                _ => {
                    buffer_map.push(channels.len());
                    // `self.channels` stays `Vec<String>` and
                    // is later used to look the source channel back up by
                    // name; a non-UTF-8 extra channel name round-trips only
                    // through its escaped `Display` form here, which will not
                    // re-match the source's real `Name` byte-for-byte. Z/
                    // ZBack/A (the only channels every deep source is
                    // guaranteed to have) are unaffected.
                    channels.push(name.to_string());
                }
            }
        }

        self.channels = channels;
        self.buffer_map = buffer_map;
        self.output_frame_buffer = frame_buffer.clone();
        Ok(())
    }

    /// Read scanlines `[start, end]` from every source and composite them
    /// into the output frame buffer -- port of `CompositeDeepScanLine::
    /// readPixels` (`Data::handleDeepFrameBuffer` + `composite_line`,
    /// single-threaded -- see the module docs).
    ///
    /// `out` is the caller-owned flat output buffer the frame buffer set via
    /// [`Self::set_frame_buffer`] addresses (pixel data is passed per call:
    /// this crate's `FrameBuffer`/`Slice` only ever record
    /// LAYOUT -- byte offsets/strides/types -- the same convention every
    /// other reader/writer in the crate uses, e.g.
    /// `scanline::input::ScanLineInputFile::read_pixels(.., out: &mut
    /// [u8])`; C++'s `FrameBuffer` instead embeds real pointers directly in
    /// each `Slice`, which this port's safe-Rust addressing model replaces
    /// with `base_offset` into this externally supplied buffer).
    ///
    /// # Errors
    /// * [`ExrError::Arg`] if a pixel's combined sample count across all
    ///   sources overflows `u32`, or exceeds
    ///   [`get_maximum_sample_count`] (when positive; see that function's
    ///   own doc for the literal, not-quite-matching-the-C++-doc-comment
    ///   semantics this ports verbatim).
    /// * Propagates each source's `set_frame_buffer`/`read_pixel_sample_counts`/
    ///   `read_pixels` errors.
    pub fn read_pixels(&mut self, start: i32, end: i32, out: &mut [u8]) -> Result<()> {
        if self.sources.is_empty() {
            // Nothing to composite -- not reached by any C++ call site
            // (which always has >=1 source by construction) but a clean,
            // documented no-op is safer than dereferencing an unset,
            // default-constructed data window.
            return Ok(());
        }
        let dw = self.data_window.expect("checked: sources is non-empty");
        let width = (dw.max.x - dw.min.x + 1) as usize;
        let rows = (end - start + 1) as usize;
        let pixel_count = width * rows;
        let num_sources = self.sources.len();

        // --- Pass 1: per-source LOCAL sample counts -------------------
        // C++ `Data::handleDeepFrameBuffer` builds a count slice whose base
        // maps (dataWindow.min.x, start) -> counts[0] -- a LOCAL raster
        // covering only the requested rows, same convention every other
        // per-range deep read in this crate uses.
        let count_x_stride = 4i32;
        let count_y_stride = (4 * width) as i32;
        let count_base_offset =
            -(start as i64 * count_y_stride as i64 + dw.min.x as i64 * count_x_stride as i64);

        let mut counts: Vec<Vec<u8>> = Vec::with_capacity(num_sources);
        for source in self.sources.iter_mut() {
            let mut sc_fb = DeepFrameBuffer::new();
            sc_fb.insert_sample_count_slice(Slice::new(
                PixelType::Uint,
                count_base_offset,
                count_x_stride as usize,
                count_y_stride as usize,
                1,
                1,
                0.0,
                false,
                false,
            )?)?;
            // The deep channel slices are unused by `read_pixel_sample_counts`
            // (only the count slice is consulted); the source's own
            // `set_frame_buffer` still requires at least the count slice to
            // be set, which is all this pass needs.
            source.set_frame_buffer(&sc_fb)?;
            let mut sample_counts = vec![0u8; pixel_count * 4];
            source.read_pixel_sample_counts(start, end, &mut sample_counts)?;
            counts.push(sample_counts);
        }

        // --- Accumulate combined per-pixel counts ----------------------
        let mut total_sizes = vec![0u32; pixel_count];
        let mut num_nonzero_sources = vec![0u32; pixel_count];
        let mut overall_sample_count: i64 = 0;
        for p in 0..pixel_count {
            let mut total: u64 = 0;
            let mut n: u32 = 0;
            for sc in &counts {
                let off = p * 4;
                let c = u32::from_le_bytes(sc[off..off + 4].try_into().map_err(|_| {
                    ExrError::Logic("CompositeDeepScanLine: count buffer slice error".into())
                })?);
                total += c as u64;
                if total > u32::MAX as u64 {
                    return Err(ExrError::Arg(
                        "Cannot composite scanline: pixel cannot have more than UINT_MAX \
                         samples"
                            .into(),
                    ));
                }
                if c > 0 {
                    n += 1;
                }
            }
            total_sizes[p] = total as u32;
            num_nonzero_sources[p] = n;
            overall_sample_count += total as i64;
        }

        let max_samples = get_maximum_sample_count();
        if max_samples > 0 && overall_sample_count > max_samples {
            return Err(ExrError::Arg(
                "Cannot composite scanline: total sample count on scanline exceeds limit \
                 set by set_maximum_sample_count()"
                    .into(),
            ));
        }

        // --- Pass 2: per-source sample DATA, in its OWN flat heap -------
        // Each channel gets its own contiguous `total_for_source * 4`-byte
        // slab within one combined buffer for that source (see the module
        // docs for why this differs from C++'s single-shared-heap pointer
        // trick).
        let mut per_source_data: Vec<Vec<u8>> = Vec::with_capacity(num_sources);
        let mut per_source_prefix: Vec<Vec<u64>> = Vec::with_capacity(num_sources);
        for (idx, source) in self.sources.iter_mut().enumerate() {
            let sc = &counts[idx];
            let mut prefix = vec![0u64; pixel_count + 1];
            for p in 0..pixel_count {
                let off = p * 4;
                let c = u32::from_le_bytes(sc[off..off + 4].try_into().unwrap_or([0; 4]));
                prefix[p + 1] = prefix[p] + c as u64;
            }
            let total_for_source = *prefix.last().unwrap_or(&0);

            let mut deep_fb = DeepFrameBuffer::new();
            deep_fb.insert_sample_count_slice(Slice::new(
                PixelType::Uint,
                count_base_offset,
                count_x_stride as usize,
                count_y_stride as usize,
                1,
                1,
                0.0,
                false,
                false,
            )?)?;
            for (ch_idx, name) in self.channels.iter().enumerate() {
                // The ZBack "channel" only exists as a genuinely separate
                // slice when `self.zback` is set; otherwise index 1 is a
                // duplicate name of "Z" and inserting it twice under the
                // SAME key would just overwrite the first -- C++ likewise
                // only allocates `pointers[1]` when `_zback` (`:176-187`).
                if ch_idx == 1 && !self.zback {
                    continue;
                }
                let base_offset = ch_idx as i64 * total_for_source as i64 * 4;
                deep_fb.insert(
                    name.clone(),
                    DeepSlice::new(
                        PixelType::Float,
                        base_offset,
                        8,
                        8 * width,
                        4,
                        1,
                        1,
                        0.0,
                        false,
                        false,
                    )?,
                )?;
            }

            let buf_channel_count = self.channels.len();
            let mut sample_data = vec![0u8; buf_channel_count * total_for_source as usize * 4];
            source.set_frame_buffer(&deep_fb)?;
            source.read_pixels(start, end, sc, &mut sample_data)?;

            per_source_data.push(sample_data);
            per_source_prefix.push(prefix);
        }

        // --- Composite each pixel and write back ------------------------
        let mut default_comp = DefaultDeepCompositing;
        let num_channels = self.channels.len();
        let mut inputs_owned: Vec<Vec<f32>> = vec![Vec::new(); num_channels];
        let mut outputs = vec![0.0f32; num_channels];
        let channel_names: Vec<&str> = self.channels.iter().map(|s| s.as_str()).collect();

        for y in start..=end {
            let row = (y - start) as usize;
            for x in dw.min.x..=dw.max.x {
                let col = (x - dw.min.x) as usize;
                let pixel = row * width + col;
                let num_samples = total_sizes[pixel] as usize;
                let sources_here = num_nonzero_sources[pixel] as usize;

                for buf in inputs_owned.iter_mut() {
                    buf.clear();
                }
                if num_samples > 0 {
                    for src_idx in 0..num_sources {
                        let prefix = &per_source_prefix[src_idx];
                        let s0 = prefix[pixel] as usize;
                        let s1 = prefix[pixel + 1] as usize;
                        if s1 == s0 {
                            continue;
                        }
                        let data = &per_source_data[src_idx];
                        let total_for_source = *prefix.last().unwrap_or(&0) as usize;
                        for (ch_idx, buf) in inputs_owned.iter_mut().enumerate() {
                            let eff_ch = if ch_idx == 1 && !self.zback { 0 } else { ch_idx };
                            let base = eff_ch * total_for_source * 4;
                            for s in s0..s1 {
                                let off = base + s * 4;
                                let bytes: [u8; 4] = data[off..off + 4].try_into().map_err(|_| {
                                    ExrError::Logic(
                                        "CompositeDeepScanLine: sample data slice error".into(),
                                    )
                                })?;
                                buf.push(f32::from_le_bytes(bytes));
                            }
                        }
                    }
                }

                let inputs: Vec<&[f32]> = inputs_owned.iter().map(|v| v.as_slice()).collect();
                let comp: &mut dyn DeepCompositing = match self.compositor.as_deref_mut() {
                    Some(c) => c,
                    None => &mut default_comp,
                };
                comp.composite_pixel(
                    &mut outputs,
                    &inputs,
                    &channel_names,
                    num_samples,
                    sources_here,
                );

                for (channel_number, (_name, slice)) in
                    self.output_frame_buffer.iter().enumerate()
                {
                    let value = outputs[self.buffer_map[channel_number]];
                    let addr = slice.base_offset
                        + y as i64 * slice.y_stride as i64
                        + x as i64 * slice.x_stride as i64;
                    if addr < 0 {
                        return Err(ExrError::Arg(
                            "CompositeDeepScanLine: output frame buffer address is out of \
                             range (negative)."
                                .into(),
                        ));
                    }
                    // C++ only handles FLOAT and HALF, silently no-oping any
                    // other type (`ImfCompositeDeepScanLine.cpp:389-400`).
                    let addr = addr as usize;
                    let out_len = out.len();
                    match slice.type_ {
                        PixelType::Float => {
                            let dst = out.get_mut(addr..addr + 4).ok_or_else(|| {
                                ExrError::Arg(format!(
                                    "CompositeDeepScanLine: output address {addr} out of range \
                                     of the output buffer (len {out_len})"
                                ))
                            })?;
                            dst.copy_from_slice(&value.to_le_bytes());
                        }
                        PixelType::Half => {
                            let h = imath_rs::Half::from(value).bits();
                            let dst = out.get_mut(addr..addr + 2).ok_or_else(|| {
                                ExrError::Arg(format!(
                                    "CompositeDeepScanLine: output address {addr} out of range \
                                     of the output buffer (len {out_len})"
                                ))
                            })?;
                            dst.copy_from_slice(&h.to_le_bytes());
                        }
                        PixelType::Uint | PixelType::NumPixelTypes => {}
                    }
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder};
    use crate::deep::DeepScanLineOutputFile;
    use crate::io::{MemIStream, MemOStream};
    use imath_rs::V2f;

    /// Build a single-scanline (`w`x1) deep header with the given channels
    /// (all HALF-independent: we always write FLOAT samples in these tests).
    fn make_header(w: i32, channels: &[&str]) -> Header {
        let mut h = Header::with_dimensions(
            w,
            1,
            1.0,
            V2f { x: 0.0, y: 0.0 },
            1.0,
            LineOrder::IncreasingY,
            Compression::NoCompression,
        )
        .unwrap();
        let mut cl = ChannelList::new();
        for name in channels {
            cl.insert(*name, Channel::new(PixelType::Float, 1, 1, false)).unwrap();
        }
        h.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        h
    }

    /// Write a single-scanline deep file: `counts[x]` samples at column `x`,
    /// `values[channel][x]` giving that channel's `Vec<f32>` samples for
    /// column `x` (length `counts[x]`, in on-disk order -- the caller decides
    /// whether that's front-to-back).
    fn write_deep_row(w: i32, channels: &[&str], counts: &[u32], values: &[Vec<Vec<f32>>]) -> Vec<u8> {
        assert_eq!(counts.len(), w as usize);
        assert_eq!(values.len(), channels.len());

        // Flat sample-count buffer: u32 LE per pixel.
        let mut sc = Vec::with_capacity(w as usize * 4);
        for &c in counts {
            sc.extend_from_slice(&c.to_le_bytes());
        }

        // Flat per-channel sample heaps, concatenated: each channel's own
        // heap is `total_samples * 4` bytes, laid out contiguously in
        // raster (here: column) order.
        let total: u32 = counts.iter().sum();
        let mut data = vec![0u8; channels.len() * total as usize * 4];
        for (ch_idx, per_x) in values.iter().enumerate() {
            let mut offset = 0usize;
            for (x, samples) in per_x.iter().enumerate() {
                assert_eq!(samples.len(), counts[x] as usize);
                for &v in samples {
                    let byte_off = (ch_idx * total as usize + offset) * 4;
                    data[byte_off..byte_off + 4].copy_from_slice(&v.to_le_bytes());
                    offset += 1;
                }
            }
        }

        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(Slice {
            type_: PixelType::Uint,
            base_offset: 0,
            x_stride: 4,
            y_stride: 4 * w as usize,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        })
        .unwrap();
        for (ch_idx, name) in channels.iter().enumerate() {
            fb.insert(
                *name,
                DeepSlice::new(
                    PixelType::Float,
                    ch_idx as i64 * total as i64 * 4,
                    8,
                    8 * w as usize,
                    4,
                    1,
                    1,
                    0.0,
                    false,
                    false,
                )
                .unwrap(),
            )
            .unwrap();
        }

        let header = make_header(w, channels);
        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, 1).unwrap();
        }
        out.into_data()
    }

    /// A flat output slot for one channel of one pixel row -- `1` pixel wide
    /// output buffer, one f32 per column.
    fn make_output(w: i32) -> (Vec<u8>, FrameBuffer) {
        let out = vec![0u8; w as usize * 4];
        let mut fb = FrameBuffer::new();
        fb.insert(
            "A",
            Slice {
                type_: PixelType::Float,
                base_offset: 0,
                x_stride: 4,
                y_stride: 4 * w as usize,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        (out, fb)
    }

    /// A single deep source's samples must be composited front-to-back
    /// WITHOUT re-sorting (`sources <= 1`) -- two half-opaque samples already
    /// in front-to-back order accumulate to `0.75` (hand-derived: `0.5 +
    /// (1-0.5)*0.5`), and the empty second pixel composites to `0`.
    #[test]
    fn composite_single_source_front_to_back() {
        let w = 2;
        let channels = ["Z", "A"];
        let counts = [2u32, 0];
        let values = vec![
            vec![vec![0.0, 1.0], vec![]], // Z
            vec![vec![0.5, 0.5], vec![]], // A
        ];
        let bytes = write_deep_row(w, &channels, &counts, &values);

        let mut is = MemIStream::new(bytes);
        let mut reader = crate::deep::DeepScanLineInputFile::new(&mut is).unwrap();

        let mut comp = CompositeDeepScanLine::new();
        comp.add_source(&mut reader).unwrap();

        let (mut out, fb) = make_output(w);
        comp.set_frame_buffer(&fb).unwrap();
        comp.read_pixels(0, 0, &mut out).unwrap();

        let a0 = f32::from_le_bytes(out[0..4].try_into().unwrap());
        let a1 = f32::from_le_bytes(out[4..8].try_into().unwrap());
        assert!((a0 - 0.75).abs() < 1e-6, "expected 0.75, got {a0}");
        assert_eq!(a1, 0.0, "an empty pixel must composite to 0");
    }

    /// Two sources with overlapping, out-of-depth-order samples at the
    /// same pixel must be re-sorted front-to-back before compositing -- the
    /// front (nearer) opaque sample must win regardless of source order,
    /// verified via a THIRD channel ("R") whose value differs per source.
    #[test]
    fn composite_multi_source_sorted_front_to_back() {
        let w = 1;
        let channels = ["Z", "A", "R"];

        // Source 0: the BACK sample (Z=1, opaque, R=9).
        let back_bytes = write_deep_row(
            w,
            &channels,
            &[1],
            &[vec![vec![1.0]], vec![vec![1.0]], vec![vec![9.0]]],
        );
        // Source 1: the FRONT sample (Z=0, opaque, R=5).
        let front_bytes = write_deep_row(
            w,
            &channels,
            &[1],
            &[vec![vec![0.0]], vec![vec![1.0]], vec![vec![5.0]]],
        );

        let mut back_is = MemIStream::new(back_bytes);
        let mut back_reader = crate::deep::DeepScanLineInputFile::new(&mut back_is).unwrap();
        let mut front_is = MemIStream::new(front_bytes);
        let mut front_reader = crate::deep::DeepScanLineInputFile::new(&mut front_is).unwrap();

        let mut comp = CompositeDeepScanLine::new();
        // Add the BACK source first (source order must not matter -- the
        // compositor must sort by Z, not by add_source call order).
        comp.add_source(&mut back_reader).unwrap();
        comp.add_source(&mut front_reader).unwrap();

        // Interleaved 2-float-per-pixel output layout (A then R).
        let mut fb = FrameBuffer::new();
        fb.insert(
            "A",
            Slice {
                type_: PixelType::Float,
                base_offset: 0,
                x_stride: 8,
                y_stride: 8 * w as usize,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        fb.insert(
            "R",
            Slice {
                type_: PixelType::Float,
                base_offset: 4,
                x_stride: 8,
                y_stride: 8 * w as usize,
                x_sampling: 1,
                y_sampling: 1,
                fill_value: 0.0,
                x_tile_coords: false,
                y_tile_coords: false,
            },
        )
        .unwrap();
        let mut out = vec![0u8; w as usize * 8];

        comp.set_frame_buffer(&fb).unwrap();
        comp.read_pixels(0, 0, &mut out).unwrap();

        let a = f32::from_le_bytes(out[0..4].try_into().unwrap());
        let r = f32::from_le_bytes(out[4..8].try_into().unwrap());
        assert!((a - 1.0).abs() < 1e-6, "expected fully opaque, got {a}");
        assert!(
            (r - 5.0).abs() < 1e-6,
            "expected the FRONT source's R value (5.0) to win, got {r}"
        );
    }

    /// Write a `w`x`h` deep file whose data window starts at line `min_y`: pixel `(x, y)` holds
    /// `(x + k * y + src) % 3` samples (`k` = 2 for source 1) with Z / A / R values derived from
    /// `src`, `y`, `x` and the sample index, so lines and sources are all distinguishable.
    fn write_deep_image(w: i32, h: i32, min_y: i32, src: i32) -> Vec<u8> {
        let k = 1 + src;
        let mut header = make_header(w, &["Z", "A", "R"]);
        let dw = Box2i {
            min: imath_rs::V2i { x: 0, y: min_y },
            max: imath_rs::V2i { x: w - 1, y: min_y + h - 1 },
        };
        *header.data_window_mut().unwrap() = dw;

        // Whole-window layouts: counts addressed by absolute pixel, one flat heap per channel.
        let ys = 4 * w as usize;
        let mut sc = vec![0u8; h as usize * ys];
        let mut samples: [Vec<f32>; 3] = Default::default();
        for y in min_y..min_y + h {
            for x in 0..w {
                let n = (x + k * y + src).rem_euclid(3);
                let off = (y - min_y) as usize * ys + x as usize * 4;
                sc[off..off + 4].copy_from_slice(&(n as u32).to_le_bytes());
                for s in 0..n {
                    samples[0].push((y * 10 + x) as f32 + s as f32 * 0.5 + src as f32 * 0.25);
                    samples[1].push(0.2 + 0.1 * s as f32);
                    samples[2].push((src * 100 + y * 10 + x + s) as f32);
                }
            }
        }
        let total = samples[0].len();
        let mut data = Vec::new();
        for ch in &samples {
            for v in ch {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut fb = DeepFrameBuffer::new();
        fb.insert_sample_count_slice(
            Slice::new(PixelType::Uint, -(min_y as i64 * ys as i64), 4, ys, 1, 1, 0.0, false, false).unwrap(),
        )
        .unwrap();
        for (i, name) in ["Z", "A", "R"].iter().enumerate() {
            fb.insert(
                *name,
                DeepSlice::new(PixelType::Float, (i * total * 4) as i64, 8, 8 * w as usize, 4, 1, 1, 0.0, false, false)
                    .unwrap(),
            )
            .unwrap();
        }
        let mut out = MemOStream::new();
        {
            let mut df = DeepScanLineOutputFile::new(&mut out, &header).unwrap();
            df.set_frame_buffer(&fb).unwrap();
            df.write_pixels(&sc, &data, h).unwrap();
        }
        out.into_data()
    }

    /// `CompositeDeepScanLine::readPixels(start, end)` is a per-line operation in C++
    /// (`ImfCompositeDeepScanLine.cpp` composites each scanline independently), so any range --
    /// a single line, an interior or trailing sub-range, not only one starting at the data
    /// window's first line -- must give exactly the pixels the whole-window read gives for those
    /// lines. Two overlapping sources, a window starting at line 5, every `[start, end]`.
    #[test]
    fn read_pixels_of_any_range_matches_the_whole_window() {
        for min_y in [-12, 5] {
        let (w, h) = (3i32, 5i32);
        let files = [write_deep_image(w, h, min_y, 0), write_deep_image(w, h, min_y, 1)];
        let row = 12 * w as usize;

        let read = |start: i32, end: i32| -> Vec<u8> {
            let mut streams: Vec<MemIStream> = files.iter().map(|f| MemIStream::new(f.clone())).collect();
            let mut readers: Vec<_> = streams
                .iter_mut()
                .map(|s| crate::deep::DeepScanLineInputFile::new(s).unwrap())
                .collect();
            let mut comp = CompositeDeepScanLine::new();
            for r in readers.iter_mut() {
                comp.add_source(r).unwrap();
            }
            // Z, A, R interleaved as three floats per pixel, line `min_y` at offset 0.
            let mut fb = FrameBuffer::new();
            for (i, name) in ["Z", "A", "R"].iter().enumerate() {
                fb.insert(
                    *name,
                    Slice {
                        type_: PixelType::Float,
                        base_offset: 4 * i as i64 - min_y as i64 * row as i64,
                        x_stride: 12,
                        y_stride: row,
                        x_sampling: 1,
                        y_sampling: 1,
                        fill_value: 0.0,
                        x_tile_coords: false,
                        y_tile_coords: false,
                    },
                )
                .unwrap();
            }
            comp.set_frame_buffer(&fb).unwrap();
            let mut out = vec![0u8; row * h as usize];
            comp.read_pixels(start, end, &mut out).unwrap();
            out
        };

        let whole = read(min_y, min_y + h - 1);
        assert!(
            whole.as_chunks::<4>().0.iter().any(|c| f32::from_le_bytes(*c) != 0.0),
            "premise: the composite is not all zero"
        );
        for start in min_y..min_y + h {
            for end in start..min_y + h {
                let part = read(start, end);
                let (a, b) = ((start - min_y) as usize * row, (end - min_y + 1) as usize * row);
                assert_eq!(part[a..b], whole[a..b], "lines [{start}, {end}]");
                assert!(
                    part[..a].iter().chain(&part[b..]).all(|&v| v == 0),
                    "lines [{start}, {end}]: wrote outside the requested lines"
                );
            }
        }
        }
    }
}
