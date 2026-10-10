//! Multipart adapter — turn each image PART of a multipart EXR into a [`Part`].
//!
//! Multipart AOV/comp renders carry one part per AOV (each with its own header,
//! channels and data window). We read every scanline/tiled IMAGE part through the
//! facade's `Image::from_{input,tiled_input}_file` helpers (single point of
//! responsibility for "reader → Image") and wrap it as a [`Part`]; DEEP parts have
//! no flat-texture mapping and are skipped (logged). A file with no mappable image
//! part returns [`GpuError::Unsupported`].

use std::path::Path;

use exr_core::header::{SCANLINE_IMAGE, TILED_IMAGE};
use exr_core::{Image, MultiPartInputFile, StdIfStream};

use crate::error::{GpuError, Result};
use crate::source::SourceImage;
use crate::Part;

/// Read all image parts of the multipart file at `path` into [`Part`]s.
pub(crate) fn load_parts(path: &Path) -> Result<Vec<Part>> {
    let path_str = path.to_str().ok_or_else(|| {
        GpuError::Unsupported(format!("path is not valid UTF-8: {}", path.display()))
    })?;

    let mut stream = StdIfStream::open(path_str).map_err(GpuError::from)?;
    let mut mp = MultiPartInputFile::new(&mut stream).map_err(GpuError::from)?;
    let n_parts = mp.parts();

    let mut parts = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    for n in 0..n_parts {
        let part_type = mp.part_type(n).map_err(GpuError::from)?.to_string();
        // Resolve the name as an OWNED string before the mutable `*_input_part`
        // borrow below (the &Header borrow must end first).
        let name = mp
            .header(n)
            .ok()
            .and_then(|h| h.name().ok().map(str::to_string))
            .unwrap_or_else(|| format!("part{n}"));

        // A subsampled scanline part keeps its channels at native sampling
        // (the same low-level reader the single-part path uses); the facade
        // would reject it.
        let subsampled = mp
            .header(n)
            .map_err(GpuError::from)?
            .channels()
            .map_err(GpuError::from)?
            .iter()
            .any(|(_, c)| (c.x_sampling, c.y_sampling) != (1, 1));
        let source = match part_type.as_str() {
            SCANLINE_IMAGE if subsampled => {
                let mut ip = mp.input_part(n).map_err(GpuError::from)?;
                SourceImage::from_reader(&mut ip)?
            }
            SCANLINE_IMAGE => {
                let mut ip = mp.input_part(n).map_err(GpuError::from)?;
                SourceImage::from_image(&Image::from_input_file(&mut ip).map_err(GpuError::from)?)?
            }
            TILED_IMAGE => {
                let mut tp = mp.tiled_input_part(n).map_err(GpuError::from)?;
                SourceImage::from_image(&Image::from_tiled_input_file(&mut tp).map_err(GpuError::from)?)?
            }
            // Deep (deepscanline / deeptile) or anything else: no 2-D texture grid.
            other => {
                log::warn!(
                    "exr-gpu: skipping part {n} \"{name}\" of type \"{other}\" \
                     (no flat-texture mapping)"
                );
                skipped.push((name, other.to_string()));
                continue;
            }
        };

        parts.push(Part::new(name, source));
    }

    if parts.is_empty() {
        return Err(GpuError::Unsupported(format!(
            "multipart file has no image parts exr-gpu can map to textures \
             (skipped {} part(s): {skipped:?})",
            skipped.len()
        )));
    }
    Ok(parts)
}
