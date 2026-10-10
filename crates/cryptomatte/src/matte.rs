//! Coverage matte extraction from ranked Cryptomatte float layers.

use std::borrow::Cow;
use std::collections::HashSet;

use exr_core::{ChannelData, Image};

use crate::channels::RankedLayer;
use crate::error::{Error, Result};

/// Flat coverage matte over the image data window (`width * height` samples).
#[derive(Clone, Debug)]
pub struct Matte {
    /// Data-window width in pixels.
    pub width: usize,
    /// Data-window height in pixels.
    pub height: usize,
    /// Row-major coverage samples in `[0, 1]` (summed across ranks).
    pub coverage: Vec<f32>,
}

impl Matte {
    /// Sum of all coverage samples (useful for non-empty smoke tests).
    pub fn sum(&self) -> f32 {
        self.coverage.iter().copied().sum()
    }
}

// `#[cfg(feature = "test-instrumentation")]`, NOT `#[cfg(test)]`: this counter
// must be visible to `tests/inject.rs`, which is a SEPARATE crate that links
// the library built WITHOUT `--cfg test` (only the library's own `#[cfg(test)]
// mod tests` gets that cfg -- an integration test in `tests/` never does).
// The feature is enabled for every `cargo test` run via this crate's own
// `[dev-dependencies]` self-reference in `Cargo.toml` (a standard, resolver-3
// -supported pattern for exactly this "test-only instrumentation" case), so
// no test is ever silently skipped for lack of a flag (this crate's own
// hygiene rule against silent skips) and the counter still costs nothing in
// a normal (non-dev) build of the library.
#[cfg(feature = "test-instrumentation")]
thread_local! {
    /// Test-only call counter for [`channel_as_f32`] — the single per-channel
    /// HALF/FLOAT conversion primitive [`crate::rank::RankedPlanes::from_channels`]
    /// is built around ("convert every ranked plane ONCE, not once per
    /// pixel"). Thread-local (not a shared atomic) so concurrently-running
    /// `#[test]`s on their own OS threads never pollute each other's counts.
    /// See `cryptomatte/tests/inject.rs::inject_half_channels_scales_linearly`
    /// (rewritten from a wall-clock threshold, which flaked under debug
    /// builds / machine load, to an operation-count ratio using this
    /// counter — the same fix pattern as exr-core's D-10).
    static CHANNEL_AS_F32_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Reset this thread's [`channel_as_f32`] call counter to zero.
#[cfg(feature = "test-instrumentation")]
pub fn reset_channel_as_f32_calls() {
    CHANNEL_AS_F32_CALLS.with(|c| c.set(0));
}

/// This thread's [`channel_as_f32`] call count since the last
/// [`reset_channel_as_f32_calls`].
#[cfg(feature = "test-instrumentation")]
pub fn channel_as_f32_calls() -> u64 {
    CHANNEL_AS_F32_CALLS.with(|c| c.get())
}

/// View a FLOAT channel, or convert a HALF one, as `f32` samples.
///
/// SSOT for Cryptomatte's channel → `f32` conversion (ID/coverage planes and
/// the CLI's coverage input). UINT is rejected: a Cryptomatte ID is a float bit
/// pattern, so widening an integer channel would silently never match.
pub fn channel_as_f32(data: &ChannelData) -> Result<Cow<'_, [f32]>> {
    #[cfg(feature = "test-instrumentation")]
    CHANNEL_AS_F32_CALLS.with(|c| c.set(c.get() + 1));
    match data {
        // Zero-copy fast path for the common case (a real FLOAT ID/coverage
        // channel); HALF widening delegates to the W4.6 SSOT
        // `ChannelData::to_f32` instead of a duplicate `.map(f32::from)`.
        ChannelData::Float(v) => Ok(Cow::Borrowed(v.as_slice())),
        ChannelData::Half(_) => Ok(Cow::Owned(data.to_f32())),
        ChannelData::Uint(_) => Err(Error::Metadata(
            "Cryptomatte ID/coverage channels must be FLOAT or HALF, not UINT".into(),
        )),
    }
}

/// View an ID channel as `f32` samples: FLOAT only.
///
/// A Cryptomatte ID is the bit pattern of a 32-bit float (spec 1.2.0,
/// "Floating point conversion"); a HALF channel keeps 16 of those bits, so its
/// widened values can never equal a hashed ID and every matte would silently
/// come out empty. HALF is therefore an error for IDs (it stays fine for
/// coverage, see [`channel_as_f32`]).
pub fn id_channel_as_f32<'a>(name: &str, data: &'a ChannelData) -> Result<Cow<'a, [f32]>> {
    if matches!(data, ChannelData::Half(_)) {
        return Err(Error::Metadata(format!(
            "Cryptomatte ID channel `{name}` is HALF; IDs are 32-bit float bit patterns \
             and must be stored as FLOAT"
        )));
    }
    channel_as_f32(data)
}

/// Borrow a channel by name, or report it missing.
pub fn require_channel<'a>(image: &'a Image, name: &str) -> Result<&'a ChannelData> {
    image.channel(name).ok_or_else(|| {
        Error::Metadata(format!("missing Cryptomatte channel `{name}`"))
    })
}

/// Sum coverage for pixels whose ID matches any selected float ID (bit-exact).
///
/// Walks every ID/coverage pair in `ranked` layers (typically from
/// [`crate::channels::ranked_layers_for`]). Channel pixels come from `get_channel`
/// so callers can use an [`Image`] or an already-decoded channel list.
pub fn extract_coverage_with<'a, F>(
    width: u32,
    height: u32,
    ranked: &[RankedLayer],
    selected_ids: &[f32],
    mut get_channel: F,
) -> Result<Matte>
where
    F: FnMut(&str) -> Result<&'a ChannelData>,
{
    let width = usize::try_from(width).map_err(|_| {
        Error::Metadata(format!("invalid image width {width}"))
    })?;
    let height = usize::try_from(height).map_err(|_| {
        Error::Metadata(format!("invalid image height {height}"))
    })?;
    let n = width
        .checked_mul(height)
        .ok_or_else(|| Error::Metadata("image pixel count overflow".into()))?;

    let selected: HashSet<u32> = selected_ids.iter().map(|id| id.to_bits()).collect();
    let mut coverage = vec![0.0f32; n];

    if selected.is_empty() || ranked.is_empty() {
        return Ok(Matte {
            width,
            height,
            coverage,
        });
    }

    for layer in ranked {
        for (id_name, cov_name) in layer.pairs() {
            let id_ch = get_channel(id_name)?;
            let cov_ch = get_channel(cov_name)?;
            let ids = id_channel_as_f32(id_name, id_ch)?;
            let covs = channel_as_f32(cov_ch)?;
            if ids.len() != n || covs.len() != n {
                return Err(Error::Metadata(format!(
                    "channel length mismatch for `{id_name}`/`{cov_name}` (got {}/{} want {n})",
                    ids.len(),
                    covs.len()
                )));
            }
            for i in 0..n {
                if selected.contains(&ids[i].to_bits()) {
                    coverage[i] += covs[i];
                }
            }
        }
    }

    Ok(Matte {
        width,
        height,
        coverage,
    })
}

/// Sum coverage using channels from an [`Image`] (bit-exact ID match).
pub fn extract_coverage(
    image: &Image,
    ranked: &[RankedLayer],
    selected_ids: &[f32],
) -> Result<Matte> {
    let width = u32::try_from(image.width()).map_err(|_| {
        Error::Metadata(format!("invalid image width {}", image.width()))
    })?;
    let height = u32::try_from(image.height()).map_err(|_| {
        Error::Metadata(format!("invalid image height {}", image.height()))
    })?;
    extract_coverage_with(width, height, ranked, selected_ids, |name| {
        require_channel(image, name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::RankedLayer;
    use exr_core::Image;
    use imath_rs::{Box2i, V2i};

    #[test]
    fn sums_matching_id_coverage() {
        let dw = Box2i {
            min: V2i { x: 0, y: 0 },
            max: V2i { x: 1, y: 0 },
        };
        let target = f32::from_bits(0x3f800000); // 1.0
        let mut image = Image::new(dw);
        image
            .insert_channel(
                "L00.R",
                ChannelData::Float(vec![target, 0.0]),
            )
            .unwrap();
        image
            .insert_channel("L00.G", ChannelData::Float(vec![0.25, 0.5]))
            .unwrap();
        image
            .insert_channel("L00.B", ChannelData::Float(vec![0.0, target]))
            .unwrap();
        image
            .insert_channel("L00.A", ChannelData::Float(vec![0.1, 0.75]))
            .unwrap();

        let ranked = [RankedLayer {
            base: "L00".into(),
            id0: Some("L00.R".into()),
            cov0: Some("L00.G".into()),
            id1: Some("L00.B".into()),
            cov1: Some("L00.A".into()),
        }];
        let matte = extract_coverage(&image, &ranked, &[target]).unwrap();
        assert_eq!(matte.coverage, vec![0.25, 0.75]);
    }

    #[test]
    fn unknown_id_yields_zero_matte() {
        let dw = Box2i {
            min: V2i { x: 0, y: 0 },
            max: V2i { x: 1, y: 0 },
        };
        let target = f32::from_bits(0x3f800000);
        let mut image = Image::new(dw);
        image
            .insert_channel("L00.R", ChannelData::Float(vec![target, target]))
            .unwrap();
        image
            .insert_channel("L00.G", ChannelData::Float(vec![0.5, 0.5]))
            .unwrap();
        let ranked = [RankedLayer {
            base: "L00".into(),
            id0: Some("L00.R".into()),
            cov0: Some("L00.G".into()),
            id1: None,
            cov1: None,
        }];
        let other = f32::from_bits(0x40000000); // 2.0
        let matte = extract_coverage(&image, &ranked, &[other]).unwrap();
        assert_eq!(matte.coverage, vec![0.0, 0.0]);
    }
}
