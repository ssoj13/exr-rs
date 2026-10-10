//! Port of OpenEXR `ImfDeepCompositing.{h,cpp}` -- `class DeepCompositing`,
//! the pluggable sort + composite engine [`crate::deep::CompositeDeepScanLine`]
//! uses to combine samples from one or more deep sources into a flat pixel.
//!
//! Port target (pinned tag v3.3.11 -- unchanged in v3.4.15, confirmed by
//! diffing the file against the oracle checkout):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepCompositing.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfDeepCompositing.h>
//!
//! # Channel layout contract (verbatim from the C++ doc comment)
//!
//! `composite_pixel`/`sort` receive one `&[f32]` per channel, `num_samples`
//! long, in this fixed order:
//!
//! | Position | Channel |
//! |---|---|
//! | 0 | Z |
//! | 1 | ZBack (if the source has no ZBack, `inputs[1] == inputs[0]`) |
//! | 2 | A (alpha) |
//! | 3.. | any other requested channel |
//!
//! # Rust adaptation
//!
//! C++ makes this an abstract base class with two overridable `virtual`
//! methods, each with a default body; callers construct a plain
//! `DeepCompositing` when they want the default behaviour, or a derived
//! class to override `sort`/`composite_pixel`. Rust has no virtual dispatch
//! on a concrete base type, so this is a trait instead: [`DeepCompositing`]
//! has default bodies for both methods (the exact C++ default algorithm), and
//! a caller who wants custom behaviour implements the trait on their own type
//! and overrides just the method(s) they need -- the same "override one
//! virtual, inherit the other" usage C++ supports. [`DefaultDeepCompositing`]
//! is the zero-sized type used when the caller supplies no compositor at all
//! (C++'s local fallback `DeepCompositing d;`).

use core::cmp::Ordering;

/// Sort + composite engine for deep samples -- port of `class DeepCompositing`.
///
/// Implement this trait to override how [`crate::deep::CompositeDeepScanLine`]
/// combines samples (C++: derive from `DeepCompositing`, override `sort` and/or
/// `composite_pixel`). Both methods have the C++ default body, so a type that
/// implements the trait with an empty body (see [`DefaultDeepCompositing`])
/// reproduces the default front-to-back "under" compositing exactly.
pub trait DeepCompositing {
    /// Composite `num_samples` samples (drawn from `sources` distinct deep
    /// sources) into `outputs` -- port of `DeepCompositing::composite_pixel`.
    ///
    /// `outputs.len() == inputs.len() == channel_names.len()` (`num_channels`
    /// in C++, always `>= 3` per the module doc's channel layout). Each
    /// `inputs[c]` holds `num_samples` values for channel `c`.
    ///
    /// The default implementation: zeroes `outputs`; returns immediately if
    /// `num_samples == 0`; calls [`sort`](Self::sort) to get a
    /// front-to-back visiting order when `sources > 1` (a single source's
    /// samples are ALREADY in file order, which for one deep image is
    /// front-to-back by construction, so C++ skips sorting a lone source);
    /// then accumulates `outputs[c] += (1 - alpha) * inputs[c][s]` in that
    /// order, stopping as soon as the running alpha (`outputs[2]`) reaches
    /// `1.0` (matches C++'s early-return "no more samples can contribute").
    fn composite_pixel(
        &mut self,
        outputs: &mut [f32],
        inputs: &[&[f32]],
        channel_names: &[&str],
        num_samples: usize,
        sources: usize,
    ) {
        for o in outputs.iter_mut() {
            *o = 0.0;
        }
        // no samples? do nothing (C++: `if (num_samples == 0) return;`).
        if num_samples == 0 {
            return;
        }

        let mut sort_order: Vec<usize> = (0..num_samples).collect();
        if sources > 1 {
            self.sort(&mut sort_order, inputs, channel_names, num_samples, sources);
        }

        #[allow(clippy::needless_range_loop)] // `i` itself is the sample index when `sources <= 1`
        for i in 0..num_samples {
            let s = if sources > 1 { sort_order[i] } else { i };
            let alpha = outputs[2];
            if alpha >= 1.0 {
                return;
            }
            for (c, output) in outputs.iter_mut().enumerate() {
                *output += (1.0 - alpha) * inputs[c][s];
            }
        }
    }

    /// Compute the front-to-back visiting order for `num_samples` samples
    /// drawn from `sources` distinct sources -- port of
    /// `DeepCompositing::sort`. `order` is populated (not the sample arrays
    /// themselves, which are left untouched) with `order[n]` = index of the
    /// `n`-th closest sample.
    ///
    /// The default implementation sorts by `inputs[0]` (Z) ascending, then by
    /// `inputs[1]` (ZBack) ascending as a tie-break, then by the ORIGINAL
    /// index (stable) as a final tie-break -- a faithful port of C++
    /// `sort_helper::operator()`'s exact 3-level chain, deliberately NOT
    /// `f32::total_cmp`/`partial_cmp` (which would order NaN differently);
    /// C++'s plain `<`/`>` on a NaN Z value is already unspecified behaviour
    /// in the oracle too, so bit-parity there is moot, but this keeps the
    /// SAME decision structure C++ has rather than substituting a different
    /// float-ordering primitive.
    fn sort(
        &mut self,
        order: &mut [usize],
        inputs: &[&[f32]],
        _channel_names: &[&str],
        _num_samples: usize,
        _sources: usize,
    ) {
        order.sort_by(|&a, &b| {
            let za = inputs[0][a];
            let zb = inputs[0][b];
            if za < zb {
                return Ordering::Less;
            }
            if za > zb {
                return Ordering::Greater;
            }
            let zba = inputs[1][a];
            let zbb = inputs[1][b];
            if zba < zbb {
                return Ordering::Less;
            }
            if zba > zbb {
                return Ordering::Greater;
            }
            a.cmp(&b)
        });
    }
}

/// The default compositor used when [`crate::deep::CompositeDeepScanLine`] is
/// given no custom one -- C++'s local fallback `DeepCompositing d;` in
/// `composite_line`. A zero-sized type whose trait impl uses both default
/// method bodies verbatim.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultDeepCompositing;

impl DeepCompositing for DefaultDeepCompositing {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_samples_zeroes_outputs() {
        let mut comp = DefaultDeepCompositing;
        let mut outputs = [1.0f32, 2.0, 3.0];
        let inputs: [&[f32]; 3] = [&[], &[], &[]];
        comp.composite_pixel(&mut outputs, &inputs, &["Z", "Z", "A"], 0, 0);
        assert_eq!(outputs, [0.0, 0.0, 0.0]);
    }

    /// Single opaque sample: output equals the sample verbatim (alpha
    /// starts at 0, so `(1-0)*value`).
    #[test]
    fn single_opaque_sample() {
        let mut comp = DefaultDeepCompositing;
        let mut outputs = [0.0f32; 3];
        let z = [1.0f32];
        let a = [1.0f32];
        let inputs: [&[f32]; 3] = [&z, &z, &a];
        comp.composite_pixel(&mut outputs, &inputs, &["Z", "Z", "A"], 1, 1);
        assert_eq!(outputs, [1.0, 1.0, 1.0]);
    }

    /// Two half-opaque samples, front then back: the classic "under"
    /// compositing accumulation, hand-derived.
    ///  front: Z=0, A=0.5; back: Z=1, A=0.5 (same for every channel value).
    ///  out[c] after front: (1-0)*0.5 = 0.5.
    ///  out[c] after back:  0.5 + (1-0.5)*0.5 = 0.75.
    #[test]
    fn two_samples_front_to_back_accumulation() {
        let mut comp = DefaultDeepCompositing;
        let mut outputs = [0.0f32; 3];
        let z = [0.0f32, 1.0];
        let a = [0.5f32, 0.5];
        let inputs: [&[f32]; 3] = [&z, &z, &a];
        comp.composite_pixel(&mut outputs, &inputs, &["Z", "Z", "A"], 2, 1);
        assert!((outputs[2] - 0.75).abs() < 1e-6);
    }

    /// Samples fed out of depth order (source order, not sorted) with
    /// `sources > 1` must be re-sorted by Z before accumulating -- the front
    /// sample (Z=0) must dominate regardless of array position.
    #[test]
    fn multi_source_samples_are_sorted_by_z() {
        let mut comp = DefaultDeepCompositing;
        let mut outputs = [0.0f32; 3];
        // Back sample listed FIRST in the array, front sample second.
        let z = [1.0f32, 0.0];
        let a = [1.0f32, 1.0]; // both fully opaque
        let inputs: [&[f32]; 3] = [&z, &z, &a];
        comp.composite_pixel(&mut outputs, &inputs, &["Z", "Z", "A"], 2, 2);
        // The FRONT (Z=0) sample must be applied first and, being fully
        // opaque, must be the only one that contributes.
        assert_eq!(outputs[0], 0.0, "Z output must come from the front sample");
    }

    /// A single source's samples must NOT be re-sorted (C++ skips `sort()`
    /// entirely when `sources <= 1`), even if they are not depth-ordered --
    /// proven by checking the alpha-early-stop happens on the FIRST array
    /// entry, not the smallest Z.
    #[test]
    fn single_source_is_never_sorted() {
        let mut comp = DefaultDeepCompositing;
        let mut outputs = [0.0f32; 3];
        // Deliberately NOT depth-ordered: back (Z=1) listed first, opaque.
        let z = [1.0f32, 0.0];
        let a = [1.0f32, 1.0];
        let inputs: [&[f32]; 3] = [&z, &z, &a];
        comp.composite_pixel(&mut outputs, &inputs, &["Z", "Z", "A"], 2, 1);
        // With sources == 1, C++ never sorts: sample 0 (Z=1) is applied
        // first and, being opaque, is the only contributor.
        assert_eq!(outputs[0], 1.0, "a lone source's samples must be used in array order");
    }
}
