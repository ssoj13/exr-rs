//! Physical output texture sizing: 256-aligned buckets with resize hysteresis.
//!
//! Avoids reallocating the panel output texture on every small egui resize:
//! the logical panel size is rounded up to a 256-aligned bucket, and shrinking
//! only triggers a realloc once the logical size drops well below the current
//! bucket (hysteresis), so dragging a dock split does not thrash GPU textures.

/// Quantize output dimensions to this alignment (bytes / warp-friendly).
pub const BUCKET_ALIGN: u32 = 256;

/// Hysteresis band when shrinking buckets (logical must drop below `current * 85 / 100`).
const SHRINK_NUM: u32 = 85;
const SHRINK_DEN: u32 = 100;

/// Round `v` up to the next multiple of [`BUCKET_ALIGN`]; minimum one bucket.
#[must_use]
pub fn round_up_bucket(v: u32) -> u32 {
    if v == 0 {
        return BUCKET_ALIGN;
    }
    v.div_ceil(BUCKET_ALIGN) * BUCKET_ALIGN
}

/// Tracks the last allocated physical bucket; avoids realloc on small logical resizes.
#[derive(Debug, Clone, Copy, Default)]
pub struct BucketSizer {
    physical: (u32, u32),
}

impl BucketSizer {
    #[must_use]
    pub fn physical(&self) -> (u32, u32) {
        self.physical
    }

    /// Returns `(bucket_w, bucket_h)` to allocate; `reallocated` is true when the bucket changed.
    pub fn update(&mut self, logical_w: u32, logical_h: u32) -> ((u32, u32), bool) {
        let want = (round_up_bucket(logical_w), round_up_bucket(logical_h));
        if self.physical == (0, 0) {
            self.physical = want;
            return (want, true);
        }
        if want == self.physical {
            return (self.physical, false);
        }
        let mut next = want;
        if want.0 < self.physical.0
            && logical_w.saturating_mul(SHRINK_DEN) > self.physical.0.saturating_mul(SHRINK_NUM)
        {
            next.0 = self.physical.0;
        }
        if want.1 < self.physical.1
            && logical_h.saturating_mul(SHRINK_DEN) > self.physical.1.saturating_mul(SHRINK_NUM)
        {
            next.1 = self.physical.1;
        }
        if next == self.physical {
            return (self.physical, false);
        }
        self.physical = next;
        (next, true)
    }
}
