//! Engagement probes: the single mechanism tests use to prove a fast path,
//! parallel path or reuse path actually RAN.
//!
//! Result-equality tests cannot tell a fast path from its fallback, because
//! both produce the same bytes. A probe counts how often a named dispatch
//! point took the branch under test, so a test can assert on it.
//!
//! Two shapes, one enum:
//! - [`hit`] / [`get`] / [`reset`] use ONE thread-local counter array, for
//!   code that runs on the caller's thread. Concurrent `#[test]`s never see
//!   each other's counts. A process-global counter is never used: one was
//!   tried for `new_compressor` and got polluted by other tests' traffic.
//! - [`SharedProbes`] is owned by an object (a pool or a parallel plan) for
//!   work that crosses rayon threads, where a thread-local would miss worker
//!   hits. Its counts are scoped to that object, never global.
//!
//! Outside `cfg(test)` every function is an empty `#[inline(always)]` body and
//! [`SharedProbes`] is zero-sized, so release builds pay nothing.

/// A named dispatch point. Add a variant per probed branch; the variant's doc
/// names the branch and the test that asserts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Probe {
    /// DWA lossy DCT block decoded with the DC-only IDCT (`last_non_zero == 0`).
    DwaDcOnlyBlock,
    /// DWA lossy DCT block decoded with the full (row-skipping) IDCT.
    DwaFullIdctBlock,
    /// DWA decode rebuilt its channel plan (cache miss on `(channels, rules)`).
    DwaPlanBuild,
    /// `misc::plan_disjoint` accepted a chunk plan, so the parallel scatter
    /// path can run (the deep readers fall back to serial otherwise).
    PlanDisjointAccepted,
    /// A flat scanline read completed on the parallel path (not declined).
    ScanReadParallel,
    /// A flat scanline write completed on the parallel path.
    ScanWriteParallel,
    /// A flat tiled read completed on the parallel path (not declined).
    TileReadParallel,
    /// A flat tiled write completed on the parallel path.
    TileWriteParallel,
    /// A deep scanline read completed on the parallel path (not declined).
    DeepScanReadParallel,
    /// A deep scanline write completed on the parallel path.
    DeepScanWriteParallel,
    /// A deep tiled read completed on the parallel path (not declined).
    DeepTileReadParallel,
    /// A deep tiled write completed on the parallel path (not declined).
    DeepTileWriteParallel,
    /// One flat scanline chunk decoded (`scanline::input::decode_chunk`).
    DecodeChunk,
    /// One low-level scanline `write_pixels` call.
    WritePixels,
    /// One `new_compressor`/`new_tile_compressor` construction.
    NewCompressor,
    /// One deep `sample_count_at` lookup (linear-vs-quadratic addressing).
    SampleCountAt,
    /// `CodecPool::checkout` constructed a codec (free list empty).
    CodecCreated,
    /// `BufPool::take` reused a pooled buffer.
    BufPoolHit,
    /// `BufPool::take` found the free list empty and allocated.
    BufPoolMiss,
    /// `misc::scatter_convert`/`scatter_fill` took the chunked path.
    ScatterTypedFast,
    /// `misc::gather_convert` took the chunked path.
    GatherTypedFast,
    /// `scanline::input::scatter_chunk_packed` serviced a chunk (interleaved,
    /// same-type, unsampled frame buffer).
    ScatterChunkPacked,
    /// `InputFile` over a tiled file read the whole data window straight
    /// into the caller's buffer (C++'s skip-the-double-copy path).
    TiledWholeWindow,
    /// The read-ahead producer handed one window to the decode worker (a
    /// parallel flat read pipelines more than one window).
    ReadAheadWindow,
    /// The tiled writer BUFFERED a tile that arrived out of line order
    /// (`tile_map` insert). A writer fed in file order (the `api::Image` facade)
    /// never hits it.
    TileBuffered,
    /// `huffman::build_encoding_table` refilled a warmed `BinaryHeap` instead of
    /// allocating one (`HufEncScratch::heap`; asserted by
    /// `huffman::tests::generic_and_heap_scratch_is_reused_and_poison_proof`).
    PizHufHeapReuse,
    /// `huffman::decode_generic` (the `nBits <= 128` path) ran on already
    /// allocated code/decoding tables (`GenScratch`; same test).
    PizHufGenericReuse,
    /// `PizCompressor::uncompress` sized its output once and scattered into it
    /// (no per-row zero-fill) over a buffer that already held that many bytes
    /// (`piz::tests::uncompress_out_is_reused_and_poison_proof`).
    PizOutKept,
    /// `Htj2kCompressor::decode_chunk` decoded into a buffer that already held
    /// the chunk's full size, leaving the stale bytes in place instead of
    /// zero-filling them (every byte is overwritten by the decode; asserted by
    /// `htj2k::tests::decode_overwrites_every_byte_for_every_chunk_shape`).
    Htj2kOutKept,
    /// HTJ2K rebuilt chunk metadata in buffers whose capacity covered the call.
    Htj2kMetadataReuse,
    /// `threading::ordered_pipeline` started emitting item `i` while a later
    /// item was still being produced on the pool (true emit/produce overlap, the
    /// C++ line-buffer ring; asserted by every parallel writer's overlap test).
    WriteOverlap,
}

impl Probe {
    /// Number of variants; sizes the counter arrays.
    #[cfg(test)]
    pub(crate) const COUNT: usize = Probe::WriteOverlap as usize + 1;
    /// Every variant, in declaration order (`ALL[i] as usize == i`).
    #[cfg(test)]
    const ALL: [Probe; Probe::COUNT] = [
        Probe::DwaDcOnlyBlock,
        Probe::DwaFullIdctBlock,
        Probe::DwaPlanBuild,
        Probe::PlanDisjointAccepted,
        Probe::ScanReadParallel,
        Probe::ScanWriteParallel,
        Probe::TileReadParallel,
        Probe::TileWriteParallel,
        Probe::DeepScanReadParallel,
        Probe::DeepScanWriteParallel,
        Probe::DeepTileReadParallel,
        Probe::DeepTileWriteParallel,
        Probe::DecodeChunk,
        Probe::WritePixels,
        Probe::NewCompressor,
        Probe::SampleCountAt,
        Probe::CodecCreated,
        Probe::BufPoolHit,
        Probe::BufPoolMiss,
        Probe::ScatterTypedFast,
        Probe::GatherTypedFast,
        Probe::ScatterChunkPacked,
        Probe::TiledWholeWindow,
        Probe::ReadAheadWindow,
        Probe::TileBuffered,
        Probe::PizHufHeapReuse,
        Probe::PizHufGenericReuse,
        Probe::PizOutKept,
        Probe::Htj2kOutKept,
        Probe::Htj2kMetadataReuse,
        Probe::WriteOverlap,
    ];
}

#[cfg(test)]
thread_local! {
    static COUNTS: [std::cell::Cell<u64>; Probe::COUNT] =
        const { [const { std::cell::Cell::new(0) }; Probe::COUNT] };
}

/// Record one hit of `p` on this thread.
#[inline(always)]
pub(crate) fn hit(p: Probe) {
    #[cfg(test)]
    COUNTS.with(|c| c[p as usize].set(c[p as usize].get() + 1));
    #[cfg(not(test))]
    let _ = p;
}

/// This thread's hit count for `p` since the last [`reset`].
#[cfg(test)]
pub(crate) fn get(p: Probe) -> u64 {
    COUNTS.with(|c| c[p as usize].get())
}

/// Zero this thread's counter for `p`.
#[cfg(test)]
pub(crate) fn reset(p: Probe) {
    COUNTS.with(|c| c[p as usize].set(0));
}

/// Assert the parallel-completion probe `p` fired iff `num_threads > 1`,
/// then reset it for the next iteration of a thread-count matrix. Byte
/// identity across thread counts cannot tell a parallel run from a silently
/// serial one; this can.
#[cfg(test)]
pub(crate) fn assert_engaged(p: Probe, num_threads: i32) {
    let hits = get(p);
    if num_threads > 1 {
        assert!(hits > 0, "num_threads={num_threads}: {p:?} never completed: the parallel path was skipped or declined");
    } else {
        assert_eq!(hits, 0, "num_threads={num_threads}: {p:?} completed on a serial thread count");
    }
    reset(p);
}

/// Run `write` (a parallel write of enough real work to have several items
/// in flight) up to three times and report whether the writer ever emitted an
/// item while a later one was still being produced
/// ([`Probe::WriteOverlap`]). Retrying only forgives a scheduler that happened
/// to finish everything before the first emit; a window-barrier writer can
/// never overlap, so it fails every attempt.
#[cfg(test)]
pub(crate) fn saw_overlap(mut write: impl FnMut()) -> bool {
    for _ in 0..3 {
        reset(Probe::WriteOverlap);
        write();
        if get(Probe::WriteOverlap) > 0 {
            return true;
        }
    }
    false
}

/// Assert the parallel-completion probe `p` never fired (a case the parallel
/// path must decline, e.g. an aliased frame buffer), then reset it.
#[cfg(test)]
pub(crate) fn assert_declined(p: Probe) {
    assert_eq!(get(p), 0, "{p:?} completed although the parallel path had to decline");
    reset(p);
}

/// Add every count of an object-owned [`SharedProbes`] to this thread's
/// counters. `BufPool`/`CodecPool` call it from `Drop` (under test), which
/// runs on the owning read/write call's thread once its workers have joined,
/// so a test on that thread sees worker hits through the same [`get`] as
/// everything else.
#[cfg(test)]
pub(crate) fn absorb(shared: &SharedProbes) {
    COUNTS.with(|c| {
        for p in Probe::ALL {
            let i = p as usize;
            c[i].set(c[i].get() + shared.get(p));
        }
    });
}

/// Probe counters owned by an object whose work fans out across rayon
/// threads. Zero-sized outside `cfg(test)`.
#[derive(Debug, Default)]
pub(crate) struct SharedProbes {
    #[cfg(test)]
    counts: [std::sync::atomic::AtomicU64; Probe::COUNT],
}

impl SharedProbes {
    /// Record one hit of `p`, from any thread.
    #[inline(always)]
    pub(crate) fn hit(&self, p: Probe) {
        #[cfg(test)]
        self.counts[p as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let _ = p;
    }

    /// Hit count for `p` on this object.
    #[cfg(test)]
    pub(crate) fn get(&self, p: Probe) -> u64 {
        self.counts[p as usize].load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Count of tasks currently inside a producer body, for the overlap probe.
/// Zero-sized outside `cfg(test)`.
#[derive(Debug, Default)]
pub(crate) struct Busy {
    #[cfg(test)]
    n: std::sync::atomic::AtomicUsize,
}

impl Busy {
    /// A task entered its producer body.
    #[inline(always)]
    pub(crate) fn enter(&self) {
        #[cfg(test)]
        self.n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// A task left its producer body.
    #[inline(always)]
    pub(crate) fn exit(&self) {
        #[cfg(test)]
        self.n.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether any task is inside its producer body right now (always `false`
    /// outside `cfg(test)`, where the only consumer, [`hit`], is a no-op).
    #[inline(always)]
    pub(crate) fn any(&self) -> bool {
        #[cfg(test)]
        return self.n.load(std::sync::atomic::Ordering::SeqCst) > 0;
        #[cfg(not(test))]
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_local_counts_are_per_thread_and_per_probe() {
        reset(Probe::DwaDcOnlyBlock);
        reset(Probe::DwaFullIdctBlock);
        hit(Probe::DwaDcOnlyBlock);
        hit(Probe::DwaDcOnlyBlock);
        std::thread::spawn(|| hit(Probe::DwaDcOnlyBlock))
            .join()
            .expect("probe thread");
        assert_eq!(get(Probe::DwaDcOnlyBlock), 2, "another thread's hit leaked in");
        assert_eq!(get(Probe::DwaFullIdctBlock), 0, "probes share a counter");
        reset(Probe::DwaDcOnlyBlock);
        assert_eq!(get(Probe::DwaDcOnlyBlock), 0);
    }

    #[test]
    fn all_lists_every_variant_in_order() {
        for (i, p) in Probe::ALL.iter().enumerate() {
            assert_eq!(*p as usize, i, "{p:?} out of order in Probe::ALL");
        }
    }

    #[test]
    fn absorb_adds_shared_counts_to_this_thread() {
        reset(Probe::CodecCreated);
        let shared = SharedProbes::default();
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| shared.hit(Probe::CodecCreated));
            }
        });
        absorb(&shared);
        absorb(&shared);
        assert_eq!(get(Probe::CodecCreated), 6);
    }

    #[test]
    fn shared_probes_count_across_threads() {
        let probes = SharedProbes::default();
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| probes.hit(Probe::DwaFullIdctBlock));
            }
        });
        assert_eq!(probes.get(Probe::DwaFullIdctBlock), 4);
        assert_eq!(probes.get(Probe::DwaDcOnlyBlock), 0);
    }
}
