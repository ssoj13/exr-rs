# Conventions

## Port and boundaries

Mirror upstream format-code structure and cite pinned reference sources.
OpenEXR 3.4.15 is the behaviour target; settle contested behaviour with the real
oracle. Rust-facing facades, GPU upload and the viewer have their own API design
without reshaping the ported format contracts.

Format codecs are Rust ports rather than C/C++ runtime wrappers. Approved shared
ports such as `jph-rs` and `codec-simd` remain separate dependencies.
GPU drivers and native OpenFX plugins have explicit platform/ABI boundaries;
the application as a whole does not promise an absence of FFI.

SIMD unsafe stays isolated behind safe APIs and scalar reference paths.
Native ABI/test-fixture boundaries also require explicit safety contracts.
Do not spread unsafe codec kernels into `exr-core`.

## Ownership and correctness

- Reuse checked geometry, layout, wire-size and sample-count boundaries.
- Keep one implementation and owner for each invariant; avoid parallel caches,
  schedulers or duplicated conversion paths.
- Writers receive pixel bytes per write call and return fallible completion.
- Retained pixel owners keep their allocation charge; copies need new admission.
- Validate untrusted indexing and propagate structured errors. Classify by error
  kind rather than display text; handle or log fallible results.
- Keep optimized paths tied to scalar/reference semantics and prove engagement.
- Explain ownership, bounds and non-obvious invariants in Rustdoc/comments.

## Git and source trees

Use scoped commits and normal pushes to `main`. Preserve other sessions' edits;
do not reset, revert, restore or stash their work. Use meaningful commit messages
without co-author trailers.

Source worktrees and durable checkpoints live outside disposable build output.
Give independent gates and mutations their own source/build directories.
Never rely on `target/` to retain uncommitted source or irreplaceable evidence.

## Review

Authoring and verification are separate passes: a change is accepted after an
independent check against the C++ oracle or a mutation-checked test, not because it
was committed. Keep stable user documentation in this book. Do not enable LTO in
debug builds or claim speed gains from loaded-machine measurements.
