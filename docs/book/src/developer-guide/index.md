# Developer guide

- [Architecture](architecture.md): format boundaries, ownership and the viewer.
- [Crates](crates.md): workspace members and shared dependencies.
- [Parity and testing](parity.md): native references, fixtures and actual acceptance.
- [Conventions](conventions.md): source, review and publication rules.

Start with the relevant source and pinned reference. For the format port, module
layout follows OpenEXR/IlmImf, Imath and libdeflate; OpenJPH lives in `jph-rs`.
Viewer/GPU APIs have separate ownership contracts on top of that core.

Current tasks have one
[roadmap](https://github.com/ssoj13/exr-rs/blob/main/plans/README.md).
The [handoff](https://github.com/ssoj13/exr-rs/blob/main/plans/HANDOFF.md) records
restart facts and frozen evidence; it is not another user manual.
