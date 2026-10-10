# Developer guide

- [Architecture](architecture.md): format boundaries, ownership and the viewer.
- [Crates](crates.md): workspace members and shared dependencies.
- [Parity and testing](parity.md): native oracles, fixtures and test gates.
- [Conventions](conventions.md): porting, correctness and review rules.

Start with the relevant source and pinned reference. For the format port, module
layout follows OpenEXR/IlmImf, Imath and libdeflate; OpenJPH lives in `jph-rs`.
The viewer and GPU crates have their own APIs on top of that core.
