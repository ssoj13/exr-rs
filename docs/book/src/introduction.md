# Introduction

`exr-rs` is a pure-Rust port of [OpenEXR](https://www.openexr.com/) together with a
GPU viewer, a wgpu-based upload layer, Cryptomatte support, and a drop-in `OpenEXR`
Python module.

The image-format code is a **1:1 port** of the upstream C/C++ libraries — Imath,
libdeflate, OpenEXR/IlmImf, and OpenJPH (the last via the `jph-rs` dependency) —
rather than a C/C++ codec wrapper. The ported core uses Rust implementations;
isolated SIMD in `deflate-rs` and the shared `codec-simd` dependency has scalar
reference paths. GPU drivers and native OpenFX plug-ins have separate platform
boundaries; the format port does not depend on the C++ libraries at runtime.

Correctness is held to the original implementations: every codec with a byte-identity
test (including the lossy DWA, B44 and PXR24) writes the same bytes as C++, and the
readers and writers are checked against real C++ OpenEXR 3.4.15 by a cross-read parity
gate and a corpus of upstream test images.

This book has two parts:

- The **User Guide** covers installing the tools, the `exr-rs` command line, the
  viewer, and using `exr-core`/`exr-gpu` as libraries.
- The **Developer Guide** covers the architecture, the crate breakdown, the parity
  harness, and the engineering conventions the port follows.

If you only want to read or write EXR files from Rust, start with
[Library API](user-guide/library.md). If you want to display or process them on the
GPU, see [GPU upload](user-guide/gpu.md) and [Viewer](user-guide/viewer.md).
