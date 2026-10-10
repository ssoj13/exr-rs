# Vendored official OpenEXR Python tests

Source: `AcademySoftwareFoundation/openexr`, `src/wrappers/python/tests/` (plus the
two fixture files `test.exr` / `unfinished_multipart.exr` from the same
directory), pinned at tag **v3.4.15** (commit `e71cdd5d30a146dcb56c5e4c576d9e9d3c45f4fb`).

License: **BSD-3-Clause**, "Copyright (c) Contributors to the OpenEXR Project."
Each `.py` file carries its own SPDX header, retained unmodified.

These are the official acceptance-gate tests for the `OpenEXR` Python module
(`crates/openexr-py`'s drop-in target). They are run
against **our** `OpenEXR` module (built via `python bootstrap.py b p`), not
against the reference C++ bindings. Do not hand-edit test logic here beyond
what's noted below — if a test is wrong for us, fix the module, not the test.

## Status

Every official test file passes unmodified against our module; none is skipped.
