"""Where exr-rs's test image set is, for the tests that read it: `EXR_TEST_DATA` (the variable
the Rust `exr-test-data` crate reads first).

Those tests are off by default, like the Rust ones behind `--features test-data`: they run only
when `EXR_TEST_DATA` is set (`bootstrap.py t` sets it), and then missing data is a failure."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ENV = "EXR_TEST_DATA"

#: Marks a test that reads the image set; skipped unless `EXR_TEST_DATA` is set.
needs_data = pytest.mark.skipif(
    not os.environ.get(ENV), reason=f"needs the exr-rs test image set (set {ENV})"
)


def root() -> Path:
    """The data root from `EXR_TEST_DATA`; asserts it holds the image set."""
    root = Path(os.environ[ENV])
    assert (root / "openexr-images").is_dir(), f"exr-rs test data not found at {root}"
    return root


def openexr_images() -> Path:
    """The openexr-images repository (e38ffb0)."""
    return root() / "openexr-images"


def cryptomatte() -> Path:
    """Cryptomatte samples (Psyop 1.4.0) with the pixel goldens and the synthetic sample."""
    return root() / "cryptomatte"
