# SPDX-License-Identifier: BSD-3-Clause
"""The `OpenEXR` package: re-exports the compiled extension so `import OpenEXR`
works unchanged, and ships the companion `Imath` module.

`Imath` lives inside this package (maturin reliably ships package contents) but is
registered as a top-level module so legacy `import Imath` keeps working once
`OpenEXR` has been imported — the usual drop-in pattern (`import OpenEXR, Imath`)."""

import sys

from .OpenEXR import *  # noqa: F401,F403  (the compiled extension's public API)
from .OpenEXR import __version__, OPENEXR_VERSION  # noqa: F401

from . import Imath as _Imath

# Expose `Imath` at top level (don't clobber a genuinely-installed one).
sys.modules.setdefault("Imath", _Imath)
Imath = _Imath
