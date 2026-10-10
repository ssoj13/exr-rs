# cryptomatte

Cryptomatte **decode and encode** helpers for [exr-rs](https://github.com/ssoj13/exr-rs):
MurmurHash3_x86_32 IDs, header metadata, JSON/sidecar manifests, ranked coverage
mattes, named-matte write path.

Not part of OpenEXR IlmImf parity. Python: `OpenEXR.cryptomatte` (exr-rs extension).

## Status

| Area | Done |
|------|------|
| ID hash / float / hex | yes |
| Metadata discover (`cryptomatte/`, legacy `exr/cryptomatte/`) | yes |
| Manifest inline + sidecar | yes |
| Ranked matte extract | yes |
| Rank / pack encode | yes |
| Inject over/under (full re-rank) | yes |
| CLI `exr-rs cryptomatte` | yes |
| Python `encode` + `inject` + sidecar | yes |
| exr-view Crypto/Matte pick overlay | yes |
| Renderer sample-filter kernels | no (non-goal) |
| Nuke Encryptomatte gizmo graph | no (non-goal) |

## Usage

```rust
use cryptomatte::{CryptomatteFile, EncodeOptions, NamedMatte, Windows, from_named_mattes};
use exr_core::Compression;

// Decode
let file = CryptomatteFile::open("shot.exr")?;
let layer = &file.layers[0];
let matte = file.extract_names(layer, &["bunny"])?;

// Encode
let opts = EncodeOptions::new("CryptoObject");
let enc = from_named_mattes(Windows::origin(w, h)?, &[NamedMatte::new("bunny", &coverage)], &opts)?;
enc.write("out.exr", Some(Compression::Zip))?;  // None: keep the encoded image's compression
```

### Python (`OpenEXR.cryptomatte`)

```python
import numpy as np
import OpenEXR.cryptomatte as crypto

matte = crypto.extract("shot.exr", names=["bunny"])  # HxW float32

crypto.encode(
    "out.exr",
    "CryptoObject",
    {"bunny": np.zeros((720, 1280), dtype=np.float32)},
    ranked_layers=3,
    # sidecar="manif.json",  # optional relative manif_file
)

crypto.inject("src.exr", "dst.exr", "newObj", coverage_hxw, operation="over")
```

## Samples and pixel goldens

The sample tests read Psyop's sample images (Cryptomatte 1.4.0) and their pixel goldens from a
separate image set; they are off by default and run with `--features test-data`.
`CRYPTOMATTE_SAMPLES` points them at another set.

`scripts/fetch_cryptomatte_samples.py` is how that set was made: it downloads the samples,
generates the synthetic one and writes the goldens (`--out <dir>`).
