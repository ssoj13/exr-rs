#!/usr/bin/env python3
"""Fetch Cryptomatte sample EXRs and/or write pixel-matte goldens.

Samples are NOT stored in-repo. Cache them locally, then point tests at the
cache via CRYPTOMATTE_SAMPLES (or the default cache path this script prints).

Examples:
  python scripts/fetch_cryptomatte_samples.py
  python scripts/fetch_cryptomatte_samples.py --out D:/data/crypto-samples
  python scripts/fetch_cryptomatte_samples.py --download-only
  python scripts/fetch_cryptomatte_samples.py --write-goldens --skip-download
  python scripts/fetch_cryptomatte_samples.py --generate-synthetic

Goldens are produced by a *reference* matte walk over OpenEXR.File channel
planes (not OpenEXR.cryptomatte.extract), so Rust/Python cryptomatte extract
is checked against an independent implementation of the same spec math.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

PSYOP_ZIP = "https://github.com/Psyop/Cryptomatte/archive/refs/heads/master.zip"
ZIP_PREFIX = "Cryptomatte-master/sample_images/"

WANTED = [
    "bunny_CryptoObject.exr",
    "sidecar_manifest/bunny_CryptoObject.exr",
    "sidecar_manifest/bunny_CryptoObject_manifest.json",
    "debug_images/special_chars.exr",
    "debug_images/multichannel.exr",
    "debug_images/sidecar_manifest.exr",
    "debug_images/sidecar_manifest.crypto_asset.json",
]

# (relative exr, object name, optional layer name)
GOLDEN_JOBS = [
    ("bunny_CryptoObject.exr", "bunny", None),
    ("sidecar_manifest/bunny_CryptoObject.exr", "bunny", None),
    ("debug_images/special_chars.exr", "равнина", None),
    ("debug_images/special_chars.exr", "name_containing/slash", None),
    ("debug_images/multichannel.exr", "toroidal_asset", "crypto_asset"),
    ("debug_images/multichannel.exr", "ball_asset", "crypto_asset"),
    ("debug_images/sidecar_manifest.exr", "leftAsset", None),
    ("debug_images/sidecar_manifest.exr", "rightAsset", None),
]

COMPONENT = {
    "r": "id0",
    "red": "id0",
    "g": "cov0",
    "green": "cov0",
    "b": "id1",
    "blue": "id1",
    "a": "cov1",
    "alpha": "cov1",
}


def default_cache_dir() -> Path:
    if os.environ.get("CRYPTOMATTE_SAMPLES"):
        return Path(os.environ["CRYPTOMATTE_SAMPLES"])
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "cryptomatte-rs-samples"


def download_samples(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {PSYOP_ZIP}")
    with urllib.request.urlopen(PSYOP_ZIP, timeout=180) as resp:
        data = resp.read()
    print(f"zip size={len(data)} bytes; extracting subset -> {out}")
    wanted = set(WANTED)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name.startswith(ZIP_PREFIX) or info.is_dir():
                continue
            rel = name[len(ZIP_PREFIX) :]
            if rel not in wanted:
                continue
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  wrote {rel} ({dest.stat().st_size} bytes)")
            wanted.discard(rel)
    if wanted:
        missing = ", ".join(sorted(wanted))
        raise SystemExit(f"missing from Psyop zip: {missing}")


def _channel_pixels(ch):
    import numpy as np

    arr = np.asarray(ch.pixels)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def _ranked_pairs(crypto_name: str, channel_names: list[str]) -> list[tuple[str, str]]:
    by_base: dict[str, dict[str, str]] = {}
    for full in channel_names:
        if "." not in full:
            continue
        base, suffix = full.rsplit(".", 1)
        if not base.startswith(crypto_name):
            continue
        digits = base[len(crypto_name) :]
        if not digits or not digits.isdigit():
            continue
        slot = COMPONENT.get(suffix.lower())
        if not slot:
            continue
        by_base.setdefault(base, {})[slot] = full
    pairs: list[tuple[str, str]] = []
    for base in sorted(by_base):
        slots = by_base[base]
        if "id0" in slots and "cov0" in slots:
            pairs.append((slots["id0"], slots["cov0"]))
        if "id1" in slots and "cov1" in slots:
            pairs.append((slots["id1"], slots["cov1"]))
    return pairs


def _discover_layers(header: dict) -> list[dict]:
    prefixes = ("cryptomatte/", "exr/cryptomatte/")
    raw: dict[str, dict] = {}
    for key, value in header.items():
        if not isinstance(key, str):
            continue
        for prefix in prefixes:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if "/" not in rest:
                break
            mid, field = rest.split("/", 1)
            entry = raw.setdefault(mid, {"metadata_id": mid, "md_prefix": prefix})
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8", errors="replace")
            entry[field] = value
            break
    layers = list(raw.values())
    layers.sort(key=lambda L: L.get("name", ""))
    return layers


def _load_manifest(exr_path: Path, layer: dict) -> dict[str, float]:
    manif_file = layer.get("manif_file") or ""
    inline = layer.get("manifest") or ""
    if manif_file:
        if "\\" in manif_file:
            raise ValueError(f"invalid sidecar path: {manif_file}")
        text = (exr_path.parent / manif_file).read_text(encoding="utf-8")
    elif inline:
        text = inline
    else:
        return {}
    raw = json.loads(text)
    out: dict[str, float] = {}
    for name, hex_id in raw.items():
        bits = int(hex_id, 16)
        out[name] = struct.unpack("<f", struct.pack("<I", bits))[0]
    return out


def reference_extract(
    exr_path: Path, object_name: str, layer_name: str | None = None
):
    """Independent Cryptomatte coverage extract (OpenEXR.File planes + spec math)."""
    import numpy as np
    import OpenEXR as E

    with E.File(str(exr_path), separate_channels=True) as f:
        header = f.header()
        channels = f.channels()

    layers = _discover_layers(header)
    if not layers:
        raise RuntimeError(f"no Cryptomatte metadata in {exr_path}")
    if layer_name:
        layer = next(L for L in layers if L.get("name") == layer_name)
    else:
        layer = layers[0]
    crypto_name = layer["name"]
    manif = _load_manifest(exr_path, layer)
    if object_name not in manif:
        raise KeyError(f"{object_name!r} not in manifest of {exr_path.name}")
    selected_bits = struct.unpack("<I", struct.pack("<f", manif[object_name]))[0]

    pairs = _ranked_pairs(crypto_name, list(channels.keys()))
    if not pairs:
        raise RuntimeError(f"no ranked channels for {crypto_name} in {exr_path}")

    sample = _channel_pixels(channels[pairs[0][0]])
    h, w = sample.shape
    coverage = np.zeros((h, w), dtype=np.float32)
    for id_name, cov_name in pairs:
        ids = _channel_pixels(channels[id_name])
        covs = _channel_pixels(channels[cov_name])
        id_bits = ids.view(np.uint32)
        mask = id_bits == selected_bits
        coverage[mask] += covs[mask]
    return coverage


def golden_stem(exr_rel: str, object_name: str) -> str:
    key = f"{exr_rel}{chr(0)}{object_name}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    exr_stem = Path(exr_rel).stem.replace(" ", "_")
    return f"{exr_stem}.{digest}"


def golden_paths(out: Path, exr_rel: str, object_name: str) -> tuple[Path, Path]:
    stem = golden_stem(exr_rel, object_name)
    base = out / "goldens" / stem
    return Path(str(base) + ".matte.json"), Path(str(base) + ".matte.f32")


def write_golden(out: Path, exr_rel: str, object_name: str, layer_name: str | None) -> dict:
    import numpy as np

    exr_path = out / exr_rel
    if not exr_path.is_file():
        raise FileNotFoundError(exr_path)
    matte = reference_extract(exr_path, object_name, layer_name)
    meta_path, bin_path = golden_paths(out, exr_rel, object_name)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.ascontiguousarray(matte, dtype=np.float32).tobytes(order="C")
    bin_path.write_bytes(raw)
    stem = golden_stem(exr_rel, object_name)
    meta = {
        "exr": exr_rel.replace("\\", "/"),
        "name": object_name,
        "layer": layer_name,
        "stem": stem,
        "width": int(matte.shape[1]),
        "height": int(matte.shape[0]),
        "dtype": "float32",
        "order": "row-major",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sum": float(matte.astype(np.float64).sum()),
        "nonzero": int(np.count_nonzero(matte > 1e-6)),
        "max": float(matte.max()) if matte.size else 0.0,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    msg = (
        f"  golden {exr_rel}::{object_name} -> {bin_path.name} "
        f"nz={meta['nonzero']} sum={meta['sum']:.4f} sha256={meta['sha256'][:12]}..."
    )
    # Windows cp1252 consoles cannot print Cyrillic object names.
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="backslashreplace").decode("ascii"))
    return {
        "exr": meta["exr"],
        "name": object_name,
        "layer": layer_name,
        "stem": stem,
        "json": f"goldens/{stem}.matte.json",
        "bin": f"goldens/{stem}.matte.f32",
    }


def write_all_goldens(out: Path) -> None:
    print(f"writing goldens under {out / 'goldens'}")
    index = []
    for exr_rel, name, layer in GOLDEN_JOBS:
        index.append(write_golden(out, exr_rel, name, layer))
    synth = out / "synthetic" / "synth_CryptoObject.exr"
    if synth.is_file():
        for name in ("objA", "objB"):
            index.append(
                write_golden(out, "synthetic/synth_CryptoObject.exr", name, "synth")
            )
    index_path = out / "goldens" / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {index_path}")


def generate_synthetic(out: Path) -> None:
    """Tiny Cryptomatte EXR + golden with known exact pixels (no network)."""
    import numpy as np
    import OpenEXR as E

    id_a = struct.unpack("<f", struct.pack("<I", 0x3F800000))[0]
    id_b = struct.unpack("<f", struct.pack("<I", 0x40000000))[0]
    hex_a = f"{0x3F800000:08x}"
    hex_b = f"{0x40000000:08x}"
    mid = "synth01"
    h, w = 4, 4
    ids0 = np.zeros((h, w), dtype=np.float32)
    cov0 = np.zeros((h, w), dtype=np.float32)
    ids1 = np.zeros((h, w), dtype=np.float32)
    cov1 = np.zeros((h, w), dtype=np.float32)
    ids0[0, 0] = id_a
    cov0[0, 0] = 1.0
    ids0[0, 1] = id_a
    cov0[0, 1] = 0.5
    ids0[1, 0] = id_b
    cov0[1, 0] = 1.0
    ids0[1, 1] = id_a
    cov0[1, 1] = 0.25
    ids1[1, 1] = id_b
    cov1[1, 1] = 0.25

    manif = json.dumps({"objA": hex_a, "objB": hex_b}, separators=(",", ":"))
    header = {
        "compression": E.ZIP_COMPRESSION,
        f"cryptomatte/{mid}/name": "synth",
        f"cryptomatte/{mid}/hash": "MurmurHash3_32",
        f"cryptomatte/{mid}/conversion": "float",
        f"cryptomatte/{mid}/manifest": manif,
    }
    channels = {
        "synth00.R": ids0,
        "synth00.G": cov0,
        "synth00.B": ids1,
        "synth00.A": cov1,
    }
    synth_dir = out / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    exr_path = synth_dir / "synth_CryptoObject.exr"
    with E.File(header, channels) as f:
        f.write(str(exr_path))
    print(f"  wrote {exr_path}")

    expect_a = np.zeros((h, w), dtype=np.float32)
    expect_a[0, 0] = 1.0
    expect_a[0, 1] = 0.5
    expect_a[1, 1] = 0.25
    expect_b = np.zeros((h, w), dtype=np.float32)
    expect_b[1, 0] = 1.0
    expect_b[1, 1] = 0.25
    for name, arr in (("objA", expect_a), ("objB", expect_b)):
        got = reference_extract(exr_path, name, "synth")
        if not np.array_equal(got, arr):
            raise SystemExit(f"synthetic reference mismatch for {name}")
    print("  synthetic reference self-check OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--write-goldens", action="store_true")
    ap.add_argument("--generate-synthetic", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args(argv)
    out = (args.out or default_cache_dir()).resolve()

    do_download = True
    do_goldens = True
    do_synthetic = True
    if args.download_only:
        do_goldens = False
        do_synthetic = False
    elif args.write_goldens:
        do_download = False
        do_synthetic = False
    elif args.generate_synthetic:
        do_download = False
        do_goldens = False
    if args.skip_download:
        do_download = False

    if do_download:
        download_samples(out)

    if do_synthetic or do_goldens:
        try:
            import OpenEXR  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as e:
            print(
                "OpenEXR + numpy required for goldens/synthetic.\n"
                "  maturin develop -m crates/openexr-py/Cargo.toml\n"
                "  or pip-install the built OpenEXR wheel + numpy",
                file=sys.stderr,
            )
            raise SystemExit(1) from e

    if do_synthetic:
        print("generating synthetic fixture")
        generate_synthetic(out)

    if do_goldens:
        write_all_goldens(out)

    print()
    print("Done.")
    print(f"  samples: {out}")
    print("  Point tests at this cache:")
    if os.name == "nt":
        print(f"    set CRYPTOMATTE_SAMPLES={out}")
    else:
        print(f'    export CRYPTOMATTE_SAMPLES="{out}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
