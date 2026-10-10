"""Hard tests for OpenEXR.cryptomatte (exr-rs extension) — not tautologies."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

import data_root


#: The sample tests read the test image set: off unless it is enabled (`EXR_TEST_DATA`) or
#: another sample set is given (`CRYPTOMATTE_SAMPLES`).
needs_samples = pytest.mark.skipif(
    not (os.environ.get(data_root.ENV) or os.environ.get("CRYPTOMATTE_SAMPLES")),
    reason=f"needs the Cryptomatte samples (set {data_root.ENV} or CRYPTOMATTE_SAMPLES)",
)


def _samples() -> Path:
    """`CRYPTOMATTE_SAMPLES` (another sample set), else the image set's `cryptomatte/`
    (the same rule as the Rust tests' `common::samples_root`)."""
    if os.environ.get("CRYPTOMATTE_SAMPLES"):
        return Path(os.environ["CRYPTOMATTE_SAMPLES"])
    return data_root.cryptomatte()


def _sample(rel: str) -> str:
    path = _samples() / rel
    assert path.is_file(), f"missing sample {path}"
    return str(path)


def _load_golden(exr_rel: str, name: str):
    import json

    SAMPLES = _samples()
    index = json.loads((SAMPLES / "goldens" / "index.json").read_text(encoding="utf-8"))
    exr_rel = exr_rel.replace("\\", "/")
    entry = next((e for e in index if e["exr"] == exr_rel and e["name"] == name), None)
    if not entry:
        return None
    meta = json.loads((SAMPLES / entry["json"]).read_text(encoding="utf-8"))
    raw = (SAMPLES / entry["bin"]).read_bytes()
    arr = np.frombuffer(raw, dtype="<f4").reshape(meta["height"], meta["width"]).copy()
    return meta, arr


def _assert_golden(matte, exr_rel: str, name: str):
    loaded = _load_golden(exr_rel, name)
    assert loaded is not None, f"no pixel golden for {exr_rel}::{name}"
    meta, expected = loaded
    assert matte.shape == (meta["height"], meta["width"])
    np.testing.assert_array_equal(matte, expected)


# Psyop / measured goldens (same as Rust cryptomatte tests).
PSYOP_HEX = {
    "hello": "248bfa47",
    "cube": "d9682f08",
    "sphere": "591e9a8d",
    "plane": "2e2136b8",
    "равнина": "9623511e",
    "mädchen": "6a4e5613",
    "bunny": "13851a76",
}
BUNNY_NZ = 68919
BUNNY_SUM = 60460.645


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def test_psyop_hex_goldens():
    import OpenEXR.cryptomatte as crypto

    for name, hex_id in PSYOP_HEX.items():
        assert crypto.float_id_to_hex(crypto.name_to_float_id(name)) == hex_id
        assert f32_bits(crypto.hex_to_float_id(hex_id)) == f32_bits(
            crypto.name_to_float_id(name)
        )


@needs_samples
def test_bunny_extract_stats_and_negative():
    import OpenEXR.cryptomatte as crypto

    BUNNY = _sample("bunny_CryptoObject.exr")

    layers = crypto.layers(BUNNY)
    assert len(layers) == 1
    assert layers[0]["name"] == "uCryptoObject"
    assert layers[0]["ranked_layers"]  # non-empty ranked list
    assert len(layers[0]["ranked_layers"]) == 3

    matte = crypto.extract(BUNNY, names=["bunny"])
    assert matte.dtype == np.float32
    assert matte.shape == (720, 1280)
    assert int(np.count_nonzero(matte > 1e-6)) == BUNNY_NZ
    # Sum in float64: numpy float32 reduction order differs slightly from Rust.
    assert abs(float(matte.astype(np.float64).sum()) - BUNNY_SUM) <= 0.5
    assert float(matte.max()) == 1.0
    assert not np.any(matte < 0.0)
    assert not np.any(matte > 1.0 + 1e-3)

    by_id = crypto.extract(
        BUNNY, ids=[crypto.hex_to_float_id(PSYOP_HEX["bunny"])]
    )
    np.testing.assert_array_equal(matte, by_id)
    _assert_golden(matte, "bunny_CryptoObject.exr", "bunny")

    junk = crypto.extract(BUNNY, names=["__no_such_cryptomatte_object__"])
    assert float(junk.sum()) == 0.0
    assert int(np.count_nonzero(junk > 1e-6)) == 0


@needs_samples
def test_special_chars_utf8():
    import OpenEXR.cryptomatte as crypto

    SPECIAL = _sample("debug_images/special_chars.exr")

    matte = crypto.extract(SPECIAL, names=["равнина"])
    assert matte.shape == (64, 64)
    assert int(np.count_nonzero(matte > 1e-6)) == 147
    assert abs(float(matte.sum()) - 108.11782) <= 0.02
    assert crypto.float_id_to_hex(crypto.name_to_float_id("равнина")) == "9623511e"
    _assert_golden(matte, "debug_images/special_chars.exr", "равнина")


@needs_samples
def test_multichannel_layers():
    import OpenEXR.cryptomatte as crypto

    MULTI = _sample("debug_images/multichannel.exr")

    layers = crypto.layers(MULTI)
    names = [L["name"] for L in layers]
    assert names == ["crypto_asset", "crypto_material", "crypto_object"]
    tor = crypto.extract(MULTI, names=["toroidal_asset"], layer="crypto_asset")
    assert int(np.count_nonzero(tor > 1e-6)) == 6665
    assert abs(float(tor.astype(np.float64).sum()) - 4459.2183) <= 0.05
    _assert_golden(tor, "debug_images/multichannel.exr", "toroidal_asset")


@needs_samples
def test_synthetic_exact_pixels():
    import OpenEXR.cryptomatte as crypto

    SYNTH = _sample("synthetic/synth_CryptoObject.exr")

    a = crypto.extract(SYNTH, names=["objA"], layer="synth")
    b = crypto.extract(SYNTH, names=["objB"], layer="synth")
    assert a.shape == (4, 4)
    assert float(a[0, 0]) == 1.0
    assert float(a[0, 1]) == 0.5
    assert float(a[1, 0]) == 0.0
    assert float(a[1, 1]) == 0.25
    assert float(b[1, 0]) == 1.0
    assert float(b[1, 1]) == 0.25
    _assert_golden(a, "synthetic/synth_CryptoObject.exr", "objA")
    _assert_golden(b, "synthetic/synth_CryptoObject.exr", "objB")


def test_encode_roundtrip(tmp_path):
    import OpenEXR.cryptomatte as crypto

    h, w = 2, 4
    bunny = np.zeros((h, w), dtype=np.float32)
    cube = np.zeros((h, w), dtype=np.float32)
    bunny[0, 0] = 1.0
    bunny[0, 1] = 0.5
    cube[0, 1] = 0.5
    cube[1, 0] = 1.0
    out = tmp_path / "encoded.exr"
    crypto.encode(
        str(out),
        "CryptoObject",
        {"bunny": bunny, "cube": cube},
        ranked_layers=2,
    )
    layers = crypto.layers(str(out))
    assert len(layers) == 1
    assert layers[0]["name"] == "CryptoObject"
    assert layers[0]["has_inline_manifest"] is True
    got = crypto.extract(str(out), names=["bunny"])
    np.testing.assert_array_equal(got, bunny)
    got_c = crypto.extract(str(out), names=["cube"])
    np.testing.assert_array_equal(got_c, cube)


def test_encode_sidecar(tmp_path):
    import OpenEXR.cryptomatte as crypto

    h, w = 2, 3
    bunny = np.zeros((h, w), dtype=np.float32)
    bunny[0, 0] = 1.0
    out = tmp_path / "sidecar.exr"
    crypto.encode(
        str(out),
        "CryptoObject",
        {"bunny": bunny},
        ranked_layers=1,
        sidecar="sidecar.json",
    )
    layers = crypto.layers(str(out))
    assert layers[0]["has_inline_manifest"] is False
    assert layers[0]["manif_file"] == "sidecar.json"
    assert (tmp_path / "sidecar.json").is_file()
    got = crypto.extract(str(out), names=["bunny"])
    np.testing.assert_array_equal(got, bunny)


def test_inject_over(tmp_path):
    import OpenEXR.cryptomatte as crypto

    h, w = 2, 2
    a = np.zeros((h, w), dtype=np.float32)
    a[0, 0] = 1.0
    src = tmp_path / "src.exr"
    crypto.encode(str(src), "CryptoObject", {"objA": a}, ranked_layers=2)
    b = np.zeros((h, w), dtype=np.float32)
    b[0, 0] = 0.5
    b[1, 1] = 1.0
    dst = tmp_path / "injected.exr"
    crypto.inject(str(src), str(dst), "objB", b, operation="over")
    got_a = crypto.extract(str(dst), names=["objA"])
    got_b = crypto.extract(str(dst), names=["objB"])
    # over replaces coverage at shared pixels for the injected id ranking
    assert float(got_b[1, 1]) == 1.0
    assert float(got_b[0, 0]) == 0.5
    assert float(got_a.sum()) > 0.0


def test_inject_rejects_a_transposed_coverage(tmp_path):
    """The (H, W) shape must match, not only the element count."""
    import OpenEXR.cryptomatte as crypto

    h, w = 2, 3
    src = tmp_path / "src.exr"
    crypto.encode(str(src), "CryptoObject", {"objA": np.ones((h, w), dtype=np.float32)})
    assert [layer["part"] for layer in crypto.layers(str(src))] == [0]
    transposed = np.ones((w, h), dtype=np.float32)
    with pytest.raises(ValueError, match=r"coverage shape \(3, 2\) does not match the image \(2, 3\)"):
        crypto.inject(str(src), str(tmp_path / "dst.exr"), "objB", transposed)
