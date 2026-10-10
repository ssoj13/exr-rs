#!/usr/bin/env python3
"""Regenerate the raw-DEFLATE binary fixtures for the deflate-rs inflater tests.

Each fixture is a pair of files in this directory:
  <name>.bin      -- the expected plaintext
  <name>.deflate  -- the raw DEFLATE stream (wbits=-15, no zlib header/trailer)

The compressed bytes are produced with Python's stdlib zlib at level 9 and
wbits=-15 (raw DEFLATE), exactly matching what `deflate_decompress` consumes.
Plaintext is fully deterministic (fixed formulas / repeating patterns) so the
fixtures are reproducible byte-for-byte and never depend on os.urandom.

Run from anywhere:
    python crates/deflate-rs/src/testdata/gen_fixtures.py
"""
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compress_raw(plain: bytes) -> bytes:
    """Raw DEFLATE (no zlib header/trailer), level 9 — wbits=-15."""
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return co.compress(plain) + co.flush()


def emit(name: str, plain: bytes) -> None:
    comp = compress_raw(plain)
    (HERE / f"{name}.bin").write_bytes(plain)
    (HERE / f"{name}.deflate").write_bytes(comp)
    print(f"{name}: plain={len(plain)} bytes  deflate={len(comp)} bytes")


# ~64 KB pseudo-random-looking but deterministic sequence (i*7+3 mod 256).
# Exercises dynamic Huffman + large distances. (Was the brittle `large_mixed`.)
emit("large_mixed", bytes([(i * 7 + 3) % 256 for i in range(65536)]))

# 4096 copies of one byte — RLE-like, exercises long matches / max length runs.
emit("all_same_byte", b"\x42" * 4096)

# 200 repetitions of "abcabc" (1200 bytes) — exercises short back-references.
emit("repetitive_backrefs", b"abcabc" * 200)
