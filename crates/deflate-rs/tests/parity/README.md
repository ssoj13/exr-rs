# deflate-rs ⟷ C libdeflate byte-exact golden parity

This directory proves `deflate-rs`'s **zlib compressor** is **byte-identical** to
the real C **libdeflate** (vcpkg, v1.26) for every compression level 1..=12 over
a fixed set of deterministic inputs.

Why this matters: OpenEXR's ZIP/ZIPS codec compresses pixel data with
`libdeflate_zlib_compress`. DEFLATE streams are **not** canonical — two correct
encoders can emit different bytes for the same input. So the **only** way an
exr-rs `.exr` file can be byte-identical to a C++ OpenEXR `.exr` file is if
deflate-rs reproduces libdeflate's exact compressor heuristics. This gate is the
proof.

## Files
- `oracle.c` — a small C program that `#include`s the vcpkg `libdeflate.h`, and
  for each fixed input × level 1..=12 calls `libdeflate_zlib_compress`, printing
  one machine-readable line `level<L>.<input> = <hex of exact compressed bytes>`.
  It also emits each input's exact bytes as `input.<name> = <hex>` so the Rust
  side can self-verify it reproduces the identical input.
- `golden.txt` — the oracle's stdout, **committed** so the Rust test runs
  standalone (no C toolchain needed for plain `cargo test`).
- `../parity_libdeflate.rs` — the Rust integration test: reproduces the same
  inputs (by the documented formulas below), runs deflate-rs `zlib_compress`, and
  asserts **byte-identity** plus a decode round-trip.

## Inputs (the shared contract — reproduced identically on both sides)
| name            | definition                                                          |
|-----------------|---------------------------------------------------------------------|
| `empty`         | 0 bytes                                                             |
| `short_str`     | ASCII `"hello, deflate-rs vs libdeflate!\n"` (no NUL)               |
| `repetitive`    | 4096 bytes, `buf[i] = b'a' + (i % 3)` (cycle a,b,c)                 |
| `all_same`      | 8192 bytes all `0x5A`                                               |
| `pseudo_random` | 65536 bytes from xorshift32, seed `0x12345678`, emit low byte/step |
| `structured`    | 1000 lines `"row {i}: the quick brown fox jumps over the lazy dog ({i})\n"` |

The `xorshift32` step is: `x ^= x<<13; x ^= x>>17; x ^= x<<5` (u32 wrapping); the
emitted byte is `x & 0xFF`. The Rust test asserts each locally-reproduced input
matches the oracle's `input.<name>` hex before comparing compressed output, so
any drift fails loudly and is never mistaken for a compression mismatch.

## How to run / regenerate

```text
cargo run -p xtask -- parity-deflate      # or: python bootstrap.py parity-deflate
```

On **Windows** this:
1. applies the MSVC toolchain to the process env via `vcv-rs` (no manual
   `vcvars64.bat`), so `cl` is on PATH;
2. compiles `oracle.c` with `cl` against the vcpkg `libdeflate.h`, linking
   `deflate.lib`;
3. runs the oracle (with vcpkg `bin/` on PATH so `deflate.dll` resolves) and
   writes its stdout to `golden.txt`;
4. runs `cargo test -p deflate-rs --test parity_libdeflate` against the fresh file.

On **non-Windows** the command is a no-op (the oracle needs MSVC + a vcpkg
libdeflate install) and exits 0. The committed `golden.txt` still lets the Rust
test run.

The vcpkg tree is picked by the shared resolver in `xtask/src/vcpkg.rs`: `VCPKG_INSTALLED`
(a full `installed/<triplet>` path) wins, else the first `x64-windows*` triplet under
`$VCPKG_ROOT/installed` (default `C:/vcpkg/installed`) that has the header. A DLL triplet links
the import library `deflate.lib` and puts vcpkg `bin/` (for `deflate.dll`) on PATH at
runtime; a static triplet links `deflatestatic.lib`.

## Byte-exact policy (and why)

**Compress is BYTE-IDENTICAL.** deflate-rs `zlib_compress` output must equal
libdeflate's output byte-for-byte, including the 2-byte zlib header, every
DEFLATE block, and the 4-byte Adler-32 trailer. This is asserted with `==` on the
full `Vec<u8>`. **Do not loosen this check** to hide a divergence — any mismatch
is a real port bug in the compressor heuristics (diagnose against libdeflate v1.26
`lib/deflate_compress.c`,
<https://github.com/ebiggers/libdeflate/blob/v1.26/lib/deflate_compress.c>, never patch the test).

**Decode is CANONICAL.** Separately, the test decodes *libdeflate's* output with
our `zlib_decompress` and asserts it recovers the original input. Any valid
DEFLATE stream must decode correctly, so this is an independent correctness check
of our inflater, not a byte-comparison.

## libdeflate version

vcpkg currently ships libdeflate **1.26** (`LIBDEFLATE_VERSION_STRING`), which is
the version `deflate-rs` is ported from (bumped 2026-09-22 from 1.25 — the only
behavioural delta was `compression_level == -1` aliasing to level 6). If vcpkg upgrades to a newer
libdeflate whose compressor heuristics changed, regenerating `golden.txt` will
surface the delta as parity failures — that is a signal to re-converge the port
to the new upstream, not to relax the gate.
