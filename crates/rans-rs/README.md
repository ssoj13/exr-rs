# rans-rs

Integer interleaved rANS entropy coder — the entropy foundation of EXV (crate name `rans`).

## What it is
A pure crate (no dependencies) implementing integer rANS with independent interleaved lanes. Bit-exact
deterministic on any CPU/GPU and parallel (N lanes ↔ N GPU threads, no cross-lane interaction). All state math is
`u32`, so the CPU reference ports 1:1 to WGSL.

## Why / where used
The bottom of the `exv-rs` dependency graph. Consumed by `exv-codec` (entropy stage of the rANS-family, lossy-wave
and EXV-RT codecs) and by `exv-gpu` / `exv-gpu-kernels`, whose WGSL decoders are bit-exact against this reference.

## Key API
- `RansConfig { lanes, table_log }` — the CPU↔WGSL contract (default 128 lanes, `table_log` 12; `new` clamps to
  `lanes >= 1`, `table_log` 8..=15).
- `encode` / `encode_into` — self-contained container (embeds its frequency table); `decode_capped(data, max_n)` /
  `decode_into(data, out)` — bounded decode (there is no unbounded decode).
- `parse` / `parse_ref` / `prepare` / `Prepared` — the unpacked model + lane streams a GPU decoder uploads.
- `build_table`, `encode_block[_into]` / `decode_block[_into]`, `Model` — shared-table, per-block HEADERLESS coding
  (one pooled model, thousands of blocks; used by EXV-RT).
- `MAGIC`, `MAX_DECODE_SYMBOLS` (2^30), `RANS_L` / `RANS_UPPER` / `RENORM_GUARD` (contract constants the WGSL
  prelude is generated from), `EncodeError`.

## Notes / gotchas
- 32-bit state, `M = 2^table_log` (power of two), alphabet ≤256 are format invariants, not implementation details.
- The CPU decoder enforces an exact-end rule per lane (final state `RANS_L`, every byte consumed) and rejects
  truncated / padded / forged lanes; the GPU path checks structure only.
- Wire format: `docs/FORMAT.md` §4.2 (container) and §4.7 (EXV-RT substreams). Golden vector: `golden_vector` test.
