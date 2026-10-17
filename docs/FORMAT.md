# EXV On-Disk Format Specification

> Status: **pre-stable** (header version byte `0x01`, container version byte `0x03`). This document
> is derived from the **code** (code is the truth); it cites function names, not line numbers, because
> line numbers rot. Where this text and the code disagree, the code wins - fix this text.

All multi-byte integers are **little-endian (LE)**, everywhere. Every parser reads fields through the
bounded reader `exv_format::Rd` (`crates/exv-format/src/rd.rs`): a short read is `None`, never a panic.

---

## 1. Overview

There is **one** file kind: the `.exv` container (`exv-video::container`). A still image is a 1-frame
`.exv` (`frame_count = 1`, a single INTRA frame); there is no separate still format.

```
.exv = prelude | frames (decode order) | outer frame index | index_off trailer
frame = frame_hdr | frame_body        frame_body = tile blobs | tile index | index_off trailer
tile blob = [chan_len[nch]:u32][per-channel blobs]   (every codec except HTJ2K)
          | [TAG_HTJ2K][one multi-component codestream] (HTJ2K) -- see §4
```

| Layer | Crate / module | Spec |
|-------|----------------|------|
| Container (`EXV1`) | `exv-video::container` | §6 |
| Header + Meta (no magic of its own) | `exv-format::header` | §2 |
| Frame body (tiles) | `exv-codec::frame` | §5 |
| Tile blobs + per-blob framing + codec payloads | `exv-format::blob`, `exv-codec` | §4 |
| Codec registry | `exv-format::types` | §7 |
| Allocation limits / decode budget | `exv-format::limits` | §8 |

Version bytes: the embedded `Header` carries `VERSION = 1` (`header::VERSION`); the container prelude
carries `VERSION = 3` (`container::VERSION`, the tiled/IBP container with CRC-32C per coded blob in the frame tile
index, §5.1; `2` was the same layout without checksums and no longer opens). Layout is not guaranteed
stable until the GPU==CPU gate.

---

## 2. `Header` (`exv-format::header`)

Serialized by `Header::write` / `Header::write_with_meta`, parsed by `Header::read` /
`Header::read_with_meta`. The same header is **embedded** in the `.exv` prelude and is authoritative for
the whole clip (dimensions, tile grid, channels, codec).

### 2.1 Byte layout

```
offset  size   field            notes
------  -----  ---------------  ----------------------------------------------------
  0      1     version : u8     = 0x01 (VERSION); anything else -> reject
  1      1     flags : u8       bit0 = a Meta TLV block follows the channels (§2.2);
                                bits 1..7 are written 0; a set bit = reject (`Corrupt`)
  2      4     width : u32      > 0
  6      4     height : u32     > 0
 10      4     frame_count : u32  >= 1: 1 for a still, N for video (must equal the container prelude's).
                                0 is only a streaming-writer placeholder patched at finish; the reader rejects it
 14      4     tile_w : u32     1..=MAX_TILE_DIM (65536); effective tile area capped (§2.4)
 18      4     tile_h : u32     1..=MAX_TILE_DIM
 22      4     codec_id : u32   see §7
 26      2     cp_len : u16     codec_params length
 28     cp_len codec_params : u8[]   codec params (e.g. [q], [kind, q]); layout per codec_id (§7)
 ..      2     nch : u16        channel count
 ..      per channel (nch times):
         1     name_len : u8
        name_len name : u8[]    UTF-8 "layer.sub.chan"
         1     dtype : u8       §3
 ..      [if flags & 1]  meta_len : u32, then meta_len bytes of Meta TLV (§2.2)
```

### 2.2 `Meta` TLV block (flags bit0)

Optional, forward-extensible clip metadata: fps / color / alpha / pixel aspect / timecode. The block is
`meta_len:u32` followed by TLV records `[tag:u8][len:u16][value]`. Empty `Meta` is not written at all
(`flags = 0`), so a header without metadata is byte-identical to the pre-Meta form. Decode skips unknown
tags and records whose length does not match the tag (forward-compatible) and stops at a truncated record.
A decoded `fps` or `pixel_aspect` with `den == 0` is a hard error (`Corrupt`), and `write_with_meta` refuses it too.

| Tag | Len | Value |
|-----|-----|-------|
| 1 | 8 | `fps`: `num:u32, den:u32` (exact rational, e.g. 24000/1001) |
| 2 | 3 | `primaries:u8, transfer:u8, range:u8` (host-defined enums, `0 = unspecified`) |
| 3 | 1 | `alpha:u8` (0 unspecified, 1 straight, 2 premultiplied) |
| 4 | 8 | `pixel_aspect`: `num:u32, den:u32` (absent = square) |
| 5 | 4 | `timecode:u32` (frames since 00:00:00:00) |
| 6 | 16 | `exr_headers`: chunked serialized OpenEXR part headers (`[n_parts:u32][len:u32, bytes]*`) |

### 2.3 Write rules (`Header::write*` returns `Result<(), WriteError>`)

A header whose fields do not fit their on-disk width is a hard error in **every** build profile (nothing
is appended to `out`): more than 65535 channels (`WriteError::Channels`), a channel name over 255 bytes
(`WriteError::Name`), `codec_params` over 65535 bytes (`WriteError::Params`), or a geometry the reader
would reject (`WriteError::Geometry`: the §2.4 dimension / tile rules), or an unreadable channel list / meta
(`WriteError::Invalid`: no channels, duplicate channel name, `den == 0`). `frame_count == 0` is the one value the
writer MAY emit (streaming placeholder, patched at finish). Silent truncation, or writing a header
the reader refuses, would produce an unreadable file.

### 2.4 Read rules (`Header::read*` returns `Result<_, DecodeError>`)

Reject (`Err(DecodeError::Corrupt(reason))`; a foreign header `version` is `Unsupported`): short input, any of `width/height/tile_w/tile_h == 0`,
`tile_w/tile_h > MAX_TILE_DIM`, an **effective tile area** `min(tile_w,width) * min(tile_h,height)` over
`MAX_TILE_AREA = 2^30` px (a tile may be larger than the image, so the EFFECTIVE tile is what decode must
hold; the cap is on PIXELS, not bytes - see §8), non-UTF-8 name, unknown dtype code, a channel plane
`width*height*dtype.size()` that overflows `usize` (`plane_len`, computed in `u64`), a truncated Meta block, any unknown `flags` bit (1..7), `frame_count == 0`, `nch == 0`, a duplicate channel
name, a Meta `fps`/`pixel_aspect` with `den == 0`.
The geometry rules are ONE function (`check_geometry`) shared by `Header::read` and `Header::write`. The
channel `Vec` capacity is clamped by the remaining input (`limits::cap`), so a forged `nch` cannot
over-reserve. The **image** `width/height` are deliberately not capped (65k² tiled is a design goal):
`read` does not judge the declared image size - decoding it is budgeted by the caller (§8).

### 2.5 Geometry helpers

`TileGrid` (and `Header`, which delegates to `Header::grid()`): `tiles_x = ceil(width/tile_w)`,
`tiles_y = ceil(height/tile_h)`, `tile_count = tiles_x*tiles_y` as **`u64`** (overflows `u32` at 65536²
with tile=1), `tile_rect(tx,ty) -> (x,y,w,h)` clamped to the image with saturating arithmetic (edge tiles
are smaller). `Header::frame_bytes()` = checked sum of every channel's plane bytes (the figure the §8 budget
is compared against).

---

## 3. Dtype table

`Dtype::{code, from_code, size, is_float}`:

| Name | Code | Element size | Float |
|------|------|--------------|-------|
| `F16`  | 0 | 2 | yes |
| `F32`  | 1 | 4 | yes |
| `Bf16` | 2 | 2 | yes |
| `U8`   | 3 | 1 | no |
| `U16`  | 4 | 2 | no |
| `U32`  | 5 | 4 | no |

Any other code is a corrupt header. `is_float` only steers encode-time stage selection; it is not stored.
Channels may have **different** dtypes within one header.

---

## 4. Tile blobs, channel blobs and codec payloads

A **tile blob** is the unit the frame index (§5) points at. Its layout depends on the file's `codec_id`
(§7) - it is NOT self-signalling, a reader derives it from the header:

| Layout | `codec_id` | Tile blob |
|--------|-----------|-----------|
| per-channel | 1, 2, 3, 4, 7 (rANS family, lossy wave, EXV-RT, BC6H) | `[chan_len[nch] : u32][ch0 blob][ch1 blob]...[ch{nch-1} blob]` |
| multi-component | 6 (HTJ2K, lossless or lossy) | `[TAG_HTJ2K = 3][one multi-component J2K codestream carrying ALL channels]` |

Per-channel: channels are in `Header.channels` order; `chan_len[i]` is the byte length of channel blob `i`
(§4.1); the reader consumes the sub-blobs in order, bounds-checked (a forged table or short blob is an error;
bytes after the last sub-blob are ignored, writers emit none). Multi-component: there is no length table - the
codestream's SIZ carries the component count (== `nch`). Both are produced by `exv-codec::encode_tile` and
read by `decode_tile` / `FrameReader`.

### 4.1 Channel blob = `[tag:u8][payload]` (`exv-format::blob`)

`classify_blob` returns `None` for an empty blob or a tag `>= 6`.

| Tag | Const | Payload | Decodable on |
|-----|-------|---------|--------------|
| 0 | `TAG_RAW`     | the raw plane bytes, exactly `w*h*esz` (the tile rect) | CPU/GPU |
| 1 | `TAG_RANS`    | a `rans-rs` container (§4.2) over the decorrelated bytes (lossy-wave: the coefficient stream, §4.4) | CPU / GPU |
| 2 | `TAG_DEFLATE` | raw DEFLATE (RFC 1951, no zlib wrapper) over the decorrelated bytes; must inflate to exactly `w*h*esz` | CPU only |
| 3 | `TAG_HTJ2K`   | a complete HTJ2K (JPEG 2000 Part 15) codestream. In a file it only appears as the tile-level tag of the multi-component layout (§4.6); the single-component form (`encode_channel` under an HTJ2K profile) is the same codestream with one component and exists for the per-channel API/tests, not the file layout | CPU (jph-rs) / GPU (jph-wgpu) |
| 4 | `TAG_EXV_RT`  | an EXV-RT block-codec payload for one **F16** channel plane (LOSSY, §4.7) | CPU / GPU (WGSL) |
| 5 | `TAG_BC6H`    | a BC6H_SF16 hardware block payload for one **F16** channel plane (with optional rANS lossless residual) | CPU / GPU (HW TMU + WGSL) |

Notes: the per-blob `TAG_RAW` (0) is unrelated to `CodecId::RAW` (0). Encode keeps a candidate only when
strictly smaller than the raw plane (no-shrink rule, +1 byte worst case); under EXV-RT every non-F16 channel is
stored `TAG_RAW` (a lossy profile keeps ints exact). `EntropyPolicy` is encode-only and **not stored**; decode is
tag-driven. A tag that does not match the profile (e.g. `TAG_HTJ2K` in a rANS file) is a corrupt blob.
`w`/`h` below are the tile rect (`TileGrid::tile_rect`, edge tiles are smaller); the decoded channel must be
exactly `w*h*esz` bytes.

### 4.2 `rans-rs` container (inside `TAG_RANS`)

```
MAGIC:u32 "RANS" | table_log:u8 (8..=15) | lanes:u8 (>=1) | n:u32 | freq[256] : u16 | lane_len[lanes] : u32 | lane streams
```
`freq` sums to `M = 2^table_log`. Lane `k` codes symbols `k, k+lanes, k+2*lanes, ...` (`ceil((n-k)/lanes)` of them).
Each lane stream starts with its 4-byte LE initial state, which must lie in `[2^23, 2^31)`. A lane is valid only
if, after decoding its symbols, the state is back at `2^23` and every byte was consumed; a decoder that tries to
read past a lane's end aborts at once. The CPU decoder rejects truncated, padded or forged lanes; the GPU
kernels do not check this (their output on such a lane is unspecified, `rans::parse` checks the same STRUCTURE
- magic, `lanes >= 1`, `n` ceiling, freq sum, lane bounds, initial states - and nothing about lane ends).
Encoder-side frequency normalization (not needed to decode): scale counts proportionally (`max(1, c*M/total)`),
then correct the sum to `M` in +-1 steps on the highest-indexed maximum.

### 4.3 Lossless payloads (`codec_id` 1, 2): `TAG_RANS` / `TAG_DEFLATE` over a decorrelated plane

The plane (row-major `w*h` elements of `esz` bytes, LE) is decorrelated, then entropy coded (rANS preferred when
it shrinks, DEFLATE otherwise or under `MaxRatio`, RAW if neither is smaller than the plane):

- `LOSSLESS_FAST` (1) - `split_delta`: for each byte significance `p = 0..esz` (ascending), emit the first
  difference (`b[e] - b[e-1]`, wrapping `u8`, previous starts at 0) over all elements' byte `p`. `esz = 1` is a
  plain first difference. Inverse: prefix sums per byte plane.
- `WAVELET53` (2) - one level of reversible integer 5/3 lifting, **tile-local, no halo**: elements are read as
  unsigned LE integers; arithmetic wraps modulo `2^(8*esz)` (sums in `u64`). Rows first, then columns. Per line of
  length `n >= 2`: predict odd `i`: `a[i] -= (a[i-1] + r) >> 1` with `r = a[i+1]` (or `a[i-1]` if `i+1 = n`);
  update even `i`: `a[i] += (l + r + 2) >> 2` with `l = a[i-1]` (or `a[1]` if `i = 0`), `r = a[i+1]` (or
  `a[i-1]` if `i+1 = n`). Output = the same `w*h` elements re-serialized LE. Inverse: columns then rows, undoing
  update then predict.

A rANS or DEFLATE payload must decode to exactly `w*h*esz` bytes.

### 4.4 Lossy wave payload (`codec_id` 3, `[q]`): `TAG_RANS` (or `TAG_RAW`) only

The rANS container's `n` symbols are the **zigzag + LEB128 varint** bytes of `w*h` signed `i64` coefficients
(row-major): `z = (c << 1) ^ (c >> 63)`, then 7-bit groups LSB first with the continuation bit `0x80`. Each
coefficient needs at least one byte, so a symbol count below `w*h` is corrupt. The coefficients are:

1. input values: `q == 0`, or an integer dtype: the element as an unsigned integer (zero-extended, so floats
   at `q == 0` keep their exact bit pattern); `q > 0` and a float dtype: the **hdr log codes** of §4.5 at
   `scale = 256` (f16/bf16 widened to f32 first);
2. a **signed** 5/3 lifting in `i64` (no wrapping, floor shifts), rows then columns, same boundary rules as §4.3
   with `a[i] -= (l+r) >> 1` and `a[i] += (l+r+2) >> 2` on signed values;
3. for `q > 0` and a **float** dtype, every **detail** coefficient (`x` odd or `y` odd in the transformed array)
   is divided by `step = 2^min(q, 62)` rounding half away from zero (`sign * ((|c| + step/2) / step)`); LL
   (both even) is exact. **Integer channels (U8/U16/U32: ids, masks, counters) are never quantized**: they use
   the same lifting with `q` ignored, so they round-trip bit-exact under every `q`.

Decode: for `q > 0` float dtypes dequantize (`* step`, wrapping), inverse lifting (columns then rows), then for
`q > 0` float dtypes `hdr` decode at scale 256, otherwise clamp each value to `[0, 2^(8*esz) - 1]` and write it
LE. Integer channels are never log-mapped nor dequantized.

### 4.5 HDR log codes (`exv-codec::hdr`; shared by §4.4, §4.6 lossy and §4.7)

A float sample maps to a signed integer **code** at `scale` codes per stop (`scale` is an integer `1..=256`;
codes fit 16-bit signed: `|code| <= 64*scale`). Everything is IEEE `+ - * /` and exact bit operations - no
libm - so every platform and the GPU agree bit for bit.
- **encode**: f16/bf16 are widened exactly to f32. `NaN -> 0`; `+-inf -> +-64*scale`; let `a` be the unbiased
  binary exponent of `|v|` and `f in [1,2)` its mantissa. `a < -32 -> 0`; `a >= 32 -> +-64*scale` (saturate);
  else `r` = number of thresholds `exp2_frac((j + 1/2)/scale)` (`j = 0..scale-1`) that are `<= f`, and
  `code = sign * ((a + 32)*scale + r)`. Consequently `-0.0` and everything below `2^(-32 + 1/(2*scale))`
  (including exactly `2^-32`) encode to code 0, which decodes to `+0.0`.
- **decode**: code 0 -> `0.0`; else `k = min(|code|, 64*scale) - 32*scale`, `a = floor(k/scale)`,
  `r = k mod scale`, `value = sign * 2^a * T[r]` with `T[r] = f32(exp2_frac(r/scale))` (exact exponent add, one
  multiply). Narrowing to f16/bf16 is round-to-nearest-even (overflow -> inf); a code never decodes to NaN.
- `exp2_frac(x)`, `x in [0,1)`: f64 Horner evaluation of `sum_{k=0..18} (x*ln2)^k / k!` (`1/k!` constants built
  with IEEE division); `hdr::exp2_table(scale)` is the table a GPU loads instead of calling `exp2`.

### 4.6 HTJ2K payload (`codec_id` 6): ONE multi-component codestream per tile

`params` select the mode (`exv_codec::Htj2kMode`): `[]` = **Lossless**; `[1, q]` = **HdrCodes** (lossy floats);
`[2, q]` = **Irrev97** (irreversible 9/7, `q` in `1..=40`, U8/U16 channels only). The old bare `[q]` form does not
exist. The tile blob is `[TAG_HTJ2K][codestream]` (§4). Codestream: image offset `(0,0)`, extent = the tile rect
`w x h`, `nch` components in header channel order, every component 1x1-subsampled, **no** colour transform
(channels are independent AOVs), reversible 5/3 (9/7 for Irrev97, see below), 5 decompositions, 64x64
code-blocks, non-planar (line-interleaved) feed. Per component `(bit_depth, signed)`:

| dtype | component | sample = | NLT |
|-------|-----------|----------|-----|
| `F16`, `Bf16` | 16-bit signed | the 16 bits as sign-extended `i16` | binary-complement (type 3) |
| `F32` | 32-bit signed | the 32 bits as `i32` | binary-complement (type 3) |
| `U32` | 32-bit unsigned | the 32 bits as `i32` (wraps) | none |
| `U16` | 16-bit unsigned | zero-extended | none |
| `U8` | 8-bit unsigned | zero-extended | none |

HdrCodes (`[1, q]`): FLOAT channels become hdr codes (§4.5) at `scale = clamp(16*q, 16, 256)` as 16-bit **signed**
components with no NLT, coded losslessly; INTEGER channels stay exactly as in the table (never quantized). A
decoder must verify the codestream's **headers** against what the tile must be, BEFORE handing the bytes to a
JPEG 2000 library (the stream's own markers must never size a buffer or drive a loop; a sub-sampled component
hangs a line puller and a 4x4 code-block exponent makes it allocate hundreds of MiB). The reference decoder
accepts exactly:

- main header = `SOC`, `SIZ` first, then `CAP` (Part 15, `Pcap = 0x00020000`, one `Ccap`), `COD`, `QCD`, any
  number of `QCC` (one per component at most) / `NLT` (one per component at most) / `COM`, then `SOT`. Any other
  marker (COC, RGN, POC, PPM/PPT, TLM, PLM/PLT, CRG, DFS, ATK, ...) is corruption;
- `SIZ`: extent `w x h`, image offset `(0,0)`, ONE tile (`XTsiz = w`, `YTsiz = h`, tile offset `(0,0)`), `Csiz = nch`,
  every component `XRsiz = YRsiz = 1` with the bit depth and signedness of the table above;
- `COD`: `Scod = 0` (no precincts, SOP or EPH), progression order `0..=4`, ONE layer, no multi-component
  transform, 5 decompositions, code-block exponents `4,4` (64x64), code-block style `0x40` (HT only), reversible 5/3;
- `QCD`/`QCC`: reversible no-quantization style, >= 1 guard bit, `3 * 5 + 1` subband exponents;
- `NLT` entries (type 3, `Snlt` = the component's `SSiz`) present on exactly the NLT components of the table
  (never on integer components or hdr code components);
- tile-parts: each `SOT` has `Lsot = 10`, `Isot = 0`, an EMPTY tile-part header (`SOD` right after), and a `Psot`
  that is 0 or lands on the next `SOT` / `EOC` inside the codestream.

A mismatch is corruption, not a reinterpretation. Resolution levels give a
mip pyramid: a Lossless or Irrev97 tile can be decoded at level `L`, dims `ceil(w / 2^L) x ceil(h / 2^L)` (`L <=` the
stream's decomposition count); an HdrCodes tile cannot (its codes go through the hdr inverse at full resolution).

**Irrev97 (`[2, q]`).** ONE wavelet per codestream (the main COD), so the mode takes tiles whose channels are ALL
`U8`/`U16` (unsigned, no NLT); a float or `U32` channel is an encode error (`CodecError::Dtype`), never quantized and never
a silent fallback. The codestream is the table above with these differences, all checked by the reader: `COD`
transform byte `0` (irreversible 9/7) instead of `1`; `QCD`/`QCC` = scalar-expounded style 2 (`Sqcd & 0x1F = 2`, >= 1
guard bit, `16`-bit `(exponent << 11 | mantissa)` per subband, `3 * 5 + 1` entries; the step itself sizes nothing and
is not validated). Quantizer step: jph-rs `delta`, NORMALISED to the sample range (its default is `2^-min(16,
bit_depth)`, one LSB): `delta = 2^(-q/2)` (encode only; the step lives in the QCD, so a decoder never needs `q`):
q=16 is about one 8-bit LSB, q=32 one 16-bit LSB, both U8 and U16 see the same relative error. Reconstruction is
rounded and CLAMPED by the decoder to `[0, 2^bits - 1]` (jph does not clamp; an unclamped 255 -> 256 overshoot would
wrap to 0 in the low byte). Measured on the reference codec: U8 `q=16` max error 1, `q=24` exact; U16 `q=32` max
error 1, `q=24` 11, `q=16` 168. NOT GPU-decodable (`gpu_kind = None`: jph-wgpu's irreversible entry points take 1 or 3
components with a colour transform); a lossy profile keeps P-frame tiles INTRA/SKIP.

### 4.7 EXV-RT payload (`codec_id` 4): per-channel, F16 channels only (LOSSY)

`params = [ver, block, p, flags]` selects the payload of each F16 channel's `[TAG_EXV_RT][payload]`; every other
dtype is a `TAG_RAW` channel blob.

**`ver = 1` Dwt** (the shipping codec; `block in 1..=64`, `p = q < 16`):
```
block:u8 | q:u8 | table_log:u8 | lanes:u8 | nbx:u16 | nby:u16 | freq[256] : u16
  | off[nbx*nby + 1] : u32 | width[nbx*nby] : u8 | substreams
```
`block`/`q` must equal the profile's (the GPU reads the profile); `nbx = ceil(w/block)`, `nby = ceil(h/block)`;
`lanes >= 1`; `off` is a monotonically non-decreasing byte offset index into the substream region, ending exactly
at its length, every substream at least `8*lanes - 4` bytes; `width[bi] in 1..=4`. The plane is mirror-padded to
`nbx*block x nby*block` (index folding with period `2(n-1)`, no edge repeat; `n <= 1 -> 0`), cut into
`block x block` blocks (row-major block order). Each block's F16 plane goes through the §4.4 front end
(`forward_quant` at `q`: raw 16-bit values at `q = 0`, hdr codes at scale 256 for `q > 0`; signed 5/3; detail
quantization), then each coefficient is zigzagged and stored LE in `width[bi]` bytes (the minimum covering the
block's largest code). The pooled code bytes of ALL blocks define ONE rANS model (`table_log = 12`, §4.2
normalization, stored as `freq`); block `bi` is a headerless `lanes`-way substream against it:
`[lane_len[lanes-1] : u32][lane streams]` (the last lane takes the rest of the slice), `n = block^2 * width[bi]`
symbols, exact-end rule per lane. Decode reverses and crops the padded plane to `w x h`.

**`ver = 0` Lerp** (reference/proxy; `block >= 1`, `p = idx_bits in 1..=8`, `flags` bit0 = log space): the plane
is tiled into `block x block` blocks (edge blocks clamp their indices), each exactly
`4 + ceil(block^2 * idx_bits / 8)` bytes: `[min:f16][max:f16][idx_bits-bit index per texel, row-major, LE
bit-packed]`; the payload length must equal `blocks * block_bytes`. `value = min + idx/(2^idx_bits - 1) *
(max - min)`; in log space the endpoints/interpolation are in `flog(v) = sign(v) * log2(1 + |v|)` and the result
is mapped back by `sign(c) * (2^|c| - 1)` (deterministic `hdr::{log2f, exp2f}`).

---

## 5. Frame body (`exv-codec::frame`)

A header-less tiled frame; the container's `Header` supplies dims / tile grid / channels / codec.

```
frame body = [tile blobs ...]            (a SKIP tile contributes 0 bytes)
           | index { n_tiles:u32, n_crc:u16,
                     per tile: kind:u8, off:u64, len:u32, crc:[u32; n_crc] }   (off/len relative to body[0])
           | index_off:u64                (trailer: last 8 bytes of the body)
```

- Tile order is row-major (`tx = t % tiles_x`, `ty = t / tiles_x`). `n_tiles` must equal `tile_count`.
- A tile blob holds the tile's channels in `Header.channels` order; its layout is §4 (per-channel vs HTJ2K
  multi-component, chosen by `codec_id`).
- Tile kinds (`TILE_*`): `0 INTRA` the tile itself; `1 PRED` the integer bit-pattern residual `cur - ref`
  (LOSSLESS profiles only; decode adds back the reference); `2 BI` reserved; `3 SKIP` `len = 0`, the tile is
  byte-identical to the reference's co-located tile.
- Every non-SKIP blob must lie inside `[0, index_off)`. A SKIP/PRED tile without a reference is an error.
- The reference is the previously **reconstructed** frame, already motion-compensated by the container.
- `n_crc` must be `1` or `nch` (else corrupt) and must equal the codec's layout (§5.1) when the codec opens.

### 5.1 Blob checksums (CRC-32C) and the `Verify` policy

`crc` = CRC-32C (Castagnoli, reflected, init/xorout `0xFFFFFFFF`, check value `crc32c("123456789") = 0xE3069283`;
`exv-format::crc`) of every CODED blob of the tile, chosen by the codec's tile layout (§4):

| Layout | `n_crc` | `crc[i]` covers |
|---|---|---|
| per-channel (codec 1-4) | `nch` | channel `i`'s sub-blob `[tag][payload]` (§4.1). The `chan_len` table is covered implicitly: a damaged length moves the sub-blob boundaries, so a split failure (`Corrupt`) or a mismatch follows. |
| multi-component (codec 6) | `1` | the whole tile blob `[TAG_HTJ2K][codestream]` |

SKIP tiles store zeros. The writer ALWAYS checksums (`write_frame_tiles` / `write_intra_frame_body_windowed`, on the
parallel encode lanes). One CRC per sub-blob (not per tile) is what keeps the read granularity: a damaged channel fails
alone (lenient region reads record just that `(tile, channel)`), and a channel-subset decode hashes only the channels
it decodes.

Reading: `DecodeLimits.verify: Verify::{On, Off}` (default **On**; `DecodeLimits::with_verify`) is the one policy,
carried by the existing decode options (`FrameReader::with_limits`, `VideoIndex::with_limits`, `ExvSource`, hosts).
Every blob accessor of `FrameReader` verifies BEFORE the bytes reach a codec or the GPU: the tile decoders (wanted
channels only), `tile_blob` (all blobs of the tile), `tile_chans` (all channels, split once), `tile_channel_blob`
(that channel only). A mismatch is `DecodeError::Checksum(Mismatch { tile, chan, expected, got })` (`chan = None` for a
multi-component tile); the frame index is attached by the frame walker (`exv_video::BadBlob`, `VideoError::Decode`).
**GPU paths** verify on the CPU while planning the upload (exv-gpu `dispatch::items`, atlas `tile_jobs`: tiles in
parallel, one hash per sub-blob), so corrupt bytes never reach a kernel. `Verify::Off` is for hosts that already
verified a file (or a hot replay loop over verified bytes); `exv verify` always scans every blob
(`exv_video::verify_checksums`, hashing only, no decode) and lists the damaged tiles. Cost: ~22 GB/s/core (SSE4.2
3-way interleaved `crc32`, `crc32c` crate = software / other-arch fallback), measured <= ~2% of a 4096x2160x3 F16
decode upper bound (single-thread scan of the compressed bytes vs single-thread decode; noisy A/B within noise).
Not covered: the container prelude / header / frame headers / indexes (structural validation, §6.3 and the fuzz
harness, guard those).

---

## 6. `.exv` container (`exv-video::container`)

### 6.1 Byte map

```
prelude : MAGIC:u32 "EXV1" (45 58 56 31) | version:u8 = 3 | gop:u32 (>= 1) | frame_count:u32
          | Header (§2, codec baked in; header.frame_count == frame_count)
frames  : N frames in DECODE order, each = frame_hdr | frame_body (§5)
index   : n_frames:u32 | per frame (decode order): kind:u8, offset:u64, display_index:u32   (13 bytes each)
trailer : index_off:u64   (last 8 bytes of the file = byte offset of `index`)
```

```
frame_hdr : kind:u8 (0 Intra, 1 Pred, 2 Bidir) | n_refs:u8 | refs:[u32; n_refs] | mvs:[(i16 dx, i16 dy); n_refs]
```
`n_refs` is fixed by `kind` (Intra 0, Pred 1, Bidir 2); a mismatch is a hard error. `refs` are DECODE indices.

### 6.2 Semantics

- **GOP / closed GOP.** A GOP starts at an Intra frame and decodes standalone; every ref must be an EARLIER
  decode index of the SAME GOP (`gop_start <= ref < self`). `gop` is the writer's I-frame cadence in display
  order (a hint for readers; the real GOP boundaries are the Intra entries of the index).
- **P**: reference = the referenced frame motion-compensated by its global MV: `out[y][x] = prev[clamp(y-dy)][clamp(x-dx)]`
  (edge-clamped, whole-plane, every channel). `mv = (0,0)` is the reference itself.
- **B**: two refs (past + future in display order, each MC by its own MV), prediction = the reversible
  per-element integer bit-pattern mean `(a+b)>>1` (computed in `u64`); emitted after both anchors.
- Temporal prediction (PRED tiles, MV != 0, B) is **LOSSLESS-only**; a lossy profile stays I/P, MV = 0, INTRA/SKIP.
- **Tile-local temporal dependency (MV = 0).** With a zero MV, tile `(tx, ty)` of a P-frame depends ONLY on the co-located tile of
  `refs[0]`: INTRA ignores it, SKIP IS its bytes, PRED adds the residual onto it. A tile therefore resolves, through its SKIP chain, to one
  INTRA tile of an ancestor frame (`base`) plus the PRED residuals met on the way, so a decoder may decode just the visible tiles of a frame
  and a clip of INTRA + SKIP tiles needs no arithmetic at all (a lossy profile always is such a clip). A non-zero MV makes SKIP/PRED a
  SHIFTED copy that reads neighbouring reference tiles, and B-frames average two such references: those are not tile-local and a decoder
  must treat them as a distinct class (`exv_video::TemporalClass::{Motion, Bidir}`), never as "SKIP = co-located bytes".
- The MV is `i16` on the wire; the encoder clamps its search radius to `i16::MAX`.
- A still is the 1-frame case: `gop = 1`, `frame_count = 1`, one Intra frame, index of one entry (`display = 0`).

### 6.3 Parse rules (`VideoIndex::parse`; any violation -> `Err(DecodeError::Corrupt/Unsupported)`)

magic/version; `gop >= 1`; `header.frame_count == prelude frame_count == index n`; the header's frame byte size
is addressable (`frame_bytes()` does not overflow - the declared image size is NOT judged here, §8);
`exv-codec::registry::open(codec_id, codec_params)` succeeds (else `DecodeError::Codec(OpenError)`); trailer present and
`head_end <= index_off <= len-8`; the index fills **exactly** `[index_off, len-8)` (`4 + 13*n` bytes - this
proves `n` before anything is allocated from it and rejects trailing garbage); every `offset` lies in
`[head_end, index_off)` and offsets strictly increase (a frame ends where the next starts; the last ends at
`index_off`); `display_index < n` and the display indices form a permutation; the first decode frame is Intra;
`kind` is 0..=2. Frame headers (n_refs, refs) are validated when the frame is decoded.

---

## 7. Codec ids and `codec_params` (`exv-format::types::CodecId`, `exv-codec::registry::open`)

| Codec | Id | `codec_params` | Profile |
|-------|----|----------------|---------|
| `RAW` | 0 | - | **no profile**: a file with `codec_id = 0` does not open (raw exists only per blob, `TAG_RAW`) |
| `LOSSLESS_FAST` | 1 | empty | reorder+delta -> rANS |
| `WAVELET53` | 2 | empty | reversible 5/3 -> rANS |
| `WAVELET53_LOSSY` | 3 | `[q:u8]` | lossy 5/3 + detail quantization |
| `EXV_RT` | 4 | `[ver, block, p, flags]` | realtime LOSSY block codec (`TAG_EXV_RT`); `ver=0` Lerp: `block>=1`, `p=idx_bits in 1..=8`, `flags bit0 = log`; `ver=1` Dwt: `block in 1..=64`, `p=q < 16`, flags ignored |
| (5) | 5 | reserved | `TEMPORAL_DELTA` - comment only, no constant, not implemented |
| `HTJ2K` | 6 | `[]` Lossless, `[1, q]` HdrCodes (any `q`), `[2, q]` Irrev97 (`q` 1..=40) | ONE multi-component HTJ2K codestream per tile, every dtype incl. U8/U16 (§4.6; needs the `exv-codec/htj2k-backend` feature) |
| `BC6H` | 7 | `[0]` Lossy, `[1]` LosslessResidual | Hardware `BC6H_SF16` 4x4 block codec (`TAG_BC6H`) with optional lossless rANS residual |

Any other id (or malformed params) makes `registry::open` fail -> the file does not open, with a typed `OpenError`
(carried as `DecodeError::Codec`; the C-ABI maps it to `UNSUPPORTED`, `BadParams` to `INVALID_ARGUMENT`):

| `OpenError` | When |
|---|---|
| `NoProfile(id)` | `id = 0` (`RAW`: a per-blob tag, not a file codec) |
| `Unknown(id)` | spec-range id that defines no codec (e.g. the reserved `5`, `7+`) |
| `Vendor(id)` | `id >= 0x8000_0000` and no such backend is linked into this build |
| `NotBuilt{id,feature}` | defined spec id whose backend is not compiled in (`HTJ2K` without `htj2k-backend`) |
| `BadParams{id,why}` | backend found, `codec_params` violate its contract (also: params on a codec that takes none) |

`CodecId >= 0x8000_0000` is the vendor/experimental range (`is_vendor`, consulted by `open`); `0..=0x7FFF_FFFF` is spec.
HDR-float lossy profiles apply a log2 value-domain transform inside the codec (internal, reversible; not a
format field).

### 7.1 Decode contract (exactness / tolerance class per codec)

"Bit-exact" is a property of the profile, not a blanket promise of the format:

| Profile | Contract | Why |
|---|---|---|
| `LOSSLESS_FAST`, `WAVELET53`, HTJ2K `Lossless` | **EXACT** to the source, on every conforming decoder (CPU or GPU) | integer lifting / table math only |
| `WAVELET53_LOSSY`, HTJ2K `HdrCodes`, `EXV_RT` | **EXACT to the quantised stream**: every decoder yields identical bytes for the same file; the loss is the encoder's quantization; integer channels stay exact to the source | integer dequantization + deterministic hdr tables (§4.5) |
| HTJ2K `Irrev97` | **TOLERANCE +-1 LSB** between independent decoders (CPU SIMD tiers, a future GPU path); the same decoder is deterministic | float 9/7 reconstruction rounded to the integer sample; differences appear only at rounding boundaries. Encode is deterministic per (build, SIMD tier) |

Consequences: `exv verify` bit-compares lossless profiles only; GPU parity gates for a tolerance codec use
`maxdiff <= 1` (as jph-wgpu's own irreversible gate does). The 9/7 is acceptable for INTEGER LDR samples and is refused
for floats, where an f32 reconstruction followed by a log decode would amplify decoder differences.

### 7.2 Rate-distortion record of HTJ2K Irrev97 (`exv-codec/tests/htj2k_rd.rs`, `--ignored`)

512x512x3 synthetic tile, bytes at equal PSNR (dB, peak 255/65535), median over `grad`+`photo`+`ui`, U8 and U16.
`lossy_wave` and `htj2k_lossy` never quantize integer channels (§4.4), so on U8/U16 input they are lossless; the fair
comparison feeds the SAME image as normalised F32 to `WAVELET53_LOSSY` / HdrCodes. At 45 dB: `photo` Irrev97 68 KB vs
HdrCodes 195 KB vs lossy-wave 402 KB (lossless HTJ2K 350 KB); `ui` 87 KB vs 90 KB vs 395 KB; `grad` 15 KB vs 171 KB vs
407 KB. Median saving of Irrev97 over lossy-wave on `grad`+`photo` at equal PSNR: **83 %** (threshold for exposing the
flag was 25 %), so `--lossy-htj2k-ldr Q` exists in `exv-cli` and `ExvCodecMode::Htj2kIrreversible97` in the C ABI.
Caveat: this measures the only lossy competitors that accept the content (float-fed); it does NOT show 9/7 beating an
integer lossy wavelet, because the project has none by design.

---

## 8. Robustness / allocation limits

Parsers never panic and never allocate from an unchecked file value. The single source is
`exv-format::limits` (+ `Rd`, `Header::frame_bytes`, `DecodeError`):

- **Opening never guesses.** `Header::read` and `VideoIndex::parse` validate structure and self-consistency only.
  They do NOT reject a file for declaring a large image: a legitimate 20000x20000 all-zero image is a ~1.6 KB
  file, indistinguishable from a bomb by any compression-ratio heuristic (the former `check_alloc(bytes, input_len)`
  ratio guard rejected exactly such files and no longer exists).
- **Decode is budgeted by the caller.** `DecodeLimits { max_alloc }` (default `DEFAULT_MAX_ALLOC` = 8 GiB; hosts
  that know their machine pass their own) bounds what one decode call may hold at once:
  the decoded planes of a frame/region + one decoded tile (`FrameReader`, `read_frame_tiles`), the resident GOP
  (`decode_frame`, `Decoder`), a whole clip's output (`read_video_with`). Exceeding it is
  `DecodeError::Limit { need, max }` BEFORE any plane is allocated. Planes are allocated fallibly
  (`limits::alloc_zeroed`, `try_reserve_exact`), so a size that passed the budget but cannot be served is
  `DecodeError::Alloc`, never an abort.
- `limits::cap(n, remaining, min_entry)` - clamp a **capacity hint** read from the file to what the remaining input
  could hold (`Vec::with_capacity(cap(..))`); the parse loop still fails on short data.
- `Header::read`: zero dims, `MAX_TILE_DIM`, effective tile area `MAX_TILE_AREA`, plane-size overflow (§2.4).
- `VideoIndex::parse`: exact index length, strictly increasing in-range offsets, permutation of display indices (§6.3).
- `rans-rs`: `n <= MAX_DECODE_SYMBOLS (2^30)` and the caller's REQUIRED `max_n` (`decode_capped`; there is no
  cap-less decode) before the output is allocated (fallibly); lanes `>= 1`; table/state ranges; exact-end check per
  lane with immediate abort on an over-read (§4.2). `encode` refuses what the decoder would refuse (`n > 2^30`).
- Codec decoders trust the tile `w`/`h` they are handed (validated header + budget) and compare every stream-declared
  geometry to them (EXV-RT `nbx/nby`, HTJ2K full header check, §4.6) instead of sizing buffers from the stream. The
  public `decode_channel`/`decode_tile` fronts re-check the tile area against `MAX_TILE_AREA`
  (`DecodeError::Request`) so an unvalidated `w`/`h` is an error, never an allocation panic; a DEFLATE payload whose
  claimed output exceeds the 1032:1 DEFLATE expansion bound is corrupt and refused before any buffer is reserved.
- **The budget is the real peak, not the pixels.** `FrameReader` charges `planes + one tile + the codec's decode
  working set`: per tile pixel, per channel and per tile, as reported by the codec registry
  (`TileCodec::work` / `work_base`: FAST 1 plane, WAVE a plane + 4 B/px, lossy wave 18 B/px, EXV-RT 2 B/px per F16
  channel, HTJ2K 4 B/px + 320 KiB (<= 16-bit) / 640 KiB (32-bit) per channel + 2.5 MiB per tile), or 2 planes for tiles
  reconstructed from a reference, whichever is larger. The whole-frame row path charges no decoded tile for HTJ2K (its
  lines go straight into the planes).
  `exv-codec/tests/mem_budget.rs` measures the process peak (bufpool::track) against the limit the reader reports for
  every profile x dtype, plus a worst-case (incompressible, many-channel) HTJ2K sweep: a decoder that allocates
  more than it declares fails that test.
- **Host budgets.** `DecodeLimits::default()` (8 GiB) is for tests/tools. A host opening untrusted files passes
  `exv_tile::Budget::auto(over)?.limits()` = half the physical RAM capped by what is available (the one formula,
  `exv_tile::usable_ram`; it returns a typed `BudgetError` when the OS cannot answer and no override is given), so a
  3 KB file declaring a 32768-square image is `DecodeError::Limit` before any plane exists.
- A tile area of `MAX_TILE_AREA` pixels is up to `4 * 2^30` bytes per channel; only the budget above (and the rANS
  symbol ceiling, which makes encoders refuse an F32 tile over 2^28 px) bounds what is held - not the area cap alone.
- Every coded blob carries a CRC-32C in its frame index (§5.1): under `Verify::On` (default) a payload flip is
  `DecodeError::Checksum`, not wrong pixels. Structure (header, indexes, frame headers) is not checksummed; the
  structural checks above and the fuzz harness `exv-video/tests/hostile_files.rs` (whose decodes run with
  `Verify::Off`, so the codecs still see hostile payloads) guard it.
- `exv-codec::frame`: tile index validated against `tile_count` and the blob region (§5), BEFORE planes are allocated.

---

## 9. Stability / limitations

- Pre-stable: no layout guarantee before the GPU==CPU gate. No compatibility with earlier layouts is maintained (old files need not open).
- Not yet in the format: multipart / mip / deep (deep would be I-frames only), `TILE_BI`, block-based motion
  (only one global MV per ref), per-group codecs (one `codec_id` per file).
- Unknown header flag bits are rejected on read (pre-stable: a set bit means a different writer, never silently mis-parsed).
