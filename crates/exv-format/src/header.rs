//! EXV container header + tile geometry (format structure, no codecs).
//!
//! LE on the wire (see `docs/FORMAT.md` §2). `version = 0x01` — pre-stable. The header has NO magic of its own:
//! it is always embedded after the `EXV1` container prelude, which is the file signature. File assembly/parsing
//! (tiling + codec + index) lives in `exv-codec::frame` / `exv-video::container` (where the dependency
//! on the codec is allowed) — here we have only the structure and (de)serialization of the header.

use std::fmt;

use crate::error::DecodeError;
use crate::limits::cap;
use crate::rd::Rd;
use crate::types::Dtype;

/// Header layout version. `0x01` = pre-stable (layout not guaranteed until the GPU==CPU gate).
pub const VERSION: u8 = 1;
/// Max tile edge (px) accepted by `Header::read`. Decode is **per-tile** (out-of-core, peak `O(tile)`),
/// so bounding the TILE bounds every codec's per-tile allocation — the fix for the decompression-bomb /
/// alloc-abort DoS (a crafted header with a giant `tile_w`/`tile_h` → `vec![0u8; O(tile)]` → uncatchable
/// `handle_alloc_error`/`abort`). The IMAGE `width`/`height` are deliberately NOT capped here (65k² tiled
/// is a design goal); 65536 is far above any real tile (typ. 256²) yet caps one F16 tile at ≤8 GiB.
pub const MAX_TILE_DIM: u32 = 65536;
/// Max EFFECTIVE tile area (px), `min(tile_w, width) * min(tile_h, height)`. The edge cap alone allows a
/// 4 G-pixel tile (~17 GiB of F32 per channel); decode materializes whole tiles, so the area is what bounds
/// per-tile memory. A tile may exceed the image (one-tile-per-image files), hence the EFFECTIVE tile: a
/// 20000x20000 image with one 20000x20000 tile (4e8 px) is legitimate.
///
/// This caps PIXELS, not bytes: a tile is up to `MAX_TILE_AREA * 4` bytes per channel. 2^30 equals the rANS
/// `MAX_DECODE_SYMBOLS` ceiling (symbols = bytes), so it is codable by the entropy coder only for 1-byte
/// dtypes: an F32 tile above 2^28 px (F16/U16 above 2^29) passes this check but the rANS encoders refuse it
/// loudly. What a decode may actually hold is governed by the caller's `DecodeLimits` plus each codec's
/// per-pixel working set (`exv_codec` `TileCodec::work_bytes`), never by this constant alone.
pub const MAX_TILE_AREA: u64 = 1 << 30;

/// Byte offset of `frame_count` inside a serialized [`Header`] (`version(1) | flags(1) | width(4) |
/// height(4) | frame_count(4) ..`). A streaming writer that does not know the count up front writes `0` and
/// patches it here at `finish` (pinned by the `frame_count_offset` test, so layout drift fails a test, not a file).
pub const FRAME_COUNT_OFF: usize = 10;

/// The single geometry rule shared by [`Header::read`] and [`Header::write`] (a writer must never emit a
/// header its own reader rejects): non-zero dims, tile edges within [`MAX_TILE_DIM`], effective tile area
/// within [`MAX_TILE_AREA`]. The image `width`/`height` are deliberately NOT capped (65k² tiled is a goal).
fn check_geometry(width: u32, height: u32, tile_w: u32, tile_h: u32) -> Result<(), &'static str> {
    // size 0 (of a tile or plane) is degenerate (no pixels / division by zero / empty geometry).
    if width == 0 || height == 0 || tile_w == 0 || tile_h == 0 {
        return Err("zero image or tile dimension");
    }
    if tile_w > MAX_TILE_DIM || tile_h > MAX_TILE_DIM {
        return Err("tile edge exceeds MAX_TILE_DIM");
    }
    if (tile_w.min(width) as u64) * (tile_h.min(height) as u64) > MAX_TILE_AREA {
        return Err("tile area exceeds MAX_TILE_AREA");
    }
    Ok(())
}

/// Highest header `flags` bit this build understands (bit0 = Meta block). Pre-stable format: a set unknown bit
/// is a file from a different writer, so it is refused instead of being mis-parsed.
const KNOWN_FLAGS: u8 = 1;

/// Channel-list rules shared by [`Header::read`] and [`Header::write`] (a writer never emits a header its reader
/// rejects): at least one channel, names unique (a duplicate makes by-name lookup ambiguous).
fn check_channels(channels: &[Channel]) -> Result<(), &'static str> {
    if channels.is_empty() {
        return Err("header has no channels");
    }
    let mut seen = std::collections::HashSet::with_capacity(channels.len());
    if channels.iter().any(|c| !seen.insert(c.name.as_str())) {
        return Err("duplicate channel name");
    }
    Ok(())
}

/// Tile-grid geometry (no codecs/data) — shared by the container and the tile engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGrid {
    pub width: u32,
    pub height: u32,
    pub tile_w: u32,
    pub tile_h: u32,
}

impl TileGrid {
    pub fn tiles_x(&self) -> u32 {
        self.width.div_ceil(self.tile_w.max(1))
    }
    pub fn tiles_y(&self) -> u32 {
        self.height.div_ceil(self.tile_h.max(1))
    }
    /// `u64` — `tiles_x*tiles_y` overflows u32 (e.g. 65536² at tile=1).
    pub fn tile_count(&self) -> u64 {
        self.tiles_x() as u64 * self.tiles_y() as u64
    }
    /// Tile rectangle `(x, y, w, h)` clamped to the edge (saturating — no underflow on out-of-range).
    pub fn tile_rect(&self, tx: u32, ty: u32) -> (u32, u32, u32, u32) {
        let x = tx.saturating_mul(self.tile_w);
        let y = ty.saturating_mul(self.tile_h);
        (x, y, self.tile_w.min(self.width.saturating_sub(x)), self.tile_h.min(self.height.saturating_sub(y)))
    }
}

/// Channel descriptor: name (hierarchy `layer.sub.chan`) + element type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Channel {
    pub name: String,
    pub dtype: Dtype,
}

/// Header of a single part (v1: one part, one frame).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    /// Frames in the stream. `1` for still (the container is video-aware from day one).
    pub frame_count: u32,
    pub tile_w: u32,
    pub tile_h: u32,
    /// Codec identifier (v1: per file; per-group — later).
    pub codec_id: u32,
    /// Codec parameters (StageDesc params; e.g. `[q]` for lossy). Empty for lossless.
    pub codec_params: Vec<u8>,
    pub channels: Vec<Channel>,
}

/// A [`Header`] field that does not fit its on-disk width (see [`Header::write`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// More than 65535 channels (count is a `u16`).
    Channels(usize),
    /// A channel name longer than 255 bytes (length is a `u8`).
    Name(String),
    /// `codec_params` longer than 65535 bytes (length is a `u16`).
    Params(usize),
    /// Image/tile geometry the reader would reject (zero dim, tile edge/area over its cap).
    Geometry(&'static str),
    /// A channel list / meta value the reader would reject (no channels, duplicate name, zero denominator).
    Invalid(&'static str),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::Channels(n) => write!(f, "{n} channels exceed the u16 header limit (65535)"),
            WriteError::Name(n) => write!(f, "channel name `{n}` exceeds 255 bytes"),
            WriteError::Params(n) => write!(f, "{n} codec_params bytes exceed the u16 header limit (65535)"),
            WriteError::Geometry(m) => write!(f, "unreadable geometry: {m}"),
            WriteError::Invalid(m) => write!(f, "unreadable header: {m}"),
        }
    }
}

impl std::error::Error for WriteError {}

/// Optional, forward-extensible clip metadata — makes a `.exv` SELF-DESCRIBING (fps, color,
/// alpha, timecode) beyond raw pixels. Stored as a flags-gated TLV block appended to the header: OLD
/// files (written with `flags=0`) parse BYTE-IDENTICALLY, and unknown future tags skip by length. It is
/// NOT a `Header` struct field (that would break every `Header { .. }` literal across the workspace) —
/// it rides alongside via [`Header::write_with_meta`]/[`Header::read_with_meta`]. All fields optional /
/// `0 = unspecified`. These describe the AUTHORED DISPLAY intent (the codec's log2 HDR transform is
/// internal + reversible, so decoded planes are already linear — this is color/transfer for viewing).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Meta {
    /// Playback rate as an exact rational `(num, den)` (e.g. 24000/1001). `None` = unspecified.
    pub fps: Option<(u32, u32)>,
    /// Color primaries / transfer / range as small host-defined enums (`0 = unspecified`).
    pub primaries: u8,
    pub transfer: u8,
    pub range: u8,
    /// Alpha: `0 = unspecified`, `1 = straight/unassociated`, `2 = premultiplied/associated`.
    pub alpha: u8,
    /// Pixel aspect ratio `(num, den)`. `None` = square (1:1).
    pub pixel_aspect: Option<(u32, u32)>,
    /// SMPTE timecode packed as frames since `00:00:00:00`. `None` = unspecified.
    pub timecode: Option<u32>,
    /// Canonical XDR wire serialization (`exr_core::Header::write_to`) of the original OpenEXR part headers.
    /// Preserves 100% of OpenEXR attributes, custom metadata, Cryptomatte manifests, Multipart, Multiview,
    /// Deep, and Tile/Level descriptions across `.exr <-> .exv` round-trips.
    pub exr_headers: Vec<Vec<u8>>,
}

/// Append one TLV (`len <= u16::MAX`).
fn put_tlv(o: &mut Vec<u8>, tag: u8, val: &[u8]) {
    debug_assert!(val.len() <= u16::MAX as usize);
    o.push(tag);
    o.extend_from_slice(&(val.len() as u16).to_le_bytes());
    o.extend_from_slice(val);
}

impl Meta {
    /// `true` when nothing is set ⇒ no block is written and the header stays byte-identical to the old form.
    pub fn is_empty(&self) -> bool {
        *self == Meta::default()
    }

    /// Serialize and store one or more native [`exr_core::header::Header`]s into [`Self::exr_headers`].
    pub fn set_exr_headers(&mut self, headers: &[exr_core::header::Header]) -> exr_core::Result<()> {
        let base_flags = if headers.len() > 1 { exr_core::container::version::MULTI_PART_FILE_FLAG } else { 0 };
        let mut out = Vec::with_capacity(headers.len());
        for h in headers {
            let mut stream = exr_core::io::MemOStream::new();
            h.write_to(&mut stream, base_flags)?;
            out.push(stream.into_data());
        }
        self.exr_headers = out;
        Ok(())
    }

    /// Reconstruct the native [`exr_core::header::Header`] list stored in [`Self::exr_headers`].
    pub fn decode_exr_headers(&self) -> exr_core::Result<Vec<exr_core::header::Header>> {
        let mut out = Vec::with_capacity(self.exr_headers.len());
        for raw in &self.exr_headers {
            let mut stream = exr_core::io::MemIStream::with_name(raw.as_slice(), "<exv-meta>");
            let (h, _) = exr_core::header::Header::read_from(&mut stream)?;
            out.push(h);
        }
        Ok(out)
    }

    /// Encode the set fields as a skippable TLV blob `[tag:u8][len:u16][val]…`.
    fn encode(&self) -> Vec<u8> {
        let mut o = Vec::new();
        if let Some((n, d)) = self.fps {
            put_tlv(&mut o, 1, &[n.to_le_bytes(), d.to_le_bytes()].concat());
        }
        if self.primaries | self.transfer | self.range != 0 {
            put_tlv(&mut o, 2, &[self.primaries, self.transfer, self.range]);
        }
        if self.alpha != 0 {
            put_tlv(&mut o, 3, &[self.alpha]);
        }
        if let Some((n, d)) = self.pixel_aspect {
            put_tlv(&mut o, 4, &[n.to_le_bytes(), d.to_le_bytes()].concat());
        }
        if let Some(tc) = self.timecode {
            put_tlv(&mut o, 5, &tc.to_le_bytes());
        }
        if !self.exr_headers.is_empty() {
            let mut raw = Vec::new();
            raw.extend_from_slice(&(self.exr_headers.len() as u32).to_le_bytes());
            for h in &self.exr_headers {
                raw.extend_from_slice(&(h.len() as u32).to_le_bytes());
                raw.extend_from_slice(h);
            }
            for chunk in raw.chunks(u16::MAX as usize) {
                put_tlv(&mut o, 6, chunk);
            }
        }
        o
    }
    /// The one value rule shared by `write_with_meta` and `read_with_meta`: a rational with `den == 0`
    /// (fps, pixel aspect) is meaningless (division by zero in every consumer), so it is refused on both sides.
    fn check(&self) -> Result<(), &'static str> {
        if matches!(self.fps, Some((_, 0))) {
            return Err("meta fps denominator is zero");
        }
        if matches!(self.pixel_aspect, Some((_, 0))) {
            return Err("meta pixel aspect denominator is zero");
        }
        if self.exr_headers.iter().any(|h| h.is_empty() || u32::try_from(h.len()).is_err()) {
            return Err("meta exr_headers entry is empty or exceeds u32");
        }
        if u32::try_from(self.exr_headers.len()).is_err() {
            return Err("meta exr_headers count exceeds u32");
        }
        Ok(())
    }
    /// Decode a TLV blob; unknown tags / wrong lengths are skipped (forward-compatible), but a decoded value
    /// that violates [`Self::check`] is a hard error (a hostile file must not smuggle `den = 0` to the host).
    fn decode(tlv: &[u8]) -> Result<Meta, &'static str> {
        let mut m = Meta::default();
        let mut exr_raw = Vec::new();
        let mut r = Rd::new(tlv);
        while let (Some(tag), Some(len)) = (r.u8(), r.u16()) {
            let Some(v) = r.take(len as usize) else { break };
            let mut f = Rd::new(v);
            match (tag, len) {
                (1, 8) => m.fps = f.u32().zip(f.u32()),
                (2, 3) => {
                    if let (Some(p), Some(t), Some(r)) = (f.u8(), f.u8(), f.u8()) {
                        (m.primaries, m.transfer, m.range) = (p, t, r);
                    }
                }
                (3, 1) => m.alpha = f.u8().unwrap_or(0),
                (4, 8) => m.pixel_aspect = f.u32().zip(f.u32()),
                (5, 4) => m.timecode = f.u32(),
                (6, _) => exr_raw.extend_from_slice(v),
                _ => {} // unknown tag / bad length — skip
            }
        }
        if !exr_raw.is_empty() {
            let mut er = Rd::new(&exr_raw);
            let count = er.u32().ok_or("corrupt meta exr_headers")? as usize;
            if count == 0 {
                return Err("corrupt meta exr_headers");
            }
            let mut headers = Vec::with_capacity(cap(count, er.left(), 4));
            for _ in 0..count {
                let len = er.u32().ok_or("corrupt meta exr_headers")? as usize;
                if len == 0 {
                    return Err("corrupt meta exr_headers");
                }
                let bytes = er.take(len).ok_or("corrupt meta exr_headers")?;
                headers.push(bytes.to_vec());
            }
            if er.left() != 0 {
                return Err("corrupt meta exr_headers");
            }
            m.exr_headers = headers;
        }
        m.check()?;
        Ok(m)
    }
}

impl Header {
    /// Tile geometry of this header.
    pub fn grid(&self) -> TileGrid {
        TileGrid { width: self.width, height: self.height, tile_w: self.tile_w, tile_h: self.tile_h }
    }

    /// Number of tiles along X/Y (including partial edge tiles) — delegates to [`TileGrid`] (one source).
    pub fn tiles_x(&self) -> u32 {
        self.grid().tiles_x()
    }
    pub fn tiles_y(&self) -> u32 {
        self.grid().tiles_y()
    }
    /// `u64` — `tiles_x*tiles_y` overflows u32 (e.g. 65536² at tile=1).
    pub fn tile_count(&self) -> u64 {
        self.grid().tile_count()
    }

    /// Tile rectangle `(x, y, w, h)` clamped (saturating — no underflow on out-of-range tx/ty).
    pub fn tile_rect(&self, tx: u32, ty: u32) -> (u32, u32, u32, u32) {
        self.grid().tile_rect(tx, ty)
    }

    /// Full byte length of a channel plane (checked: `None` on usize overflow).
    ///
    /// Why: untrusted `width*height*esz` is multiplied in usize further down the stack
    /// (video/image/codec). On a hostile header (`width=height=u32::MAX`) this overflows usize → silent
    /// wrap in release / panic in debug, after which the size guards let a buffer of the wrong length
    /// through and lead to OOB. We compute in u64 and narrow to usize via `try_from` — this is the single source
    /// of truth about the bound, so the format checks it once instead of every consumer.
    /// Legitimately large sizes (65536²) are NOT cut — only a real overflow is rejected.
    pub fn plane_len(&self, dtype: Dtype) -> Option<usize> {
        let bytes = (self.width as u64)
            .checked_mul(self.height as u64)?
            .checked_mul(dtype.size() as u64)?;
        usize::try_from(bytes).ok()
    }

    /// Maximum plane length across channels (checked). `None` on usize overflow.
    /// Empty channel list → `Some(0)`. Handy for precomputing the decoder's worst-case buffer.
    pub fn max_plane_len(&self) -> Option<usize> {
        let mut max = 0usize;
        for ch in &self.channels {
            max = max.max(self.plane_len(ch.dtype)?);
        }
        Some(max)
    }

    /// Total byte length of ONE frame (sum of every channel plane), checked. `None` on overflow.
    /// The single input for decode-allocation budgets: a decoder materializes whole-frame planes, so this is the
    /// figure compared against the caller's `DecodeLimits::max_alloc` (never against the input size: no ratio guess).
    pub fn frame_bytes(&self) -> Option<u64> {
        self.channels.iter().try_fold(0u64, |a, c| a.checked_add(self.plane_len(c.dtype)? as u64))
    }

    /// Serialize the header into `out` (LE).
    ///
    /// Fails (appending nothing) when a field does not fit its on-disk width — more than 65535
    /// channels, a channel name over 255 bytes, `codec_params` over 65535 bytes. Silent truncation here
    /// would emit a file this crate cannot read back, so it is a hard error in every build profile.
    pub fn write(&self, out: &mut Vec<u8>) -> Result<(), WriteError> {
        self.write_inner(0, &[], out)
    }

    /// Serialize with optional [`Meta`] appended as a flags-gated TLV block. Empty meta → byte-identical
    /// to [`Self::write`] (`flags=0`), so meta is purely additive.
    pub fn write_with_meta(&self, meta: &Meta, out: &mut Vec<u8>) -> Result<(), WriteError> {
        meta.check().map_err(WriteError::Invalid)?;
        let tlv = meta.encode();
        if tlv.is_empty() { self.write_inner(0, &[], out) } else { self.write_inner(1, &tlv, out) }
    }

    fn write_inner(&self, flags: u8, meta_tlv: &[u8], out: &mut Vec<u8>) -> Result<(), WriteError> {
        // Validate everything first so a failure leaves `out` untouched.
        check_geometry(self.width, self.height, self.tile_w, self.tile_h).map_err(WriteError::Geometry)?;
        let nch = u16::try_from(self.channels.len()).map_err(|_| WriteError::Channels(self.channels.len()))?;
        let cp_len = u16::try_from(self.codec_params.len()).map_err(|_| WriteError::Params(self.codec_params.len()))?;
        for c in &self.channels {
            if c.name.len() > u8::MAX as usize {
                return Err(WriteError::Name(c.name.clone()));
            }
        }
        check_channels(&self.channels).map_err(WriteError::Invalid)?;
        // `frame_count == 0` is deliberately NOT checked here: a streaming writer emits 0 and patches the real
        // count at `finish` (see `FRAME_COUNT_OFF`); only the READER refuses 0, so an unfinished clip never opens.
        out.push(VERSION);
        out.push(flags); // bit0 = a Meta TLV block follows the channels
        for v in [self.width, self.height, self.frame_count, self.tile_w, self.tile_h, self.codec_id] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&cp_len.to_le_bytes());
        out.extend_from_slice(&self.codec_params);
        out.extend_from_slice(&nch.to_le_bytes());
        for ch in &self.channels {
            out.push(ch.name.len() as u8); // <= 255, validated above
            out.extend_from_slice(ch.name.as_bytes());
            out.push(ch.dtype.code());
        }
        if flags & 1 != 0 {
            out.extend_from_slice(&(meta_tlv.len() as u32).to_le_bytes());
            out.extend_from_slice(meta_tlv);
        }
        Ok(())
    }

    /// Parse the header. Returns `(Header, consumed_bytes)`, or the typed reason it was refused
    /// ([`DecodeError::Corrupt`] names the failed rule; [`DecodeError::Unsupported`] a foreign header version).
    /// Skips any [`Meta`] block; use [`Self::read_with_meta`] to also get the metadata.
    pub fn read(data: &[u8]) -> Result<(Header, usize), DecodeError> {
        Self::read_with_meta(data).map(|(h, _m, consumed)| (h, consumed))
    }

    /// Parse the header + its optional [`Meta`] block. Returns `(Header, meta, consumed)`;
    /// `meta` is `None` when no block is present (flags bit0 clear — every pre-meta `.exv`).
    pub fn read_with_meta(data: &[u8]) -> Result<(Header, Option<Meta>, usize), DecodeError> {
        // Every short read is the same cause: the input ends inside the header.
        fn t<T>(v: Option<T>) -> Result<T, DecodeError> {
            v.ok_or(DecodeError::Corrupt("header truncated"))
        }
        let mut c = Rd::new(data);
        if t(c.u8())? != VERSION {
            return Err(DecodeError::Unsupported("header version differs from this build"));
        }
        let flags = t(c.u8())?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(DecodeError::Corrupt("unknown header flag bits"));
        }
        let width = t(c.u32())?;
        let height = t(c.u32())?;
        let frame_count = t(c.u32())?;
        let tile_w = t(c.u32())?;
        let tile_h = t(c.u32())?;
        // A zero-frame stream is an unfinished/forged file (streaming writers patch the real count at finish).
        if frame_count == 0 {
            return Err(DecodeError::Corrupt("frame_count is zero"));
        }
        // Geometry caps bound every codec's per-tile allocation (decompression-bomb DoS guard): a tiny
        // hostile blob can otherwise declare a 262144² tile → ~137 GiB alloc → abort (uncatchable).
        check_geometry(width, height, tile_w, tile_h).map_err(DecodeError::Corrupt)?;
        let codec_id = t(c.u32())?;
        let cp_len = t(c.u16())? as usize;
        let codec_params = t(c.take(cp_len))?.to_vec();
        let nch = t(c.u16())? as usize;
        // Each channel needs >= 3 bytes (len, dtype + name) so a forged `nch` cannot over-reserve.
        let mut channels = Vec::with_capacity(cap(nch, c.left(), 2));
        for _ in 0..nch {
            let nlen = t(c.u8())? as usize;
            let nbytes = t(c.take(nlen))?;
            let name = String::from_utf8(nbytes.to_vec()).map_err(|_| DecodeError::Corrupt("channel name is not UTF-8"))?;
            let dtype = Dtype::from_code(t(c.u8())?).ok_or(DecodeError::Corrupt("unknown dtype code"))?;
            channels.push(Channel { name, dtype });
        }
        check_channels(&channels).map_err(DecodeError::Corrupt)?;
        let header = Header { width, height, frame_count, tile_w, tile_h, codec_id, codec_params, channels };
        // Bound at the source: the full plane of each channel (width*height*esz) must fit
        // in usize. Otherwise a downstream multiplication overflows (see plane_len) → OOB/panic. We reject
        // a hostile header here ONCE — all consumers of Header::read inherit the invariant.
        for ch in &header.channels {
            header.plane_len(ch.dtype).ok_or(DecodeError::Corrupt("plane size overflows"))?;
        }
        let meta = if flags & 1 != 0 {
            let meta_len = t(c.u32())? as usize;
            Some(Meta::decode(t(c.take(meta_len))?).map_err(DecodeError::Corrupt)?)
        } else {
            None
        };
        Ok((header, meta, c.pos()))
    }

    /// Build an EXV [`Header`] + [`Meta`] from one or more native [`exr_core::header::Header`]s, embedding the full
    /// OpenEXR part headers into [`Meta::exr_headers`] so 100% of OpenEXR attributes, custom metadata,
    /// Cryptomatte manifests, Multipart, Multiview, Deep, and Tile/Level descriptions round-trip losslessly.
    pub fn from_exr_headers(
        exr_headers: &[exr_core::header::Header],
        frame_count: u32,
        default_tile: (u32, u32),
        codec_id: u32,
        codec_params: Vec<u8>,
    ) -> exr_core::Result<(Header, Meta)> {
        let first = exr_headers
            .first()
            .ok_or_else(|| exr_core::ExrError::Arg("exr_headers must not be empty".into()))?;
        let dw = first.data_window()?;
        let width = u32::try_from(i64::from(dw.max.x) - i64::from(dw.min.x) + 1)
            .map_err(|_| exr_core::ExrError::Arg("invalid dataWindow width".into()))?;
        let height = u32::try_from(i64::from(dw.max.y) - i64::from(dw.min.y) + 1)
            .map_err(|_| exr_core::ExrError::Arg("invalid dataWindow height".into()))?;
        let (tile_w, tile_h) = if first.has_tile_description() {
            let td = first.tile_description()?;
            (td.x_size, td.y_size)
        } else {
            default_tile
        };
        let mut channels = Vec::new();
        for (part_idx, h) in exr_headers.iter().enumerate() {
            let part_prefix = if exr_headers.len() > 1 {
                h.name().ok().map(|s| s.to_owned()).unwrap_or_else(|| format!("part{part_idx}"))
            } else {
                String::new()
            };
            for (name, ch) in h.channels()?.iter() {
                let dtype = match ch.type_ {
                    exr_core::attr::PixelType::Half => Dtype::F16,
                    exr_core::attr::PixelType::Float => Dtype::F32,
                    exr_core::attr::PixelType::Uint => Dtype::U32,
                    _ => return Err(exr_core::ExrError::Arg("invalid EXR channel pixel type".into())),
                };
                let ch_name = name.to_str().unwrap_or("chan");
                let full_name = if part_prefix.is_empty() {
                    ch_name.to_owned()
                } else {
                    format!("{part_prefix}.{ch_name}")
                };
                channels.push(Channel { name: full_name, dtype });
            }
        }
        let mut meta = Meta::default();
        if let Ok(fps) = first.frames_per_second()
            && fps.d != 0
            && fps.n >= 0
        {
            meta.fps = Some((fps.n as u32, fps.d));
        }
        meta.set_exr_headers(exr_headers)?;
        let header = Header {
            width,
            height,
            frame_count,
            tile_w,
            tile_h,
            codec_id,
            codec_params,
            channels,
        };
        Ok((header, meta))
    }

    /// Reconstruct the native [`exr_core::header::Header`]s from `meta` (when `meta.exr_headers` is populated) or
    /// synthesize a valid single-part [`exr_core::header::Header`] matching this EXV [`Header`].
    pub fn to_exr_headers(&self, meta: Option<&Meta>) -> exr_core::Result<Vec<exr_core::header::Header>> {
        if let Some(m) = meta
            && !m.exr_headers.is_empty()
        {
            return m.decode_exr_headers();
        }
        let w = i32::try_from(self.width).map_err(|_| exr_core::ExrError::Arg("width exceeds i32".into()))?;
        let h = i32::try_from(self.height).map_err(|_| exr_core::ExrError::Arg("height exceeds i32".into()))?;
        let par = meta
            .and_then(|m| m.pixel_aspect)
            .filter(|&(_, d)| d > 0)
            .map(|(n, d)| n as f32 / d as f32)
            .unwrap_or(1.0);
        let mut exr = exr_core::header::Header::with_dimensions(
            w,
            h,
            par,
            imath_rs::V2f::new(0.0, 0.0),
            1.0,
            exr_core::attr::LineOrder::IncreasingY,
            exr_core::attr::Compression::Zip,
        )?;
        for ch in &self.channels {
            let pt = match ch.dtype {
                Dtype::F16 | Dtype::Bf16 | Dtype::U16 | Dtype::U8 => exr_core::attr::PixelType::Half,
                Dtype::F32 => exr_core::attr::PixelType::Float,
                Dtype::U32 => exr_core::attr::PixelType::Uint,
            };
            exr.channels_mut()?.insert(&ch.name, exr_core::attr::Channel::new(pt, 1, 1, false))?;
        }
        if let Some(m) = meta
            && let Some((n, d)) = m.fps
            && let Ok(ni) = i32::try_from(n)
        {
            exr.add_frames_per_second(exr_core::attr::Rational::new(ni, d))?;
        }
        Ok(vec![exr])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exr_headers_roundtrip_in_meta() {
        let mut exr = exr_core::header::Header::with_dimensions(
            1920,
            1080,
            1.0,
            imath_rs::V2f::new(0.0, 0.0),
            1.0,
            exr_core::attr::LineOrder::IncreasingY,
            exr_core::attr::Compression::Piz,
        )
        .expect("exr header");
        exr.channels_mut()
            .expect("ch")
            .insert("R", exr_core::attr::Channel::new(exr_core::attr::PixelType::Half, 1, 1, false))
            .expect("insert R");
        exr.channels_mut()
            .expect("ch")
            .insert("crypto_asset00.r", exr_core::attr::Channel::new(exr_core::attr::PixelType::Float, 1, 1, false))
            .expect("insert crypto");
        exr.insert(
            "cryptomatte/abcdef0/manifest",
            exr_core::attr::StringAttribute::new("{\"hero\":\"12345678\"}".into()),
        )
        .expect("insert manifest");
        exr.add_frames_per_second(exr_core::attr::Rational::new(24000, 1001))
            .expect("fps");

        let (hdr, meta) = Header::from_exr_headers(&[exr], 48, (256, 256), 7, vec![1]).expect("from_exr_headers");
        assert_eq!(hdr.width, 1920);
        assert_eq!(hdr.height, 1080);
        assert_eq!(hdr.frame_count, 48);
        assert_eq!(meta.fps, Some((24000, 1001)));

        let mut bytes = Vec::new();
        hdr.write_with_meta(&meta, &mut bytes).expect("write");
        let (hdr2, meta2, consumed) = Header::read_with_meta(&bytes).expect("read");
        assert_eq!(consumed, bytes.len());
        assert_eq!(hdr2, hdr);
        let restored = hdr2.to_exr_headers(meta2.as_ref()).expect("to_exr_headers");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].compression().expect("comp"), exr_core::attr::Compression::Piz);
        let manifest = restored[0]
            .typed_attribute::<exr_core::attr::ExrString>("cryptomatte/abcdef0/manifest")
            .expect("manifest attr");
        assert_eq!(manifest.value.as_str().expect("utf8"), "{\"hero\":\"12345678\"}");
    }

    /// `FRAME_COUNT_OFF` really addresses `frame_count` (sentinel value, with and without a meta block).
    #[test]
    fn frame_count_offset() {
        let h = Header {
            width: 8,
            height: 4,
            frame_count: 0xA1B2_C3D4,
            tile_w: 8,
            tile_h: 8,
            codec_id: 1,
            codec_params: vec![],
            channels: vec![Channel { name: "R".into(), dtype: Dtype::F16 }],
        };
        for meta in [Meta::default(), Meta { fps: Some((24, 1)), ..Meta::default() }] {
            let mut b = Vec::new();
            h.write_with_meta(&meta, &mut b).expect("write");
            assert_eq!(b[FRAME_COUNT_OFF..FRAME_COUNT_OFF + 4], 0xA1B2_C3D4u32.to_le_bytes());
        }
    }

    /// `Meta` round-trips; empty meta is byte-identical to a plain header (no flag), and a
    /// plain `read` still parses a meta-carrying header (skips the block, same `consumed`).
    #[test]
    fn meta_roundtrip_and_backcompat() {
        let h = Header {
            width: 8,
            height: 4,
            frame_count: 1,
            tile_w: 8,
            tile_h: 8,
            codec_id: 1,
            codec_params: vec![],
            channels: vec![Channel { name: "R".into(), dtype: Dtype::F16 }],
        };
        // No meta → write == write_with_meta(empty) byte-for-byte; read_with_meta gives None.
        let mut plain = Vec::new();
        h.write(&mut plain).expect("write");
        let mut empty = Vec::new();
        h.write_with_meta(&Meta::default(), &mut empty).expect("write");
        assert_eq!(empty, plain, "empty meta is byte-identical (no flag)");
        assert_eq!(Header::read_with_meta(&plain).unwrap(), (h.clone(), None, plain.len()));
        // With meta → round-trips; plain `read` still parses (skips the block) with the same consumed.
        let meta = Meta {
            fps: Some((24000, 1001)),
            primaries: 1,
            transfer: 3,
            range: 0,
            alpha: 2,
            pixel_aspect: Some((2, 1)),
            timecode: Some(90),
            exr_headers: Vec::new(),
        };
        let mut buf = Vec::new();
        h.write_with_meta(&meta, &mut buf).expect("write");
        assert!(buf.len() > plain.len(), "meta block adds bytes");
        assert_eq!(Header::read_with_meta(&buf).unwrap(), (h.clone(), Some(meta), buf.len()));
        assert_eq!(Header::read(&buf).unwrap(), (h, buf.len()), "plain read skips the meta block");
    }

    #[test]
    fn header_roundtrip() {
        let h = Header {
            width: 1920,
            height: 1080,
            frame_count: 1,
            tile_w: 256,
            tile_h: 256,
            codec_id: 3,
            codec_params: vec![5u8], // e.g. q=5 for lossy — verify params roundtrip
            channels: vec![
                Channel { name: "R".into(), dtype: Dtype::F16 },
                Channel { name: "beauty.Z".into(), dtype: Dtype::F32 },
                Channel { name: "id".into(), dtype: Dtype::U32 },
            ],
        };
        let mut buf = Vec::new();
        h.write(&mut buf).expect("write");
        let (h2, consumed) = Header::read(&buf).expect("parse");
        assert_eq!(h, h2);
        assert_eq!(consumed, buf.len());
        assert_eq!(h.tiles_x(), 8); // ceil(1920/256)
        assert_eq!(h.tiles_y(), 5); // ceil(1080/256)
        assert_eq!(h.tile_rect(7, 4), (1792, 1024, 128, 56)); // edge tile
    }

    #[test]
    fn header_rejects_garbage() {
        assert!(Header::read(&[]).is_err());
        assert!(Header::read(&[0, 1, 2, 3, 4]).is_err());
    }

    /// The refusal carries its cause: truncation, a foreign version, a geometry rule, a bad dtype / name.
    #[test]
    fn read_errors_name_the_cause() {
        let ok = forged(8, 8, 8, 8, Dtype::F16);
        assert_eq!(Header::read(&[]), Err(DecodeError::Corrupt("header truncated")));
        assert_eq!(Header::read(&ok[..ok.len() - 1]), Err(DecodeError::Corrupt("header truncated")));
        let mut bad = ok.clone();
        bad[0] = VERSION + 1;
        assert!(matches!(Header::read(&bad), Err(DecodeError::Unsupported(_))));
        assert_eq!(Header::read(&forged(0, 8, 8, 8, Dtype::F16)), Err(DecodeError::Corrupt("zero image or tile dimension")));
        let mut bad = ok.clone();
        *bad.last_mut().expect("dtype byte") = 0xEE; // the channel's dtype code is the last byte (no meta)
        assert_eq!(Header::read(&bad), Err(DecodeError::Corrupt("unknown dtype code")));
        let mut bad = ok;
        let name_at = bad.len() - 2; // 1-byte name "R"
        bad[name_at] = 0xFF;
        assert_eq!(Header::read(&bad), Err(DecodeError::Corrupt("channel name is not UTF-8")));
    }

    /// Wire offsets of the geometry fields (`version1 + flags1` precede them).
    const AT_W: usize = 2;
    const AT_H: usize = 6;
    const AT_TW: usize = 14;
    const AT_TH: usize = 18;

    /// A valid 1-channel header whose geometry fields are patched IN THE BYTES: `write()` itself refuses
    /// unreadable geometry, so a hostile file can only be forged below the API.
    fn forged(width: u32, height: u32, tile_w: u32, tile_h: u32, dtype: Dtype) -> Vec<u8> {
        let h = Header { channels: vec![Channel { name: "R".into(), dtype }], ..base() };
        let mut buf = Vec::new();
        h.write(&mut buf).expect("write");
        for (at, v) in [(AT_W, width), (AT_H, height), (AT_TW, tile_w), (AT_TH, tile_h)] {
            buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn header_rejects_zero_dims() {
        // width==0 / height==0 — degenerate, reject (CLASS-A: consistent with the tile guard).
        assert!(Header::read(&forged(0, 1080, 64, 64, Dtype::F32)).is_err());
        assert!(Header::read(&forged(1920, 0, 64, 64, Dtype::F32)).is_err());
        assert!(Header::read(&forged(0, 0, 64, 64, Dtype::F32)).is_err());
    }

    #[test]
    fn header_rejects_plane_overflow() {
        // w=h=u32::MAX with F32 (esz=4): width*height*esz overflows usize → corrupt header → None.
        assert!(Header::read(&forged(u32::MAX, u32::MAX, 64, 64, Dtype::F32)).is_err());
    }

    #[test]
    fn header_rejects_oversized_tile() {
        // Decompression-bomb guard: a tile edge beyond MAX_TILE_DIM is rejected here, so per-tile decode
        // can never be handed a geometry that allocates `O(tile)` → abort. Reviewer PoC: 262144² (≈137 GiB).
        let mk = |tw: u32, th: u32| forged(1920, 1080, tw, th, Dtype::F16);
        assert!(Header::read(&mk(262144, 262144)).is_err()); // the PoC
        assert!(Header::read(&mk(MAX_TILE_DIM + 1, 256)).is_err());
        assert!(Header::read(&mk(256, MAX_TILE_DIM + 1)).is_err());
        // Boundary: a tile may exceed the image (effective tile = the image here).
        assert!(Header::read(&mk(MAX_TILE_DIM, MAX_TILE_DIM)).is_ok());
    }

    /// The tile AREA cap applies to the EFFECTIVE tile `min(tile, image)`: huge tiles on a big image are
    /// rejected even when each edge is within `MAX_TILE_DIM`; the same tile on a small image is fine; a
    /// legitimate 20000² single-tile image (4e8 px) is accepted.
    #[test]
    fn header_caps_effective_tile_area() {
        let side = (MAX_TILE_AREA as f64).sqrt() as u32; // 32768: exactly at the cap when squared
        assert!(Header::read(&forged(65536, 65536, side, side, Dtype::U8)).is_ok(), "area == cap");
        assert!(Header::read(&forged(65536, 65536, side + 1, side, Dtype::U8)).is_err(), "area > cap");
        assert!(Header::read(&forged(65536, 65536, 65536, 65536, Dtype::U8)).is_err(), "4 Gpx tile");
        assert!(Header::read(&forged(1920, 1080, 65536, 65536, Dtype::F32)).is_ok(), "tile > image is clamped");
        assert!(Header::read(&forged(20000, 20000, 20000, 20000, Dtype::U8)).is_ok(), "legit one-tile image");
    }

    /// A writer never emits a header its own reader rejects.
    #[test]
    fn write_rejects_unreadable_geometry() {
        let mut out = vec![0xAA];
        for (w, h, tw, th) in [(0, 8, 8, 8), (8, 8, 0, 8), (65536, 65536, 65536, 65536), (8, 8, MAX_TILE_DIM + 1, 8)] {
            let hd = Header { width: w, height: h, tile_w: tw, tile_h: th, ..base() };
            assert!(matches!(hd.write(&mut out), Err(WriteError::Geometry(_))), "{w}x{h} tile {tw}x{th}");
        }
        assert_eq!(out, [0xAA], "failed write must not append partial bytes");
    }

    #[test]
    fn plane_len_values() {
        let h = Header {
            width: 1920,
            height: 1080,
            frame_count: 1,
            tile_w: 64,
            tile_h: 64,
            codec_id: 0,
            codec_params: vec![],
            channels: vec![
                Channel { name: "R".into(), dtype: Dtype::U8 },
                Channel { name: "Z".into(), dtype: Dtype::F32 },
            ],
        };
        // normal sizes — exact value.
        assert_eq!(h.plane_len(Dtype::F32), Some(1920 * 1080 * 4));
        assert_eq!(h.plane_len(Dtype::U8), Some(1920 * 1080));
        assert_eq!(h.max_plane_len(), Some(1920 * 1080 * 4)); // max across channels = F32

        // usize overflow → None (NOT a panic, NOT a wrap).
        let big = Header { width: u32::MAX, height: u32::MAX, ..h.clone() };
        assert_eq!(big.plane_len(Dtype::F32), None);
        assert_eq!(big.max_plane_len(), None);
    }

    fn base() -> Header {
        Header {
            width: 8,
            height: 8,
            frame_count: 1,
            tile_w: 8,
            tile_h: 8,
            codec_id: 1,
            codec_params: vec![],
            channels: vec![Channel { name: "R".into(), dtype: Dtype::F16 }],
        }
    }

    /// Fields that do not fit their on-disk width are a hard error and leave `out` untouched.
    #[test]
    fn write_rejects_unrepresentable_fields() {
        let mut out = vec![0xAA];
        let long = Header { channels: vec![Channel { name: "x".repeat(256), dtype: Dtype::U8 }], ..base() };
        assert!(matches!(long.write(&mut out), Err(WriteError::Name(_))));
        let many = Header {
            channels: (0..=u16::MAX as usize).map(|_| Channel { name: String::new(), dtype: Dtype::U8 }).collect(),
            ..base()
        };
        assert_eq!(many.write(&mut out), Err(WriteError::Channels(65536)));
        let params = Header { codec_params: vec![0; 65536], ..base() };
        assert_eq!(params.write(&mut out), Err(WriteError::Params(65536)));
        assert_eq!(out, [0xAA], "failed write must not append partial bytes");
        // Boundary values are accepted and round-trip.
        let ok = Header { channels: vec![Channel { name: "n".repeat(255), dtype: Dtype::U8 }], ..base() };
        let mut buf = Vec::new();
        ok.write(&mut buf).expect("255-byte name fits");
        assert_eq!(Header::read(&buf).map(|(h, _)| h), Ok(ok));
    }

    /// A forged channel count must not over-reserve and must fail cleanly on the short body.
    #[test]
    fn read_rejects_forged_channel_count() {
        let mut buf = Vec::new();
        base().write(&mut buf).expect("write");
        // nch lives right after the (empty) codec params: ver1+flags1+6*4+cp_len2 = 28.
        buf[28] = 0xff;
        buf[29] = 0xff;
        assert!(Header::read(&buf).is_err());
    }

    #[test]
    fn frame_bytes_sums_planes() {
        let h = Header {
            channels: vec![
                Channel { name: "a".into(), dtype: Dtype::F16 },
                Channel { name: "b".into(), dtype: Dtype::F32 },
            ],
            ..base()
        };
        assert_eq!(h.frame_bytes(), Some(8 * 8 * 2 + 8 * 8 * 4));
        let big = Header { width: u32::MAX, height: u32::MAX, ..h };
        assert_eq!(big.frame_bytes(), None);
    }

    /// Serialized valid base header (1 channel "R", no meta) for byte-level forging.
    fn base_bytes() -> Vec<u8> {
        let mut b = Vec::new();
        base().write(&mut b).expect("write");
        b
    }

    /// Unknown flag bits (1..7) are a typed refusal, each bit on its own.
    #[test]
    fn read_rejects_unknown_flag_bits() {
        for bit in 1..8 {
            let mut b = base_bytes();
            b[1] |= 1 << bit;
            assert_eq!(Header::read(&b), Err(DecodeError::Corrupt("unknown header flag bits")), "bit {bit}");
        }
    }

    #[test]
    fn read_rejects_zero_frame_count() {
        let mut b = base_bytes();
        b[FRAME_COUNT_OFF..FRAME_COUNT_OFF + 4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(Header::read(&b), Err(DecodeError::Corrupt("frame_count is zero")));
    }

    /// The streaming placeholder (`frame_count = 0`) must still be WRITABLE: `ClipWriter` patches it at finish.
    #[test]
    fn write_allows_streaming_zero_frame_count() {
        let mut b = Vec::new();
        Header { frame_count: 0, ..base() }.write(&mut b).expect("placeholder header writes");
        assert_eq!(b[FRAME_COUNT_OFF..FRAME_COUNT_OFF + 4], 0u32.to_le_bytes());
    }

    #[test]
    fn read_rejects_zero_channels() {
        let mut b = base_bytes();
        b.truncate(28 + 2); // keep up to nch, drop the one channel record
        b[28] = 0;
        b[29] = 0;
        assert_eq!(Header::read(&b), Err(DecodeError::Corrupt("header has no channels")));
    }

    #[test]
    fn read_rejects_duplicate_channel_names() {
        let h = Header {
            channels: vec![
                Channel { name: "R".into(), dtype: Dtype::F16 },
                Channel { name: "R".into(), dtype: Dtype::F32 },
            ],
            ..base()
        };
        // Forge below the API: `write` refuses a duplicate, so build the bytes from a unique pair and patch the name.
        let uniq = Header { channels: vec![h.channels[0].clone(), Channel { name: "G".into(), dtype: Dtype::F32 }], ..base() };
        let mut b = Vec::new();
        uniq.write(&mut b).expect("write");
        let at = b.iter().rposition(|&x| x == b'G').expect("second name");
        b[at] = b'R';
        assert_eq!(Header::read(&b), Err(DecodeError::Corrupt("duplicate channel name")));
    }

    /// The writer refuses what the reader refuses (channels + meta), leaving `out` untouched.
    #[test]
    fn write_rejects_unreadable_channels_and_meta() {
        let mut out = vec![0xAA];
        let none = Header { channels: vec![], ..base() };
        assert_eq!(none.write(&mut out), Err(WriteError::Invalid("header has no channels")));
        let dup = Header {
            channels: vec![Channel { name: "R".into(), dtype: Dtype::F16 }; 2],
            ..base()
        };
        assert_eq!(dup.write(&mut out), Err(WriteError::Invalid("duplicate channel name")));
        for meta in [
            Meta { fps: Some((24, 0)), ..Meta::default() },
            Meta { pixel_aspect: Some((1, 0)), ..Meta::default() },
        ] {
            assert!(matches!(base().write_with_meta(&meta, &mut out), Err(WriteError::Invalid(_))), "{meta:?}");
        }
        assert_eq!(out, [0xAA], "failed write must not append partial bytes");
    }

    /// A forged `den = 0` in the Meta TLV is a typed refusal (fps and pixel aspect).
    #[test]
    fn read_rejects_zero_meta_den() {
        for (field, msg) in [(0, "meta fps denominator is zero"), (1, "meta pixel aspect denominator is zero")] {
            let ok = if field == 0 { Meta { fps: Some((24, 1)), ..Meta::default() } } else { Meta { pixel_aspect: Some((2, 1)), ..Meta::default() } };
            let mut b = Vec::new();
            base().write_with_meta(&ok, &mut b).expect("write");
            let n = b.len();
            b[n - 4..].copy_from_slice(&0u32.to_le_bytes()); // den is the last u32 of the single TLV
            assert_eq!(Header::read_with_meta(&b), Err(DecodeError::Corrupt(msg)));
        }
    }

    /// Byte-mutation sanity: valid headers (with and without meta) still round-trip, and every single-byte
    /// mutation either parses or fails with a typed error (never panics).
    #[test]
    fn valid_headers_roundtrip_and_mutations_never_panic() {
        let meta = Meta { fps: Some((24000, 1001)), pixel_aspect: Some((2, 1)), alpha: 1, ..Meta::default() };
        let multi = Header {
            channels: vec![
                Channel { name: "R".into(), dtype: Dtype::F16 },
                Channel { name: "G".into(), dtype: Dtype::F16 },
                Channel { name: "Z".into(), dtype: Dtype::F32 },
            ],
            ..base()
        };
        for (h, m) in [(base(), Meta::default()), (multi, meta)] {
            let mut b = Vec::new();
            h.write_with_meta(&m, &mut b).expect("write");
            let (h2, m2, n) = Header::read_with_meta(&b).expect("valid header parses");
            assert_eq!((h2, n), (h.clone(), b.len()));
            assert_eq!(m2.is_some(), !m.is_empty());
            for i in 0..b.len() {
                for x in [0x00u8, 0x01, 0x80, 0xFF] {
                    let mut bad = b.clone();
                    bad[i] ^= x;
                    drop(Header::read_with_meta(&bad)); // must return (Ok or Err), not panic
                }
            }
        }
    }
}
