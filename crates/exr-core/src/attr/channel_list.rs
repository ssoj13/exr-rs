//! Port of OpenEXR `ImfPixelType.h`, `ImfChannelList.h/.cpp` and
//! `ImfChannelListAttribute.h/.cpp`.
//!
//! Defines [`PixelType`] (the per-channel sample type), [`Channel`] (type +
//! sub-sampling + linearity hint), and [`ChannelList`] — an **ordered** map
//! name → [`Channel`].  C++ uses `std::map<Name, Channel>`, which keeps the
//! channels sorted by name; we mirror that with a [`BTreeMap`] so iteration
//! order — and therefore the on-disk byte order — is byte-identical.
//!
//! `"channels"` is present in every EXR file, so [`ChannelList`] serialisation
//! must be byte-exact.  The wire format (see `ChannelListAttribute::writeValueTo`)
//! is, per channel in sorted order:
//!   * the channel name as a `\0`-terminated string,
//!   * `type` as `i32`,
//!   * `pLinear` as one byte, then **3 reserved/pad bytes (written as 0)**,
//!   * `xSampling` as `i32`,
//!   * `ySampling` as `i32`.
//!
//! The list is terminated by a single empty (`\0`) name.
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfPixelType.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfChannelList.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfChannelList.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfChannelListAttribute.cpp>

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::attr::attribute::{AttrValue, TypedAttribute};
use crate::attr::name::Name;
use crate::container::xdr;
use crate::error::{ExrError, Result};
use crate::io::stream::{IStream, OStream};

/// `Name::SIZE` / `Name::MAX_LENGTH` from `ImfName.h`.
///
/// A channel name is stored in a fixed `char[256]` buffer, so the maximum
/// number of payload characters (excluding the terminating `\0`) is 255.
const NAME_MAX_LENGTH: usize = 255;

// ---------------------------------------------------------------------------
// PixelType — mirrors `enum PixelType` in ImfPixelType.h
// ---------------------------------------------------------------------------

/// Per-channel sample data type — port of `enum PixelType`.
///
/// On disk the discriminant is written as an `i32` (`UINT=0, HALF=1, FLOAT=2`).
/// `NUM_PIXELTYPES` is the "number of pixel types" sentinel; it is also used by
/// `readValueFrom` to flag an unrecognised type so that `Header::sanityCheck`
/// can reject it later (it is an *invalid* pixel type, never a valid sample).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PixelType {
    /// Unsigned 32-bit integer (`UINT = 0`).
    Uint = 0,
    /// 16-bit floating point (`HALF = 1`).
    Half = 1,
    /// 32-bit floating point (`FLOAT = 2`).
    Float = 2,
    /// Sentinel for "number of pixel types" / unrecognised type (`NUM_PIXELTYPES = 3`).
    ///
    /// Never a valid sample type; only ever built by hand (the parser rejects an
    /// unknown type discriminant, as C++ 3.4.15 core does).
    NumPixelTypes = 3,
}

impl PixelType {
    /// The on-disk `i32` discriminant — mirrors casting the enum to `int`.
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Decode an `i32` discriminant.
    ///
    /// Mirrors `ImfChannelListAttribute.cpp readValueFrom`: any value other
    /// than `UINT`/`HALF`/`FLOAT` is forced to `NUM_PIXELTYPES`, an invalid
    /// pixel type that `Header::sanityCheck` rejects.
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => PixelType::Uint,
            1 => PixelType::Half,
            2 => PixelType::Float,
            _ => PixelType::NumPixelTypes,
        }
    }

    // `size()` (a fallible-free `pixelTypeSize` returning `0` for the
    // sentinel) was removed: the sentinel is constructible
    // from a crafted file (`from_i32` above forces any unrecognised
    // discriminant to it), so silently sizing it as `0` bytes was a
    // silent-fallback hazard. `crate::misc::pixel_type_size(t) -> Result
    // <usize>` is the one canonical implementation — it is the same
    // `pixelTypeSize` port but `Err(Arg)`s on the sentinel, matching what
    // C++'s callers actually do with an invalid type (reject, not size-0).
}

// ---------------------------------------------------------------------------
// Channel — mirrors `struct Channel` in ImfChannelList.h
// ---------------------------------------------------------------------------

/// A single image channel — port of `struct Channel`.
///
/// `pixel (x, y)` is present in the channel only if
/// `x % x_sampling == 0 && y % y_sampling == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channel {
    /// Sample data type; see [`PixelType`].
    pub type_: PixelType,
    /// Horizontal sub-sampling factor.
    pub x_sampling: i32,
    /// Vertical sub-sampling factor.
    pub y_sampling: i32,
    /// Lossy-compression linearity hint (perceptually linear vs. logarithmic).
    pub p_linear: bool,
}

impl Channel {
    /// Construct a channel — mirrors `Channel(PixelType, int, int, bool)`.
    ///
    /// C++ defaults are `type=HALF, xSampling=1, ySampling=1, pLinear=false`;
    /// callers that want those should use [`Channel::default`].
    pub fn new(type_: PixelType, x_sampling: i32, y_sampling: i32, p_linear: bool) -> Self {
        Self {
            type_,
            x_sampling,
            y_sampling,
            p_linear,
        }
    }
}

impl Default for Channel {
    /// C++ default `Channel()`: `type=HALF, xSampling=1, ySampling=1, pLinear=false`.
    fn default() -> Self {
        Self {
            type_: PixelType::Half,
            x_sampling: 1,
            y_sampling: 1,
            p_linear: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelList — mirrors `class ChannelList` in ImfChannelList.h/.cpp
// ---------------------------------------------------------------------------

/// Ordered map name → [`Channel`] — port of `class ChannelList`.
///
/// Backed by a [`BTreeMap`] so the channels stay sorted by name exactly as the
/// C++ `std::map<Name, Channel>` does; this ordering is what makes the on-disk
/// byte layout deterministic and byte-exact.
///
/// `PartialEq`/`Eq` are hand-written (not derived) to reproduce a C++ quirk:
/// `ChannelList::operator==` (`ImfChannelList.cpp:239-253`) walks both
/// lists in sorted order and compares only the `Channel` VALUES pairwise —
/// never the names. Two lists with the same channel values under different
/// names compare equal in C++, and a derived `PartialEq` (which also compares
/// keys) would disagree.
#[derive(Clone, Debug, Default)]
pub struct ChannelList {
    /// The sorted name → channel map (`ChannelList::_map`).
    map: BTreeMap<Name, Channel>,
}

impl PartialEq for ChannelList {
    /// Port of `ChannelList::operator==` — see the struct doc.
    fn eq(&self, other: &Self) -> bool {
        self.map.len() == other.map.len()
            && self
                .map
                .values()
                .zip(other.map.values())
                .all(|(a, b)| a == b)
    }
}

impl Eq for ChannelList {}

impl ChannelList {
    /// Construct an empty channel list.
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Add or replace a channel — mirrors `ChannelList::insert`.
    ///
    /// Returns `ArgExc` if `name` is empty, matching the C++ `THROW(ArgExc,…)`.
    ///
    /// `name` accepts anything convertible to a byte-string [`Name`]
    /// (`&str`, `String`, `&[u8]`, `Vec<u8>`, `Name`); [`Name::new`] applies
    /// the exact C++ `strncpy(_text, text, MAX_LENGTH)` cut (first NUL, then
    /// 255 bytes) instead of the old UTF-8-boundary-respecting
    /// `xdr::truncate_name`.
    pub fn insert(&mut self, name: impl Into<Name>, channel: Channel) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(ExrError::Arg(
                "Image channel name cannot be an empty string.".into(),
            ));
        }
        self.map.insert(name, channel);
        Ok(())
    }

    /// Look up a channel by name — mirrors `ChannelList::findChannel` (returns
    /// `None` when absent, like the pointer-returning C++ overload).
    pub fn find_channel<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Option<&Channel> {
        self.map.get(Name::normalize(name.as_ref()))
    }

    /// Mutable lookup — mirrors the non-const `ChannelList::findChannel`.
    pub fn find_channel_mut<K: AsRef<[u8]> + ?Sized>(&mut self, name: &K) -> Option<&mut Channel> {
        self.map.get_mut(Name::normalize(name.as_ref()))
    }

    /// Throwing lookup — mirrors `ChannelList::operator[](const char[]) const`.
    /// Returns [`ExrError::Arg`] ("Cannot find image
    /// channel …") for a missing name, matching C++'s `THROW(ArgExc, …)`,
    /// unlike [`find_channel`][Self::find_channel]'s `Option`.
    pub fn get<K: AsRef<[u8]> + ?Sized>(&self, name: &K) -> Result<&Channel> {
        let bytes = Name::normalize(name.as_ref());
        self.map.get(bytes).ok_or_else(|| {
            ExrError::Arg(format!(
                "Cannot find image channel \"{}\".",
                String::from_utf8_lossy(bytes)
            ))
        })
    }

    /// Mutable counterpart of [`get`][Self::get] — mirrors the non-const
    /// `ChannelList::operator[](const char[])`.
    pub fn get_mut<K: AsRef<[u8]> + ?Sized>(&mut self, name: &K) -> Result<&mut Channel> {
        let bytes = Name::normalize(name.as_ref());
        match self.map.get_mut(bytes) {
            Some(channel) => Ok(channel),
            None => Err(ExrError::Arg(format!(
                "Cannot find image channel \"{}\".",
                String::from_utf8_lossy(bytes)
            ))),
        }
    }

    /// Number of channels in the list.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate over `(name, channel)` pairs in sorted name order.
    ///
    /// Mirrors `ChannelList::begin()`/`end()` (a `std::map` iterator, which is
    /// ordered by key).
    pub fn iter(&self) -> impl Iterator<Item = (&Name, &Channel)> {
        self.map.iter()
    }

    /// Collect the set of layer names — port of `ChannelList::layers`.
    ///
    /// Channel `L.C` belongs to layer `L`.  A name with no `.`, or a `.` only
    /// at the very beginning or very end, is not part of any layer.  Mirrors
    /// the C++ `rfind('.')` logic exactly (`pos != npos && pos != 0 && pos + 1
    /// < size`), operating on raw bytes so a non-UTF-8 name still splits
    /// correctly on the ASCII `.` byte.
    pub fn layers(&self) -> BTreeSet<Name> {
        let mut layer_names = BTreeSet::new();
        for name in self.map.keys() {
            let bytes = name.as_bytes();
            if let Some(pos) = bytes.iter().rposition(|&b| b == b'.') {
                // pos != 0 (not leading) and pos + 1 < len (not trailing).
                if pos != 0 && pos + 1 < bytes.len() {
                    layer_names.insert(Name::new(&bytes[..pos]));
                }
            }
        }
        layer_names
    }

    /// All channels whose name begins with `prefix`, in sorted order.
    ///
    /// Port of `ChannelList::channelsWithPrefix`.  The C++ uses
    /// `lower_bound(prefix)` then advances while `strncmp(name, prefix, n) <= 0`;
    /// because the map is sorted, that is exactly the contiguous run of names
    /// that start with `prefix`, which a byte `starts_with` selects.
    pub fn channels_with_prefix<'a>(
        &'a self,
        prefix: &'a str,
    ) -> impl Iterator<Item = (&'a Name, &'a Channel)> {
        let prefix_bytes = prefix.as_bytes();
        self.map
            .range(Name::new(prefix)..)
            .take_while(move |(name, _)| name.as_bytes().starts_with(prefix_bytes))
    }

    /// All channels in layer `layer_name`, in sorted order.
    ///
    /// Port of `ChannelList::channelsInLayer`, which calls
    /// `channelsWithPrefix(layerName + '.', …)`.
    pub fn channels_in_layer<'a>(
        &'a self,
        layer_name: &str,
    ) -> impl Iterator<Item = (&'a Name, &'a Channel)> {
        let mut prefix = layer_name.to_string();
        prefix.push('.');
        let prefix_name = Name::new(&prefix);
        // Materialise into a Vec so the temporary `prefix` is not borrowed by
        // the returned iterator (channels_with_prefix borrows its argument).
        let out: Vec<(&'a Name, &'a Channel)> = self
            .map
            .range(prefix_name..)
            .take_while(move |(name, _)| name.as_bytes().starts_with(prefix.as_bytes()))
            .collect();
        out.into_iter()
    }
}

// ---------------------------------------------------------------------------
// AttrValue — byte-exact `"chlist"` serialisation
// ---------------------------------------------------------------------------

impl AttrValue for ChannelList {
    fn static_type_name() -> &'static str {
        // ImfChannelListAttribute.cpp: return "chlist";
        "chlist"
    }

    /// Mirrors `ChannelListAttribute::writeValueTo`.
    ///
    /// For each channel in sorted name order:
    ///   name (NUL-terminated) · type (i32) · pLinear (1 byte) · 3 pad bytes ·
    ///   xSampling (i32) · ySampling (i32),
    /// then a single empty (`\0`) name as the end-of-list marker.
    fn write_value(&self, o: &mut dyn OStream, _version: i32) -> Result<()> {
        for (name, channel) in self.map.iter() {
            // Names longer than 255 bytes cannot round-trip through the
            // fixed-size `char[256]` read buffer; reject rather than truncate.
            // `Name::new` already cut a longer input at insert time, so this
            // only fires for a `Name` built by some other path.
            if name.len() > NAME_MAX_LENGTH {
                return Err(ExrError::Input(format!(
                    "Invalid channel name: it is more than {NAME_MAX_LENGTH} characters long."
                )));
            }

            // Xdr::write<StreamIO>(os, i.name());  — zero-terminated string,
            // raw bytes, no encoding assumed.
            xdr::write_name(o, name.as_bytes())?;

            // Xdr::write<StreamIO>(os, int(i.channel().type));
            xdr::write_i32(o, channel.type_.as_i32())?;
            // Xdr::write<StreamIO>(os, i.channel().pLinear);  — one byte.
            xdr::write_bool(o, channel.p_linear)?;
            // Xdr::pad<StreamIO>(os, 3);  — 3 reserved bytes, written as 0.
            xdr::pad(o, 3)?;
            // Xdr::write<StreamIO>(os, i.channel().xSampling);
            xdr::write_i32(o, channel.x_sampling)?;
            // Xdr::write<StreamIO>(os, i.channel().ySampling);
            xdr::write_i32(o, channel.y_sampling)?;
        }

        // Xdr::write<StreamIO>(os, "");  — end-of-list marker (single \0).
        xdr::write_string(o, "")
    }

    /// Mirrors `ChannelListAttribute::readValueFrom`, via the one core-faithful parser
    /// [`ChannelList::read_into`] (a rejected list is an error here).
    fn read_value(i: &mut dyn IStream, size: i32, _version: i32) -> Result<Self> {
        let mut list = ChannelList::new();
        match list.read_into(i, size)? {
            None => Ok(list),
            Some(e) => Err(e),
        }
    }
}

impl ChannelList {
    /// Port of C++ 3.4.15 core `extract_attr_chlist` (`parse_header.c:352-418`) plus the
    /// per-channel checks of `exr_attr_chlist_add_with_length` (`channel_list.c:69-160`),
    /// appending to `self` as it goes exactly like C++: channels added before a failure STAY
    /// (that is how a repeated `channels` attribute keeps the first list's entries).
    ///
    /// The parse is bounded by the declared `size`: every name costs `len + 1`, every
    /// channel 16 more, and the loop ends when the budget is spent or an empty name is seen
    /// (a list without a terminator inside the declared size is fine). Whatever the
    /// outcome, C++ then `sequential_skip`s the remaining budget FROM WHERE THE CURSOR IS,
    /// so a failure part-way leaves the stream at a position `size`-dependent but not at
    /// the declared end -- reproduced here.
    ///
    /// Returns `Err` for what is fatal in C++ (stream failure, a budget overrun whose skip
    /// runs off the end of the file) and `Ok(Some(e))` when the attribute is rejected but
    /// the header parse goes on from the stream's current position.
    pub(crate) fn read_into(&mut self, i: &mut dyn IStream, size: i32) -> Result<Option<ExrError>> {
        let mut left = i64::from(size);
        let mut rv: Option<ExrError> = if size < 0 {
            Some(ExrError::Input(format!("Invalid negative size {size} for a channel list")))
        } else {
            None
        };
        let mut name_buf: Vec<u8> = Vec::with_capacity(NAME_MAX_LENGTH);

        while rv.is_none() && left > 0 {
            match xdr::read_name(i, &mut name_buf, NAME_MAX_LENGTH) {
                Ok(true) => {}
                Ok(false) => {
                    rv = Some(ExrError::Input("channel name is too long".into()));
                    break;
                }
                Err(e) => {
                    rv = Some(e);
                    break;
                }
            }
            left -= name_buf.len() as i64 + 1;
            if name_buf.is_empty() {
                break;
            }
            if left < 16 {
                skip_budget(i, left)?;
                return Ok(Some(ExrError::Input(format!(
                    "Out of data parsing channels, last channel '{}'",
                    String::from_utf8_lossy(&name_buf)
                ))));
            }

            // type, pLinear + 3 reserved bytes, xSampling, ySampling.
            let fields = (|| -> Result<(i32, u8, i32, i32)> {
                let ty = xdr::read_i32(i)?;
                let lin = xdr::read_u8_val(i)?;
                xdr::skip(i, 3)?;
                Ok((ty, lin, xdr::read_i32(i)?, xdr::read_i32(i)?))
            })();
            let (ty, lin, xs, ys) = match fields {
                Ok(f) => f,
                Err(e) => {
                    rv = Some(e);
                    break;
                }
            };
            left -= 16;

            // `exr_attr_chlist_add_with_length`'s rejections; each ends the loop.
            let name = Name::new(name_buf.as_slice());
            rv = if !(0..=2).contains(&ty) {
                Some(format!("invalid pixel type {ty} for channel \"{name}\""))
            } else if lin > 1 {
                Some(format!("invalid pLinear value {lin} for channel \"{name}\" (must be 0 or 1)"))
            } else if xs <= 0 || ys <= 0 {
                Some(format!("invalid sampling ({xs}, {ys}) for channel \"{name}\""))
            } else if self.map.contains_key(&name) {
                Some(format!("duplicate channel name \"{name}\" in channel list"))
            } else {
                None
            }
            .map(ExrError::Input);
            if rv.is_none() {
                let ch = Channel::new(PixelType::from_i32(ty), xs, ys, lin != 0);
                if let Err(e) = self.insert(&name, ch) {
                    rv = Some(e);
                }
            }
        }
        skip_budget(i, left)?;
        Ok(rv)
    }
}

/// C++ `sequential_skip(attrsz)` after the channel loop: forward from the cursor; a
/// negative budget (a name that ran past the declared size) is a skip of ~4 GiB that hits
/// EOF, which is fatal for the rest of the header.
fn skip_budget(i: &mut dyn IStream, left: i64) -> Result<()> {
    match u64::try_from(left) {
        Ok(0) => Ok(()),
        Ok(n) => {
            let at = i.tellg()?;
            i.seekg(at + n)
        }
        Err(_) => Err(ExrError::Input("channel list overran its declared size".into())),
    }
}

/// `typedef TypedAttribute<ChannelList> ChannelListAttribute;`
pub type ChannelListAttribute = TypedAttribute<ChannelList>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::std_io::{MemIStream, MemOStream};

    fn write_bytes(list: &ChannelList) -> Vec<u8> {
        let mut o = MemOStream::new();
        list.write_value(&mut o, 0).expect("write");
        o.into_data()
    }

    // `PixelType::size()` was removed; its boundary coverage lives in
    // `misc.rs`'s `pixel_type_size_*` tests for the canonical
    // `misc::pixel_type_size`.

    #[test]
    fn type_name_is_chlist() {
        assert_eq!(ChannelList::static_type_name(), "chlist");
    }

    /// `ChannelList::operator==` compares only the `Channel` values in
    /// sorted-position order, never the names: two lists whose channels have
    /// swapped names but the same values-in-order must compare equal, and a
    /// differing channel count must compare unequal.
    #[test]
    fn eq_compares_values_by_position_not_names() {
        let mut a = ChannelList::new();
        a.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        a.insert("B", Channel::new(PixelType::Float, 1, 1, true)).unwrap();

        // Same two Channel values in the same sorted-position order, but under
        // different names — C++ considers this equal.
        let mut b = ChannelList::new();
        b.insert("X", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        b.insert("Y", Channel::new(PixelType::Float, 1, 1, true)).unwrap();
        assert_eq!(a, b);

        // A different channel COUNT must never compare equal.
        let mut c = ChannelList::new();
        c.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        assert_ne!(a, c);

        // A different value at the same position must compare unequal.
        let mut d = ChannelList::new();
        d.insert("A", Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        d.insert("B", Channel::new(PixelType::Uint, 1, 1, true)).unwrap();
        assert_ne!(a, d);
    }

    /// A non-UTF-8 channel name (C++ `Name`/`parse_header.c`'s
    /// `read_text` never checks encoding) round-trips byte-exact through the
    /// real wire encode/decode, not just through the in-memory map. Oracle
    /// cross-read coverage (a real C++-written file with such a name) is
    /// tracked as follow-up work.
    #[test]
    fn non_utf8_channel_name_round_trips_through_the_wire() {
        let raw_name: &[u8] = &[0x5a, 0xe9]; // "Z\xE9" (invalid UTF-8)
        let mut list = ChannelList::new();
        list.insert(Name::new(raw_name), Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();

        let bytes = write_bytes(&list);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = ChannelList::read_value(&mut i, size, 0).unwrap();

        let (name, ch) = back.iter().next().expect("one channel");
        assert_eq!(name.as_bytes(), raw_name);
        assert_eq!(ch.type_, PixelType::Half);
    }

    /// A channel name with an interior NUL is truncated at insert
    /// time, matching C++ `Name::operator=`, instead of corrupting the wire
    /// framing on write.
    #[test]
    fn insert_truncates_name_at_interior_nul() {
        let mut list = ChannelList::new();
        list.insert("R\0garbage", Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        assert!(list.find_channel("R").is_some());
        // A LOOKUP key is normalized the exact same way an insert key
        // is (C++ `map::find(const char*)` builds a temporary `Name` via the
        // same `strncpy` truncation `operator=` uses for insert), so a query
        // with an embedded NUL finds the SAME channel the truncated insert
        // stored — it is not "not found", it is an alias for "R".
        assert!(list.find_channel("R\0garbage").is_some());
    }

    /// A channel name over 255 bytes is truncated to `Name::MAX_LENGTH`.
    #[test]
    fn insert_truncates_overlong_name() {
        let mut list = ChannelList::new();
        let long = "x".repeat(300);
        list.insert(&long, Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        assert_eq!(list.iter().next().unwrap().0.len(), NAME_MAX_LENGTH);
    }

    /// `get`/`get_mut` are the throwing-lookup equivalent of
    /// `ChannelList::operator[]`: `Ok` for a present channel, `Err` for a
    /// missing one (never a panic, unlike a real `Index` impl would need).
    #[test]
    fn get_and_get_mut_throwing_lookup() {
        let mut list = ChannelList::new();
        list.insert("R", Channel::new(PixelType::Half, 1, 1, false)).unwrap();

        assert_eq!(list.get("R").unwrap().type_, PixelType::Half);
        assert!(list.get("Z").is_err());

        list.get_mut("R").unwrap().type_ = PixelType::Float;
        assert_eq!(list.get("R").unwrap().type_, PixelType::Float);
        assert!(list.get_mut("Z").is_err());
    }

    /// The load-bearing byte-exact test: `{"A": HALF, "B": FLOAT}` must serialise
    /// in sorted order with NUL-terminated names, the 3 reserved bytes after
    /// `pLinear`, and a trailing `\0` end-of-list marker.
    #[test]
    fn channel_list_exact_bytes() {
        let mut list = ChannelList::new();
        // Default sampling = 1, pLinear = false.
        list.insert("A", Channel::new(PixelType::Half, 1, 1, false))
            .unwrap();
        list.insert("B", Channel::new(PixelType::Float, 1, 1, false))
            .unwrap();

        let bytes = write_bytes(&list);

        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            // Channel "A": name "A\0"
            b'A', 0x00,
            // type = HALF (1) as i32 LE
            0x01, 0x00, 0x00, 0x00,
            // pLinear = false (1 byte) + 3 reserved/pad bytes
            0x00, 0x00, 0x00, 0x00,
            // xSampling = 1 LE
            0x01, 0x00, 0x00, 0x00,
            // ySampling = 1 LE
            0x01, 0x00, 0x00, 0x00,
            // Channel "B": name "B\0"
            b'B', 0x00,
            // type = FLOAT (2) as i32 LE
            0x02, 0x00, 0x00, 0x00,
            // pLinear = false + 3 reserved
            0x00, 0x00, 0x00, 0x00,
            // xSampling = 1
            0x01, 0x00, 0x00, 0x00,
            // ySampling = 1
            0x01, 0x00, 0x00, 0x00,
            // end-of-list marker: empty name "\0"
            0x00,
        ];

        assert_eq!(bytes, expected);
    }

    #[test]
    fn channel_list_round_trip() {
        let mut list = ChannelList::new();
        list.insert("R", Channel::new(PixelType::Half, 1, 1, true))
            .unwrap();
        list.insert("G", Channel::new(PixelType::Half, 2, 2, false))
            .unwrap();
        list.insert("Z", Channel::new(PixelType::Float, 1, 1, false))
            .unwrap();
        list.insert("id", Channel::new(PixelType::Uint, 1, 1, false))
            .unwrap();

        let bytes = write_bytes(&list);
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        let back = ChannelList::read_value(&mut i, size, 0).unwrap();
        assert_eq!(back, list);
    }

    #[test]
    fn empty_name_rejected() {
        let mut list = ChannelList::new();
        assert!(list.insert("", Channel::default()).is_err());
    }

    #[test]
    fn layers_and_prefix() {
        let mut list = ChannelList::new();
        list.insert("light1.R", Channel::default()).unwrap();
        list.insert("light1.G", Channel::default()).unwrap();
        list.insert("light2.B", Channel::default()).unwrap();
        list.insert("Z", Channel::default()).unwrap();
        list.insert(".leading", Channel::default()).unwrap();
        list.insert("trailing.", Channel::default()).unwrap();

        let layers = list.layers();
        assert!(layers.contains(Name::normalize(b"light1")));
        assert!(layers.contains(Name::normalize(b"light2")));
        // ".leading" (leading dot) and "trailing." (trailing dot) are NOT layers.
        assert!(!layers.contains(Name::normalize(b"")));
        assert_eq!(layers.len(), 2);

        let in_light1: Vec<_> = list
            .channels_with_prefix("light1.")
            .map(|(n, _)| n.to_string())
            .collect();
        assert_eq!(in_light1, vec!["light1.G".to_string(), "light1.R".to_string()]);

        let in_layer: Vec<_> = list
            .channels_in_layer("light2")
            .map(|(n, _)| n.to_string())
            .collect();
        assert_eq!(in_layer, vec!["light2.B".to_string()]);
    }

    #[test]
    fn unknown_pixel_type_rejects_the_list() {
        // Build a channel list payload by hand with an out-of-range type (7).
        let mut o = MemOStream::new();
        xdr::write_string(&mut o, "X").unwrap();
        xdr::write_i32(&mut o, 7).unwrap(); // bogus type
        xdr::write_bool(&mut o, false).unwrap();
        xdr::pad(&mut o, 3).unwrap();
        xdr::write_i32(&mut o, 1).unwrap();
        xdr::write_i32(&mut o, 1).unwrap();
        xdr::write_string(&mut o, "").unwrap();
        let bytes = o.into_data();
        let size = bytes.len() as i32;
        let mut i = MemIStream::new(bytes);
        // C++ 3.4.15 core `exr_attr_chlist_add_with_length` rejects the channel, and with it
        // the whole attribute (`channel_list.c:99-107`).
        assert!(ChannelList::read_value(&mut i, size, 0).is_err());
    }
}
