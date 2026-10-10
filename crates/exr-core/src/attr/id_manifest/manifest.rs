//! Port of the structured `IDManifest`/`ChannelGroupManifest` model —
//! `ImfIDManifest.h`'s class declarations.
//!
//! # Rust idiom vs. C++ 1:1
//! C++ exposes `ChannelGroupManifest::{Iterator,ConstIterator}` classes and
//! `operator<<` streaming insertion. This port uses `BTreeMap<u64,
//! Vec<String>>` directly (`.iter()`/`.iter_mut()` already give the same
//! read/write iteration C++'s custom iterator classes exist to provide, more
//! idiomatically — the same choice this crate already made for
//! [`crate::attr::ChannelList`]) and skips `operator<<` (a C++-only ergonomic
//! device for building a table with a "current insertion" state machine that
//! throws if used incorrectly; [`ChannelGroupManifest::insert`] gives the
//! same capability without that state machine or its exception path).
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.h>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.cpp>

use std::collections::{BTreeMap, BTreeSet};

use crate::attr::id_manifest::hash::{murmur_hash32, murmur_hash64};
use crate::error::{ExrError, Result};

// ---------------------------------------------------------------------------
// Scheme name constants — `IDManifest::{UNKNOWN,NOTHASHED,CUSTOMHASH,
// MURMURHASH3_32,MURMURHASH3_64,ID_SCHEME,ID2_SCHEME}` (`ImfIDManifest.cpp:41-48`).
// ---------------------------------------------------------------------------

/// Default value for encoding/hash scheme — should be changed by the caller.
pub const UNKNOWN: &str = "unknown";
/// Hash scheme: no relationship between text and ID.
pub const NOTHASHED: &str = "none";
/// Hash scheme: text is hashed using a caller-defined scheme.
pub const CUSTOMHASH: &str = "custom";
/// Hash scheme: MurmurHash3 32-bit.
pub const MURMURHASH3_32: &str = "MurmurHash3_32";
/// Hash scheme: bottom 8 bytes of MurmurHash3_128 (64-bit architecture version).
pub const MURMURHASH3_64: &str = "MurmurHash3_64";
/// Encoding scheme: 32-bit ID stored directly in a UINT channel.
pub const ID_SCHEME: &str = "id";
/// Encoding scheme: 64-bit ID stored in two channels, per the `ChannelGroup`.
pub const ID2_SCHEME: &str = "id2";

/// How long a mapping between an ID number and its text holds for — mirrors
/// `enum IDManifest::IdLifetime`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IdLifetime {
    /// The mapping may change every frame.
    #[default]
    Frame = 0,
    /// The mapping is consistent for every frame of a shot.
    Shot = 1,
    /// The mapping is consistent for all time.
    Stable = 2,
}

impl IdLifetime {
    /// Construct from the raw on-disk byte — C++ casts unchecked
    /// (`IdLifetime(lifetime)` in `init`, `ImfIDManifest.cpp:509`); mirrored
    /// here as a checked conversion so a corrupt manifest errors instead of
    /// constructing an out-of-range enum value.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Frame),
            1 => Ok(Self::Shot),
            2 => Ok(Self::Stable),
            _ => Err(ExrError::Input(format!("Invalid IdLifetime value {v} in IDManifest"))),
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelGroupManifest — port of `class IDManifest::ChannelGroupManifest`
// ---------------------------------------------------------------------------

/// Describes the information represented by one group of channels — port of
/// `IDManifest::ChannelGroupManifest`.
#[derive(Clone, Debug, Default)]
pub struct ChannelGroupManifest {
    channels: BTreeSet<String>,
    components: Vec<String>,
    life_time: IdLifetime,
    hash_scheme: String,
    encoding_scheme: String,
    table: BTreeMap<u64, Vec<String>>,
}

impl PartialEq for ChannelGroupManifest {
    /// Port of `ChannelGroupManifest::operator==`
    /// (`ImfIDManifest.cpp:1410-1417`): compares `life_time`, `components`,
    /// `hash_scheme` and `table` — **not** `channels` or `encoding_scheme`,
    /// matching the C++ omission exactly (the C++ body also compares
    /// `_components` twice; the duplicate is redundant and not repeated here).
    fn eq(&self, other: &Self) -> bool {
        self.life_time == other.life_time
            && self.components == other.components
            && self.hash_scheme == other.hash_scheme
            && self.table == other.table
    }
}

impl ChannelGroupManifest {
    /// `ChannelGroupManifest()` — `hashScheme`/`encodingScheme` default to
    /// [`UNKNOWN`], `lifeTime` defaults to [`IdLifetime::Stable`].
    pub fn new() -> Self {
        Self {
            channels: BTreeSet::new(),
            components: Vec::new(),
            life_time: IdLifetime::Stable,
            hash_scheme: UNKNOWN.to_string(),
            encoding_scheme: UNKNOWN.to_string(),
            table: BTreeMap::new(),
        }
    }

    /// `const set<string>& getChannels() const`.
    pub fn channels(&self) -> &BTreeSet<String> {
        &self.channels
    }

    /// `set<string>& getChannels()`.
    pub fn channels_mut(&mut self) -> &mut BTreeSet<String> {
        &mut self.channels
    }

    /// `void setChannels(const set<string>&)`.
    pub fn set_channels(&mut self, channels: BTreeSet<String>) {
        self.channels = channels;
    }

    /// `void setChannel(const string&)` — replaces the whole set with a
    /// single channel.
    pub fn set_channel(&mut self, channel: &str) {
        self.channels.clear();
        self.channels.insert(channel.to_string());
    }

    /// `const vector<string>& getComponents() const`.
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// `void setComponents(const vector<string>&)` — rejects a length change
    /// once the table has entries, matching C++'s `THROW(ArgExc, …)`.
    pub fn set_components(&mut self, components: Vec<String>) -> Result<()> {
        if !self.table.is_empty() && components.len() != self.components.len() {
            return Err(ExrError::Arg(
                "attempt to change number of components in manifest once entries have been added"
                    .into(),
            ));
        }
        self.components = components;
        Ok(())
    }

    /// `void setComponent(const string&)` — single-component convenience,
    /// via [`set_components`][Self::set_components].
    pub fn set_component(&mut self, component: &str) -> Result<()> {
        self.set_components(vec![component.to_string()])
    }

    /// `IdLifetime getLifetime() const`.
    pub fn lifetime(&self) -> IdLifetime {
        self.life_time
    }

    /// `void setLifetime(const IdLifetime&)`.
    pub fn set_lifetime(&mut self, life_time: IdLifetime) {
        self.life_time = life_time;
    }

    /// `const string& getHashScheme() const`.
    pub fn hash_scheme(&self) -> &str {
        &self.hash_scheme
    }

    /// `void setHashScheme(const string&)`.
    pub fn set_hash_scheme(&mut self, scheme: &str) {
        self.hash_scheme = scheme.to_string();
    }

    /// `const string& getEncodingScheme() const`.
    pub fn encoding_scheme(&self) -> &str {
        &self.encoding_scheme
    }

    /// `void setEncodingScheme(const string&)`.
    pub fn set_encoding_scheme(&mut self, scheme: &str) {
        self.encoding_scheme = scheme.to_string();
    }

    /// `size_t size() const` — number of table entries (may be 0).
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Read-only iteration in ID order — replaces C++'s `begin()`/`end()` /
    /// `ConstIterator` (see the module doc).
    pub fn iter(&self) -> impl Iterator<Item = (u64, &[String])> {
        self.table.iter().map(|(&id, text)| (id, text.as_slice()))
    }

    /// Mutable iteration in ID order — replaces C++'s non-const
    /// `begin()`/`end()` / `Iterator`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u64, &mut Vec<String>)> {
        self.table.iter_mut().map(|(&id, text)| (id, text))
    }

    /// `ConstIterator find(uint64_t) const` — port returns `Option` rather
    /// than an end-sentinel iterator.
    pub fn find(&self, id_value: u64) -> Option<&[String]> {
        self.table.get(&id_value).map(Vec::as_slice)
    }

    /// `Iterator find(uint64_t)` (mutable).
    pub fn find_mut(&mut self, id_value: u64) -> Option<&mut Vec<String>> {
        self.table.get_mut(&id_value)
    }

    /// `void erase(uint64_t)`.
    pub fn erase(&mut self, id_value: u64) {
        self.table.remove(&id_value);
    }

    /// `vector<string>& operator[](uint64_t)` — "adds the mapping to the
    /// vector if it doesn't exist" (a default-constructed empty `Vec`).
    pub fn entry(&mut self, id_value: u64) -> &mut Vec<String> {
        self.table.entry(id_value).or_default()
    }

    /// `Iterator insert(uint64_t, const vector<string>&)` — rejects a `text`
    /// whose length disagrees with [`components`][Self::components].
    pub fn insert(&mut self, id_value: u64, text: Vec<String>) -> Result<()> {
        if self.components.len() != text.len() {
            return Err(ExrError::Arg(
                "mismatch between number of components in manifest and number of components \
                 in inserted entry"
                    .into(),
            ));
        }
        self.table.insert(id_value, text);
        Ok(())
    }

    /// `Iterator insert(uint64_t, const string&)` — requires exactly one
    /// component.
    pub fn insert_single(&mut self, id_value: u64, text: &str) -> Result<()> {
        if self.components.len() != 1 {
            return Err(ExrError::Arg(
                "Cannot insert single component attribute into manifest with multiple components"
                    .into(),
            ));
        }
        self.table.insert(id_value, vec![text.to_string()]);
        Ok(())
    }

    /// `uint64_t insert(const vector<string>&)` — computes the ID from
    /// `text` via [`hash_scheme`][Self::hash_scheme] ([`MURMURHASH3_32`] or
    /// [`MURMURHASH3_64`] only; any other scheme errors, matching C++'s
    /// `THROW(ArgExc, "Cannot compute hash: unknown hashing scheme")`),
    /// inserts, and returns the computed hash.
    pub fn insert_hashed(&mut self, text: Vec<String>) -> Result<u64> {
        let hash = self.compute_hash(&text)?;
        self.insert(hash, text)?;
        Ok(hash)
    }

    /// `uint64_t insert(const string&)` — single-component convenience for
    /// [`insert_hashed`][Self::insert_hashed].
    pub fn insert_hashed_single(&mut self, text: &str) -> Result<u64> {
        self.insert_hashed(vec![text.to_string()])
    }

    fn compute_hash(&self, text: &[String]) -> Result<u64> {
        if self.hash_scheme == MURMURHASH3_32 {
            Ok(murmur_hash32(text) as u64)
        } else if self.hash_scheme == MURMURHASH3_64 {
            Ok(murmur_hash64(text))
        } else {
            Err(ExrError::Arg("Cannot compute hash: unknown hashing scheme".into()))
        }
    }

    /// Direct access to the underlying table, for the codec module
    /// (`init`/`serialize` need the raw `BTreeMap` for the string-frequency
    /// pass) — `pub(super)` since it bypasses the length/hash validation the
    /// public `insert*` methods enforce.
    pub(super) fn table(&self) -> &BTreeMap<u64, Vec<String>> {
        &self.table
    }

    pub(super) fn table_mut(&mut self) -> &mut BTreeMap<u64, Vec<String>> {
        &mut self.table
    }
}

// ---------------------------------------------------------------------------
// IDManifest — port of `class IDManifest`
// ---------------------------------------------------------------------------

/// A table mapping ID numbers to text, grouped by the channel(s) they apply
/// to — port of `class IDManifest`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IDManifest {
    manifest: Vec<ChannelGroupManifest>,
}

impl IDManifest {
    /// `IDManifest()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `size_t size() const`.
    pub fn len(&self) -> usize {
        self.manifest.len()
    }

    /// Whether there are no channel-group manifests at all.
    pub fn is_empty(&self) -> bool {
        self.manifest.is_empty()
    }

    /// Read-only iteration over the channel-group manifests, in the order
    /// they were added — replaces indexing via `operator[]` for the common
    /// "look at everything" case.
    pub fn iter(&self) -> impl Iterator<Item = &ChannelGroupManifest> {
        self.manifest.iter()
    }

    /// `size_t find(const string& channel) const` — the index of the first
    /// `ChannelGroupManifest` whose channel set contains `channel`, if any.
    /// Returns `None` where C++ returns `size()` (not found).
    pub fn find_channel(&self, channel: &str) -> Option<usize> {
        self.manifest.iter().position(|m| m.channels().contains(channel))
    }

    /// `const ChannelGroupManifest& operator[](size_t) const` — a checked
    /// `Result` instead of C++'s unchecked `vector::operator[]` (UB on an
    /// out-of-range index in C++; this port never has UB).
    pub fn get(&self, index: usize) -> Result<&ChannelGroupManifest> {
        self.manifest
            .get(index)
            .ok_or_else(|| ExrError::Arg(format!("IDManifest index {index} out of range")))
    }

    /// Mutable counterpart of [`get`][Self::get].
    pub fn get_mut(&mut self, index: usize) -> Result<&mut ChannelGroupManifest> {
        self.manifest
            .get_mut(index)
            .ok_or_else(|| ExrError::Arg(format!("IDManifest index {index} out of range")))
    }

    /// `ChannelGroupManifest& add(const ChannelGroupManifest&)` — appends a
    /// copy and returns a reference to it for further modification.
    pub fn add(&mut self, table: ChannelGroupManifest) -> &mut ChannelGroupManifest {
        self.manifest.push(table);
        self.manifest.last_mut().expect("just pushed")
    }

    /// `ChannelGroupManifest& add(const set<string>&)` — an empty table
    /// definition for a group of channels.
    pub fn add_group(&mut self, channels: BTreeSet<String>) -> &mut ChannelGroupManifest {
        let mut m = ChannelGroupManifest::new();
        m.set_channels(channels);
        self.add(m)
    }

    /// `ChannelGroupManifest& add(const string&)` — an empty table
    /// definition for a single channel.
    pub fn add_channel(&mut self, channel: &str) -> &mut ChannelGroupManifest {
        let mut m = ChannelGroupManifest::new();
        m.set_channel(channel);
        self.add(m)
    }

    /// `bool merge(const IDManifest&)` — port of `ImfIDManifest.cpp:1063-1103`.
    ///
    /// For each of `other`'s channel-group manifests: if `self` has no
    /// manifest with the same `channels` set, `other`'s is appended
    /// unchanged. If one does exist with the same channels:
    /// * a `components` mismatch is a conflict (that group is left alone);
    /// * otherwise, entries `other` has that `self` lacks are copied in, and
    ///   an ID present in both with DIFFERENT text is a conflict (the
    ///   existing entry in `self` is left unchanged either way — C++ never
    ///   overwrites on conflict, it just records that one occurred).
    ///
    /// Returns `true` if any conflict occurred (C++'s `bool conflict`).
    pub fn merge(&mut self, other: &IDManifest) -> bool {
        let mut conflict = false;
        for other_group in &other.manifest {
            let mut merged = false;
            for this_group in &mut self.manifest {
                if this_group.channels() != other_group.channels() {
                    continue;
                }
                merged = true;
                if this_group.components() != other_group.components() {
                    conflict = true;
                    continue;
                }
                for (id, text) in other_group.iter() {
                    match this_group.find(id) {
                        None => {
                            // `insert` here cannot fail: components already
                            // matched above, so the length check always passes.
                            let _ = this_group.insert(id, text.to_vec());
                        }
                        Some(ours) if ours != text => conflict = true,
                        Some(_) => {}
                    }
                }
            }
            if !merged {
                self.manifest.push(other_group.clone());
            }
        }
        conflict
    }

    pub(super) fn groups(&self) -> &[ChannelGroupManifest] {
        &self.manifest
    }

    pub(super) fn groups_mut(&mut self) -> &mut Vec<ChannelGroupManifest> {
        &mut self.manifest
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_group_defaults() {
        let m = ChannelGroupManifest::new();
        assert_eq!(m.hash_scheme(), UNKNOWN);
        assert_eq!(m.encoding_scheme(), UNKNOWN);
        assert_eq!(m.lifetime(), IdLifetime::Stable);
        assert!(m.is_empty());
    }

    #[test]
    fn set_components_rejects_length_change_once_populated() {
        let mut m = ChannelGroupManifest::new();
        m.set_components(vec!["a".into()]).unwrap();
        m.insert(1, vec!["x".into()]).unwrap();
        assert!(m.set_components(vec!["a".into(), "b".into()]).is_err());
        // Same length is fine.
        m.set_components(vec!["c".into()]).unwrap();
    }

    #[test]
    fn insert_validates_component_count() {
        let mut m = ChannelGroupManifest::new();
        m.set_components(vec!["a".into(), "b".into()]).unwrap();
        assert!(m.insert(1, vec!["only-one".into()]).is_err());
        m.insert(1, vec!["x".into(), "y".into()]).unwrap();
        assert_eq!(m.find(1), Some(&["x".to_string(), "y".to_string()][..]));
    }

    #[test]
    fn insert_single_requires_exactly_one_component() {
        let mut m = ChannelGroupManifest::new();
        m.set_component("only").unwrap();
        m.insert_single(5, "hello").unwrap();
        assert_eq!(m.find(5), Some(&["hello".to_string()][..]));

        let mut multi = ChannelGroupManifest::new();
        multi.set_components(vec!["a".into(), "b".into()]).unwrap();
        assert!(multi.insert_single(1, "x").is_err());
    }

    #[test]
    fn insert_hashed_uses_the_configured_scheme() {
        let mut m = ChannelGroupManifest::new();
        m.set_component("name").unwrap();
        m.set_hash_scheme(MURMURHASH3_32);
        let id = m.insert_hashed_single("hello").unwrap();
        assert_eq!(m.find(id), Some(&["hello".to_string()][..]));

        let mut unknown_scheme = ChannelGroupManifest::new();
        unknown_scheme.set_component("name").unwrap();
        assert!(unknown_scheme.insert_hashed_single("hello").is_err());
    }

    #[test]
    fn entry_creates_empty_vec_if_missing() {
        let mut m = ChannelGroupManifest::new();
        assert!(m.entry(42).is_empty());
        m.entry(42).push("x".into());
        assert_eq!(m.find(42), Some(&["x".to_string()][..]));
    }

    #[test]
    fn erase_removes_entry() {
        let mut m = ChannelGroupManifest::new();
        m.set_component("c").unwrap();
        m.insert_single(1, "a").unwrap();
        m.erase(1);
        assert_eq!(m.find(1), None);
    }

    /// C++'s `operator==` ignores `channels` and `encoding_scheme` — two
    /// groups differing ONLY in those fields must still compare equal.
    #[test]
    fn eq_ignores_channels_and_encoding_scheme() {
        let mut a = ChannelGroupManifest::new();
        a.set_channel("R");
        a.set_encoding_scheme("id");
        a.set_component("c").unwrap();
        a.insert_single(1, "x").unwrap();

        let mut b = a.clone();
        b.set_channel("G");
        b.set_encoding_scheme("id2");
        assert_eq!(a, b);

        let mut c = a.clone();
        c.insert_single(2, "y").unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn idmanifest_add_and_find_channel() {
        let mut d = IDManifest::new();
        d.add_channel("R");
        d.add_group(BTreeSet::from(["G".to_string(), "B".to_string()]));
        assert_eq!(d.find_channel("R"), Some(0));
        assert_eq!(d.find_channel("G"), Some(1));
        assert_eq!(d.find_channel("B"), Some(1));
        assert_eq!(d.find_channel("Z"), None);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn idmanifest_get_out_of_range_errors() {
        let d = IDManifest::new();
        assert!(d.get(0).is_err());
    }

    #[test]
    fn merge_appends_disjoint_groups() {
        let mut a = IDManifest::new();
        a.add_channel("R");

        let mut b = IDManifest::new();
        b.add_channel("G");

        let conflict = a.merge(&b);
        assert!(!conflict);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn merge_combines_entries_for_the_same_channel_group() {
        let mut a = IDManifest::new();
        {
            let g = a.add_channel("R");
            g.set_component("c").unwrap();
            g.insert_single(1, "one").unwrap();
        }

        let mut b = IDManifest::new();
        {
            let g = b.add_channel("R");
            g.set_component("c").unwrap();
            g.insert_single(2, "two").unwrap();
        }

        let conflict = a.merge(&b);
        assert!(!conflict);
        assert_eq!(a.len(), 1);
        assert_eq!(a.get(0).unwrap().find(1), Some(&["one".to_string()][..]));
        assert_eq!(a.get(0).unwrap().find(2), Some(&["two".to_string()][..]));
    }

    #[test]
    fn merge_reports_conflict_on_differing_components() {
        let mut a = IDManifest::new();
        a.add_channel("R").set_component("c").unwrap();

        let mut b = IDManifest::new();
        b.add_channel("R").set_components(vec!["x".into(), "y".into()]).unwrap();

        assert!(a.merge(&b));
    }

    #[test]
    fn merge_reports_conflict_on_differing_text_for_same_id() {
        let mut a = IDManifest::new();
        {
            let g = a.add_channel("R");
            g.set_component("c").unwrap();
            g.insert_single(1, "one").unwrap();
        }

        let mut b = IDManifest::new();
        {
            let g = b.add_channel("R");
            g.set_component("c").unwrap();
            g.insert_single(1, "ONE-DIFFERENT").unwrap();
        }

        assert!(a.merge(&b));
        // The original entry is kept, not overwritten.
        assert_eq!(a.get(0).unwrap().find(1), Some(&["one".to_string()][..]));
    }
}
