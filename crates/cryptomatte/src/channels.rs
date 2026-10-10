//! Ranked Cryptomatte EXR channel layout — SSOT for metadata + matte extract.
//!
//! Dot-split uses [`exr_core::attr::layer_prefix`] / [`channel_suffix`] (shared
//! with the rest of exr-rs). Spec packing per numbered layer `NameNN`:
//! - `.R` / `.r` / `.red` — ID 0
//! - `.G` / `.g` / `.green` — coverage 0
//! - `.B` / `.b` / `.blue` — ID 1
//! - `.A` / `.a` / `.alpha` — coverage 1
//!
//! Preview layers (`Name` with no trailing digits) are ignored (deprecated).
//!
//! The rank index is exactly two digits (`Name00` .. `Name99`): the spec's
//! layout table numbers layers `{typename}00`, `{typename}01`, ... and every
//! reference writer formats it `{0:02d}` (Nuke `cryptomatte_utilities.py`).
//! Accepting any digit run would let type `A` claim type `A1`'s `A100.R`.

use std::collections::BTreeMap;

use exr_core::attr::{channel_suffix, layer_prefix};

/// One RGBA component of a ranked Cryptomatte layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    /// Red — first ID slot.
    R,
    /// Green — coverage for first ID.
    G,
    /// Blue — second ID slot.
    B,
    /// Alpha — coverage for second ID.
    A,
}

impl Component {
    /// Parse a channel suffix (case-insensitive; Nuke + OpenEXR spellings).
    pub fn parse(suffix: &str) -> Option<Self> {
        match suffix.to_ascii_lowercase().as_str() {
            "r" | "red" => Some(Self::R),
            "g" | "green" => Some(Self::G),
            "b" | "blue" => Some(Self::B),
            "a" | "alpha" => Some(Self::A),
            _ => None,
        }
    }

}

/// How a type spells its ranked-channel suffixes.
///
/// The spec's table writes `.r .g .b .a`, yet real files use all three families:
/// Psyop's `bunny_*` samples `.red .green .blue .alpha`, the `cornellBox` sample
/// `.R .G .B .A`. A reader accepts every one ([`Component::parse`]); a writer that
/// re-encodes a type in place keeps the one the file already uses
/// ([`Spelling::detect`]) so the other channels and tools that match names
/// literally are not surprised. A type written from scratch gets the default (`.R`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Spelling {
    /// `.R .G .B .A` (default; OpenEXR's own RGBA names).
    #[default]
    Upper,
    /// `.r .g .b .a` (the spec's table).
    Lower,
    /// `.red .green .blue .alpha` (Nuke).
    Long,
}

impl Spelling {
    /// Channel suffix of `comp` in this spelling.
    pub fn suffix(self, comp: Component) -> &'static str {
        match (self, comp) {
            (Self::Upper, Component::R) => "R",
            (Self::Upper, Component::G) => "G",
            (Self::Upper, Component::B) => "B",
            (Self::Upper, Component::A) => "A",
            (Self::Lower, Component::R) => "r",
            (Self::Lower, Component::G) => "g",
            (Self::Lower, Component::B) => "b",
            (Self::Lower, Component::A) => "a",
            (Self::Long, Component::R) => "red",
            (Self::Long, Component::G) => "green",
            (Self::Long, Component::B) => "blue",
            (Self::Long, Component::A) => "alpha",
        }
    }

    /// The spelling of the first ranked channel of `crypto_name` among `names`,
    /// or `None` when it has none. A file mixing spellings keeps its first.
    pub fn detect<'a>(crypto_name: &str, names: impl IntoIterator<Item = &'a str>) -> Option<Self> {
        names.into_iter().find_map(|full| {
            ranked_channel_of(crypto_name, full)?;
            let s = channel_suffix(full);
            Some(match (s.len() > 1, s.starts_with(|c: char| c.is_ascii_uppercase())) {
                (true, _) => Self::Long,
                (false, true) => Self::Upper,
                (false, false) => Self::Lower,
            })
        })
    }
}

/// Number of rank-index digits in a ranked layer name (`Name00`).
pub const RANK_DIGITS: usize = 2;

/// Most ranked layers a type can have: two-digit indices `00..=99`.
pub const MAX_RANKED_LAYERS: usize = 100;

/// Ranked layer base name: `{crypto_name}{index:02}` (e.g. `CryptoObject00`).
///
/// `index` must be below [`MAX_RANKED_LAYERS`]; the encoder checks that before
/// naming any layer.
#[inline]
pub fn ranked_base(crypto_name: &str, index: usize) -> String {
    format!("{crypto_name}{index:02}")
}

/// Rank index of a ranked layer base (`CryptoObject07` -> 7), if `base` is
/// `crypto_name` followed by exactly [`RANK_DIGITS`] ASCII digits.
pub fn rank_index(crypto_name: &str, base: &str) -> Option<usize> {
    let digits = base.strip_prefix(crypto_name)?;
    if digits.len() != RANK_DIGITS || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Full channel name for a ranked base + component in `spelling`.
#[inline]
pub fn ranked_channel_name(base: &str, comp: Component, spelling: Spelling) -> String {
    format!("{base}.{}", spelling.suffix(comp))
}

/// One numbered ranked layer (`CryptoObject00`) with up to two ID/coverage pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankedLayer {
    /// Full layer base without component, e.g. `uCryptoObject00`.
    pub base: String,
    /// Full channel name for ID slot 0 (`.R`), if present.
    pub id0: Option<String>,
    /// Full channel name for coverage 0 (`.G`), if present.
    pub cov0: Option<String>,
    /// Full channel name for ID slot 1 (`.B`), if present.
    pub id1: Option<String>,
    /// Full channel name for coverage 1 (`.A`), if present.
    pub cov1: Option<String>,
}

impl RankedLayer {
    /// ID/coverage channel name pairs present in this layer (slot 0 then 1).
    pub fn pairs(&self) -> Vec<(&str, &str)> {
        let mut out = Vec::with_capacity(2);
        if let (Some(id), Some(cov)) = (self.id0.as_deref(), self.cov0.as_deref()) {
            out.push((id, cov));
        }
        if let (Some(id), Some(cov)) = (self.id1.as_deref(), self.cov1.as_deref()) {
            out.push((id, cov));
        }
        out
    }
}

/// Split a full channel name into `(ranked base, component)` when it belongs to
/// the ranked layers of `crypto_name` (`CryptoObject00.R` → `CryptoObject00`, R).
///
/// SSOT for "does this channel belong to this Cryptomatte type": used both to
/// collect layers ([`ranked_layers_for`]) and to drop a type's stale channels
/// when re-encoding it in place.
pub fn ranked_channel_of<'a>(crypto_name: &str, full: &'a str) -> Option<(&'a str, Component)> {
    let base = layer_prefix(full);
    // Preview layers, other types and any non-two-digit suffix are not ranks.
    rank_index(crypto_name, base)?;
    Component::parse(channel_suffix(full)).map(|comp| (base, comp))
}

/// Collect sorted ranked layers for a Cryptomatte type name (e.g. `uCryptoObject`).
///
/// `channel_names` is any iterator of full EXR channel names (`layer.comp`).
pub fn ranked_layers_for<'a, I>(crypto_name: &str, channel_names: I) -> Vec<RankedLayer>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut by_base: BTreeMap<String, RankedLayer> = BTreeMap::new();

    for full in channel_names {
        let Some((base, comp)) = ranked_channel_of(crypto_name, full) else {
            continue;
        };

        let entry = by_base.entry(base.to_string()).or_insert_with(|| RankedLayer {
            base: base.to_string(),
            id0: None,
            cov0: None,
            id1: None,
            cov1: None,
        });
        match comp {
            Component::R => entry.id0 = Some(full.to_string()),
            Component::G => entry.cov0 = Some(full.to_string()),
            Component::B => entry.id1 = Some(full.to_string()),
            Component::A => entry.cov1 = Some(full.to_string()),
        }
    }

    by_base.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_ranked_and_skips_preview() {
        let names = [
            "uCryptoObject.R",
            "uCryptoObject00.R",
            "uCryptoObject00.G",
            "uCryptoObject00.B",
            "uCryptoObject00.A",
            "uCryptoObject01.red",
            "uCryptoObject01.green",
            "beauty.R",
        ];
        let layers = ranked_layers_for("uCryptoObject", names);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].base, "uCryptoObject00");
        assert_eq!(layers[0].pairs().len(), 2);
        assert_eq!(layers[1].base, "uCryptoObject01");
        assert_eq!(layers[1].pairs().len(), 1);
    }

    #[test]
    fn spelling_follows_the_files_own_channels() {
        let names = ["uCryptoObject.red", "uCryptoObject00.red", "beauty.R"];
        assert_eq!(Spelling::detect("uCryptoObject", names), Some(Spelling::Long));
        assert_eq!(Spelling::detect("uCryptoObject", ["X00.R"]), None);
        assert_eq!(Spelling::detect("X", ["X00.R"]), Some(Spelling::Upper));
        assert_eq!(Spelling::detect("X", ["X00.g"]), Some(Spelling::Lower));
        for sp in [Spelling::Upper, Spelling::Lower, Spelling::Long] {
            for c in [Component::R, Component::G, Component::B, Component::A] {
                assert_eq!(Component::parse(sp.suffix(c)), Some(c));
            }
        }
    }

    /// C14: type `A` must not claim type `A1`'s `A100.*` (rank index is exactly
    /// two digits), nor one- or three-digit suffixes.
    #[test]
    fn rank_suffix_is_exactly_two_digits() {
        let names = ["A00.R", "A00.G", "A100.R", "A100.G", "A1.R", "A001.R", "A1x.R"];
        let a: Vec<String> = ranked_layers_for("A", names).into_iter().map(|l| l.base).collect();
        assert_eq!(a, ["A00"]);
        let a1: Vec<String> = ranked_layers_for("A1", names).into_iter().map(|l| l.base).collect();
        assert_eq!(a1, ["A100"]);
        assert_eq!(rank_index("A", "A07"), Some(7));
        assert_eq!(rank_index("A", "A7"), None);
        assert_eq!(rank_index("A", "A007"), None);
        assert_eq!(rank_index("A", "B07"), None);
    }
}
