//! Cryptomatte JSON manifest (inline header or sidecar file).

use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};
use crate::id::hex_to_float_id;

/// Name ↔ float-ID maps from a Cryptomatte manifest.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    /// Object/material name → float ID.
    pub by_name: HashMap<String, f32>,
    /// Float ID bit pattern → name. On a hash collision the lexicographically
    /// smallest name wins, so the answer never depends on map iteration order.
    pub by_id_bits: HashMap<u32, String>,
}

impl Manifest {
    /// Build from a JSON object `{ "name": "aabbccdd", ... }` (8 hex digits).
    pub fn from_json_str(json: &str) -> Result<Self> {
        let raw: HashMap<String, String> = serde_json::from_str(json)?;
        Self::from_name_hex_map(raw)
    }

    /// Build from already-parsed name → hex map.
    pub fn from_name_hex_map(raw: HashMap<String, String>) -> Result<Self> {
        let mut by_name = HashMap::with_capacity(raw.len());
        for (name, hex) in raw {
            let id = hex_to_float_id(&hex).ok_or_else(|| {
                Error::Manifest(format!(
                    "invalid id `{hex}` for `{name}` (want exactly 8 hex digits)"
                ))
            })?;
            by_name.insert(name, id);
        }
        Ok(Self::from_by_name(by_name))
    }

    /// Build both maps from name → ID, resolving ID collisions
    /// deterministically (smallest name wins, with a warning).
    pub fn from_by_name(by_name: HashMap<String, f32>) -> Self {
        let mut names: Vec<(&String, &f32)> = by_name.iter().collect();
        names.sort_by(|a, b| a.0.cmp(b.0));
        let mut by_id_bits: HashMap<u32, String> = HashMap::with_capacity(names.len());
        for (name, id) in names {
            match by_id_bits.get(&id.to_bits()) {
                Some(kept) => log::warn!(
                    "cryptomatte manifest: `{name}` and `{kept}` share id {:08x}; \
                     `{kept}` names it",
                    id.to_bits()
                ),
                None => {
                    by_id_bits.insert(id.to_bits(), name.clone());
                }
            }
        }
        Self { by_name, by_id_bits }
    }

    /// Add one name, keeping the collision rule of [`Manifest::from_by_name`].
    pub fn insert(&mut self, name: &str, id: f32) {
        self.by_name.insert(name.to_string(), id);
        let slot = self.by_id_bits.entry(id.to_bits()).or_insert_with(|| name.to_string());
        if name < slot.as_str() {
            *slot = name.to_string();
        }
    }

    /// Float ID for `name`: the manifest entry, else the hashed name.
    ///
    /// SSOT for name → ID resolution. The manifest is optional and, per spec
    /// 1.2.0, its IDs are exactly the name hashes — the reference decoder does
    /// not consult it at all (Nuke `cryptomatte_utilities.py`: `name_to_ID` is
    /// `mm3hash_float(name)`), so a missing entry is not an error.
    pub fn id_for_name(&self, name: &str) -> f32 {
        self.by_name
            .get(name)
            .copied()
            .unwrap_or_else(|| crate::id::name_to_float_id(name))
    }

    /// [`Manifest::id_for_name`] over several names, in order.
    pub fn ids_for_names<'a, I>(&self, names: I) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a str>,
    {
        names.into_iter().map(|n| self.id_for_name(n)).collect()
    }

    /// Look up a name for a float ID (bit-exact).
    pub fn name_for_id(&self, id: f32) -> Option<&str> {
        self.by_id_bits.get(&id.to_bits()).map(|s| s.as_str())
    }

    /// Serialize to compact JSON `{ "name": "aabbccdd", ... }` with sorted keys.
    pub fn to_json_str(&self) -> Result<String> {
        use crate::id::float_id_to_hex;
        use std::collections::BTreeMap;
        let map: BTreeMap<&str, String> = self
            .by_name
            .iter()
            .map(|(n, id)| (n.as_str(), float_id_to_hex(*id)))
            .collect();
        Ok(serde_json::to_string(&map)?)
    }
}

/// Check a sidecar path from a file header (spec 1.2.0 p.3): "always relative
/// to the directory the EXR file is found in", forward slashes only.
///
/// Rejects backslashes (as Nuke does), absolute or rooted paths and drive
/// prefixes (`Path::join` would replace the EXR's directory with them), and
/// any `..` component (the spec forbids a leading `../`; one further in
/// escapes the directory just the same). A `./` component is tolerated on
/// read: it cannot leave the directory. Writers use the stricter
/// [`validate_sidecar_for_write`].
pub fn validate_sidecar(sidecar_rel: &str) -> Result<()> {
    let bad = |why: &str| Err(Error::Manifest(format!("invalid sidecar path `{sidecar_rel}`: {why}")));
    if sidecar_rel.is_empty() {
        return bad("empty");
    }
    if sidecar_rel.contains('\\') {
        return bad("backslashes are not allowed");
    }
    for comp in Path::new(sidecar_rel).components() {
        match comp {
            Component::Prefix(_) | Component::RootDir => {
                return bad("must be relative to the EXR's directory");
            }
            Component::ParentDir => return bad("`..` is not allowed"),
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    Ok(())
}

/// [`validate_sidecar`] plus the rest of spec 1.2.0 p.3 that a writer must
/// honour: no `./` component ("should not start with "../" or "./"") and a
/// `.json` extension ("The file should have the file extension ".json"").
pub fn validate_sidecar_for_write(sidecar_rel: &str) -> Result<()> {
    validate_sidecar(sidecar_rel)?;
    let bad = |why: &str| Err(Error::Manifest(format!("invalid sidecar path `{sidecar_rel}`: {why}")));
    // By segment, not `Path::components()`: that drops a `.` in mid-path.
    if sidecar_rel.split('/').any(|seg| seg == ".") {
        return bad("`./` is not allowed");
    }
    if Path::new(sidecar_rel).extension().and_then(|e| e.to_str()) != Some("json") {
        return bad("the extension must be `.json`");
    }
    Ok(())
}

/// Sidecar location for an EXR: `sidecar_rel` (checked by [`validate_sidecar`])
/// joined to the EXR's directory.
pub fn resolve_sidecar(exr_path: &Path, sidecar_rel: &str) -> Result<PathBuf> {
    validate_sidecar(sidecar_rel)?;
    let parent = exr_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent.join(sidecar_rel))
}

/// Load manifest: prefer sidecar `manif_file` when set, else inline JSON string.
pub fn load_manifest(
    exr_path: &Path,
    manif_file: Option<&str>,
    inline_json: Option<&str>,
) -> Result<Manifest> {
    if let Some(rel) = manif_file.filter(|s| !s.is_empty()) {
        let path = resolve_sidecar(exr_path, rel)?;
        let text = fs::read_to_string(&path).map_err(|e| {
            Error::Manifest(format!("cannot read sidecar {}: {e}", path.display()))
        })?;
        return Manifest::from_json_str(&text);
    }
    if let Some(json) = inline_json.filter(|s| !s.is_empty()) {
        return Manifest::from_json_str(json);
    }
    Ok(Manifest::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{float_id_to_hex, name_to_float_id};

    #[test]
    fn json_roundtrip_bunny() {
        let hex = float_id_to_hex(name_to_float_id("bunny"));
        let json = format!(r#"{{"bunny":"{hex}"}}"#);
        let m = Manifest::from_json_str(&json).unwrap();
        assert_eq!(
            m.by_name["bunny"].to_bits(),
            name_to_float_id("bunny").to_bits()
        );
        assert_eq!(m.name_for_id(name_to_float_id("bunny")), Some("bunny"));
        let round = Manifest::from_json_str(&m.to_json_str().unwrap()).unwrap();
        assert_eq!(
            round.by_name["bunny"].to_bits(),
            name_to_float_id("bunny").to_bits()
        );
    }

    #[test]
    fn sidecar_rejects_backslash() {
        let err = resolve_sidecar(Path::new("a/b.exr"), "dir\\m.json").unwrap_err();
        assert!(matches!(err, Error::Manifest(_)));
    }

    /// The header cannot point outside the EXR's directory.
    #[test]
    fn sidecar_paths_stay_under_the_exr_directory() {
        let exr = Path::new("renders/shot.exr");
        assert_eq!(
            resolve_sidecar(exr, "m/cryptoObject.json").unwrap(),
            Path::new("renders").join("m/cryptoObject.json")
        );
        assert!(resolve_sidecar(exr, "./m.json").is_ok(), "`./` is harmless on read");
        for bad in ["", "../m.json", "m/../../x.json", "/etc/m.json", "//host/share/m.json"] {
            assert!(resolve_sidecar(exr, bad).is_err(), "{bad:?} must be rejected");
        }
        // A drive prefix is only a prefix where the OS has drives; elsewhere
        // `C:` is an ordinary directory name under the EXR's directory.
        #[cfg(windows)]
        assert!(resolve_sidecar(exr, "C:/m.json").is_err());
    }

    #[test]
    fn sidecar_write_rules_are_stricter() {
        assert!(validate_sidecar_for_write("m/cryptoObject.json").is_ok());
        for bad in ["./m.json", "m/./x.json", "m.txt", "m", "../m.json"] {
            assert!(validate_sidecar_for_write(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    /// A collision resolves to the same name whatever the input order.
    #[test]
    fn id_collision_is_deterministic() {
        let id = name_to_float_id("bunny");
        let hex = float_id_to_hex(id);
        for json in [
            format!(r#"{{"zeta":"{hex}","alpha":"{hex}"}}"#),
            format!(r#"{{"alpha":"{hex}","zeta":"{hex}"}}"#),
        ] {
            let m = Manifest::from_json_str(&json).unwrap();
            assert_eq!(m.name_for_id(id), Some("alpha"));
        }
        let mut m = Manifest::default();
        m.insert("zeta", id);
        m.insert("alpha", id);
        assert_eq!(m.name_for_id(id), Some("alpha"));
        m.insert("omega", id);
        assert_eq!(m.name_for_id(id), Some("alpha"));
    }
}
