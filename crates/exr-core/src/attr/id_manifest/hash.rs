//! Port of the hashing half of `ImfIDManifest.cpp` — `IDManifest::MurmurHash32`
//! / `MurmurHash64` and the private `catString` helper.
//!
//! Uses the `murmur3` crate (already an approved git dependency, used the
//! same way by `cryptomatte`) rather than re-implementing MurmurHash3 —
//! MurmurHash3 is not itself part of the OpenEXR *format*; it's a
//! general-purpose hash OpenEXR happens to call, unlike `half`/zlib/OpenEXR
//! itself, which this repo's "port everything ourselves" rule is about.
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.cpp>
//! (the anonymous-namespace `MurmurHash3_x86_32`/`MurmurHash3_x64_128` +
//! `catString`, and `IDManifest::MurmurHash32`/`MurmurHash64`).

/// `catString`: join text components with `;` — `ImfIDManifest.cpp:1682-1690`.
fn cat_string(text: &[String]) -> String {
    text.join(";")
}

/// `IDManifest::MurmurHash32(const std::string&)` — MurmurHash3_x86_32, seed 0.
pub fn murmur_hash32_str(s: &str) -> u32 {
    murmur3::x86_32::hash(s.as_bytes(), 0)
}

/// `IDManifest::MurmurHash64(const std::string&)` — the low 64 bits
/// (`out[0]`) of MurmurHash3_x64_128, seed 0.
pub fn murmur_hash64_str(s: &str) -> u64 {
    murmur3::x64_128::hash(s.as_bytes(), Some(0), None).0
}

/// `IDManifest::MurmurHash32(const vector<string>&)` — `0` for an empty
/// vector (matching C++'s early return), else hashes `catString(text)`.
pub fn murmur_hash32(text: &[String]) -> u32 {
    if text.is_empty() {
        return 0;
    }
    murmur_hash32_str(&cat_string(text))
}

/// `IDManifest::MurmurHash64(const vector<string>&)` — `0` for an empty
/// vector, else hashes `catString(text)`.
pub fn murmur_hash64(text: &[String]) -> u64 {
    if text.is_empty() {
        return 0;
    }
    murmur_hash64_str(&cat_string(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vector_hashes_to_zero() {
        assert_eq!(murmur_hash32(&[]), 0);
        assert_eq!(murmur_hash64(&[]), 0);
    }

    #[test]
    fn cat_string_joins_with_semicolon() {
        assert_eq!(cat_string(&["a".into(), "b".into(), "c".into()]), "a;b;c");
        assert_eq!(cat_string(&["only".into()]), "only");
    }

    #[test]
    fn single_and_multi_component_hash_agree_via_cat_string() {
        let single = vec!["hello".to_string()];
        assert_eq!(murmur_hash32(&single), murmur_hash32_str("hello"));
        assert_eq!(murmur_hash64(&single), murmur_hash64_str("hello"));

        let multi = vec!["a".to_string(), "b".to_string()];
        assert_eq!(murmur_hash32(&multi), murmur_hash32_str("a;b"));
    }

    /// Deterministic, reproducible: hashing is stable across calls (a smoke
    /// test that the crate's algorithm is being exercised at all, not that a
    /// specific digest is "correct" per se — that's covered by the crate's
    /// own tests plus its use in cryptomatte).
    #[test]
    fn hashing_is_deterministic() {
        assert_eq!(murmur_hash32_str("repeat-me"), murmur_hash32_str("repeat-me"));
        assert_eq!(murmur_hash64_str("repeat-me"), murmur_hash64_str("repeat-me"));
        assert_ne!(murmur_hash32_str("a"), murmur_hash32_str("b"));
    }
}
