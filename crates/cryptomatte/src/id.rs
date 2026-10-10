//! Cryptomatte ID generation: MurmurHash3_x86_32 → float bits.
//!
//! Spec 1.2.0 / Psyop `mm3hash_float`: hash UTF-8 name bytes with seed 0, then
//! bit-cast to `f32` while clamping the IEEE exponent out of `{0, 255}`.

use murmur3::x86_32;

/// MurmurHash3_x86_32 of `name` (UTF-8 bytes), Cryptomatte seed `0`.
#[inline]
pub fn name_to_hash32(name: &str) -> u32 {
    x86_32::hash(name.as_bytes(), 0)
}

/// Convert a 32-bit Murmur digest to the Cryptomatte float ID.
///
/// If all exponent bits are 0 or 255, flip bit 23 so the value is a finite
/// normal/subnormal-safe float for compositors.
#[inline]
pub fn hash32_to_float_id(mut hash_32: u32) -> f32 {
    let exp = (hash_32 >> 23) & 255;
    if exp == 0 || exp == 255 {
        hash_32 ^= 1 << 23;
    }
    f32::from_bits(hash_32)
}

/// Name → Cryptomatte float ID (hash + exponent clamp).
#[inline]
pub fn name_to_float_id(name: &str) -> f32 {
    hash32_to_float_id(name_to_hash32(name))
}

/// Float ID → 8 lowercase hex digits (little-endian bit pattern), as in manifests.
#[inline]
pub fn float_id_to_hex(id: f32) -> String {
    format!("{:08x}", id.to_bits())
}

/// Parse an 8-digit hex manifest value into a float ID.
///
/// Exactly 8 ASCII hex digits, nothing else: spec 1.2.0 p.4 says manifest
/// hashes "are always 8 digits of hexadecimal". `from_str_radix` alone would
/// also take 1-7 digits, a leading `+`, and (with the old `trim`) padding.
pub fn hex_to_float_id(hex: &str) -> Option<f32> {
    if hex.len() != 8 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let bits = u32::from_str_radix(hex, 16).ok()?;
    Some(f32::from_bits(bits))
}

/// Layer metadata key suffix: first 7 hex chars of `float_id_to_hex(name_to_float_id(layer))`.
/// Matches Nuke `layer_hash` (`id_to_hex(...)[:-1]`).
#[inline]
pub fn layer_metadata_id(layer_name: &str) -> String {
    let mut h = float_id_to_hex(name_to_float_id(layer_name));
    h.pop();
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Psyop `CryptoHashing.mm3hash_float_values` (after f32 narrowing).
    #[test]
    #[allow(clippy::excessive_precision)] // goldens copied from Psyop reference literals
    fn psyop_float_goldens() {
        let cases: &[(&str, f32)] = &[
            ("hello", 6.0705627102400005616e-17),
            ("cube", -4.08461912519e15),
            ("sphere", 2.79018604383e15),
            ("plane", 3.66557617593e-11),
            ("равнина", -1.3192631212399999468e-25),
            ("mädchen", 6.2361298211599995797e25),
        ];
        for &(name, expected) in cases {
            let got = name_to_float_id(name);
            assert_eq!(got.to_bits(), expected.to_bits(), "name={name:?}");
        }
    }

    #[test]
    fn psyop_hex_goldens() {
        let cases: &[(&str, &str)] = &[
            ("hello", "248bfa47"),
            ("cube", "d9682f08"),
            ("sphere", "591e9a8d"),
            ("plane", "2e2136b8"),
            ("равнина", "9623511e"),
            ("mädchen", "6a4e5613"),
            ("bunny", "13851a76"),
        ];
        for &(name, hex) in cases {
            assert_eq!(float_id_to_hex(name_to_float_id(name)), hex, "name={name:?}");
            assert_eq!(
                hex_to_float_id(hex).unwrap().to_bits(),
                name_to_float_id(name).to_bits()
            );
        }
    }

    #[test]
    fn hex_roundtrip() {
        let id = name_to_float_id("bunny");
        let hex = float_id_to_hex(id);
        assert_eq!(hex.len(), 8);
        assert_eq!(hex_to_float_id(&hex).unwrap().to_bits(), id.to_bits());
    }

    /// C13: only exactly 8 hex digits parse.
    #[test]
    fn hex_parse_is_strict() {
        assert!(hex_to_float_id("13851a76").is_some());
        assert!(hex_to_float_id("13851A76").is_some());
        for bad in ["3851a76", "013851a76", "+3851a76", " 13851a76", "13851a7g", ""] {
            assert!(hex_to_float_id(bad).is_none(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn layer_metadata_id_is_7_hex() {
        let id = layer_metadata_id("cryptoObject");
        assert_eq!(id.len(), 7);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
