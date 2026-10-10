//! Port of `IDManifest::init` (parse) and `IDManifest::serialize` (write) —
//! the binary format of the DECOMPRESSED manifest payload.
//!
//! Both directions share the same prefix-compressed string table + RLE index
//! remapping scheme documented inline in `ImfIDManifest.cpp`; see the
//! rustdoc on [`serialize`] for the shape.
//!
//! Port target (pinned tag v3.4.15):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.cpp#L349-L586>
//! (`init`) and
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfIDManifest.cpp#L643-L1030>
//! (`serialize`).

use std::collections::BTreeMap;

use crate::attr::id_manifest::manifest::{ChannelGroupManifest, IDManifest, IdLifetime};
use crate::attr::id_manifest::varint::{
    read_byte_string_list, read_i32, read_pascal_string, read_string_list, read_u8,
    write_byte_string_list, write_i32, write_pascal_string, write_string_list, write_u8,
    write_varint, read_varint, varint_size,
};
use crate::error::{ExrError, Result};

/// Serialize an [`IDManifest`] to the decompressed binary payload —
/// port of `IDManifest::serialize` (`ImfIDManifest.cpp:643-1030`).
///
/// Unlike C++ (which must pre-compute the exact output size before writing
/// into a fixed `vector<char>`, since it writes through raw pointer
/// arithmetic), this builds the output with a growing `Vec<u8>`, which drops
/// the size-precomputation pass entirely (`getStringListSize`/`getStringSize`
/// have no Rust equivalent — the size was an allocation optimisation, not a
/// format requirement) while writing the exact same bytes in the exact same
/// order.
///
/// # Format (decompressed payload)
/// 1. version (`i32`, always `0`)
/// 2. the prefix-compressed table of every distinct text string used
///    anywhere in the manifest, sorted by descending frequency (ties broken
///    by string value) then grouped by the varint width its compact index
///    needs — a byte-string list ([`write_byte_string_list`])
/// 3. an RLE-encoded remapping table: `count` then `count` `(first, last)`
///    `i32` pairs
/// 4. `count` of channel-group manifests, then for each:
///    `channels` (string list) · `components` (string list) · `lifetime`
///    (`u8`) · `hashScheme` (Pascal string) · `encodingScheme` (Pascal
///    string) · ID storage scheme (`u8`: `0` = 8 bytes/ID, `1` = 4 bytes/ID,
///    `2` = variable) · entry count (`i32`) · the entries themselves (each:
///    a **delta**-encoded ID in the chosen scheme, then one varint per
///    component giving that string's compact index)
pub fn serialize(manifest: &IDManifest) -> Result<Vec<u8>> {
    // --- Step 1: count every text string's occurrences, across every group.
    let mut string_set: BTreeMap<String, i32> = BTreeMap::new();
    for group in manifest.groups() {
        for text in group.table().values() {
            for s in text {
                *string_set.entry(s.clone()).or_insert(0) += 1;
            }
        }
    }

    // --- Step 2: prefix-compress the string table (sorted key order, same
    // as the C++ `std::map<string,int>` iteration) and record, for each
    // ORIGINAL index, `(sortKey, index)` so the next steps can reorder by
    // frequency without losing which original string each entry was.
    let mut prefixed: Vec<Vec<u8>> = Vec::with_capacity(string_set.len());
    let mut sorted_indices: Vec<(i32, i32)> = Vec::with_capacity(string_set.len());
    // "Repurpose stringSet" (C++ comment) to map string -> its original index,
    // needed later when writing each table entry's component strings.
    let mut string_to_index: BTreeMap<String, i32> = BTreeMap::new();

    let mut prev: &str = "";
    for (idx, (key, &count)) in string_set.iter().enumerate() {
        let key_bytes = key.as_bytes();
        if idx == 0 {
            prefixed.push(key_bytes.to_vec());
        } else {
            let prev_bytes = prev.as_bytes();
            let mut common = 0usize;
            while common < 65535
                && common < prev_bytes.len()
                && common < key_bytes.len()
                && prev_bytes[common] == key_bytes[common]
            {
                common += 1;
            }
            let mut entry = Vec::with_capacity(2 + key_bytes.len() - common);
            if prev_bytes.len() > 255 {
                entry.push((common >> 8) as u8);
                entry.push((common & 255) as u8);
            } else {
                entry.push(common as u8);
            }
            entry.extend_from_slice(&key_bytes[common..]);
            prefixed.push(entry);
        }
        prev = key;
        sorted_indices.push((-count, idx as i32));
        string_to_index.insert(key.clone(), idx as i32);
    }

    // --- Step 3: sort by descending frequency (ties by original index).
    sorted_indices.sort();

    // --- Step 4: re-tag each entry's sort key with the varint-width category
    // its FINAL POSITION implies (so the most-varint-friendly strings land at
    // the front once re-sorted), then re-sort by (category, original index).
    {
        let mut i = 0usize;
        let len = sorted_indices.len();
        for &(limit, category) in &[
            (1usize << 7, 1i32),
            (1 << 14, 2),
            (1 << 21, 3),
            (1 << 28, 4),
        ] {
            while i < len && i < limit {
                sorted_indices[i].0 = category;
                i += 1;
            }
        }
        while i < len {
            sorted_indices[i].0 = 5;
            i += 1;
        }
    }
    sorted_indices.sort();

    // --- Step 5: RLE-encode the resulting index order, and build
    // `string_indices[original_index] -> compact_position` for step 6.
    let mut rle: Vec<(i32, i32)> = Vec::new();
    let mut string_indices: Vec<i32> = vec![-1; sorted_indices.len()];
    if let Some(&(_, first)) = sorted_indices.first() {
        rle.push((first, first));
        string_indices[first as usize] = 0;
        let mut next_to_include = first + 1;
        for (compact_pos, &(_, original)) in sorted_indices.iter().enumerate().skip(1) {
            if original == next_to_include {
                rle.last_mut().expect("just pushed the first entry").1 = original;
            } else {
                rle.push((original, original));
            }
            string_indices[original as usize] = compact_pos as i32;
            next_to_include = original + 1;
            while (next_to_include as usize) < string_indices.len()
                && string_indices[next_to_include as usize] >= 0
            {
                next_to_include += 1;
            }
        }
    }

    // --- Step 6: emit the header + string table + RLE block.
    let mut out = Vec::new();
    write_i32(&mut out, 0); // version
    write_byte_string_list(&mut out, &prefixed)?;
    write_i32(&mut out, rle.len() as i32);
    for &(a, b) in &rle {
        write_i32(&mut out, a);
        write_i32(&mut out, b);
    }

    // --- Step 7: emit every channel-group manifest.
    write_i32(&mut out, manifest.groups().len() as i32);
    for group in manifest.groups() {
        let channels: Vec<String> = group.channels().iter().cloned().collect();
        write_string_list(&mut out, &channels, None)?;
        write_string_list(&mut out, group.components(), None)?;
        write_u8(&mut out, group.lifetime() as u8);
        write_pascal_string(&mut out, group.hash_scheme());
        write_pascal_string(&mut out, group.encoding_scheme());

        // Pick the cheapest per-entry ID storage scheme: 0 = fixed 8 bytes,
        // 1 = fixed 4 bytes (only if every delta fits), 2 = variable length.
        let mut previous_id = 0u64;
        let mut variable_size = 0usize;
        let mut can_use_32_bits = true;
        for &id in group.table().keys() {
            let delta = id.wrapping_sub(previous_id);
            variable_size += varint_size(delta);
            if delta >= (1u64 << 32) {
                can_use_32_bits = false;
            }
            previous_id = id;
        }
        let table_len = group.table().len();
        let scheme: u8 = if can_use_32_bits {
            if variable_size < table_len * 4 { 2 } else { 1 }
        } else if variable_size < table_len * 8 {
            2
        } else {
            0
        };
        write_u8(&mut out, scheme);
        write_i32(&mut out, table_len as i32);

        let mut previous_id = 0u64;
        for (&id, text) in group.table() {
            let delta = id.wrapping_sub(previous_id);
            match scheme {
                0 => out.extend_from_slice(&delta.to_le_bytes()),
                1 => out.extend_from_slice(&(delta as u32).to_le_bytes()),
                _ => write_varint(&mut out, delta),
            }
            previous_id = id;

            for s in text {
                let original = *string_to_index
                    .get(s)
                    .expect("every table string was counted in string_set above");
                let compact = string_indices[original as usize];
                write_varint(&mut out, compact as u64);
            }
        }
    }

    Ok(out)
}

/// Parse the decompressed binary payload into a structured [`IDManifest`] —
/// port of `IDManifest::init` (`ImfIDManifest.cpp:349-586`).
///
/// Every bounds/consistency check C++ performs is reproduced (rejecting a
/// non-zero version, a bad common-prefix length, an out-of-range RLE entry,
/// a negative or overflowing count, a duplicate ID within one group, and a
/// bad string index) — each returns [`ExrError::Input`] rather than
/// panicking or reading out of bounds, matching this crate's no-panic rule
/// (C++ throws `Iex::InputExc` for every one of these).
pub fn parse(data: &[u8]) -> Result<IDManifest> {
    let mut pos = 0usize;

    let version = read_i32(data, &mut pos)?;
    if version != 0 {
        return Err(ExrError::Input("Unrecognized IDmanifest version".into()));
    }

    // --- Step 1: the raw prefix-compressed string table.
    let raw_list = read_byte_string_list(data, &mut pos)?;

    // --- Step 2: expand each entry against the PREVIOUS (already-expanded)
    // string — each after the first starts with a 1- or 2-byte "characters in
    // common with the previous string" header (2 bytes iff the previous
    // string was over 255 bytes long).
    let mut string_list: Vec<String> = Vec::with_capacity(raw_list.len());
    for (i, raw) in raw_list.iter().enumerate() {
        if i == 0 {
            let s = String::from_utf8(raw.clone())
                .map_err(|e| ExrError::Input(format!("Non-UTF-8 string in IDManifest: {e}")))?;
            string_list.push(s);
            continue;
        }
        let prev_len = string_list[i - 1].len();
        let min_prefix_len = if prev_len > 255 { 2 } else { 1 };
        if raw.len() < min_prefix_len {
            return Err(ExrError::Input(
                "IDManifest string too small for common prefix length".into(),
            ));
        }
        let (common, tail_start) = if prev_len > 255 {
            (((raw[0] as usize) << 8) + raw[1] as usize, 2)
        } else {
            (raw[0] as usize, 1)
        };
        if common > prev_len {
            return Err(ExrError::Input(
                "Bad common string length in IDmanifest string table".into(),
            ));
        }
        let mut expanded = string_list[i - 1].as_bytes()[..common].to_vec();
        expanded.extend_from_slice(&raw[tail_start..]);
        let s = String::from_utf8(expanded)
            .map_err(|e| ExrError::Input(format!("Non-UTF-8 string in IDManifest: {e}")))?;
        string_list.push(s);
    }

    // --- Step 3: RLE-encoded index remapping table (compact position ->
    // original `string_list` index).
    let rle_len = read_i32(data, &mut pos)?;
    if rle_len < 0 {
        return Err(ExrError::Input("Negative RLE table length in IDManifest".into()));
    }
    let mut mapping = vec![0i32; string_list.len()];
    let mut seen = vec![false; string_list.len()];
    let mut current_index = 0usize;
    for _ in 0..rle_len {
        let first = read_i32(data, &mut pos)?;
        let last = read_i32(data, &mut pos)?;
        if first < 0
            || last < 0
            || first > last
            || first as usize >= string_list.len()
            || last as usize >= string_list.len()
        {
            return Err(ExrError::Input("Bad mapping table entry in IDManifest".into()));
        }
        for entry in first..=last {
            let entry = entry as usize;
            if !seen[entry] {
                if current_index >= mapping.len() {
                    return Err(ExrError::Input("IDManifest mapping table overflow".into()));
                }
                mapping[current_index] = entry as i32;
                seen[entry] = true;
                current_index += 1;
            }
        }
    }

    // --- Step 4: the channel-group manifests themselves.
    let manifest_entries = read_i32(data, &mut pos)?;
    if manifest_entries < 0 {
        return Err(ExrError::Input(
            "bad number of ChannelGroupsManifests in IDManifest".into(),
        ));
    }

    let mut result = IDManifest::new();
    for _ in 0..manifest_entries {
        let mut m = ChannelGroupManifest::new();

        let channels = read_string_list(data, &mut pos, None)?;
        m.set_channels(channels.into_iter().collect());
        let components = read_string_list(data, &mut pos, None)?;
        // The table is empty on a freshly-constructed manifest, so this
        // never trips the "already has entries" guard.
        m.set_components(components)?;

        let lifetime_byte = read_u8(data, &mut pos)?;
        m.set_lifetime(IdLifetime::from_u8(lifetime_byte)?);
        let hash_scheme = read_pascal_string(data, &mut pos)?;
        m.set_hash_scheme(&hash_scheme);
        let encoding_scheme = read_pascal_string(data, &mut pos)?;
        m.set_encoding_scheme(&encoding_scheme);

        let storage_scheme = read_u8(data, &mut pos)?;
        let table_size = read_i32(data, &mut pos)?;
        if table_size < 0 {
            return Err(ExrError::Input("Negative ID table size in IDManifest".into()));
        }

        let mut previous_id: u64 = 0;
        for _ in 0..table_size {
            let delta: u64 = match storage_scheme {
                0 => {
                    let bytes = data
                        .get(pos..pos + 8)
                        .ok_or_else(|| ExrError::Input("IDManifest too small".into()))?;
                    pos += 8;
                    u64::from_le_bytes(bytes.try_into().expect("checked 8-byte slice"))
                }
                1 => {
                    let bytes = data
                        .get(pos..pos + 4)
                        .ok_or_else(|| ExrError::Input("IDManifest too small".into()))?;
                    pos += 4;
                    u32::from_le_bytes(bytes.try_into().expect("checked 4-byte slice")) as u64
                }
                _ => read_varint(data, &mut pos)?,
            };
            let id = previous_id.wrapping_add(delta);
            previous_id = id;

            if m.table().contains_key(&id) {
                return Err(ExrError::Input(
                    "ID manifest contains multiple entries for the same ID".into(),
                ));
            }

            let mut text = Vec::with_capacity(m.components().len());
            for _ in 0..m.components().len() {
                let string_index = read_varint(data, &mut pos)?;
                let string_index = usize::try_from(string_index)
                    .map_err(|_| ExrError::Input("Bad string index in IDManifest".into()))?;
                let original = *mapping
                    .get(string_index)
                    .ok_or_else(|| ExrError::Input("Bad string index in IDManifest".into()))?
                    as usize;
                let s = string_list
                    .get(original)
                    .ok_or_else(|| ExrError::Input("Bad string index in IDManifest".into()))?;
                text.push(s.clone());
            }
            m.table_mut().insert(id, text);
        }

        result.groups_mut().push(m);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn round_trip(manifest: &IDManifest) -> IDManifest {
        let bytes = serialize(manifest).unwrap();
        parse(&bytes).unwrap()
    }

    #[test]
    fn empty_manifest_round_trips() {
        let m = IDManifest::new();
        let back = round_trip(&m);
        assert_eq!(back, m);
    }

    #[test]
    fn single_group_single_entry_round_trips() {
        let mut m = IDManifest::new();
        {
            let g = m.add_channel("R");
            g.set_component("name").unwrap();
            g.set_hash_scheme("MurmurHash3_32");
            g.insert_single(42, "asset_01").unwrap();
        }
        let back = round_trip(&m);
        assert_eq!(back, m);
        assert_eq!(back.get(0).unwrap().find(42), Some(&["asset_01".to_string()][..]));
    }

    #[test]
    fn multi_component_multi_entry_round_trips() {
        let mut m = IDManifest::new();
        {
            let g = m.add_group(BTreeSet::from(["R".to_string(), "G".to_string()]));
            g.set_components(vec!["asset".into(), "part".into()]).unwrap();
            g.insert(1, vec!["hero".into(), "body".into()]).unwrap();
            g.insert(2, vec!["hero".into(), "head".into()]).unwrap();
            g.insert(1000, vec!["prop".into(), "sword".into()]).unwrap();
        }
        let back = round_trip(&m);
        assert_eq!(back, m);
    }

    #[test]
    fn many_groups_with_shared_strings_round_trip() {
        // Shared strings across groups exercise the string-frequency /
        // dedup / prefix-compression machinery.
        let mut m = IDManifest::new();
        for i in 0..5 {
            let g = m.add_channel(&format!("layer{i}"));
            g.set_component("name").unwrap();
            for j in 0..20u64 {
                g.insert_single(i as u64 * 100 + j, &format!("asset_{j:03}")).unwrap();
            }
        }
        let back = round_trip(&m);
        assert_eq!(back, m);
        assert_eq!(back.len(), 5);
    }

    #[test]
    fn large_id_values_and_long_strings_round_trip() {
        let mut m = IDManifest::new();
        let g = m.add_channel("id");
        g.set_component("name").unwrap();
        g.insert_single(0, "").unwrap();
        g.insert_single(u64::MAX, &"x".repeat(400)).unwrap();
        g.insert_single(1 << 40, "mid-range").unwrap();
        let back = round_trip(&m);
        assert_eq!(back, m);
    }

    #[test]
    fn parse_rejects_bad_version() {
        let mut bytes = Vec::new();
        write_i32(&mut bytes, 1); // version != 0
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn parse_rejects_truncated_input() {
        assert!(parse(&[0u8, 0, 0]).is_err());
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn parse_rejects_duplicate_id_in_one_group() {
        // Hand-craft a payload: version=0, 1 string "a" (no prefix table
        // entries needed since the group deliberately has 0 components so no
        // strings are referenced), 0 RLE entries, 1 group with a duplicate ID.
        let mut m = IDManifest::new();
        {
            let g = m.add_channel("R");
            // 0 components -> table entries carry no per-entry strings.
            g.insert(1, vec![]).unwrap();
        }
        let mut bytes = serialize(&m).unwrap();
        // Manually duplicate the single table entry by patching the entry
        // count from 1 to 2 (the two entries will then have the SAME id,
        // since the second entry's bytes are whatever followed — but easier:
        // construct the corrupt buffer directly, see below.
        // Simpler: build directly instead of mutating serialized bytes.
        let _ = &mut bytes;

        let mut raw = Vec::new();
        write_i32(&mut raw, 0); // version
        write_byte_string_list(&mut raw, &[]).unwrap(); // no strings
        write_i32(&mut raw, 0); // no RLE entries
        write_i32(&mut raw, 1); // 1 group
        write_string_list(&mut raw, &["R".to_string()], None).unwrap(); // channels
        write_string_list(&mut raw, &[], None).unwrap(); // 0 components
        write_u8(&mut raw, IdLifetime::Stable as u8);
        write_pascal_string(&mut raw, "unknown");
        write_pascal_string(&mut raw, "unknown");
        write_u8(&mut raw, 1); // storage scheme: 4 bytes/ID
        write_i32(&mut raw, 2); // table size: 2 (both will be id 0 via delta 0)
        raw.extend_from_slice(&0u32.to_le_bytes()); // first id delta = 0
        raw.extend_from_slice(&0u32.to_le_bytes()); // second id delta = 0 -> same id!

        assert!(parse(&raw).is_err());
    }
}
