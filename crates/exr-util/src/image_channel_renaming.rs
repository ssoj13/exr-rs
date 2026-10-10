//! Port of `ImfImageChannelRenaming.h`: the old-name -> new-name map of `Image::renameChannels`.

use std::collections::BTreeMap;

use exr_core::attr::Name;

/// `typedef std::map<std::string, std::string> RenamingMap`. Channel names are byte strings, as in
/// the file format (`Name`), not necessarily UTF-8.
pub type RenamingMap = BTreeMap<Name, Name>;

/// `renameChannelsInMap`: rebuild `channels` under the names `old_to_new` assigns; channels without
/// an entry keep their name. When two channels would get the same name the later one (in name
/// order) wins, exactly like the C++ `renamedChannels[newName] = ...`; `Image::rename_channels`
/// rejects such maps before it gets here.
pub fn rename_in_map<V>(old_to_new: &RenamingMap, channels: &mut BTreeMap<Name, V>) {
    let old = std::mem::take(channels);
    for (name, v) in old {
        let new = old_to_new.get(&name).cloned().unwrap_or(name);
        channels.insert(new, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renames_and_keeps_the_rest() {
        let mut m: BTreeMap<Name, i32> = [("R", 1), ("G", 2), ("Z", 3)].map(|(k, v)| (Name::from(k), v)).into();
        let map: RenamingMap = [(Name::from("R"), Name::from("red"))].into();
        rename_in_map(&map, &mut m);
        assert_eq!(m.keys().map(|k| k.to_string()).collect::<Vec<_>>(), ["G", "Z", "red"]);
        assert_eq!(m[&Name::from("red")], 1);
    }
}
