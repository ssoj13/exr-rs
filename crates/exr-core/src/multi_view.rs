//! Multi-view channel/view helpers — 1:1 port of `ImfMultiView.{cpp,h}`
//! (OpenEXR v3.3.11).
//!
//! Port target (pinned tag v3.3.11):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMultiView.cpp>
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.11/src/lib/OpenEXR/ImfMultiView.h>
//!
//! A multi-view image file contains two or more views of the same scene, as seen
//! from different viewpoints (e.g. a left-eye and a right-eye view for stereo
//! displays).  The `multiView` attribute (an `ImfStringVectorAttribute`, see
//! [`crate::standard_attributes`] `multi_view`/`has_multi_view`/`add_multi_view`)
//! lists the view names; channel names of the form `layer.view.channel` map
//! channels to views.
//!
//! ## Naming convention (verbatim from `ImfMultiView.h`)
//! For compatibility with single-view images, the **first** view listed in
//! `multiView` is the *default view*, and channels that have no periods in their
//! names are considered part of the default view.  Given views `"left"`,
//! `"right"` (in that order, so `"left"` is the default view):
//! * `"R"`, `"left.Z"`, `"diffuse.left.R"` belong to the `"left"` view;
//! * `"right.R"`, `"right.Z"`, `"diffuse.right.R"` belong to the `"right"` view;
//! * `"tmp.R"`, `"right.diffuse.R"`, `"diffuse.tmp.R"` belong to no view at all
//!   (the *view name* must be the **penultimate** dotted component).
//!
//! ## Byte fidelity
//! The C++ `StringVector` is `std::vector<std::string>`; in this crate that is
//! `Vec<ExrString>` (an `ExrString` is raw bytes, like `std::string`).  Channel
//! names in [`ChannelList`] are `String` keys.  All the matching logic here runs
//! on raw bytes (`&[u8]`) so it reproduces the C++ byte-for-byte comparisons and
//! ASCII-space (`0x20`) stripping exactly — independent of UTF-8 validity.
//!
//! ## Return-value convention
//! The C++ functions that "return a view name" use the empty string `""` as the
//! *no view* sentinel.  The ergonomic Rust wrappers map that sentinel to
//! `Option`: `Some(view)` for a real view (including the default view) and `None`
//! for "no view".  The byte-level cores ([`view_from_channel_bytes`] etc.) keep
//! the C++ sentinel (an empty slice == "no view") for callers that need exact
//! 1:1 behaviour.

use crate::attr::{ChannelList, ExrString};

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers (ImfMultiView.cpp lines 20-86)
// ---------------------------------------------------------------------------

/// Split `name` on byte `c`, stripping leading/trailing ASCII spaces from each
/// section — 1:1 port of the file-local `parseString` (ImfMultiView.cpp:23-65).
///
/// Operates on raw bytes so it matches the C++ `std::string` semantics exactly
/// (the C++ strips only `' '` (0x20)).  The `while (name.size() > 0)` loop
/// consumes the separator and re-tests the remaining `name`, so a *trailing*
/// separator does NOT produce a final empty section (`"a."` -> `["a"]`), while a
/// *leading* or *internal* run of separators does (`".a"` -> `["", "a"]`,
/// `"a..b"` -> `["a", "", "b"]`).
fn parse_string(name: &[u8], c: u8) -> Vec<Vec<u8>> {
    let mut r: Vec<Vec<u8>> = Vec::new();
    let mut name = name.to_vec();

    while !name.is_empty() {
        // size_t s = name.find(c); string sec = name.substr(0, s);
        let s = name.iter().position(|&b| b == c);
        let mut sec: Vec<u8> = match s {
            Some(idx) => name[..idx].to_vec(),
            None => name.clone(),
        };

        // Strip spaces from beginning.
        while !sec.is_empty() && sec[0] == b' ' {
            sec.remove(0);
        }
        // Strip spaces from end.
        while !sec.is_empty() && sec[sec.len() - 1] == b' ' {
            sec.pop();
        }

        r.push(sec);

        // Strip off name including the ending 'c'.
        match s {
            None => name.clear(),
            Some(idx) => name = name[idx + 1..].to_vec(),
        }
    }

    r
}

/// Index of the view called `view` within `multi_view`, or `-1` if none —
/// 1:1 port of file-local `viewNum` (ImfMultiView.cpp:67-84).
///
/// Returns `0` if `view` is the default (first) view, a positive index for any
/// other valid view, and `-1` if `view` is not a view name at all.  Compares
/// raw bytes (the C++ `std::string operator==`).
fn view_num(view: &[u8], multi_view: &[ExrString]) -> i32 {
    for (i, v) in multi_view.iter().enumerate() {
        if v.as_bytes() == view {
            return i as i32;
        }
    }
    -1
}

// ---------------------------------------------------------------------------
// Byte-level cores (exact C++ sentinel: empty slice == "no view")
// ---------------------------------------------------------------------------

/// Byte-exact core of [`view_from_channel_name`] — 1:1 port of
/// `viewFromChannelName` (ImfMultiView.cpp:97-139).
///
/// Returns the raw bytes of the owning view name, or an empty slice for
/// "no view" (the C++ `""` sentinel).  The view name is the **penultimate**
/// dotted section of the channel name; a channel with no periods belongs to the
/// default view (`multiView[0]`).
pub fn view_from_channel_bytes<'a>(channel: &[u8], multi_view: &'a [ExrString]) -> &'a [u8] {
    let s = parse_string(channel, b'.');

    if s.is_empty() {
        return b""; // nothing in, nothing out
    }

    if s.len() == 1 {
        // The rules say ALL channels with no periods in the name belong to the
        // default view, i.e. multiView[0].  C++ dereferences multiView[0]
        // unconditionally (UB on an empty vector); we map the empty-vector case
        // to the "" sentinel, which is what defaultViewName() would yield.
        return match multi_view.first() {
            Some(v) => v.as_bytes(),
            None => b"",
        };
    }

    // size >= 2: the last part is the channel name, the next-to-last part is the
    // candidate view name.  Return it iff it is a valid view.
    let view_name = &s[s.len() - 2];
    if view_num(view_name, multi_view) >= 0 {
        // Return a slice borrowed from `multi_view` (not the owned `s`) so the
        // lifetime ties to the view list, matching the C++ which returns a copy.
        let idx = view_num(view_name, multi_view) as usize;
        multi_view[idx].as_bytes()
    } else {
        b"" // not associated with any particular view
    }
}

/// Byte-exact core of [`are_counterparts`] — 1:1 port of `areCounterparts`
/// (ImfMultiView.cpp:183-257).  See that wrapper for the rule description.
pub fn are_counterparts_bytes(
    channel1: &[u8],
    channel2: &[u8],
    multi_view: &[ExrString],
) -> bool {
    let chan1 = parse_string(channel1, b'.');
    let size1 = chan1.len();

    let chan2 = parse_string(channel2, b'.');
    let size2 = chan2.len();

    if size1 == 0 || size2 == 0 {
        return false;
    }

    // Can't be counterparts if either channel is in no view.
    if size1 > 1 && view_num(&chan1[size1 - 2], multi_view) == -1 {
        return false;
    }
    if size2 > 1 && view_num(&chan2[size2 - 2], multi_view) == -1 {
        return false;
    }

    // Not counterparts if they are in the same view.
    if view_from_channel_bytes(channel1, multi_view)
        == view_from_channel_bytes(channel2, multi_view)
    {
        return false;
    }

    if size1 == 1 {
        // channel1 is a default channel — counterparts only if channel2 is of
        // the form <view>.<channel1>.
        return size2 == 2 && chan1[0] == chan2[1];
    }

    if size2 == 1 {
        // channel2 is a default channel — counterparts only if channel1 is of
        // the form <view>.<channel2>.
        return size1 == 2 && chan2[0] == chan1[1];
    }

    // Neither channel is a default channel.  To be counterparts both names must
    // have the same number of components, and all components except the
    // penultimate one must be equal.
    if size1 != size2 {
        return false;
    }

    for i in 0..size1 {
        if i != size1 - 2 && chan1[i] != chan2[i] {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Public free functions (ImfMultiView.h surface) — ergonomic Option/&str API.
//
// The C++ "" sentinel is mapped to `None`; a real view name (including the
// default view) is `Some(view)`.  Inputs that are not valid UTF-8 still match
// byte-wise, but the &str-returning wrappers can only surface a name when its
// bytes are valid UTF-8 (see the *_bytes cores for the raw form).
// ---------------------------------------------------------------------------

/// Name of the default view — 1:1 port of `defaultViewName`
/// (ImfMultiView.cpp:88-95): the first element of `multi_view`, or `None`
/// (the C++ `""`) when the list is empty.
///
/// C-08b: returns raw bytes, not `&str` — a view name is a byte string in
/// C++ (`std::string`, no encoding check), so a non-UTF-8 view name (however
/// unlikely in practice) is representable here, matching every other
/// byte-level helper in this module.
pub fn default_view_name(multi_view: &[ExrString]) -> Option<&[u8]> {
    multi_view.first().map(|v| v.as_bytes())
}

/// View to which channel `channel` belongs, or `None` if it is a member of no
/// named view — 1:1 port of `viewFromChannelName` (ImfMultiView.cpp:97-139).
///
/// The view name is the penultimate dotted section of the channel name; a
/// channel with no periods belongs to the default view.  Returns `None` for the
/// C++ `""` sentinel.
///
/// C-08b: `channel` and the return value are raw bytes, matching C++
/// (`viewFromChannelName` takes/returns `const char*`/`std::string`, never
/// checking encoding) — a non-UTF-8 channel or view name is handled exactly
/// like a UTF-8 one, not silently turned into "no view".
pub fn view_from_channel_name<'a>(channel: &[u8], multi_view: &'a [ExrString]) -> Option<&'a [u8]> {
    let v = view_from_channel_bytes(channel, multi_view);
    if v.is_empty() { None } else { Some(v) }
}

/// Whether `channel1` and `channel2` are the same channel viewed in two
/// different views — 1:1 port of `areCounterparts` (ImfMultiView.cpp:183-257).
///
/// Returns `false` if either channel belongs to no view, or if both channels
/// belong to the same view.
pub fn are_counterparts(channel1: &[u8], channel2: &[u8], multi_view: &[ExrString]) -> bool {
    are_counterparts_bytes(channel1, channel2, multi_view)
}

/// All channels belonging to view `view_name` — 1:1 port of `channelsInView`
/// (ImfMultiView.cpp:141-171).
///
/// A channel is included iff [`view_from_channel_bytes`] for its name equals
/// `view_name` (byte-wise).  Passing an empty `view_name` selects the
/// "no view" channels — exactly the C++ `channelsInView("", …)` used by
/// [`channels_in_no_view`].  Returns a new [`ChannelList`] preserving the
/// channels' attributes, mirroring `q.insert(name, channel)`.
pub fn channels_in_view(
    view_name: &[u8],
    channel_list: &ChannelList,
    multi_view: &[ExrString],
) -> ChannelList {
    let mut q = ChannelList::new();
    for (name, channel) in channel_list.iter() {
        let view = view_from_channel_bytes(name.as_bytes(), multi_view);
        if view == view_name {
            // `insert` only errors on an empty name; the source list's keys are
            // always non-empty (ChannelList::insert rejects empty), so an error
            // here is impossible — `if let Err` skips it instead of panicking.
            if let Err(_e) = q.insert(name, *channel) {
                continue;
            }
        }
    }
    q
}

/// All channels not associated with any named view — 1:1 port of
/// `channelsInNoView` (ImfMultiView.cpp:173-181), which is
/// `channelsInView("", …)`.
pub fn channels_in_no_view(channel_list: &ChannelList, multi_view: &[ExrString]) -> ChannelList {
    channels_in_view(b"", channel_list, multi_view)
}

/// Given a channel name, the same channel in all views — 1:1 port of
/// `channelInAllViews` (ImfMultiView.cpp:259-284).
///
/// Includes every channel whose name equals `channel_name` or is a counterpart
/// of it (see [`are_counterparts`]).
pub fn channel_in_all_views(
    channel_name: &[u8],
    channel_list: &ChannelList,
    multi_view: &[ExrString],
) -> ChannelList {
    let mut q = ChannelList::new();
    let target = channel_name;
    for (name, channel) in channel_list.iter() {
        let matches =
            name.as_bytes() == target || are_counterparts_bytes(name.as_bytes(), target, multi_view);
        // See `channels_in_view`: insert cannot fail for a non-empty key.
        if matches && q.insert(name, *channel).is_err() {
            continue;
        }
    }
    q
}

/// Corresponding channel name for view `other_view_name`, given a channel in
/// some other view — 1:1 port of `channelInOtherView`
/// (ImfMultiView.cpp:286-310).  Returns `None` (the C++ `""`) if no
/// corresponding channel exists in `other_view_name`, or if that view does not
/// exist.
pub fn channel_in_other_view<'a>(
    channel_name: &[u8],
    channel_list: &'a ChannelList,
    multi_view: &[ExrString],
    other_view_name: &[u8],
) -> Option<&'a [u8]> {
    let target = channel_name;
    for (name, _channel) in channel_list.iter() {
        if view_from_channel_bytes(name.as_bytes(), multi_view) == other_view_name
            && are_counterparts_bytes(name.as_bytes(), target, multi_view)
        {
            return Some(name.as_bytes());
        }
    }
    None
}

/// Insert `multi_view[i]` into a (view-free) channel name at the correct
/// position — 1:1 port of `insertViewName` (ImfMultiView.cpp:312-348).
///
/// The view name becomes the penultimate dotted section.  Special case: if
/// `i == 0` and the channel name has no periods (default view, single section),
/// the name is returned unchanged (no view name inserted).  Returns the empty
/// string for an empty input (the C++ `""` "nothing in, nothing out").
///
/// `i` is a `usize` index into `multi_view` (the C++ `int i` parameter); the
/// caller must pass a valid index, mirroring the C++ `multiView[i]` access.
pub fn insert_view_name(channel: &[u8], multi_view: &[ExrString], i: usize) -> Vec<u8> {
    let s = parse_string(channel, b'.');

    if s.is_empty() {
        return Vec::new(); // nothing in, nothing out
    }

    if s.len() == 1 && i == 0 {
        // Channel in the default view, with no periods in its name: do not
        // insert the view name.
        return channel.to_vec();
    }

    // View name becomes the penultimate section of the new channel name.
    let view = match multi_view.get(i) {
        Some(v) => v.as_bytes(),
        None => return Vec::new(), // out-of-range index: C++ is UB; we no-op.
    };

    let mut new_name: Vec<u8> = Vec::new();
    let n = s.len();
    for (j, sec) in s.iter().enumerate() {
        if j < n - 1 {
            new_name.extend_from_slice(sec);
            new_name.push(b'.');
        } else {
            new_name.extend_from_slice(view);
            new_name.push(b'.');
            new_name.extend_from_slice(sec);
        }
    }

    new_name
}

/// Remove view `view` from a channel name, if present in the correct
/// (penultimate) position — 1:1 port of `removeViewName`
/// (ImfMultiView.cpp:350-380).
///
/// If the channel name does not contain the view name in the penultimate
/// position, it is returned unaltered.  A name with no periods (default view)
/// has no view name to remove and is returned unchanged.
pub fn remove_view_name(channel: &[u8], view: &[u8]) -> Vec<u8> {
    let s = parse_string(channel, b'.');

    if s.is_empty() {
        return Vec::new(); // nothing in, nothing out
    }

    if s.len() == 1 {
        // Channel in the default view (no periods): no viewname to remove.
        return channel.to_vec();
    }

    let n = s.len();
    let mut new_name: Vec<u8> = Vec::new();
    for (j, sec) in s.iter().enumerate() {
        // Only add the penultimate string part if it doesn't match the view name.
        // C++: if (j + 2 != s.size() || s[j] != view)
        if j + 2 != n || sec.as_slice() != view {
            new_name.extend_from_slice(sec);
            // C++: if (j + 1 != s.size()) newName += ".";
            if j + 1 != n {
                new_name.push(b'.');
            }
        }
    }

    new_name
}

// ---------------------------------------------------------------------------
// Tests — expected results derived by TRACING the C++ logic (cited inline).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, PixelType};

    /// Build a `multiView` StringVector from string slices.
    fn mv(views: &[&str]) -> Vec<ExrString> {
        views.iter().map(|s| ExrString::from(*s)).collect()
    }

    // multiView = ["right", "left"]  =>  default view = "right".
    fn rl() -> Vec<ExrString> {
        mv(&["right", "left"])
    }

    #[test]
    fn parse_string_basic() {
        // C++ parseString("left.R", '.') -> ["left","R"] (size 2).
        assert_eq!(parse_string(b"left.R", b'.'), vec![b"left".to_vec(), b"R".to_vec()]);
        // "R" -> ["R"] (size 1).
        assert_eq!(parse_string(b"R", b'.'), vec![b"R".to_vec()]);
        // "diffuse.left.R" -> ["diffuse","left","R"] (size 3).
        assert_eq!(
            parse_string(b"diffuse.left.R", b'.'),
            vec![b"diffuse".to_vec(), b"left".to_vec(), b"R".to_vec()]
        );
        // "" -> [] (size 0).
        assert_eq!(parse_string(b"", b'.'), Vec::<Vec<u8>>::new());
        // Trailing separator: the `while (name.size() > 0)` loop consumes the
        // '.' and then exits (name becomes empty), so NO trailing empty section
        // is produced: "a." -> ["a"].
        assert_eq!(parse_string(b"a.", b'.'), vec![b"a".to_vec()]);
        // Leading/trailing spaces stripped per section: " a . b " -> ["a","b"].
        assert_eq!(parse_string(b" a . b ", b'.'), vec![b"a".to_vec(), b"b".to_vec()]);
    }

    #[test]
    fn default_view_name_first_or_none() {
        // C++ defaultViewName(["right","left"]) -> "right" (multiView[0]).
        assert_eq!(default_view_name(&rl()), Some(b"right" as &[u8]));
        // Empty vector -> "" (mapped to None).
        assert_eq!(default_view_name(&[]), None);
    }

    #[test]
    fn view_from_channel_name_rules() {
        let v = rl(); // default = "right"

        // "left.R": parseString -> ["left","R"] (size 2). Penultimate = "left",
        // viewNum("left", ["right","left"]) = 1 >= 0 -> returns "left".
        assert_eq!(view_from_channel_name(b"left.R", &v), Some(b"left" as &[u8]));

        // "R": size 1 -> default view = multiView[0] = "right".
        assert_eq!(view_from_channel_name(b"R", &v), Some(b"right" as &[u8]));

        // "Z": size 1 -> default view "right" (same default-view rule as "R").
        assert_eq!(view_from_channel_name(b"Z", &v), Some(b"right" as &[u8]));

        // "right.R": ["right","R"], penultimate "right", viewNum=0 -> "right".
        assert_eq!(view_from_channel_name(b"right.R", &v), Some(b"right" as &[u8]));

        // "diffuse.left.R": ["diffuse","left","R"] (size 3). Penultimate = "left"
        // (valid view) -> "left".  (Header doc example: belongs to "left" view.)
        assert_eq!(view_from_channel_name(b"diffuse.left.R", &v), Some(b"left" as &[u8]));

        // "diffuse.tmp.R": penultimate "tmp" is not a view -> "" (None).
        // (Header doc example: belongs to no view.)
        assert_eq!(view_from_channel_name(b"diffuse.tmp.R", &v), None);

        // "tmp.R": penultimate "tmp" not a view -> "" (None).
        assert_eq!(view_from_channel_name(b"tmp.R", &v), None);

        // "right.diffuse.R": penultimate "diffuse" not a view -> "" (None).
        assert_eq!(view_from_channel_name(b"right.diffuse.R", &v), None);

        // "" : parseString -> [] -> "" (None).
        assert_eq!(view_from_channel_name(b"", &v), None);
    }

    #[test]
    fn channel_in_view_via_channels_in_view() {
        // Build a channel list mixing default-view and explicit-view channels.
        let mut cl = ChannelList::new();
        let ch = Channel::new(PixelType::Half, 1, 1, false);
        for n in ["R", "left.R", "right.R", "diffuse.left.R", "tmp.R"] {
            cl.insert(n, ch).unwrap();
        }
        let v = rl(); // default = "right"

        // channelsInView("right", …): "R" (default view "right") and "right.R".
        let right = channels_in_view(b"right", &cl, &v);
        let right_names: Vec<&str> = right.iter().map(|(n, _)| n.to_str().unwrap_or("")).collect();
        assert_eq!(right_names, vec!["R", "right.R"]);

        // channelsInView("left", …): "left.R" and "diffuse.left.R".
        let left = channels_in_view(b"left", &cl, &v);
        let left_names: Vec<&str> = left.iter().map(|(n, _)| n.to_str().unwrap_or("")).collect();
        assert_eq!(left_names, vec!["diffuse.left.R", "left.R"]);

        // channelsInNoView: "tmp.R" (penultimate "tmp" not a view).
        let none = channels_in_no_view(&cl, &v);
        let none_names: Vec<&str> = none.iter().map(|(n, _)| n.to_str().unwrap_or("")).collect();
        assert_eq!(none_names, vec!["tmp.R"]);
    }

    #[test]
    fn are_counterparts_rules() {
        let v = rl(); // ["right","left"]

        // "right.R" vs "left.R": different valid views, both size 2, all parts
        // except penultimate equal ("R"=="R") -> counterparts.
        assert!(are_counterparts(b"right.R", b"left.R", &v));

        // "R" (default "right") vs "left.R": channel1 size 1 default; channel2 is
        // <view>.<channel1> = "left"."R" and chan1[0]("R")==chan2[1]("R") -> true.
        assert!(are_counterparts(b"R", b"left.R", &v));

        // "R" vs "right.R": both resolve to view "right" (same view) -> false.
        assert!(!are_counterparts(b"R", b"right.R", &v));

        // "tmp.R" vs "left.R": "tmp" is not a view (size1>1 & viewNum==-1) -> false.
        assert!(!are_counterparts(b"tmp.R", b"left.R", &v));

        // "diffuse.right.R" vs "diffuse.left.R": size 3 each, components equal
        // except penultimate (right/left) -> counterparts.
        assert!(are_counterparts(b"diffuse.right.R", b"diffuse.left.R", &v));

        // "diffuse.right.R" vs "other.left.R": leading component differs -> false.
        assert!(!are_counterparts(b"diffuse.right.R", b"other.left.R", &v));
    }

    #[test]
    fn channel_in_all_views_and_other_view() {
        let mut cl = ChannelList::new();
        let ch = Channel::new(PixelType::Half, 1, 1, false);
        for n in ["R", "left.R", "right.diffuse.R"] {
            cl.insert(n, ch).unwrap();
        }
        let v = rl(); // default = "right"

        // channelInAllViews("R", …): "R" itself, plus counterparts of "R".
        // "left.R" is a counterpart of default "R"; "right.diffuse.R" is not.
        let all = channel_in_all_views(b"R", &cl, &v);
        let names: Vec<&str> = all.iter().map(|(n, _)| n.to_str().unwrap_or("")).collect();
        assert_eq!(names, vec!["R", "left.R"]);

        // channelInOtherView("R", …, "left"): the left counterpart of "R" is
        // "left.R".
        assert_eq!(channel_in_other_view(b"R", &cl, &v, b"left"), Some(b"left.R" as &[u8]));

        // No counterpart in a non-existent view -> None.
        assert_eq!(channel_in_other_view(b"R", &cl, &v, b"middle"), None);
    }

    #[test]
    fn insert_view_name_rules() {
        let v = rl(); // ["right","left"]

        // i==0, no periods ("R"): default view, return unchanged.
        assert_eq!(insert_view_name(b"R", &v, 0), b"R".to_vec());

        // i==1 ("left"), "R" -> view becomes penultimate: "left.R".
        // (s=["R"], size 1; loop: j==0 is last -> "left" + "." + "R".)
        assert_eq!(insert_view_name(b"R", &v, 1), b"left.R".to_vec());

        // i==0 ("right"), "diffuse.R" (size 2): NOT (size1 && i==0) special case
        // since size==2, so insert -> "diffuse.right.R".
        assert_eq!(insert_view_name(b"diffuse.R", &v, 0), b"diffuse.right.R".to_vec());

        // Empty input -> empty.
        assert_eq!(insert_view_name(b"", &v, 0), Vec::<u8>::new());
    }

    #[test]
    fn remove_view_name_rules() {
        // "left.R" remove "left": penultimate (j=0, j+2==2==size) matches -> drop
        // it; result "R".
        assert_eq!(remove_view_name(b"left.R", b"left"), b"R".to_vec());

        // "diffuse.left.R" remove "left": penultimate is "left" (j=1, j+2==3==size)
        // -> dropped; result "diffuse.R".
        assert_eq!(remove_view_name(b"diffuse.left.R", b"left"), b"diffuse.R".to_vec());

        // "left.R" remove "right": penultimate "left" != "right" -> unchanged.
        assert_eq!(remove_view_name(b"left.R", b"right"), b"left.R".to_vec());

        // "R" (no periods): nothing to remove -> unchanged.
        assert_eq!(remove_view_name(b"R", b"left"), b"R".to_vec());

        // "" -> "".
        assert_eq!(remove_view_name(b"", b"left"), Vec::<u8>::new());
    }
}
