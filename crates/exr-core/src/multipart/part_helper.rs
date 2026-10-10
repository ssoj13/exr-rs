//! Port of OpenEXR's `ImfPartHelper.h` — functions to split a flat list of
//! (possibly multi-view) channels into separate multipart parts, and to
//! enumerate an already-open multipart file's channels back the same way.
//!
//! Port target (pinned tag v3.4.15 — the repo's behavioural target; verified
//! via `git diff v3.3.11 v3.4.15 -- src/lib/OpenEXR/ImfPartHelper.h` that the
//! header is BYTE-FOR-BYTE identical between the two tags, so nothing here
//! needed a behavioural update, only this citation):
//! <https://github.com/AcademySoftwareFoundation/openexr/blob/v3.4.15/src/lib/OpenEXR/ImfPartHelper.h>
//!
//! This is header-only in C++ (no `.cpp`): two class-template-parameterised
//! free functions (`SplitChannels`, `GetChannelsInMultiPartFile`) operating
//! on an iterator range / a generic output container of `MultiViewChannelName`
//! objects. Rust has no equivalent "any random-access container of a struct
//! with these four public fields" template constraint, so both are ported as
//! plain functions over `&mut [MultiViewChannelName]` / returning
//! `Vec<MultiViewChannelName>` — the concrete shape every real C++ caller
//! (`exr2aces`-style dialogs, per the header's own module doc) already uses.
//!
//! **Workflow (verbatim from the C++ module doc):** to offer a "create
//! multipart file" checkbox in a write dialog, populate a list of
//! [`MultiViewChannelName`], call [`split_channels`] with whether a single-
//! or multi-part file is wanted, then write `SplitChannels`'s returned part
//! count using each channel's `internal_name` in the `ChannelList`/frame
//! buffer — no different code path is needed for single-part vs multipart.
//! Reading back, [`get_channels_in_multi_part_file`] enumerates every channel
//! of an open [`crate::multipart::MultiPartInputFile`], again keyed by
//! `internal_name` for frame-buffer lookups.

use std::collections::{BTreeMap, BTreeSet};

use crate::attr::{ExrString, Name};
use crate::error::Result;
use crate::multi_view::{remove_view_name, view_from_channel_name};
use crate::multipart::input::MultiPartInputFile;

/// Port of `struct MultiViewChannelName` (`ImfPartHelper.h:45-70`).
///
/// On entry to [`split_channels`], `name`/`view` must already be set (`view`
/// may be empty for a non-multiview channel); `split_channels` fills in
/// `part_number`/`internal_name`. [`get_channels_in_multi_part_file`] fills
/// in all four fields itself.
///
/// C-08b: `name`/`view`/`internal_name` are [`Name`] (raw bytes), not
/// `String` — C++'s `MultiViewChannelName` fields are `std::string`, never
/// encoding-checked, so a non-UTF-8 channel or view name works exactly like
/// a UTF-8 one here too. `Name` still accepts a `&str`/`String` literal via
/// `impl Into<Name>`/`From`, so ASCII call sites are unaffected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MultiViewChannelName {
    /// Name of the channel (without any view component once
    /// `get_channels_in_multi_part_file` has stripped it — see its doc).
    pub name: Name,
    /// View this channel belongs to (empty if not multi-view).
    pub view: Name,
    /// Part number: updated by [`split_channels`] / set by
    /// [`get_channels_in_multi_part_file`].
    pub part_number: i32,
    /// Name used in headers: in single-part mode this may have a view name
    /// spliced in; in multipart mode it is the caller's original `name`.
    pub internal_name: Name,
}

impl MultiViewChannelName {
    /// Port of `getLayer`: everything before the LAST `.`, or `""` if there
    /// is none (`ImfPartHelper.h:56-62`). C-08b: raw bytes, matching C++'s
    /// `rfind('.')` over `std::string`.
    pub fn layer(&self) -> &[u8] {
        let bytes = self.name.as_bytes();
        match bytes.iter().rposition(|&b| b == b'.') {
            Some(q) => &bytes[..q],
            None => b"",
        }
    }

    /// Port of `getSuffix`: everything after the last `.`, or the whole name
    /// if there is none (`ImfPartHelper.h:64-69`).
    pub fn suffix(&self) -> &[u8] {
        let bytes = self.name.as_bytes();
        match bytes.iter().rposition(|&b| b == b'.') {
            Some(q) => &bytes[q + 1..],
            None => bytes,
        }
    }
}

/// Port of `template <typename T> int SplitChannels(const T& begin, const T&
/// end, bool multipart, const std::string& heroView)`
/// (`ImfPartHelper.h:85-180`), operating on the whole slice in place (the
/// Rust equivalent of the `[begin, end)` iterator range).
///
/// On entry every `chans[i].name`/`.view` must already be set. Returns the
/// total number of parts required.
///
/// * `multipart = false`: every channel gets `part_number = 0`;
///   `internal_name` gets the view name spliced in (except for the hero
///   view, and except when the channel has no view at all), so a single
///   part's flat `ChannelList` still disambiguates views by name. Always
///   returns `1`.
/// * `multipart = true`: channels are grouped by `(layer, view)` and each
///   distinct group gets its own part number — the hero view (if present in
///   a layer) is assigned first within that layer, matching C++'s two-pass
///   `layerToPart` construction exactly (this ordering is externally
///   observable: it is the actual part numbering, not an implementation
///   detail). `internal_name` is left equal to `name` (no splicing needed —
///   each view already lives in its own part). Returns the number of
///   distinct `(layer, view)` groups.
pub fn split_channels(chans: &mut [MultiViewChannelName], multipart: bool, hero_view: &[u8]) -> i32 {
    if !multipart {
        for c in chans.iter_mut() {
            c.part_number = 0;

            if c.view.is_empty() {
                c.internal_name = c.name.clone();
            } else {
                let lname = c.layer().to_vec();
                if lname.is_empty() {
                    // No layer: only non-hero views get the view name spliced in.
                    c.internal_name = if c.view.as_bytes() == hero_view {
                        c.name.clone()
                    } else {
                        let mut spliced = c.view.as_bytes().to_vec();
                        spliced.push(b'.');
                        spliced.extend_from_slice(c.name.as_bytes());
                        Name::new(spliced)
                    };
                } else {
                    let mut spliced = lname;
                    spliced.push(b'.');
                    spliced.extend_from_slice(c.view.as_bytes());
                    spliced.push(b'.');
                    spliced.extend_from_slice(c.suffix());
                    c.internal_name = Name::new(spliced);
                }
            }
        }
        return 1;
    }

    // Step 1: for each layer, enumerate which views are active.
    let mut views_in_layers: BTreeMap<Vec<u8>, BTreeSet<Name>> = BTreeMap::new();
    for c in chans.iter() {
        views_in_layers.entry(c.layer().to_vec()).or_default().insert(c.view.clone());
    }

    // Step 2: assign a part number to each (layer, view) pair -- hero view
    // first within its layer, if present, exactly matching C++'s ordering.
    let mut layer_to_part: BTreeMap<(Vec<u8>, Name), i32> = BTreeMap::new();
    let mut part_count = 0i32;
    for (layer, views) in &views_in_layers {
        if views.contains(hero_view) {
            layer_to_part.insert((layer.clone(), Name::new(hero_view)), part_count);
            part_count += 1;
        }
        for view in views {
            if view.as_bytes() != hero_view {
                layer_to_part.insert((layer.clone(), view.clone()), part_count);
                part_count += 1;
            }
        }
    }

    // Step 3: update every channel's part number (every (layer, view) pair
    // was inserted in step 2, so the lookup always succeeds; `unwrap_or(0)`
    // is defensive only, never actually exercised by construction).
    for c in chans.iter_mut() {
        c.internal_name = c.name.clone();
        let key = (c.layer().to_vec(), c.view.clone());
        c.part_number = layer_to_part.get(&key).copied().unwrap_or(0);
    }

    part_count
}

/// Port of `template <class T> void GetChannelsInMultiPartFile(const
/// MultiPartInputFile& file, T& chans)` (`ImfPartHelper.h:186-223`).
///
/// Enumerates every channel of every part of `file`, in part order (and
/// within a part, in the part's own `ChannelList` order — alphabetical,
/// matching `ChannelList::ConstIterator`). If `file` is single-part AND that
/// one part carries a `multiView` attribute, each channel's view is
/// extracted via [`view_from_channel_name`] and stripped from `name` via
/// [`remove_view_name`] (both called unconditionally, exactly like C++ —
/// `view_from_channel_name` returning "no view" still routes through
/// `remove_view_name` with an empty view, which is a no-op there). Otherwise
/// each part's own `view` attribute (or `""`) is used verbatim and `name` is
/// left untouched.
pub fn get_channels_in_multi_part_file<'a, S>(
    file: &MultiPartInputFile<'a, S>,
) -> Result<Vec<MultiViewChannelName>>
where
    S: std::ops::DerefMut<Target = dyn crate::io::IStream + 'a>,
{
    let mut chans = Vec::new();

    let mut has_multiview = false;
    let mut mview: Vec<ExrString> = Vec::new();
    if file.parts() == 1 {
        let h0 = file.header(0)?;
        if h0.has_multi_view() {
            mview = h0.multi_view()?.clone();
            has_multiview = true;
        }
    }

    for p in 0..file.parts() {
        let h = file.header(p)?;
        let view: Name = if h.has_view() { Name::new(h.view()?) } else { Name::default() };

        for (name, _channel) in h.channels()?.iter() {
            let internal_name = name.clone();
            // C-08b: bytes throughout, matching C++ (`viewFromChannelName`/
            // `removeViewName` never check encoding), so a non-UTF-8 channel
            // name splits exactly like a UTF-8 one.
            let (final_name, final_view) = if has_multiview {
                // C++ sentinel: viewFromChannelName returns "" for "no view";
                // removeViewName is called with that same (possibly empty)
                // string either way.
                let v = view_from_channel_name(internal_name.as_bytes(), &mview).unwrap_or(b"");
                let stripped = remove_view_name(internal_name.as_bytes(), v);
                (Name::new(stripped), Name::new(v))
            } else {
                (internal_name.clone(), view.clone())
            };

            chans.push(MultiViewChannelName {
                name: final_name,
                view: final_view,
                part_number: p,
                internal_name,
            });
        }
    }

    Ok(chans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{Channel, ChannelList, ChannelListAttribute, Compression, LineOrder, PixelType};
    use crate::frame_buffer::{FrameBuffer, Slice};
    use crate::header::Header;
    use crate::io::{MemIStream, MemOStream};
    use crate::multipart::output::MultiPartOutputFile;
    use imath_rs::V2f;

    fn chan(name: &str, view: &str) -> MultiViewChannelName {
        MultiViewChannelName { name: name.into(), view: view.into(), part_number: -1, internal_name: Name::default() }
    }

    #[test]
    fn layer_and_suffix_split_on_the_last_dot() {
        let c = chan("diffuse.left.R", "");
        assert_eq!(c.layer(), b"diffuse.left");
        assert_eq!(c.suffix(), b"R");

        let plain = chan("R", "");
        assert_eq!(plain.layer(), b"");
        assert_eq!(plain.suffix(), b"R");
    }

    /// Non-multipart: every channel collapses to part 0; the view is spliced
    /// into `internal_name` per the three `ImfPartHelper.h` cases (no view;
    /// view + no layer, hero vs non-hero; view + layer).
    #[test]
    fn split_channels_single_part_splices_view_names() {
        let mut chans = vec![
            chan("R", ""),                 // no view at all
            chan("R", "left"),             // no layer, hero view
            chan("R", "right"),            // no layer, non-hero view
            chan("diffuse.R", "right"),    // layer + view
        ];
        let n = split_channels(&mut chans, false, b"left");
        assert_eq!(n, 1);
        assert!(chans.iter().all(|c| c.part_number == 0));
        assert_eq!(chans[0].internal_name, "R");
        assert_eq!(chans[1].internal_name, "R"); // hero view: no splice
        assert_eq!(chans[2].internal_name, "right.R");
        assert_eq!(chans[3].internal_name, "diffuse.right.R");
    }

    /// Multipart: channels group by `(layer, view)`; the hero view gets the
    /// first part number within its layer.
    #[test]
    fn split_channels_multipart_groups_by_layer_and_view_hero_first() {
        let mut chans = vec![
            chan("R", "right"),
            chan("R", "left"),
            chan("diffuse.R", "left"),
            chan("diffuse.R", "right"),
        ];
        let n = split_channels(&mut chans, true, b"left");
        assert_eq!(n, 4);
        // internal_name is left as the original name in multipart mode.
        assert!(chans.iter().all(|c| c.internal_name == c.name));

        // Within the "" layer, hero ("left") must get the lower part number.
        let base_right = chans[0].part_number;
        let base_left = chans[1].part_number;
        assert!(base_left < base_right, "hero view must be assigned first");

        // Within the "diffuse" layer, hero ("left") must also come first.
        let diffuse_left = chans[2].part_number;
        let diffuse_right = chans[3].part_number;
        assert!(diffuse_left < diffuse_right, "hero view must be assigned first");

        // All four part numbers must be distinct (4 groups -> 4 parts).
        let mut nums: Vec<i32> = chans.iter().map(|c| c.part_number).collect();
        nums.sort_unstable();
        nums.dedup();
        assert_eq!(nums.len(), 4);
    }

    /// A single-view (no `view` set at all) list still gets one part per
    /// distinct layer in multipart mode.
    #[test]
    fn split_channels_multipart_no_views_still_splits_by_layer() {
        let mut chans = vec![chan("R", ""), chan("diffuse.R", ""), chan("diffuse.G", "")];
        let n = split_channels(&mut chans, true, b"");
        assert_eq!(n, 2); // "" layer and "diffuse" layer
        assert_eq!(chans[1].part_number, chans[2].part_number, "same layer -> same part");
        assert_ne!(chans[0].part_number, chans[1].part_number);
    }

    fn part_header(name: &str, w: i32, h: i32, channels: &[&str], multi_view: Option<Vec<&str>>) -> Header {
        let mut hdr =
            Header::with_dimensions(w, h, 1.0, V2f { x: 0.0, y: 0.0 }, 1.0, LineOrder::IncreasingY, Compression::NoCompression)
                .unwrap();
        let mut cl = ChannelList::new();
        for ch in channels {
            cl.insert(*ch, Channel::new(PixelType::Half, 1, 1, false)).unwrap();
        }
        hdr.insert("channels", ChannelListAttribute::new(cl)).unwrap();
        hdr.insert("name", crate::attr::StringAttribute::new(ExrString::from(name))).unwrap();
        hdr.insert("type", crate::attr::StringAttribute::new(ExrString::from("scanlineimage"))).unwrap();
        if let Some(views) = multi_view {
            hdr.add_multi_view(views.into_iter().map(ExrString::from).collect()).unwrap();
        }
        hdr
    }

    /// Half-typed, unsampled slice into a private contiguous segment of the
    /// shared pixel buffer -- `base_offset` picks out this channel's own
    /// `w * h * 2`-byte region so multiple channels can share one buffer
    /// without overlapping.
    fn half_channel_slice(base_offset: i64, w: i32) -> Slice {
        Slice {
            type_: PixelType::Half,
            base_offset,
            x_stride: 2,
            y_stride: (w as usize) * 2,
            x_sampling: 1,
            y_sampling: 1,
            fill_value: 0.0,
            x_tile_coords: false,
            y_tile_coords: false,
        }
    }

    /// Write one part's chunks: every channel's pixels are zeroed (this
    /// module only asserts on header/channel-name metadata, never pixel
    /// values), each in its own contiguous segment of one shared buffer.
    fn write_part(mp: &mut MultiPartOutputFile, part: i32, channels: &[&str], w: i32, h: i32) {
        let mut p = mp.output_part(part).expect("output_part");
        let mut fb = FrameBuffer::new();
        let seg = (w as usize) * (h as usize) * 2;
        for (i, name) in channels.iter().enumerate() {
            fb.insert((*name).to_string(), half_channel_slice((i * seg) as i64, w)).expect("insert slice");
        }
        p.set_frame_buffer(&fb).expect("set_frame_buffer");
        let pixels = vec![0u8; seg * channels.len()];
        p.write_pixels(&pixels, h).expect("write_pixels");
    }

    /// A single-part multiView file: `get_channels_in_multi_part_file` must
    /// split the view out of the channel name.
    #[test]
    fn get_channels_splits_view_out_of_a_single_part_multiview_file() {
        const W: i32 = 4;
        const H: i32 = 4;
        let channels = ["G", "R", "right.G", "right.R"]; // alphabetical, matching ChannelList order
        let hdr = part_header("view", W, H, &channels, Some(vec!["left", "right"]));

        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[hdr]).expect("mp out");
            write_part(&mut mp, 0, &channels, W, H);
        }
        let bytes = out.into_data();
        let mut stream = MemIStream::new(bytes);
        let file = MultiPartInputFile::new(&mut stream).expect("mp in");

        let chans = get_channels_in_multi_part_file(&file).expect("get_channels");
        assert_eq!(chans.len(), 4);
        for c in &chans {
            assert_eq!(c.part_number, 0);
        }
        let left_r = chans.iter().find(|c| c.internal_name == "R" && c.view == "left").expect("left R");
        assert_eq!(left_r.name, "R");
        let right_r = chans.iter().find(|c| c.internal_name == "right.R").expect("right.R");
        assert_eq!(right_r.name, "R");
        assert_eq!(right_r.view, "right");
    }

    /// A plain (non-multiview) multi-part file: each part's own `view`
    /// (absent here) is used verbatim, and names are untouched.
    #[test]
    fn get_channels_uses_part_view_when_not_multiview() {
        const W: i32 = 4;
        const H: i32 = 4;
        let c0 = ["G", "R"];
        let c1 = ["A", "B"];
        let h0 = part_header("left", W, H, &c0, None);
        let h1 = part_header("right", W, H, &c1, None);

        let mut out = MemOStream::new();
        {
            let mut mp = MultiPartOutputFile::new(&mut out, &[h0, h1]).expect("mp out");
            write_part(&mut mp, 0, &c0, W, H);
            write_part(&mut mp, 1, &c1, W, H);
        }
        let bytes = out.into_data();
        let mut stream = MemIStream::new(bytes);
        let file = MultiPartInputFile::new(&mut stream).expect("mp in");

        let chans = get_channels_in_multi_part_file(&file).expect("get_channels");
        assert_eq!(chans.len(), 4);
        assert!(chans.iter().all(|c| c.view.is_empty()));
        assert!(chans.iter().all(|c| c.name == c.internal_name));
        assert_eq!(chans.iter().filter(|c| c.part_number == 0).count(), 2);
        assert_eq!(chans.iter().filter(|c| c.part_number == 1).count(), 2);
    }

    #[test]
    fn owned_and_borrowed_channel_enumeration_match() {
        for multiview in [false, true] {
            let channels = ["G", "right.R"];
            let headers = if multiview {
                vec![part_header(
                    "view",
                    2,
                    2,
                    &channels,
                    Some(vec!["left", "right"]),
                )]
            } else {
                vec![
                    part_header("first", 2, 2, &channels, None),
                    part_header("second", 2, 2, &channels, None),
                ]
            };
            let mut stream = MemOStream::new();
            {
                let mut output = MultiPartOutputFile::new(&mut stream, &headers).unwrap();
                for part in 0..headers.len() {
                    write_part(&mut output, part as i32, &channels, 2, 2);
                }
            }
            let bytes = stream.into_data();
            let mut borrowed_stream = MemIStream::new(bytes.clone());
            let borrowed = MultiPartInputFile::new(&mut borrowed_stream).unwrap();
            let owned = MultiPartInputFile::new_owned(Box::new(MemIStream::new(bytes))).unwrap();
            assert_eq!(
                get_channels_in_multi_part_file(&borrowed).unwrap(),
                get_channels_in_multi_part_file(&owned).unwrap()
            );
        }
    }
}
