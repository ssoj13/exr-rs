//! Golden container bytes: `write_video` output is pinned (length + FNV-1a-64) over a grid of
//! profile x gop x MV search x B-frames x tile grid. The writer refactor (ClipWriter, parallel frame
//! encode, streaming) MUST reproduce these exactly: the format is frozen, only the code path changes.
//!
//! Last regenerated: container v3 (CRC-32C per coded blob in the frame index, FORMAT.md 5): every length grew by
//! exactly `frames * (2 + 4 * tiles * n_crc)` (n_crc = 3 channels per-channel, 1 HTJ2K): t16 +518 / HTJ2K +182,
//! t64 +98 / HTJ2K +42; the tile blobs themselves are unchanged (exv-codec `golden_blobs` untouched).
//!
//! Regenerate ONLY on an intentional format/codec change: `EXV_GOLDEN_PRINT=1 cargo test -p exv-video
//! --test golden_bytes -- --nocapture` prints the table.

mod common;

use exv_codec::Profile;
use exv_codec::image::Image;
use exv_video::write_video;

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn profiles() -> Vec<(&'static str, Profile)> {
    let mut v = vec![
        ("fast", Profile::lossless_fast()),
        ("wave", Profile::lossless_wave()),
        ("lossy_wave2", Profile::lossy_wave(2)),
    ];
    #[cfg(feature = "htj2k-backend")]
    v.push(("htj2k", Profile::htj2k()));
    v
}

/// `(key, len, fnv)` of every grid point, in a fixed order.
fn grid() -> Vec<(String, usize, u64)> {
    let mut out = Vec::new();
    for (name, prof) in profiles() {
        for tile in [16u32, 64] {
            let frames: Vec<Image> = common::clip(7, tile);
            for gop in [1u32, 4, 8] {
                for search in [0i32, 4] {
                    for b in [0u32, 1] {
                        let enc = write_video(&frames, &prof, gop, search, b).expect("encode");
                        out.push((format!("{name}/t{tile}/g{gop}/s{search}/b{b}"), enc.len(), fnv(&enc)));
                    }
                }
            }
        }
    }
    out
}

#[test]
fn golden_bytes() {
    let got = grid();
    if std::env::var_os("EXV_GOLDEN_PRINT").is_some() {
        for (k, l, h) in &got {
            println!("    (\"{k}\", {l}, {h:#018x}),");
        }
        return;
    }
    for (k, l, h) in &got {
        let want = GOLDEN.iter().find(|(gk, ..)| gk == k).unwrap_or_else(|| panic!("no golden entry for {k}"));
        assert_eq!((*l, *h), (want.1, want.2), "container bytes changed for {k}");
    }
}

const GOLDEN: &[(&str, usize, u64)] = &[
    ("fast/t16/g1/s0/b0", 8133, 0x82f99cc6484b0aec),
    ("fast/t16/g1/s0/b1", 8133, 0x82f99cc6484b0aec),
    ("fast/t16/g1/s4/b0", 8133, 0x82f99cc6484b0aec),
    ("fast/t16/g1/s4/b1", 8133, 0x82f99cc6484b0aec),
    ("fast/t16/g4/s0/b0", 6340, 0x7a0e9ef4d29f963e),
    ("fast/t16/g4/s0/b1", 6125, 0x5499b18797e60353),
    ("fast/t16/g4/s4/b0", 3987, 0xb799b91fccbf06f5),
    ("fast/t16/g4/s4/b1", 4295, 0xc9d70304baf6ebff),
    ("fast/t16/g8/s0/b0", 5980, 0xaa6374d1e1e021a8),
    ("fast/t16/g8/s0/b1", 5642, 0x69e711d63ad5740e),
    ("fast/t16/g8/s4/b0", 3172, 0x7f1f10d6ac5f408b),
    ("fast/t16/g8/s4/b1", 3628, 0x9444a17561abae1d),
    ("fast/t64/g1/s0/b0", 50118, 0x3d36d659f751b6db),
    ("fast/t64/g1/s0/b1", 50118, 0x3d36d659f751b6db),
    ("fast/t64/g1/s4/b0", 50118, 0x3d36d659f751b6db),
    ("fast/t64/g1/s4/b1", 50118, 0x3d36d659f751b6db),
    ("fast/t64/g4/s0/b0", 39253, 0x9b063a7b8d64125a),
    ("fast/t64/g4/s0/b1", 39624, 0x42bf62e35afddc1c),
    ("fast/t64/g4/s4/b0", 39218, 0xb1892302b523b2a8),
    ("fast/t64/g4/s4/b1", 39824, 0x785f36381b03cc27),
    ("fast/t64/g8/s0/b0", 37074, 0x9e3a50bf66617f99),
    ("fast/t64/g8/s0/b1", 37625, 0xc9a14983d8dba628),
    ("fast/t64/g8/s4/b0", 37042, 0x6077c126fb2fdf70),
    ("fast/t64/g8/s4/b1", 37949, 0x2027adf2ed998ce9),
    ("wave/t16/g1/s0/b0", 29911, 0x5d045e88ec48e8eb),
    ("wave/t16/g1/s0/b1", 29911, 0x5d045e88ec48e8eb),
    ("wave/t16/g1/s4/b0", 29911, 0x5d045e88ec48e8eb),
    ("wave/t16/g1/s4/b1", 29911, 0x5d045e88ec48e8eb),
    ("wave/t16/g4/s0/b0", 17904, 0xad27f14f43ccc6b4),
    ("wave/t16/g4/s0/b1", 17373, 0x2b166fd51acc7041),
    ("wave/t16/g4/s4/b0", 10154, 0x80a6539848f86955),
    ("wave/t16/g4/s4/b1", 10814, 0x82a87f393ce0e6bb),
    ("wave/t16/g8/s0/b0", 15547, 0x3f15a43acda0a81f),
    ("wave/t16/g8/s0/b1", 14780, 0x8711de83b53c96a1),
    ("wave/t16/g8/s4/b0", 6311, 0xed58058b075ac894),
    ("wave/t16/g8/s4/b1", 7242, 0x8a6ebffae9a0cdc1),
    ("wave/t64/g1/s0/b0", 55333, 0x37c3f4b64c820f40),
    ("wave/t64/g1/s0/b1", 55333, 0x37c3f4b64c820f40),
    ("wave/t64/g1/s4/b0", 55333, 0x37c3f4b64c820f40),
    ("wave/t64/g1/s4/b1", 55333, 0x37c3f4b64c820f40),
    ("wave/t64/g4/s0/b0", 48890, 0x1ba8a74efa094718),
    ("wave/t64/g4/s0/b1", 47237, 0x5a0a3b3da1061cc7),
    ("wave/t64/g4/s4/b0", 39950, 0x309c5c2ae10f972d),
    ("wave/t64/g4/s4/b1", 41140, 0x1742d78bb5524f0a),
    ("wave/t64/g8/s0/b0", 47685, 0x8ab065023196bd86),
    ("wave/t64/g8/s0/b1", 45206, 0x265a94747e81944f),
    ("wave/t64/g8/s4/b0", 36981, 0x4a8d7fba6304d641),
    ("wave/t64/g8/s4/b1", 38753, 0xe73922c3963bb3f2),
    ("lossy_wave2/t16/g1/s0/b0", 87964, 0x033a2eb0a25b7f33),
    ("lossy_wave2/t16/g1/s0/b1", 87964, 0x033a2eb0a25b7f33),
    ("lossy_wave2/t16/g1/s4/b0", 87964, 0x033a2eb0a25b7f33),
    ("lossy_wave2/t16/g1/s4/b1", 87964, 0x033a2eb0a25b7f33),
    ("lossy_wave2/t16/g4/s0/b0", 88004, 0x05689ba9e01fad81),
    ("lossy_wave2/t16/g4/s0/b1", 88004, 0x05689ba9e01fad81),
    ("lossy_wave2/t16/g4/s4/b0", 88004, 0x05689ba9e01fad81),
    ("lossy_wave2/t16/g4/s4/b1", 88004, 0x05689ba9e01fad81),
    ("lossy_wave2/t16/g8/s0/b0", 88012, 0x9d7da1a80fe06a89),
    ("lossy_wave2/t16/g8/s0/b1", 88012, 0x9d7da1a80fe06a89),
    ("lossy_wave2/t16/g8/s4/b0", 88012, 0x9d7da1a80fe06a89),
    ("lossy_wave2/t16/g8/s4/b1", 88012, 0x9d7da1a80fe06a89),
    ("lossy_wave2/t64/g1/s0/b0", 52669, 0x7154e4dce2e71226),
    ("lossy_wave2/t64/g1/s0/b1", 52669, 0x7154e4dce2e71226),
    ("lossy_wave2/t64/g1/s4/b0", 52669, 0x7154e4dce2e71226),
    ("lossy_wave2/t64/g1/s4/b1", 52669, 0x7154e4dce2e71226),
    ("lossy_wave2/t64/g4/s0/b0", 52709, 0xa7020f3692f63898),
    ("lossy_wave2/t64/g4/s0/b1", 52709, 0xa7020f3692f63898),
    ("lossy_wave2/t64/g4/s4/b0", 52709, 0xa7020f3692f63898),
    ("lossy_wave2/t64/g4/s4/b1", 52709, 0xa7020f3692f63898),
    ("lossy_wave2/t64/g8/s0/b0", 52717, 0x126ed8822ad7af74),
    ("lossy_wave2/t64/g8/s0/b1", 52717, 0x126ed8822ad7af74),
    ("lossy_wave2/t64/g8/s4/b0", 52717, 0x126ed8822ad7af74),
    ("lossy_wave2/t64/g8/s4/b1", 52717, 0x126ed8822ad7af74),
    ("htj2k/t16/g1/s0/b0", 33475, 0x08ed6fe1c633c640),
    ("htj2k/t16/g1/s0/b1", 33475, 0x08ed6fe1c633c640),
    ("htj2k/t16/g1/s4/b0", 33475, 0x08ed6fe1c633c640),
    ("htj2k/t16/g1/s4/b1", 33475, 0x08ed6fe1c633c640),
    ("htj2k/t16/g4/s0/b0", 26484, 0xbc9af11a61b2a0b4),
    ("htj2k/t16/g4/s0/b1", 26422, 0x8d409558aea8d4f5),
    ("htj2k/t16/g4/s4/b0", 14276, 0x4951764de35f0d59),
    ("htj2k/t16/g4/s4/b1", 16907, 0x9479ca1b47381dda),
    ("htj2k/t16/g8/s0/b0", 25099, 0xa7472010d76f9d09),
    ("htj2k/t16/g8/s0/b1", 24935, 0xb48d0847ce5d44c4),
    ("htj2k/t16/g8/s4/b0", 10475, 0xcad12f53352e0b6c),
    ("htj2k/t16/g8/s4/b1", 14316, 0x5a05b07ad5c903b7),
    ("htj2k/t64/g1/s0/b0", 15515, 0xe15a4bf303c589cc),
    ("htj2k/t64/g1/s0/b1", 15515, 0xe15a4bf303c589cc),
    ("htj2k/t64/g1/s4/b0", 15515, 0xe15a4bf303c589cc),
    ("htj2k/t64/g1/s4/b1", 15515, 0xe15a4bf303c589cc),
    ("htj2k/t64/g4/s0/b0", 13700, 0xa40d1973ef27e2e5),
    ("htj2k/t64/g4/s0/b1", 14057, 0x963962d61d999666),
    ("htj2k/t64/g4/s4/b0", 7264, 0xde90479f308f7ec7),
    ("htj2k/t64/g4/s4/b1", 8654, 0x8b4c6e4bb12c9dfc),
    ("htj2k/t64/g8/s0/b0", 13347, 0xf9bb998ec8c7ae74),
    ("htj2k/t64/g8/s0/b1", 13867, 0x89ec87f4f0d9a22d),
    ("htj2k/t64/g8/s4/b0", 5670, 0x94a23837c0b8222d),
    ("htj2k/t64/g8/s4/b1", 7730, 0x502f60e1eb816c7b),
];
