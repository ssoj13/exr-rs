//! Golden hashes of the encoded blob BYTES for a fixed deterministic plane set across every profile and dtype.
//!
//! Why: the codec dispatch (registry) is a refactor-prone seam; the on-disk bytes must never drift by accident
//! (determinism is a format invariant). A change that legitimately alters blobs must update `GOLDEN` knowingly.
//! Entries are keyed by name, so a `--no-default-features` build (no HTJ2K) simply checks the subset it has.

use exv_codec::{Profile, encode_channel, encode_tile};
use exv_format::Dtype;

const W: u32 = 48;
const H: u32 = 40;

/// FNV-1a 64: tiny, dependency-free, stable across platforms.
fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Deterministic smooth-ish plane of `dtype` (gradient + xorshift noise) so every codec actually compresses it.
fn plane(dtype: Dtype, seed: u64) -> Vec<u8> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ seed;
    let mut noise = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s & 0xf) as u32
    };
    let n = (W * H) as usize;
    let mut out = Vec::new();
    for i in 0..n {
        let (x, y) = ((i as u32) % W, (i as u32) / W);
        let g = x * 3 + y * 5 + noise();
        match dtype {
            Dtype::U8 => out.push((g & 0xff) as u8),
            Dtype::U16 => out.extend_from_slice(&((g * 37) as u16).to_le_bytes()),
            Dtype::U32 => out.extend_from_slice(&(g * 1009).to_le_bytes()),
            // finite f16 just above 1.0 (exponent 0x3c00): smooth + never inf/NaN
            Dtype::F16 => out.extend_from_slice(&(0x3c00u16 + (g as u16 & 0x3ff)).to_le_bytes()),
            Dtype::Bf16 => out.extend_from_slice(&(0x3f80u16 + (g as u16 & 0x7f)).to_le_bytes()),
            Dtype::F32 => out.extend_from_slice(&(1.0f32 + g as f32 * 0.0137).to_le_bytes()),
        }
    }
    out
}

fn profiles() -> Vec<(&'static str, Profile)> {
    let mut v = vec![
        ("fast", Profile::lossless_fast()),
        ("fast_maxratio", Profile::lossless_fast().max_ratio()),
        ("wave", Profile::lossless_wave()),
        ("wave_maxratio", Profile::lossless_wave().max_ratio()),
        ("lossy_wave8", Profile::lossy_wave(8)),
        ("rt_lerp", Profile::exv_rt(4, 4, true)),
        ("rt_dwt_q2", Profile::exv_rt_dwt(16, 2)),
        ("rt_dwt_q0", Profile::exv_rt_dwt(16, 0)),
    ];
    #[cfg(feature = "htj2k-backend")]
    {
        v.push(("htj2k", Profile::htj2k()));
        v.push(("htj2k_lossy8", Profile::htj2k_lossy(8)));
    }
    v
}

const DTYPES: [Dtype; 6] = [Dtype::F16, Dtype::Bf16, Dtype::F32, Dtype::U8, Dtype::U16, Dtype::U32];

/// Hash table of every (profile, dtype) channel blob + one mixed-dtype tile blob per profile.
fn compute() -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let planes: Vec<(Dtype, Vec<u8>)> = DTYPES.iter().enumerate().map(|(i, &d)| (d, plane(d, i as u64))).collect();
    for (name, prof) in profiles() {
        for (d, p) in &planes {
            let blob = encode_channel(p, *d, W, H, &prof).expect("encode_channel");
            out.push((format!("{name}/{d:?}"), fnv(&blob)));
        }
        let chans: Vec<&[u8]> = planes.iter().map(|(_, p)| p.as_slice()).collect();
        let dts: Vec<Dtype> = planes.iter().map(|(d, _)| *d).collect();
        let tile = encode_tile(&chans, &dts, W, H, &prof).expect("encode_tile");
        out.push((format!("{name}/tile"), fnv(&tile)));
    }
    out
}

/// `(name, fnv1a64(blob))` recorded BEFORE the registry refactor (S2 of design_registry.md).
const GOLDEN: &[(&str, u64)] = &[
    ("fast/F16", 0x31eb36f54cc2135e),
    ("fast/Bf16", 0xe78b909609f10aaf),
    ("fast/F32", 0x1c9bb7cbabd3a10e),
    ("fast/U8", 0xb7c5b58c92e2f7e6),
    ("fast/U16", 0xac0a7a6cf6fcf674),
    ("fast/U32", 0xfe72497adf725c5a),
    ("fast/tile", 0xf29536380504ddb5),
    ("fast_maxratio/F16", 0x78b6fbead3a63f94),
    ("fast_maxratio/Bf16", 0x8cc81266e7fb2e2c),
    ("fast_maxratio/F32", 0x11910d08e6293b47),
    ("fast_maxratio/U8", 0xb7c5b58c92e2f7e6),
    ("fast_maxratio/U16", 0xac1256b2bbe43fc8),
    ("fast_maxratio/U32", 0xb4b8a8a30b6275a7),
    ("fast_maxratio/tile", 0xd9afca82cbdfb035),
    ("wave/F16", 0xd2a301e58f553ce1),
    ("wave/Bf16", 0x16281cf1e13e634f),
    ("wave/F32", 0xac74a058c4a0138f),
    ("wave/U8", 0x41fe500284a79c2d),
    ("wave/U16", 0xb5fd80ac68e880e1),
    ("wave/U32", 0xf195096d185f463f),
    ("wave/tile", 0x7bdd27d5e5cea984),
    ("wave_maxratio/F16", 0xd2a301e58f553ce1),
    ("wave_maxratio/Bf16", 0x16281cf1e13e634f),
    ("wave_maxratio/F32", 0xac74a058c4a0138f),
    ("wave_maxratio/U8", 0x41fe500284a79c2d),
    ("wave_maxratio/U16", 0xb5fd80ac68e880e1),
    ("wave_maxratio/U32", 0xf195096d185f463f),
    ("wave_maxratio/tile", 0x7bdd27d5e5cea984),
    ("lossy_wave8/F16", 0xafc5c9d98666839d),
    ("lossy_wave8/Bf16", 0x1b4efa6b70f987f4),
    ("lossy_wave8/F32", 0x58506d52c08696d4),
    ("lossy_wave8/U8", 0x8dd1188a963385da),
    ("lossy_wave8/U16", 0x990c224b96258dcc),
    ("lossy_wave8/U32", 0xe2abc535432dd87e),
    ("lossy_wave8/tile", 0x3e6344f350e257ef),
    ("rt_lerp/F16", 0x6e257a49d3209253),
    ("rt_lerp/Bf16", 0x331a3646653127c4),
    ("rt_lerp/F32", 0x5a37172b1ebc2183),
    ("rt_lerp/U8", 0x8dd1188a963385da),
    ("rt_lerp/U16", 0x990c224b96258dcc),
    ("rt_lerp/U32", 0x09e858a703e00533),
    ("rt_lerp/tile", 0xce4fceb694d9c2b4),
    ("rt_dwt_q2/F16", 0x9bb1e8c4ce424dca),
    ("rt_dwt_q2/Bf16", 0x331a3646653127c4),
    ("rt_dwt_q2/F32", 0x5a37172b1ebc2183),
    ("rt_dwt_q2/U8", 0x8dd1188a963385da),
    ("rt_dwt_q2/U16", 0x990c224b96258dcc),
    ("rt_dwt_q2/U32", 0x09e858a703e00533),
    ("rt_dwt_q2/tile", 0xc75e06c2ef01a968),
    ("rt_dwt_q0/F16", 0xd58645e28948ea1b),
    ("rt_dwt_q0/Bf16", 0x331a3646653127c4),
    ("rt_dwt_q0/F32", 0x5a37172b1ebc2183),
    ("rt_dwt_q0/U8", 0x8dd1188a963385da),
    ("rt_dwt_q0/U16", 0x990c224b96258dcc),
    ("rt_dwt_q0/U32", 0x09e858a703e00533),
    ("rt_dwt_q0/tile", 0x8f3f12d6ae101424),
    ("htj2k/F16", 0x750790d2e3c30abe),
    ("htj2k/Bf16", 0xdcdfb22dffc876d4),
    ("htj2k/F32", 0xd2a7aee89ffb5c95),
    ("htj2k/U8", 0xa4384e9c098534c9),
    ("htj2k/U16", 0xd863b8cb0911e5b5),
    ("htj2k/U32", 0x9897ae7d8f86f67e),
    ("htj2k/tile", 0x5ae3872b7c46961f),
    ("htj2k_lossy8/F16", 0x3552b0d71e39be15),
    ("htj2k_lossy8/Bf16", 0x6b41b012501032eb),
    ("htj2k_lossy8/F32", 0xd5d92b2e560b1426),
    ("htj2k_lossy8/U8", 0xa4384e9c098534c9),
    ("htj2k_lossy8/U16", 0xd863b8cb0911e5b5),
    ("htj2k_lossy8/U32", 0x9897ae7d8f86f67e),
    ("htj2k_lossy8/tile", 0x7a8bab21be097c6b),
];

#[test]
fn blob_bytes_are_stable() {
    let got = compute();
    let mut bad = Vec::new();
    for (k, h) in &got {
        match GOLDEN.iter().find(|(n, _)| n == k) {
            Some((_, g)) if g == h => {}
            _ => bad.push(format!("    (\"{k}\", {h:#018x}),")),
        }
    }
    assert!(bad.is_empty(), "blob bytes drifted / missing golden:\n{}", bad.join("\n"));
}
